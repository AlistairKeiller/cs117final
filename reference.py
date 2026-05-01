"""
Cartographer-style 2D graph SLAM for F1Tenth.
Front-end (GPU/Warp): ray-casting, optional CSM, refinement-cost evaluation, BBS pyramids.
Back-end (CPU/GTSAM): pose graph with intra-submap and loop-closure edges, LM optimization.

Closely follows Hess et al. "Real-Time Loop Closure in 2D LIDAR SLAM" (ICRA 2016)
and the cartographer-project source. Notable choices:

  Map updates (probability_grid_range_data_inserter_2d.cc::ApplyLookupTable):
    Each cell receives at most ONE update per scan, with hits taking priority over
    misses. Implemented in two GPU passes: k_mark walks rays and atomic-max's a
    `seen` grid (HIT=2 always wins over MISS=1); k_apply_marks then applies the
    correct delta exactly once per cell, clamps the log-odds, and resets `seen`.

  BBS (paper Eq. 17):
    Same-resolution sliding-window-max precomputed grids. M_h has the same shape
    as M_0; cell M_h[i,j] = max over [i, i+2^h) x [j, j+2^h) of M_0. The upper
    bound is exact for arbitrarily-positioned candidates -- no coarse-grid
    alignment trick required.

  BBS angular step (paper Eq. 7):
    dtheta = arccos(1 - r^2 / (2 * d_max^2))
    chosen so that scan points at d_max move <= one cell per angular step.

  Ceres-style refinement (CeresScanMatcher2D::Match):
    Cost = (occupied_space_weight / sqrt(N)) * sum_k (1 - M_smooth(T h_k))^2
         + translation_weight^2 * ||T.xy - prior.xy||^2
         + rotation_weight^2 * wrap(T.theta - prior.theta)^2
    The 1/sqrt(N) scaling on the occupied-space cost is critical: it keeps the
    prior weights consistent across different scan sizes. Same cost is used for
    local SLAM (prior = extrapolated odometry / CSM) and loop closure
    (prior = BBS hit).

  Submap rotation (active_submaps_2d.cc::InsertRangeData):
    Each submap receives 2 * NUM_RANGE_DATA scans -- the first NUM_RANGE_DATA
    while it's the "new" submap, the next NUM_RANGE_DATA while it's the "old"
    matching target. A scan goes into BOTH active submaps simultaneously.

  Adaptive voxel filter (paper IV-D, sensor::AdaptiveVoxelFilter):
    Reduces a ~1080-beam Hokuyo to <= MATCH_VOXEL_MAX_POINTS for local matching
    and <= LC_VOXEL_MAX_POINTS for BBS / loop-closure refinement. The full scan
    is still inserted into submaps for map fidelity. Filtering happens once per
    scan at the Slam level; matchers assume input is pre-filtered.

  Motion filter (mapping::MotionFilter):
    Scans within MIN_MOTION_DISTANCE / MIN_MOTION_ANGLE of the last accepted
    scan are matched but NOT inserted as submap data nor added as graph nodes.
    An idle-scan timeout (MAX_SCANS_BETWEEN_NODES) forces acceptance during
    long stationary periods so the map stays current.

  Loop-closure search (constraint_builder_2d.cc):
    Sweep over recent nodes against finalized submaps with spatial gating.
    Successful pairs are recorded in _lc_found and never re-added (avoids
    duplicate edges); unsuccessful pairs in _lc_attempted are retried after
    each global optimization (since pose updates change gating decisions).

  CSM (RealTimeCorrelativeScanMatcher2D::ScoreCandidate):
    Nearest-neighbor probability lookup (matching ProbabilityGrid::GetProbability).
    Refinement uses bicubic-spline interpolation for sub-cell-accurate gradients,
    with the prefilter cached on finalized submaps for fast LC.
"""

import gtsam
import numpy as np
import warp as wp
from scipy.ndimage import map_coordinates, spline_filter
from scipy.optimize import least_squares

wp.init()
DEV = "cuda" if wp.is_cuda_available() else "cpu"

# =============================================================================
# Configuration
# =============================================================================
# Resolution and grid
RES = 0.05  # m / cell
L_HIT, L_MISS = 0.85, -0.40
L_MIN, L_MAX = -2.2, 2.2
GRID_HW = 400  # 20m x 20m submap

# Submap budget (Cartographer 2D: 2 * num_range_data)
NUM_RANGE_DATA = 90
SUBMAP_TOTAL_SCANS = 2 * NUM_RANGE_DATA  # 180

# Adaptive voxel filter sizes
# Local matching uses a denser sampling; LC uses a sparser one to keep BBS fast.
# (Cartographer: adaptive_voxel_filter_options.max_num_points = 200,
#  loop_closure_adaptive_voxel_filter_options.max_num_points = 100.)
MATCH_VOXEL_MAX_POINTS = 200
LC_VOXEL_MAX_POINTS = 100
VOXEL_MIN_SIZE = 0.025  # m (~half a cell)
VOXEL_MAX_SIZE = 0.5  # m

# Motion filter
MIN_MOTION_DISTANCE = 0.20  # m
MIN_MOTION_ANGLE = np.deg2rad(1.0)  # rad
MAX_SCANS_BETWEEN_NODES = 100  # idle-scan timeout (~2s at 50Hz)

# Local scan matching
USE_CSM = False  # off by default (Cartographer also defaults this off)
CSM_WIN_XY, CSM_STEP_XY = 0.15, 0.025
CSM_WIN_TH, CSM_STEP_TH = 0.17, 0.005
CSM_MIN_SCORE = 0.30  # below this, fall back to extrapolated pose for refinement init
REFINE_ITERS = 20
# Cartographer defaults: occupied_space_weight=1, translation_weight=10, rotation_weight=40.
# Critical: the occupied-space residual is divided by sqrt(N), so these weights stay
# meaningful regardless of scan size. See refine().
REFINE_W_T = 10.0
REFINE_W_R = 40.0

# BBS / loop closure
BBS_LEVELS = 6  # precomputed grid levels
LC_MIN_SCORE = 0.55
LC_WIN_XY = 2.0  # m, half-window for local LC search
LC_WIN_TH = 0.5  # rad, half-window for local LC search
LC_MAX_DISTANCE = 15.0  # m, spatial gating for LC pairs

# Pose graph noise (sigmas)
SIGMA_INTRA = np.array([0.025, 0.025, 0.005])
SIGMA_LC = np.array([0.05, 0.05, 0.02])
SIGMA_PRIOR = np.array([1e-6, 1e-6, 1e-8])
HUBER_K = 1.0

# Loop-closure scheduling
OPTIMIZE_EVERY = 30
LC_SKIP_NODES = 50
LC_RECENT_WINDOW = 50


# =============================================================================
# SE(2) helpers (host)
# =============================================================================
def wrap(a):
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    """SE(2) composition a . b (rotate b's translation by a, sum)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array(
        [a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])]
    )


def inverse(a):
    """SE(2) inverse: T such that compose(a, T) == identity."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]])


def between(a, b):
    """Relative pose: b expressed in a's frame, == compose(inverse(a), b)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])


# =============================================================================
# Adaptive voxel filter (sensor::AdaptiveVoxelFilter)
# =============================================================================
def _voxel_filter(scan, voxel_size):
    """Bin scan points into voxels of given edge length; keep one per voxel.

    Uses int64 packing of the 2D cell key for fast np.unique() (avoids axis=0,
    which is O(N log N) sort with row comparisons).
    """
    if len(scan) == 0:
        return scan
    keys = np.floor(scan / voxel_size).astype(np.int64)
    packed = (keys[:, 0] << 32) | (keys[:, 1] & 0xFFFFFFFF)
    _, idx = np.unique(packed, return_index=True)
    return scan[idx]


def adaptive_voxel_filter(
    scan,
    max_points,
    min_size=VOXEL_MIN_SIZE,
    max_size=VOXEL_MAX_SIZE,
):
    """Reduce a scan to <= max_points by binary-searching the voxel size.

    Returns the smallest-voxel result that fits the budget (preserves more
    detail than picking an arbitrarily larger size). If even max_size doesn't
    fit the budget, returns the result at max_size (best effort).
    """
    if len(scan) <= max_points:
        return scan
    if len(_voxel_filter(scan, min_size)) <= max_points:
        return _voxel_filter(scan, min_size)

    lo, hi = min_size, max_size
    # 7 binary-search iterations -> resolution ~ (max-min)/128 ~ 4mm
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        if len(_voxel_filter(scan, mid)) <= max_points:
            hi = mid
        else:
            lo = mid
    return _voxel_filter(scan, hi)


# =============================================================================
# Motion filter (mapping::MotionFilter)
# =============================================================================
class MotionFilter:
    """Drop scans whose pose is too close to the last accepted scan.

    A scan is "similar" (drop) if BOTH translation and rotation deltas are below
    thresholds. Either condition alone allows the scan through. The Slam owner
    layers an idle-scan timeout (MAX_SCANS_BETWEEN_NODES) on top of this.
    """

    def __init__(self, min_distance=MIN_MOTION_DISTANCE, min_angle=MIN_MOTION_ANGLE):
        self.min_distance = min_distance
        self.min_angle = min_angle
        self.last = None

    def is_similar(self, pose):
        if self.last is None:
            return False
        d = float(np.linalg.norm(pose[:2] - self.last[:2]))
        a = abs(wrap(pose[2] - self.last[2]))
        return d < self.min_distance and a < self.min_angle

    def update(self, pose):
        self.last = pose.copy()


# =============================================================================
# Warp kernels (GPU)
# =============================================================================


@wp.kernel
def k_mark(
    pts: wp.array(dtype=wp.vec2),
    origin: wp.vec2,
    seen: wp.array2d(dtype=wp.float32),
    gx: float,
    gy: float,
    res: float,
    H: int,
    W: int,
):
    """Bresenham ray-walk that MARKS cells in `seen` (no log update yet).

    Pass 1 of Cartographer's per-scan update (ApplyLookupTable). Each cell gets
    at most one update per scan, with hits dominating misses regardless of ray
    arrival order. Encoded with atomic_max:
      - Cells along a ray (excluding sensor and endpoint): atomic_max(1.0) [MISS]
      - Endpoint cell:                                     atomic_max(2.0) [HIT]

    The sensor's own cell is excluded by the pre-step before the loop.
    """
    i = wp.tid()
    p = pts[i]
    x0 = int((origin[0] - gx) / res)
    y0 = int((origin[1] - gy) / res)
    x1 = int((p[0] - gx) / res)
    y1 = int((p[1] - gy) / res)
    dx = x1 - x0
    dy = y1 - y0
    sx = 1
    sy = 1
    if dx < 0:
        sx = -1
        dx = -dx
    if dy < 0:
        sy = -1
        dy = -dy
    err = dx - dy
    x = x0
    y = y0

    # Pre-step once so the loop body never touches the sensor's own cell.
    e2 = 2 * err
    if e2 > -dy:
        err -= dy
        x += sx
    if e2 < dx:
        err += dx
        y += sy

    for _ in range(dx + dy):
        if x == x1 and y == y1:
            break
        if x >= 0 and x < W and y >= 0 and y < H:
            wp.atomic_max(seen, y, x, 1.0)
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    if x1 >= 0 and x1 < W and y1 >= 0 and y1 < H:
        wp.atomic_max(seen, y1, x1, 2.0)


@wp.kernel
def k_apply_marks(
    seen: wp.array2d(dtype=wp.float32),
    log: wp.array2d(dtype=wp.float32),
    lhit: float,
    lmiss: float,
    lo: float,
    hi: float,
):
    """Pass 2 of the per-scan update: apply marks, clamp log, reset seen.

      seen >= 1.5  -> log += lhit  (HIT)
      seen >= 0.5  -> log += lmiss (MISS)
      else         -> no change
    Then clamp log to [lo, hi] and zero seen for the next scan. One thread per
    cell; no atomics needed.
    """
    i, j = wp.tid()
    s = seen[i, j]
    v = log[i, j]
    if s > 1.5:
        v += lhit
    elif s > 0.5:
        v += lmiss
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    log[i, j] = v
    seen[i, j] = 0.0


@wp.kernel
def k_logodds_to_prob(
    lo: wp.array2d(dtype=wp.float32), pr: wp.array2d(dtype=wp.float32)
):
    """Element-wise sigmoid: pr = 1 / (1 + exp(-lo))."""
    i, j = wp.tid()
    pr[i, j] = 1.0 / (1.0 + wp.exp(-lo[i, j]))


@wp.kernel
def k_csm(
    pts: wp.array(dtype=wp.vec2),
    grid: wp.array2d(dtype=wp.float32),
    gx: float,
    gy: float,
    res: float,
    H: int,
    W: int,
    px: float,
    py: float,
    pth: float,
    nx: int,
    ny: int,
    nth: int,
    sxy: float,
    sth: float,
    scores: wp.array3d(dtype=wp.float32),
):
    """Score every (dtheta, dy, dx) candidate around prior with NEAREST-neighbor
    probability lookup. Matches RealTimeCorrelativeScanMatcher2D::ScoreCandidate
    which uses ProbabilityGrid::GetProbability (nearest). One thread per candidate.
    """
    it, iy, ix = wp.tid()
    dth = (float(it) - float(nth) / 2.0) * sth
    dx = (float(ix) - float(nx) / 2.0) * sxy
    dy = (float(iy) - float(ny) / 2.0) * sxy
    th = pth + dth
    cx = px + dx
    cy = py + dy
    c = wp.cos(th)
    s = wp.sin(th)
    n = pts.shape[0]
    acc = float(0.0)
    for k in range(n):
        p = pts[k]
        wx = cx + c * p[0] - s * p[1]
        wy = cy + s * p[0] + c * p[1]
        cell_x = int((wx - gx) / res)
        cell_y = int((wy - gy) / res)
        if cell_x >= 0 and cell_x < W and cell_y >= 0 and cell_y < H:
            acc += grid[cell_y, cell_x]
    scores[it, iy, ix] = acc / float(n)


@wp.kernel
def k_pyramid_step(
    src: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    offset: int,
    H: int,
    W: int,
):
    """Cartographer's same-resolution precomputation grid (paper Eq. 17).

    M_h[i, j] = max over [i, i+2^h) x [j, j+2^h) of M_0
              = max(M_{h-1}[i, j],
                    M_{h-1}[i+offset, j],
                    M_{h-1}[i, j+offset],
                    M_{h-1}[i+offset, j+offset])
    where offset = 2^{h-1}. Out-of-bounds reads return 0.

    Output has the SAME shape as M_0; this is what makes the BBS upper bound
    exact for arbitrary candidate positions.
    """
    i, j = wp.tid()
    a = src[i, j]
    b = float(0.0)
    c = float(0.0)
    d = float(0.0)
    if i + offset < H:
        b = src[i + offset, j]
    if j + offset < W:
        c = src[i, j + offset]
    if i + offset < H and j + offset < W:
        d = src[i + offset, j + offset]
    m = a
    if b > m:
        m = b
    if c > m:
        m = c
    if d > m:
        m = d
    dst[i, j] = m


@wp.kernel
def k_bbs_score(
    pts: wp.array(dtype=wp.vec2),
    grid: wp.array2d(dtype=wp.float32),
    gx: float,
    gy: float,
    res: float,
    H: int,
    W: int,
    cands: wp.array(dtype=wp.vec3),
    scores: wp.array(dtype=wp.float32),
):
    """Score (x, y, theta) candidates against a precomputed grid (nearest-neighbor).

    Same indexing as M_0 (no level-dependent stride): the precomputed grid
    handles spatial windowing internally via the sliding-window max.
    """
    k = wp.tid()
    cand = cands[k]
    cx = cand[0]
    cy = cand[1]
    th = cand[2]
    c = wp.cos(th)
    s = wp.sin(th)
    n = pts.shape[0]
    acc = float(0.0)
    for i in range(n):
        p = pts[i]
        wx = cx + c * p[0] - s * p[1]
        wy = cy + s * p[0] + c * p[1]
        ix = int((wx - gx) / res)
        iy = int((wy - gy) / res)
        if ix >= 0 and ix < W and iy >= 0 and iy < H:
            acc += grid[iy, ix]
    scores[k] = acc / float(n)


# =============================================================================
# Submap: log-odds grid + probability grid + (when finalized) precomputed grids
# =============================================================================
class Submap:
    """A 2D occupancy submap.

    Origin is the world pose at submap creation. The grid is in submap-LOCAL
    coordinates: cell (0, 0) sits at submap-local (-W*r/2, -H*r/2). When the
    submap origin moves due to global optimization, the grid contents stay
    fixed in submap-local coords -- only the origin pose moves in the world.

    Lifecycle:
      * Active: log grid is updated as scans arrive; prob is rebuilt on demand.
      * Finalized: log/seen freed, BBS pyramid built, bicubic spline prefilter
        cached for fast LC refinement.
    """

    _next_id = 0

    def __init__(self, origin_pose):
        self.id = Submap._next_id
        Submap._next_id += 1
        self.origin = origin_pose.copy()
        self.gx = -GRID_HW * RES / 2.0
        self.gy = -GRID_HW * RES / 2.0
        # Three GPU buffers: log-odds, probability, per-scan "seen" mark grid.
        self.log = wp.zeros((GRID_HW, GRID_HW), dtype=wp.float32, device=DEV)
        self.prob = wp.zeros_like(self.log)
        self.seen = wp.zeros_like(
            self.log
        )  # cleared after each insert by k_apply_marks
        self.precomp = []  # built on finalize; precomp[0] aliases self.prob
        self.n = 0
        self.finalized = False
        self._prob_dirty = True
        self._filtered_prob = (
            None  # bicubic-prefiltered prob (numpy), cached on finalize
        )

    def insert(self, sensor_pose_world, scan_local):
        """Insert a scan into the submap at the given world sensor pose.

        Uses the FULL scan (not voxel-filtered) for best map fidelity.
        """
        if self.finalized:
            raise RuntimeError("Cannot insert into a finalized submap")
        if len(scan_local) == 0:
            return

        # Sensor pose in submap-local frame
        sensor_local = between(self.origin, sensor_pose_world)
        # Scan in submap-local frame: rotate by sensor_local.theta, then translate
        c, s = np.cos(sensor_local[2]), np.sin(sensor_local[2])
        R = np.array([[c, -s], [s, c]])
        scan_submap = scan_local @ R.T + sensor_local[:2]

        pts = wp.array(scan_submap, dtype=wp.vec2, device=DEV)
        wp.launch(
            k_mark,
            dim=len(scan_submap),
            inputs=[
                pts,
                wp.vec2(float(sensor_local[0]), float(sensor_local[1])),
                self.seen,
                self.gx,
                self.gy,
                RES,
                GRID_HW,
                GRID_HW,
            ],
            device=DEV,
        )
        wp.launch(
            k_apply_marks,
            dim=(GRID_HW, GRID_HW),
            inputs=[self.seen, self.log, L_HIT, L_MISS, L_MIN, L_MAX],
            device=DEV,
        )

        self.n += 1
        self._prob_dirty = True

    def refresh_prob(self):
        """Rebuild self.prob from self.log if dirty (no-op for finalized submaps)."""
        if self._prob_dirty:
            wp.launch(
                k_logodds_to_prob,
                dim=(GRID_HW, GRID_HW),
                inputs=[self.log, self.prob],
                device=DEV,
            )
            self._prob_dirty = False

    def finalize(self):
        """Build BBS pyramid and cache spline-prefiltered prob for LC refinement."""
        if self.finalized:
            return
        self.refresh_prob()
        self.precomp = [self.prob]
        for h in range(1, BBS_LEVELS):
            prev = self.precomp[-1]
            offset = 2 ** (h - 1)
            nxt = wp.zeros((GRID_HW, GRID_HW), dtype=wp.float32, device=DEV)
            wp.launch(
                k_pyramid_step,
                dim=(GRID_HW, GRID_HW),
                inputs=[prev, nxt, offset, GRID_HW, GRID_HW],
                device=DEV,
            )
            self.precomp.append(nxt)
        # Cache bicubic-prefiltered prob (used many times during LC refinement).
        self._filtered_prob = spline_filter(self.prob.numpy(), order=3, mode="constant")
        self.finalized = True
        # Free buffers we no longer need
        self.log = None
        self.seen = None


# =============================================================================
# Correlative Scan Matcher (CSM): brute-force 3D search over (dx, dy, dtheta).
# Optional - off by default. Use as a more robust prior when odometry is noisy.
# =============================================================================
def csm(submap, prior_local, scan):
    """Match a (pre-voxel-filtered) scan against a submap's probability grid.
    Returns (best_pose_local, best_score) in submap-local frame.

    `scan` is assumed to already be voxel-filtered to MATCH_VOXEL_MAX_POINTS.
    Uses nearest-neighbor probability lookup (matches RealTimeCorrelativeScanMatcher2D).
    """
    if len(scan) == 0:
        return prior_local.copy(), 0.0
    submap.refresh_prob()
    nx = int(2 * CSM_WIN_XY / CSM_STEP_XY) + 1
    ny = nx
    nth = int(2 * CSM_WIN_TH / CSM_STEP_TH) + 1

    pts = wp.array(scan, dtype=wp.vec2, device=DEV)
    scores = wp.zeros((nth, ny, nx), dtype=wp.float32, device=DEV)
    wp.launch(
        k_csm,
        dim=(nth, ny, nx),
        inputs=[
            pts,
            submap.prob,
            submap.gx,
            submap.gy,
            RES,
            GRID_HW,
            GRID_HW,
            float(prior_local[0]),
            float(prior_local[1]),
            float(prior_local[2]),
            nx,
            ny,
            nth,
            CSM_STEP_XY,
            CSM_STEP_TH,
            scores,
        ],
        device=DEV,
    )
    s = scores.numpy()
    it, iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
    dth = (it - nth // 2) * CSM_STEP_TH
    dx = (ix - nx // 2) * CSM_STEP_XY
    dy = (iy - ny // 2) * CSM_STEP_XY
    best = np.array(
        [prior_local[0] + dx, prior_local[1] + dy, wrap(prior_local[2] + dth)]
    )
    return best, float(s[it, iy, ix])


# =============================================================================
# Ceres-style refinement on a smooth probability grid.
# Same cost function for local SLAM and loop closure (matches CeresScanMatcher2D::Match);
# only the source of the initial pose differs.
# =============================================================================
def _make_sample_fn(submap):
    """Return a function f(world_xy_Nx2) -> probability_N using bicubic interpolation.

    Out-of-grid points return 0.5 (neutral). Finalized submaps use a cached
    spline prefilter; active submaps recompute it (bug for thought: this is the
    spline_filter that runs on every local match, so it's a meaningful cost).
    """
    if submap._filtered_prob is not None:
        filtered = submap._filtered_prob
    else:
        submap.refresh_prob()
        filtered = spline_filter(submap.prob.numpy(), order=3, mode="constant")

    def sample(world_xy):
        fx = (world_xy[:, 0] - submap.gx) / RES - 0.5
        fy = (world_xy[:, 1] - submap.gy) / RES - 0.5
        coords = np.vstack([fy, fx])  # map_coordinates wants (row, col)
        return map_coordinates(
            filtered, coords, order=3, mode="constant", cval=0.5, prefilter=False
        )

    return sample


def refine(submap, init_pose_local, scan):
    """Levenberg-Marquardt scan match. The initial pose is also the prior.

    Cost (matches CeresScanMatcher2D::Match):
        sum_k (1 / sqrt(N) * (1 - M_smooth(T h_k)))^2
        + (W_T * (T.x - init.x))^2
        + (W_T * (T.y - init.y))^2
        + (W_R * wrap(T.theta - init.theta))^2

    The 1/sqrt(N) scaling makes the occupied-space term independent of scan
    size, so REFINE_W_T / REFINE_W_R have consistent meaning regardless of how
    aggressively the voxel filter trimmed the input. Cartographer source:
    occupied_space_cost_function_2d.h::scaling_factor_.

    `scan` is assumed pre-voxel-filtered (MATCH_VOXEL_MAX_POINTS for local,
    LC_VOXEL_MAX_POINTS for loop closure).
    """
    if len(scan) == 0:
        return init_pose_local.copy()
    sample = _make_sample_fn(submap)
    inv_sqrt_n = 1.0 / np.sqrt(len(scan))

    def residuals(T):
        c, s = np.cos(T[2]), np.sin(T[2])
        wx = T[0] + c * scan[:, 0] - s * scan[:, 1]
        wy = T[1] + s * scan[:, 0] + c * scan[:, 1]
        m = sample(np.column_stack([wx, wy]))
        rp = inv_sqrt_n * (1.0 - m)
        rt_x = REFINE_W_T * (T[0] - init_pose_local[0])
        rt_y = REFINE_W_T * (T[1] - init_pose_local[1])
        rr = REFINE_W_R * wrap(T[2] - init_pose_local[2])
        return np.concatenate([rp, [rt_x, rt_y, rr]])

    res = least_squares(
        residuals, init_pose_local.copy(), method="lm", max_nfev=REFINE_ITERS * 4
    )
    out = res.x.copy()
    out[2] = wrap(out[2])
    return out


# =============================================================================
# Branch-and-Bound Search (BBS) for loop closure
# =============================================================================
def _angular_step(scan, res):
    """Paper Eq. 7: dtheta such that scan points at d_max move <= one cell.

    dtheta = arccos(1 - r^2 / (2 * d_max^2))
    """
    if len(scan) == 0:
        return 0.01
    d_max = float(np.linalg.norm(scan, axis=1).max())
    if d_max < 2 * res:
        return 0.1
    return float(np.arccos(max(-1.0, 1.0 - res * res / (2.0 * d_max * d_max))))


def bbs(submap, scan, prior_local, win_xy, win_th, min_score=LC_MIN_SCORE):
    """Branch-and-bound search for the best (x, y, theta) match in submap-local frame.

    Returns (best_pose_local, best_score), or (None, 0.0) if no match exceeds min_score.

    `scan` is assumed pre-voxel-filtered to LC_VOXEL_MAX_POINTS.

    Algorithm (paper Section V-B):
      1. Pick angular step from Eq. 7.
      2. Enumerate angles. For each angle, generate root spatial candidates at
         the coarsest level's resolution covering the search window.
      3. Score all roots in one batch, then DFS with pruning. At each non-leaf
         node, the score is an exact upper bound on any descendant (paper Eq. 17).
    """
    if not submap.finalized or len(scan) == 0:
        return None, 0.0

    L = BBS_LEVELS - 1
    coarsest_res = RES * (2**L)

    # Allocate scan buffer once; reused across all level scoring calls.
    pts = wp.array(scan, dtype=wp.vec2, device=DEV)

    def score_at_level(lvl, candidates_xyz):
        """Score (Nx3) candidates against precomp[lvl]. Returns numpy scores of len N."""
        grid = submap.precomp[lvl]
        H, W = grid.shape
        cands = wp.array(candidates_xyz.astype(np.float32), dtype=wp.vec3, device=DEV)
        scores = wp.zeros(len(candidates_xyz), dtype=wp.float32, device=DEV)
        wp.launch(
            k_bbs_score,
            dim=len(candidates_xyz),
            inputs=[pts, grid, submap.gx, submap.gy, RES, H, W, cands, scores],
            device=DEV,
        )
        return scores.numpy()

    # ---- Build root candidates (coarsest level, vectorized) ----
    dtheta = _angular_step(scan, RES)
    n_th = int(np.ceil(2 * win_th / dtheta)) + 1
    thetas = np.linspace(prior_local[2] - win_th, prior_local[2] + win_th, n_th)

    # Symmetrically tile the search box at coarsest_res spacing.
    half_n = int(np.ceil(win_xy / coarsest_res))
    offsets = (np.arange(2 * half_n + 1) - half_n) * coarsest_res
    xs = prior_local[0] + offsets
    ys = prior_local[1] + offsets

    # Vectorized root generation via meshgrid (40x faster than nested-loop append).
    TH, Y, X = np.meshgrid(thetas, ys, xs, indexing="ij")
    root_xyz = np.column_stack([X.ravel(), Y.ravel(), wrap(TH.ravel())])
    if len(root_xyz) == 0:
        return None, 0.0

    root_scores = score_at_level(L, root_xyz)

    # ---- DFS with pruning ----
    # Stack of (score, x, y, th, lvl); .pop() yields highest-score first.
    order = np.argsort(root_scores)  # ascending so .pop() gets highest
    stack = [
        (
            float(root_scores[k]),
            float(root_xyz[k, 0]),
            float(root_xyz[k, 1]),
            float(root_xyz[k, 2]),
            L,
        )
        for k in order
    ]

    best_score = min_score
    best_pose = None

    while stack:
        score, x, y, th, lvl = stack.pop()
        if score < best_score:
            continue  # prune: no descendant can beat current best
        if lvl == 0:
            best_score = score
            best_pose = np.array([x, y, th])
            continue
        # Branch into 4 children at the next finer level (each covers a quadrant).
        finer_lvl = lvl - 1
        finer_res = RES * (2**finer_lvl)
        child_xyz = np.array(
            [
                [x, y, th],
                [x + finer_res, y, th],
                [x, y + finer_res, th],
                [x + finer_res, y + finer_res, th],
            ]
        )
        c_scores = score_at_level(finer_lvl, child_xyz)
        # Push lowest-score child first so highest is processed next.
        c_order = np.argsort(c_scores)
        for k in c_order:
            if c_scores[k] >= best_score:
                stack.append(
                    (
                        float(c_scores[k]),
                        float(child_xyz[k, 0]),
                        float(child_xyz[k, 1]),
                        float(child_xyz[k, 2]),
                        finer_lvl,
                    )
                )

    if best_pose is None:
        return None, 0.0
    return best_pose, best_score


# =============================================================================
# Pose graph (GTSAM wrapper)
# =============================================================================
class PoseGraph:
    """Pose graph with two kinds of nodes: trajectory nodes and submap origins.

    Both are Pose2 variables in GTSAM. Disjoint integer ID spaces:
      * Trajectory nodes:  IDs 0      .. 999_999
      * Submap origin IDs: IDs 1_000_000 .. 1_999_999
    """

    SUBMAP_ID_OFFSET = 1_000_000

    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self._next_traj = 0
        self._next_submap = 0
        self._intra_noise = gtsam.noiseModel.Diagonal.Sigmas(SIGMA_INTRA)
        self._lc_base = gtsam.noiseModel.Diagonal.Sigmas(SIGMA_LC)
        self._lc_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(HUBER_K), self._lc_base
        )
        self._prior = gtsam.noiseModel.Diagonal.Sigmas(SIGMA_PRIOR)
        self._prior_anchored = False

    def add_trajectory_node(self, pose):
        nid = self._next_traj
        self._next_traj += 1
        self.values.insert(nid, gtsam.Pose2(*pose))
        if not self._prior_anchored:
            self.graph.add(gtsam.PriorFactorPose2(nid, gtsam.Pose2(*pose), self._prior))
            self._prior_anchored = True
        return nid

    def add_submap_origin(self, pose):
        sid = self.SUBMAP_ID_OFFSET + self._next_submap
        self._next_submap += 1
        self.values.insert(sid, gtsam.Pose2(*pose))
        return sid

    def add_intra_submap(self, submap_key, node_key, relative_pose):
        """Edge from submap origin to trajectory node, in submap-local frame."""
        self.graph.add(
            gtsam.BetweenFactorPose2(
                submap_key, node_key, gtsam.Pose2(*relative_pose), self._intra_noise
            )
        )

    def add_loop_closure(self, submap_key, node_key, relative_pose):
        """Loop-closure edge: same form as intra-submap but Huber-robust noise."""
        self.graph.add(
            gtsam.BetweenFactorPose2(
                submap_key, node_key, gtsam.Pose2(*relative_pose), self._lc_noise
            )
        )

    def optimize(self):
        """Run Levenberg-Marquardt to global optimum (paper Eq. SPA)."""
        opt = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values)
        self.values = opt.optimize()

    def pose(self, key):
        p = self.values.atPose2(key)
        return np.array([p.x(), p.y(), p.theta()])


# =============================================================================
# SubmapStack: two-active rotation, each submap receives 2*N scans.
# =============================================================================
class SubmapStack:
    """Manages the rotation of active submaps.

    Cartographer 2D (active_submaps_2d.cc::InsertRangeData): each submap receives
    2 * NUM_RANGE_DATA scans. A new submap is started every NUM_RANGE_DATA scans,
    so two submaps are always active simultaneously -- one acting as the "old"
    matching target (already has NUM_RANGE_DATA scans of context), one
    accumulating as the new.
    """

    def __init__(self):
        self.submaps = []
        # Maintained alongside self.submaps for O(1) lookups in hot paths.
        self._by_id = {}
        self._active = []  # subset of self.submaps that aren't finalized
        self._finalized = []  # subset that are finalized (LC search target list)

    def by_id(self, sid):
        return self._by_id[sid]

    def matching_target(self):
        """The older active submap (first in active list) is the matching target."""
        return self._active[0] if self._active else None

    def finalized_submaps(self):
        return self._finalized

    def add_scan(self, sensor_pose_world, scan_local):
        """Insert into all active submaps. Returns list of submap IDs touched.
        May trigger finalization of the older active submap.
        """
        if not self._active:
            self._spawn(sensor_pose_world)
        elif len(self._active) == 1 and self._active[0].n >= NUM_RANGE_DATA:
            # The older submap is "halfway through": time to start the next one.
            self._spawn(sensor_pose_world)

        ids = []
        for sm in self._active:
            sm.insert(sensor_pose_world, scan_local)
            ids.append(sm.id)

        older = self._active[0]
        if older.n >= SUBMAP_TOTAL_SCANS:
            older.finalize()
            self._active.pop(0)
            self._finalized.append(older)
        return ids

    def _spawn(self, origin_pose):
        sm = Submap(origin_pose)
        self.submaps.append(sm)
        self._active.append(sm)
        self._by_id[sm.id] = sm


# =============================================================================
# Top-level SLAM
# =============================================================================
class Slam:
    """End-to-end Cartographer-style 2D SLAM.

    Public API:
        slam.add_scan(scan_local, odom_delta) -> world_pose
        slam.current_pose() -> world_pose
        slam.snapshot() -> dict of state for visualization

    Data flow per scan:
        scan_local (full, ~1080 pts)
          -> match_scan = adaptive_voxel_filter(MATCH_VOXEL_MAX_POINTS)  [for matching]
          -> matched_world via csm() and/or refine()
          -> motion-filter check; if pass:
                * insert FULL scan_local into active submaps (best map fidelity)
                * store match_scan in traj_meta (for future LC; smaller memory)
                * add trajectory node + intra-submap edges
          -> periodically: LC sweep + global optimization
    """

    def __init__(self):
        self.stack = SubmapStack()
        self.graph = PoseGraph()
        self.motion_filter = MotionFilter()
        self.last_pose = np.zeros(3)
        self.scans_since_optimize = 0
        self._scans_since_node = 0  # for motion-filter idle timeout
        # submap_id -> graph_key for its origin
        self.submap_keys = {}
        # node_key -> {"submap_ids": [...], "scan": Nx2 (voxel-filtered)}
        self.traj_meta = {}
        # (node_key, submap_id) pairs already attempted but failed: skip until
        # a global optimization shifts poses (when we clear this set).
        self._lc_attempted = set()
        # Pairs that produced a constraint: skip forever (avoid duplicate edges).
        self._lc_found = set()

    def _ensure_submap_in_graph(self, submap):
        if submap.id not in self.submap_keys:
            key = self.graph.add_submap_origin(submap.origin)
            self.submap_keys[submap.id] = key
        return self.submap_keys[submap.id]

    def add_scan(self, scan_local, odom_delta):
        """Process one scan. Returns the (possibly post-optimization) world pose."""
        # Empty scan: nothing to match against; just propagate the prior.
        if len(scan_local) == 0:
            self.last_pose = compose(self.last_pose, odom_delta)
            return self.last_pose.copy()

        # Voxel-filter once for matching (full scan is still used for insertion).
        match_scan = adaptive_voxel_filter(scan_local, MATCH_VOXEL_MAX_POINTS)

        # 1. Motion prior in world frame
        prior_world = compose(self.last_pose, odom_delta)

        # 2. Match against current matching target
        target = self.stack.matching_target()
        if target is None or target.n < 1:
            matched_world = prior_world  # not enough data to match against
        else:
            prior_local = between(target.origin, prior_world)
            if USE_CSM:
                csm_local, score = csm(target, prior_local, match_scan)
                init_local = csm_local if score > CSM_MIN_SCORE else prior_local
            else:
                init_local = prior_local
            refined_local = refine(target, init_local, match_scan)
            matched_world = compose(target.origin, refined_local)

        self.last_pose = matched_world

        # 3. Motion filter: skip submap insertion + graph node for redundant scans,
        #    but force-accept after MAX_SCANS_BETWEEN_NODES idle scans (idle timeout).
        self._scans_since_node += 1
        if (
            self.motion_filter.is_similar(matched_world)
            and self._scans_since_node < MAX_SCANS_BETWEEN_NODES
        ):
            return matched_world.copy()
        self._scans_since_node = 0
        self.motion_filter.update(matched_world)

        # 4. Insert into all active submaps (full scan, for map fidelity)
        submap_ids = self.stack.add_scan(matched_world, scan_local)

        # 5. Add trajectory node + intra-submap edges; store filtered scan for LC
        node_key = self.graph.add_trajectory_node(matched_world)
        self.traj_meta[node_key] = {
            "submap_ids": list(submap_ids),
            "scan": match_scan.copy(),  # ~200 pts; LC re-filters to ~100
        }
        for sid in submap_ids:
            sm = self.stack.by_id(sid)
            sk = self._ensure_submap_in_graph(sm)
            rel = between(sm.origin, matched_world)
            self.graph.add_intra_submap(sk, node_key, rel)

        self.scans_since_optimize += 1

        # 6. Periodic loop-closure sweep + global optimization
        if self.scans_since_optimize >= OPTIMIZE_EVERY and node_key > LC_SKIP_NODES:
            n_added = self._sweep_loop_closures(node_key)
            self.scans_since_optimize = 0
            if n_added > 0:
                self.graph.optimize()
                # Refresh submap origins and current pose from optimization
                for sid, sk in self.submap_keys.items():
                    self.stack.by_id(sid).origin = self.graph.pose(sk)
                matched_world = self.graph.pose(node_key)
                self.last_pose = matched_world
                # Pose updates may make previously-rejected pairs viable; clear cache.
                # _lc_found is preserved (those edges are already in the graph).
                self._lc_attempted.clear()

        return matched_world.copy()

    def _sweep_loop_closures(self, current_node_key):
        """Test (recent_node, finalized_submap) pairs that pass spatial gating.

        Approximates Cartographer's background dispatch: a window of recent
        nodes gets re-tried after each optimization, and successful pairs
        (in _lc_found) are skipped permanently to avoid duplicate edges.
        """
        node_keys = sorted(self.traj_meta.keys())
        recent = node_keys[-LC_RECENT_WINDOW:]
        if current_node_key not in recent:
            recent.append(current_node_key)

        added = 0
        finalized = self.stack.finalized_submaps()
        for nk in recent:
            if nk <= LC_SKIP_NODES:
                continue
            node_pose = self.graph.pose(nk)
            # Re-filter the stored match-scan to LC density (cheap; usually a no-op
            # past the first call since it's already small).
            lc_scan = adaptive_voxel_filter(
                self.traj_meta[nk]["scan"], LC_VOXEL_MAX_POINTS
            )
            node_submap_ids = set(self.traj_meta[nk]["submap_ids"])
            for sm in finalized:
                if sm.id in node_submap_ids:
                    continue  # node belongs to this submap: not a loop closure
                pair = (nk, sm.id)
                if pair in self._lc_found or pair in self._lc_attempted:
                    continue
                # Spatial gating
                if np.linalg.norm(sm.origin[:2] - node_pose[:2]) > LC_MAX_DISTANCE:
                    self._lc_attempted.add(pair)
                    continue
                prior_local = between(sm.origin, node_pose)
                best, _ = bbs(
                    sm,
                    lc_scan,
                    prior_local,
                    win_xy=LC_WIN_XY,
                    win_th=LC_WIN_TH,
                    min_score=LC_MIN_SCORE,
                )
                if best is None:
                    self._lc_attempted.add(pair)
                    continue
                # Refine the BBS hit (use it as both init AND prior).
                refined = refine(sm, best, lc_scan)
                self.graph.add_loop_closure(self.submap_keys[sm.id], nk, refined)
                self._lc_found.add(pair)
                added += 1
        return added

    def current_pose(self):
        return self.last_pose.copy()

    def snapshot(self):
        """Return a dict suitable for visualization."""
        out_submaps = []
        for s in self.stack.submaps:
            if not s.finalized:
                s.refresh_prob()
            out_submaps.append((s.id, s.origin.copy(), s.finalized, s.prob.numpy()))
        return {
            "submaps": out_submaps,
            "trajectory": [self.graph.pose(k) for k in sorted(self.traj_meta.keys())],
        }


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    print(f"Warp device: {DEV}")
    print(f"Submap size: {GRID_HW}x{GRID_HW} @ {RES}m/cell ({GRID_HW * RES}m)")
    print(f"Submap budget: {SUBMAP_TOTAL_SCANS} scans (Cartographer 2*num_range_data)")
    print(f"BBS pyramid: {BBS_LEVELS} levels, same-resolution sliding-window max")
    print(
        f"Match voxel filter: <= {MATCH_VOXEL_MAX_POINTS} pts;"
        f"  LC voxel filter: <= {LC_VOXEL_MAX_POINTS} pts"
    )
    print(
        f"Motion filter: dx > {MIN_MOTION_DISTANCE}m"
        f" or dtheta > {np.rad2deg(MIN_MOTION_ANGLE):.1f}deg"
        f" or {MAX_SCANS_BETWEEN_NODES} idle scans"
    )
    print(f"Use CSM (real-time correlative): {USE_CSM}")

    slam = Slam()
    rng = np.random.default_rng(0)

    # Quick fake scan: 8m square room
    walls_pts = []
    for u in np.linspace(-4, 4, 200):
        walls_pts += [[u, -4], [u, 4], [-4, u], [4, u]]
    walls_pts = np.array(walls_pts)

    pose = np.zeros(3)
    for step in range(5):
        delta = np.array([0.1, 0.0, 0.1])
        pose = compose(pose, delta)
        c, s = np.cos(pose[2]), np.sin(pose[2])
        rel = (walls_pts - pose[:2]) @ np.array([[c, s], [-s, c]]).T
        ranges = np.linalg.norm(rel, axis=1)
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        scan = []
        for b in np.linspace(-np.pi, np.pi, 90, endpoint=False):
            mask = np.abs(((ang - b + np.pi) % (2 * np.pi)) - np.pi) < 0.04
            if mask.any():
                pick = rel[mask][np.argmin(ranges[mask])]
                scan.append(pick)
        scan = np.array(scan) + rng.normal(0, 0.02, (len(scan), 2))
        est = slam.add_scan(scan, delta)
        print(f"  step {step}: truth={pose.round(2)}  est={est.round(2)}")

    print(
        f"\nstate: {len(slam.stack.submaps)} submaps, "
        f"{len(slam.traj_meta)} trajectory nodes, "
        f"{slam.graph.graph.size()} factors"
    )
    print("Smoke test passed.")
