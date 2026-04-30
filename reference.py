"""
Cartographer-style 2D graph SLAM for F1Tenth.
Front-end (GPU/Warp): ray-casting, optional CSM, refinement-cost evaluation, BBS pyramids.
Back-end (CPU/GTSAM): pose graph with intra-submap and loop-closure edges, LM optimization.

Closely follows Hess et al. "Real-Time Loop Closure in 2D LIDAR SLAM" (ICRA 2016) and
the cartographer-project source. Notable choices (see component docstrings for details):

  * BBS uses Cartographer's same-resolution sliding-window-max precomputed grids
    (paper Eq. 17). M_h has the same shape as M_0; each cell stores max over
    [i, i+2^h) x [j, j+2^h). This makes the upper bound EXACT for arbitrarily
    positioned candidates -- no coarse-grid alignment required.
  * BBS angular step is derived from the paper's Eq. 7:
        dtheta = arccos(1 - r^2 / (2 * d_max^2))
    so scan points at d_max move by <= one cell per angular step.
  * LC refinement does NOT regularize toward odometry (the LC observation should
    override drift); only the smooth-grid scan-match cost is minimized.
  * Each submap receives 2 * NUM_RANGE_DATA scans, matching Cartographer 2D's
    "old + new" rotation: a submap is the "new" one for NUM_RANGE_DATA scans,
    then the "old matching target" for another NUM_RANGE_DATA scans, then frozen.
  * Loop-closure search runs over a window of recent nodes, not just the current
    one, and is retried after each global optimization (poses change -> gating
    decisions change). This approximates Cartographer's background dispatch.
  * Refinement uses bicubic-spline interpolation on the probability grid
    (paper IV-C). Out-of-grid samples return 0.5 (no information; neutral).
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

# Local scan matching
USE_CSM = False  # off by default; enable if odometry is unreliable
CSM_WIN_XY, CSM_STEP_XY = 0.15, 0.025
CSM_WIN_TH, CSM_STEP_TH = 0.17, 0.005
CSM_MIN_SCORE = 0.30  # below this, fall back to odometry-only init for refinement
REFINE_ITERS = 20
REFINE_W_T = 10.0
REFINE_W_R = 40.0

# BBS / loop closure
BBS_LEVELS = 6  # precomputed grid levels (covers 2^5 = 32 cells = 1.6m windowing)
LC_MIN_SCORE = 0.55
LC_WIN_XY = 2.0  # m, half-window for local LC search
LC_WIN_TH = 0.5  # rad, half-window for local LC search
LC_MAX_DISTANCE = 15.0  # m, spatial gating

# Pose graph noise (sigmas)
SIGMA_INTRA = np.array([0.025, 0.025, 0.005])
SIGMA_LC = np.array([0.05, 0.05, 0.02])
SIGMA_PRIOR = np.array([1e-6, 1e-6, 1e-8])
HUBER_K = 1.0

# Loop-closure scheduling
OPTIMIZE_EVERY = 30
LC_SKIP_NODES = 50  # don't try LC on the very first nodes
LC_RECENT_WINDOW = 50  # how many recent nodes to try in each LC sweep


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
# Warp kernels (GPU)
# =============================================================================


@wp.kernel
def k_cast(
    pts: wp.array(dtype=wp.vec2),
    origin: wp.vec2,
    grid: wp.array2d(dtype=wp.float32),
    gx: float,
    gy: float,
    res: float,
    lhit: float,
    lmiss: float,
    H: int,
    W: int,
):
    """Bresenham ray-cast: subtract lmiss along the ray (excluding the sensor cell),
    add lhit at the endpoint. Atomic adds let many rays update the same cell safely.

    The pre-step before the loop ensures we start applying lmiss at the cell after
    the sensor, matching the paper's "between origin and scan point" convention.
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
            wp.atomic_add(grid, y, x, lmiss)
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    if x1 >= 0 and x1 < W and y1 >= 0 and y1 < H:
        wp.atomic_add(grid, y1, x1, lhit)


@wp.kernel
def k_clamp(grid: wp.array2d(dtype=wp.float32), lo: float, hi: float):
    """Clamp grid values into [lo, hi] in place."""
    i, j = wp.tid()
    v = grid[i, j]
    if v < lo:
        grid[i, j] = lo
    if v > hi:
        grid[i, j] = hi


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
    """Score every (dtheta, dy, dx) candidate around prior with bilinear-interp
    probability. One thread per candidate. Half-cell offset because cell values
    are stored at the cell CENTER for interpolation.
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
        fx = (wx - gx) / res - 0.5
        fy = (wy - gy) / res - 0.5
        x0 = int(wp.floor(fx))
        y0 = int(wp.floor(fy))
        tx = fx - float(x0)
        ty = fy - float(y0)
        if x0 >= 0 and x0 + 1 < W and y0 >= 0 and y0 + 1 < H:
            v = (grid[y0, x0] * (1.0 - tx) + grid[y0, x0 + 1] * tx) * (1.0 - ty) + (
                grid[y0 + 1, x0] * (1.0 - tx) + grid[y0 + 1, x0 + 1] * tx
            ) * ty
            acc += v
    scores[it, iy, ix] = acc / float(n)


@wp.kernel
def k_pyramid_step(
    src: wp.array2d(dtype=wp.float32),  # M_{h-1}
    dst: wp.array2d(dtype=wp.float32),  # M_h
    offset: int,  # 2^(h-1)
    H: int,
    W: int,
):
    """Cartographer's same-resolution precomputation grid (paper Section V-B.3, Eq. 17).

    M_h[i, j] = max over [i, i+2^h) x [j, j+2^h) of M_0
              = max(M_{h-1}[i, j],
                    M_{h-1}[i+offset, j],
                    M_{h-1}[i, j+offset],
                    M_{h-1}[i+offset, j+offset])
    where offset = 2^{h-1}. Out-of-bounds reads return 0 (no contribution).

    Output has the SAME shape as M_0 -- this is what makes the BBS upper bound
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
    """Score (x, y, theta) candidates against a precomputed grid.

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
      * Active: log-odds grid is updated as scans arrive; prob grid is rebuilt
        on demand for matching.
      * Finalized: log-odds is no longer mutated. The BBS precomputed grid stack
        is built once. The log-odds buffer is freed.
    """

    _next_id = 0

    def __init__(self, origin_pose):
        self.id = Submap._next_id
        Submap._next_id += 1
        self.origin = origin_pose.copy()
        self.gx = -GRID_HW * RES / 2.0
        self.gy = -GRID_HW * RES / 2.0
        self.log = wp.zeros((GRID_HW, GRID_HW), dtype=wp.float32, device=DEV)
        self.prob = wp.zeros_like(self.log)
        self.precomp = []  # list of precomputed grids; precomp[0] aliases self.prob
        self.scans = []
        self.n = 0
        self.finalized = False
        self._prob_dirty = True

    def insert(self, sensor_pose_world, scan_local):
        """Insert a scan into the submap at the given world sensor pose."""
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
            k_cast,
            dim=len(scan_submap),
            inputs=[
                pts,
                wp.vec2(float(sensor_local[0]), float(sensor_local[1])),
                self.log,
                self.gx,
                self.gy,
                RES,
                L_HIT,
                L_MISS,
                GRID_HW,
                GRID_HW,
            ],
            device=DEV,
        )
        wp.launch(
            k_clamp, dim=(GRID_HW, GRID_HW), inputs=[self.log, L_MIN, L_MAX], device=DEV
        )

        self.scans.append((sensor_pose_world.copy(), scan_local.copy()))
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
        """Build the same-resolution precomputed grid stack for BBS.

        precomp[0] = M_0 (probability grid, aliases self.prob)
        precomp[h] = sliding-window max with window size 2^h, same resolution as M_0
        """
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
        self.finalized = True
        # Log-odds buffer is no longer needed
        self.log = None


# =============================================================================
# Correlative Scan Matcher (CSM): brute-force 3D search over (dx, dy, dtheta).
# Optional - off by default. Use as a more robust prior when odometry is noisy.
# =============================================================================
def csm(submap, prior_local, scan_local):
    """Match a scan against a submap's probability grid via brute-force search.
    Returns (best_pose_local, best_score) in submap-local frame.
    """
    if len(scan_local) == 0:
        return prior_local.copy(), 0.0
    submap.refresh_prob()
    nx = int(2 * CSM_WIN_XY / CSM_STEP_XY) + 1
    ny = nx
    nth = int(2 * CSM_WIN_TH / CSM_STEP_TH) + 1

    pts = wp.array(scan_local, dtype=wp.vec2, device=DEV)
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
# Ceres-style refinement on a smooth probability grid (paper Eq. CS).
# Bicubic-spline interpolation gives sub-cell accuracy with continuous gradients.
# =============================================================================
def _make_sample_fn(submap):
    """Return a function f(world_xy_Nx2) -> probability_N using bicubic interpolation.

    Out-of-grid points return 0.5 (neutral; no information). The spline-prefiltered
    grid is computed once per refine() call and reused across LM iterations.
    """
    submap.refresh_prob()
    prob = submap.prob.numpy()
    # Pre-filter once; map_coordinates can then use prefilter=False for speed.
    filtered = spline_filter(prob, order=3, mode="constant")

    def sample(world_xy):
        # Cell-center pixel coordinates (with -0.5 offset)
        fx = (world_xy[:, 0] - submap.gx) / RES - 0.5
        fy = (world_xy[:, 1] - submap.gy) / RES - 0.5
        coords = np.vstack([fy, fx])  # map_coordinates wants (row, col)
        return map_coordinates(
            filtered, coords, order=3, mode="constant", cval=0.5, prefilter=False
        )

    return sample


def refine_local(submap, init_pose_local, prior_local, scan_local):
    """Local-SLAM refinement: bicubic-spline cost + odometry-prior regularization.

    Cost = sum_k (1 - M_smooth(T h_k))^2
         + (W_T * (T.x - prior.x))^2 + (W_T * (T.y - prior.y))^2
         + (W_R * wrap(T.theta - prior.theta))^2

    The odometry prior keeps the solution near the extrapolated pose, which is
    appropriate when the matcher has weak signal (e.g. featureless corridors).
    """
    return _refine(submap, init_pose_local, scan_local, prior_local=prior_local)


def refine_lc(submap, init_pose_local, scan_local):
    """Loop-closure refinement: NO odometry prior. The LC observation must be
    free to override drifted odometry.
    """
    return _refine(submap, init_pose_local, scan_local, prior_local=None)


def _refine(submap, init_pose_local, scan_local, prior_local):
    """LM refinement. If prior_local is None, no prior regularizer is applied."""
    if len(scan_local) == 0:
        return init_pose_local.copy()
    sample = _make_sample_fn(submap)

    def residuals(T):
        c, s = np.cos(T[2]), np.sin(T[2])
        wx = T[0] + c * scan_local[:, 0] - s * scan_local[:, 1]
        wy = T[1] + s * scan_local[:, 0] + c * scan_local[:, 1]
        m = sample(np.column_stack([wx, wy]))
        rp = 1.0 - m
        if prior_local is None:
            return rp
        rt_x = REFINE_W_T * (T[0] - prior_local[0])
        rt_y = REFINE_W_T * (T[1] - prior_local[1])
        rr = REFINE_W_R * wrap(T[2] - prior_local[2])
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
def _angular_step(scan_local, res):
    """Paper Eq. 7: dtheta such that scan points at d_max move <= one cell.

        dtheta = arccos(1 - r^2 / (2 * d_max^2))

    With this step, the spatial precomputed grid for each angle gives an
    exact upper bound for BBS.
    """
    if len(scan_local) == 0:
        return 0.01
    d_max = float(np.linalg.norm(scan_local, axis=1).max())
    if d_max < 2 * res:
        return 0.1
    cos_arg = max(-1.0, 1.0 - res * res / (2.0 * d_max * d_max))
    return float(np.arccos(cos_arg))


def _score_at_level(submap, level_idx, scan_local, candidates):
    """Score (x, y, theta) candidates against the precomputed grid at `level_idx`."""
    grid = submap.precomp[level_idx]
    H, W = grid.shape
    cands = wp.array(candidates.astype(np.float32), dtype=wp.vec3, device=DEV)
    scores = wp.zeros(len(candidates), dtype=wp.float32, device=DEV)
    pts = wp.array(scan_local, dtype=wp.vec2, device=DEV)
    wp.launch(
        k_bbs_score,
        dim=len(candidates),
        inputs=[pts, grid, submap.gx, submap.gy, RES, H, W, cands, scores],
        device=DEV,
    )
    return scores.numpy()


def bbs(submap, scan_local, prior_local, win_xy, win_th, min_score=LC_MIN_SCORE):
    """Branch-and-bound search for the best (x, y, theta) match in submap-local frame.

    Returns (best_pose_local, best_score), or (None, 0.0) if no match exceeds min_score.

    Algorithm (paper Section V-B):
      1. Pick angular step from Eq. 7.
      2. Enumerate angles. For each angle, generate root spatial candidates at
         the coarsest level's resolution covering the search window.
      3. DFS with pruning. Each candidate is (x_corner, y_corner, theta, level).
         At a non-leaf level, the score is an exact upper bound on any leaf in
         the subtree. Children at a finer level tile the parent's spatial extent
         (4 children, each at half the spatial extent).
    """
    if not submap.finalized or len(scan_local) == 0:
        return None, 0.0

    L = BBS_LEVELS - 1
    coarsest_res = RES * (2**L)

    dtheta = _angular_step(scan_local, RES)
    n_th = int(np.ceil(2 * win_th / dtheta)) + 1
    thetas = np.linspace(prior_local[2] - win_th, prior_local[2] + win_th, n_th)

    # Spatial candidates at coarsest level: tile [-win_xy, win_xy] symmetrically.
    # Each candidate (cx, cy) at level L represents poses in [cx, cx + coarsest_res).
    half_n = int(np.ceil(win_xy / coarsest_res))
    offsets = (np.arange(2 * half_n + 1) - half_n) * coarsest_res
    xs = prior_local[0] + offsets
    ys = prior_local[1] + offsets

    # Build root candidates and score them in one batch on the coarsest grid.
    root_cands = []
    for th in thetas:
        for y in ys:
            for x in xs:
                root_cands.append((x, y, wrap(th), L))
    if not root_cands:
        return None, 0.0
    cand_xyz = np.array([(c[0], c[1], c[2]) for c in root_cands])
    root_scores = _score_at_level(submap, L, scan_local, cand_xyz)

    # DFS: process highest-scoring candidates first; prune anything below current best.
    indexed = sorted(zip(root_scores.tolist(), root_cands), key=lambda t: t[0])
    stack = list(indexed)  # lowest first; .pop() yields highest first

    best_score = min_score
    best_pose = None

    while stack:
        score, cand = stack.pop()
        if score < best_score:
            continue  # prune: no descendant can beat current best
        x, y, th, lvl = cand
        if lvl == 0:
            # Leaf score is the exact match score (1x1 cell window).
            best_score = score
            best_pose = np.array([x, y, th])
            continue
        # Branch into 4 children at the next finer level, each covering a quadrant.
        finer_lvl = lvl - 1
        finer_res = RES * (2**finer_lvl)
        children = [
            (x, y, th, finer_lvl),
            (x + finer_res, y, th, finer_lvl),
            (x, y + finer_res, th, finer_lvl),
            (x + finer_res, y + finer_res, th, finer_lvl),
        ]
        c_xyz = np.array([(cc[0], cc[1], cc[2]) for cc in children])
        c_scores = _score_at_level(submap, finer_lvl, scan_local, c_xyz)
        # Push lowest-score child first so highest is processed next.
        for s, cc in sorted(zip(c_scores.tolist(), children), key=lambda t: t[0]):
            if s >= best_score:
                stack.append((s, cc))

    if best_pose is None:
        return None, 0.0
    return best_pose, best_score


# =============================================================================
# Pose graph (GTSAM wrapper)
# =============================================================================
class PoseGraph:
    """Pose graph with two kinds of nodes: trajectory nodes and submap origins.

    Both are Pose2 variables in GTSAM. Disjoint integer ID spaces let one int
    identify both kind and index.
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

    Following Cartographer 2D: each submap receives 2 * NUM_RANGE_DATA scans.
    A new submap is started every NUM_RANGE_DATA scans, so two submaps are always
    active simultaneously -- one acting as the "old" matching target (already has
    NUM_RANGE_DATA scans of context), one accumulating as the "new" submap.
    """

    def __init__(self):
        self.submaps = []

    def all_active(self):
        return [s for s in self.submaps if not s.finalized]

    def matching_target(self):
        """The older active submap is the matching target (more accumulated data)."""
        active = self.all_active()
        return active[0] if active else None

    def add_scan(self, sensor_pose_world, scan_local):
        """Insert into all active submaps. Returns list of submap IDs touched.
        May trigger finalization of the older active submap.
        """
        active = self.all_active()
        if not active:
            new = Submap(sensor_pose_world)
            self.submaps.append(new)
            active = [new]
        elif len(active) == 1 and active[0].n >= NUM_RANGE_DATA:
            # The older submap is "halfway through": time to start the next one.
            new = Submap(sensor_pose_world)
            self.submaps.append(new)
            active.append(new)

        ids = []
        for sm in active:
            sm.insert(sensor_pose_world, scan_local)
            ids.append(sm.id)

        older = active[0]
        if older.n >= SUBMAP_TOTAL_SCANS:
            older.finalize()
        return ids


# =============================================================================
# Top-level SLAM
# =============================================================================
class Slam:
    """End-to-end Cartographer-style 2D SLAM.

    Public API:
        slam.add_scan(scan_local, odom_delta) -> world_pose
        slam.current_pose() -> world_pose
        slam.snapshot() -> dict of state for visualization
    """

    def __init__(self):
        self.stack = SubmapStack()
        self.graph = PoseGraph()
        self.last_pose = np.zeros(3)
        self.scans_since_optimize = 0
        # submap_id -> graph_key for its origin
        self.submap_keys = {}
        # node_key -> {"submap_ids": [...], "scan": Nx2}
        self.traj_meta = {}
        # Tracks (node_key, submap_id) pairs already attempted for LC and rejected.
        # Cleared after each optimization since poses (and gating) change.
        self._lc_attempted = set()

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

        # 1. Motion prior in world frame
        prior_world = compose(self.last_pose, odom_delta)

        # 2. Match against current matching target
        target = self.stack.matching_target()
        if target is None or target.n < 2:
            # No data yet -- just accept the prior.
            matched_world = prior_world
        else:
            prior_local = between(target.origin, prior_world)
            if USE_CSM:
                csm_local, score = csm(target, prior_local, scan_local)
                init_local = csm_local if score > CSM_MIN_SCORE else prior_local
            else:
                init_local = prior_local
            refined_local = refine_local(target, init_local, prior_local, scan_local)
            matched_world = compose(target.origin, refined_local)

        # 3. Insert into all active submaps (may finalize the older submap)
        submap_ids = self.stack.add_scan(matched_world, scan_local)

        # 4. Add trajectory node and intra-submap edges
        node_key = self.graph.add_trajectory_node(matched_world)
        self.traj_meta[node_key] = {
            "submap_ids": list(submap_ids),
            "scan": scan_local.copy(),
        }
        for sid in submap_ids:
            sm = next(s for s in self.stack.submaps if s.id == sid)
            sk = self._ensure_submap_in_graph(sm)
            rel = between(sm.origin, matched_world)
            self.graph.add_intra_submap(sk, node_key, rel)

        self.scans_since_optimize += 1

        # 5. Periodic loop-closure sweep + global optimization.
        if self.scans_since_optimize >= OPTIMIZE_EVERY and node_key > LC_SKIP_NODES:
            n_added = self._sweep_loop_closures(node_key)
            self.scans_since_optimize = 0
            if n_added > 0:
                self.graph.optimize()
                # Refresh submap origins and current pose from the optimization.
                for sid, sk in self.submap_keys.items():
                    sm = next(s for s in self.stack.submaps if s.id == sid)
                    sm.origin = self.graph.pose(sk)
                matched_world = self.graph.pose(node_key)
                # Optimization moves poses; previously rejected pairs may now
                # pass spatial gating, so clear the rejection cache.
                self._lc_attempted.clear()

        self.last_pose = matched_world
        return matched_world

    def _sweep_loop_closures(self, current_node_key):
        """Test (recent_node, finalized_submap) pairs that pass spatial gating.

        Trying a window of recent nodes (not just the current one) approximates
        Cartographer's background dispatch: nodes get re-tried after optimization
        has shifted their world poses, and a node that just missed the gating
        cutoff before may pass after a closure tightens the trajectory.
        """
        node_keys = sorted(self.traj_meta.keys())
        recent = node_keys[-LC_RECENT_WINDOW:]
        if current_node_key not in recent:
            recent.append(current_node_key)

        added = 0
        for nk in recent:
            if nk <= LC_SKIP_NODES:
                continue
            node_pose = self.graph.pose(nk)
            scan_local = self.traj_meta[nk]["scan"]
            node_submap_ids = set(self.traj_meta[nk]["submap_ids"])
            for sm in self.stack.submaps:
                if not sm.finalized:
                    continue
                if sm.id in node_submap_ids:
                    continue
                pair = (nk, sm.id)
                if pair in self._lc_attempted:
                    continue
                # Spatial gating
                d = np.linalg.norm(sm.origin[:2] - node_pose[:2])
                if d > LC_MAX_DISTANCE:
                    self._lc_attempted.add(pair)
                    continue
                prior_local = between(sm.origin, node_pose)
                best, score = bbs(
                    sm,
                    scan_local,
                    prior_local,
                    win_xy=LC_WIN_XY,
                    win_th=LC_WIN_TH,
                    min_score=LC_MIN_SCORE,
                )
                if best is None:
                    self._lc_attempted.add(pair)
                    continue
                # Refine the BBS hit WITHOUT odometry prior -- LC must override drift.
                refined = refine_lc(sm, best, scan_local)
                self.graph.add_loop_closure(self.submap_keys[sm.id], nk, refined)
                added += 1
                # Don't add to _lc_attempted: pair may be revisited if poses shift.
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
        rng_ = np.linalg.norm(rel, axis=1)
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        scan = []
        for b in np.linspace(-np.pi, np.pi, 90, endpoint=False):
            mask = np.abs(((ang - b + np.pi) % (2 * np.pi)) - np.pi) < 0.04
            if mask.any():
                pick = rel[mask][np.argmin(rng_[mask])]
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
