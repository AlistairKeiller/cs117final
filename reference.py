"""
Cartographer-style 2D graph SLAM for F1Tenth.
Front-end (GPU/Warp): ray-casting, CSM, refinement-cost evaluation, BBS pyramids.
Back-end (CPU/GTSAM): pose graph with intra-submap and loop-closure edges, LM optimization.
"""

import gtsam
import numpy as np
import warp as wp
from scipy.optimize import least_squares

wp.init()
DEV = "cuda" if wp.is_cuda_available() else "cpu"

# =============================================================================
# Configuration
# =============================================================================
RES = 0.05  # m / cell
L_HIT, L_MISS = 0.85, -0.40
L_MIN, L_MAX = -2.2, 2.2
GRID_HW = 400  # 20m x 20m submap
SUBMAP_N = 90  # scans before submap finalizes

# CSM
WIN_XY, STEP_XY = 0.15, 0.025
WIN_TH, STEP_TH = 0.17, 0.005

# Refinement
REFINE_ITERS = 20
REFINE_W_T = 10.0
REFINE_W_R = 40.0

# BBS
BBS_LEVELS = 6  # pyramid depth
BBS_WIN_XY = 7.0
BBS_WIN_TH = np.pi
BBS_STEP_TH = 0.02  # angular candidates spacing
LC_MIN_SCORE = 0.55

# Pose graph noise (sigmas)
SIGMA_INTRA = np.array([0.025, 0.025, 0.005])
SIGMA_LC = np.array([0.05, 0.05, 0.02])
SIGMA_PRIOR = np.array([1e-6, 1e-6, 1e-8])
SIGMA_ODOM = np.array([0.1, 0.1, 0.05])  # fallback when matcher fails
HUBER_K = 1.0

# Scheduling
LC_SKIP_NODES = 50
OPTIMIZE_EVERY = 30


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
    """Bresenham ray-cast: subtract lmiss along the ray, add lhit at endpoint.

    Math: Bresenham picks integer cell trajectory minimizing distance from
    continuous ray. Atomic adds let many rays update the same cell safely.
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
    """Score every (dtheta, dy, dx) candidate around prior. One thread per candidate.

    Math per thread:
      1. candidate = prior + (dx, dy, dtheta)
      2. for each scan point p_k (sensor frame):
           world = candidate.translation + R(candidate.theta) @ p_k
           sample probability grid bilinearly (with HALF-CELL OFFSET)
           sum into accumulator
      3. score = sum / N (normalize for cross-scan comparability)
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
        # half-cell offset: cell value sits at the cell CENTER
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
def k_max_pool_2x2(
    src: wp.array2d(dtype=wp.float32), dst: wp.array2d(dtype=wp.float32)
):
    """Pyramid level: dst[i,j] = max of src[2i:2i+2, 2j:2j+2].

    Why max (not average): BBS needs the coarse-level value to be an UPPER
    BOUND on every fine-level value it covers. Max preserves this; average
    doesn't. The upper-bound property is what justifies pruning in BBS.
    """
    i, j = wp.tid()
    a = src[2 * i, 2 * j]
    b = src[2 * i, 2 * j + 1]
    c = src[2 * i + 1, 2 * j]
    d = src[2 * i + 1, 2 * j + 1]
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
    level_res: float,
    H: int,
    W: int,
    cands: wp.array(dtype=wp.vec3),
    scores: wp.array(dtype=wp.float32),
):
    """Score a batch of candidate (x, y, theta) poses against a pyramid level.
    Same scoring math as k_csm but with arbitrary candidate list.
    No half-cell offset here -- BBS uses cell-center sampling at the level's
    resolution, where the upper-bound property holds via direct cell lookup.
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
        ix = int((wx - gx) / level_res)
        iy = int((wy - gy) / level_res)
        if ix >= 0 and ix < W and iy >= 0 and iy < H:
            acc += grid[iy, ix]
    scores[k] = acc / float(n)


# =============================================================================
# Submap: log-odds grid + probability grid + (when finalized) pyramid
# =============================================================================
class Submap:
    """A 2D occupancy submap.

    Origin is the world pose at submap creation. The grid is in *submap-local*
    coordinates: cell (0,0) sits at submap-local (-W*r/2, -H*r/2). When the
    submap origin moves due to global optimization, the grid contents stay
    fixed in submap-local coords -- only the origin pose moves in the world.
    """

    _next_id = 0

    def __init__(self, origin_pose):
        self.id = Submap._next_id
        Submap._next_id += 1
        self.origin = origin_pose.copy()
        # local grid spans [-W*r/2, W*r/2] x [-H*r/2, H*r/2]
        self.gx = -GRID_HW * RES / 2.0
        self.gy = -GRID_HW * RES / 2.0
        self.log = wp.zeros((GRID_HW, GRID_HW), dtype=wp.float32, device=DEV)
        self.prob = wp.zeros_like(self.log)
        self.pyramid = []  # built at finalize()
        self.scans = []  # list of (sensor_pose_world, scan_local) for LC
        self.n = 0
        self.finalized = False
        self._prob_dirty = True

    def insert(self, sensor_pose_world, scan_local):
        """Insert a scan, recording it for later (LC) and rasterizing into the grid.

        sensor_pose_world: SE(2) pose of sensor in WORLD frame.
        scan_local: Nx2 endpoints in SENSOR frame.
        Internally we transform scan to submap-local frame for rasterization.
        """
        if self.finalized:
            raise RuntimeError("Cannot insert into a finalized submap")

        # 1. sensor pose in submap-local frame
        sensor_local = between(self.origin, sensor_pose_world)
        # 2. scan in submap-local frame: rotate by sensor_local.theta, translate by sensor_local.xy
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
        """Rebuild self.prob from self.log if dirty."""
        if self._prob_dirty:
            wp.launch(
                k_logodds_to_prob,
                dim=(GRID_HW, GRID_HW),
                inputs=[self.log, self.prob],
                device=DEV,
            )
            self._prob_dirty = False

    def finalize(self):
        """Build the multi-resolution max-pool pyramid for BBS."""
        if self.finalized:
            return
        self.refresh_prob()
        self.pyramid = [self.prob]
        for lvl in range(1, BBS_LEVELS):
            prev = self.pyramid[-1]
            ph, pw = prev.shape[0] // 2, prev.shape[1] // 2
            nxt = wp.zeros((ph, pw), dtype=wp.float32, device=DEV)
            wp.launch(k_max_pool_2x2, dim=(ph, pw), inputs=[prev, nxt], device=DEV)
            self.pyramid.append(nxt)
        self.finalized = True


# =============================================================================
# Correlative Scan Matcher (CSM): coarse pose recovery via brute-force search
# =============================================================================
def csm(submap, prior_local, scan_local):
    """Match a scan against a submap's probability grid.

    Inputs:
      submap: target submap (its prob grid will be refreshed)
      prior_local: SE(2) prior in submap-LOCAL frame (3,)
      scan_local: Nx2 scan in sensor frame
    Outputs:
      (best_pose_local, best_score) -- pose in submap-local frame
    """
    submap.refresh_prob()
    nx = int(2 * WIN_XY / STEP_XY) + 1
    ny = nx
    nth = int(2 * WIN_TH / STEP_TH) + 1

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
            STEP_XY,
            STEP_TH,
            scores,
        ],
        device=DEV,
    )

    s = scores.numpy()
    it, iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
    dth = (it - nth // 2) * STEP_TH
    dx = (ix - nx // 2) * STEP_XY
    dy = (iy - ny // 2) * STEP_XY
    best = np.array(
        [prior_local[0] + dx, prior_local[1] + dy, wrap(prior_local[2] + dth)]
    )
    return best, float(s[it, iy, ix])


# =============================================================================
# Ceres-style refinement: post-CSM polish to sub-grid accuracy
# =============================================================================
def refine(submap, csm_pose_local, prior_local, scan_local):
    """Nonlinear least-squares pose refinement on a smooth interpolated grid.

    Cost = sum_k (1 - M_smooth(T . p_k))^2
         + (W_T * (T.x - prior.x))^2 + (W_T * (T.y - prior.y))^2
         + (W_R * wrap(T.theta - prior.theta))^2

    Solved by Levenberg-Marquardt (scipy). Initialized at csm_pose_local.
    Operates entirely in submap-local frame.
    """
    submap.refresh_prob()
    prob = submap.prob.numpy()  # one host copy; reused across LM iterations

    def sample(world_xy):
        """Bilinear-interpolated probability at (x, y) in submap-local frame.
        Returns 0 outside the grid (treat as "no information")."""
        fx = (world_xy[:, 0] - submap.gx) / RES - 0.5
        fy = (world_xy[:, 1] - submap.gy) / RES - 0.5
        x0 = np.floor(fx).astype(int)
        y0 = np.floor(fy).astype(int)
        tx = fx - x0
        ty = fy - y0
        ok = (x0 >= 0) & (x0 + 1 < GRID_HW) & (y0 >= 0) & (y0 + 1 < GRID_HW)
        v = np.zeros_like(fx)
        if ok.any():
            xs0, ys0 = x0[ok], y0[ok]
            tx_, ty_ = tx[ok], ty[ok]
            v[ok] = (prob[ys0, xs0] * (1 - tx_) + prob[ys0, xs0 + 1] * tx_) * (
                1 - ty_
            ) + (prob[ys0 + 1, xs0] * (1 - tx_) + prob[ys0 + 1, xs0 + 1] * tx_) * ty_
        return v

    def residuals(T):
        c, s = np.cos(T[2]), np.sin(T[2])
        # transform each scan point to submap-local
        wx = T[0] + c * scan_local[:, 0] - s * scan_local[:, 1]
        wy = T[1] + s * scan_local[:, 0] + c * scan_local[:, 1]
        m = sample(np.column_stack([wx, wy]))
        # per-point residual: 1 - probability
        rp = 1.0 - m
        # prior residuals
        rt_x = REFINE_W_T * (T[0] - prior_local[0])
        rt_y = REFINE_W_T * (T[1] - prior_local[1])
        rr = REFINE_W_R * wrap(T[2] - prior_local[2])
        return np.concatenate([rp, [rt_x, rt_y, rr]])

    res = least_squares(
        residuals, csm_pose_local.copy(), method="lm", max_nfev=REFINE_ITERS * 4
    )
    out = res.x.copy()
    out[2] = wrap(out[2])
    return out


# =============================================================================
# Branch-and-Bound Search (BBS) for loop-closure
# =============================================================================
def _score_at_level(submap, level_idx, scan_local, candidates):
    """Score a list of (x, y, theta) candidates at a given pyramid level.

    Returns scores as numpy array, length len(candidates).
    """
    grid = submap.pyramid[level_idx]
    H, W = grid.shape
    # cell size at this level: a level-l cell covers 2^l native cells
    level_res = RES * (2**level_idx)
    # gx, gy of the level grid: same world origin, but level grid covers same
    # world extent in fewer larger cells, so origin is the same
    gx, gy = submap.gx, submap.gy

    cands = wp.array(candidates.astype(np.float32), dtype=wp.vec3, device=DEV)
    scores = wp.zeros(len(candidates), dtype=wp.float32, device=DEV)
    pts = wp.array(scan_local, dtype=wp.vec2, device=DEV)
    wp.launch(
        k_bbs_score,
        dim=len(candidates),
        inputs=[pts, grid, gx, gy, level_res, H, W, cands, scores],
        device=DEV,
    )
    return scores.numpy()


def bbs(
    submap,
    scan_local,
    prior_local,
    win_xy=BBS_WIN_XY,
    win_th=BBS_WIN_TH,
    min_score=LC_MIN_SCORE,
):
    """Branch-and-bound search for the best (x,y,theta) match in submap-local frame.

    Returns (best_pose_local, best_score) or (None, 0) if no match exceeds min_score.

    Algorithm:
      1. Generate root candidates: for each angle in dense angular sweep,
         use coarsest pyramid level, tile (x,y) at that level's cell size.
      2. Score them all in one batch on GPU.
      3. Sort descending. DFS: for each candidate above current best,
         either subdivide (4 finer-resolution sub-cells) or accept (if at
         finest level).
      4. The score at coarse levels upper-bounds any descendant, so any
         candidate below current best is pruned.
    """
    if not submap.finalized:
        return None, 0.0

    L = BBS_LEVELS - 1  # coarsest level index
    coarsest_res = RES * (2**L)  # m per coarse cell

    # Angular sweep: at the coarsest level, pose ambiguity in theta is bounded
    # by ~asin(coarsest_res / scan_max_range), but for simplicity we use a
    # fixed angular step. Cartographer derives this from scan extent.
    n_th = int(2 * win_th / BBS_STEP_TH) + 1
    thetas = np.linspace(prior_local[2] - win_th, prior_local[2] + win_th, n_th)

    # XY candidates at coarsest level: tile at coarsest_res
    n_xy = int(2 * win_xy / coarsest_res) + 1
    dxs = (np.arange(n_xy) - n_xy // 2) * coarsest_res
    xs = prior_local[0] + dxs
    ys = prior_local[1] + dxs

    # Build root candidates: (theta, y, x) cartesian product
    # Each element stores (x, y, theta, level_index, x_index_in_level, y_index_in_level)
    root_cands = []
    for th in thetas:
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                root_cands.append((x, y, wrap(th), L))

    # Score root candidates as a batch
    cand_xyz = np.array([(c[0], c[1], c[2]) for c in root_cands])
    if len(cand_xyz) == 0:
        return None, 0.0
    root_scores = _score_at_level(submap, L, scan_local, cand_xyz)

    # Pair with metadata, sort descending by score
    indexed = sorted(
        [(root_scores[i], root_cands[i]) for i in range(len(root_cands))],
        key=lambda t: -t[0],
    )

    # DFS with best-first ordering and pruning
    best_score = min_score
    best_pose = None
    # Use a list as a stack; entries: (score_upper_bound, cand_tuple)
    stack = [(s, c) for s, c in indexed]
    # already sorted highest-first; pop from end gives lowest first; we want
    # highest first so process from the front
    stack.reverse()  # so .pop() yields highest score

    while stack:
        score, cand = stack.pop()
        if score < best_score:
            continue  # prune: no descendant beats current best
        x, y, th, lvl = cand
        if lvl == 0:
            # leaf: accept
            best_score = score
            best_pose = np.array([x, y, th])
            continue
        # branch: subdivide (x,y) into 4 children at finer level
        finer_lvl = lvl - 1
        finer_res = RES * (2**finer_lvl)
        # Each parent cell covers 2x2 finer cells; children sit at the centers
        # of those finer cells. Parent (x,y) was the corner of a coarse cell;
        # finer cells offset by 0 or finer_res in each axis.
        children = [
            (x, y, th, finer_lvl),
            (x + finer_res, y, th, finer_lvl),
            (x, y + finer_res, th, finer_lvl),
            (x + finer_res, y + finer_res, th, finer_lvl),
        ]
        c_xyz = np.array([(c[0], c[1], c[2]) for c in children])
        c_scores = _score_at_level(submap, finer_lvl, scan_local, c_xyz)
        # Push children, lowest score first so highest is processed next
        for s, c in sorted(zip(c_scores, children), key=lambda t: t[0]):
            if s >= best_score:
                stack.append((s, c))

    if best_pose is None:
        return None, 0.0
    return best_pose, best_score


# =============================================================================
# Pose graph (GTSAM wrapper)
# =============================================================================
class PoseGraph:
    """Pose graph with two kinds of nodes: trajectory nodes and submap origins.

    Both are Pose2 variables in GTSAM. We use disjoint integer ID spaces so
    a single int identifies both kind and index.
    Trajectory nodes:  IDs 0      .. 999_999
    Submap origin IDs: IDs 1_000_000 .. 1_999_999
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
        """Run Levenberg-Marquardt to global optimum.

        GTSAM internally:
          1. linearizes every factor at current values, computing analytic
             SE(2) Jacobians (Grisetti tutorial appendix)
          2. assembles sparse H = J^T Omega J and b = J^T Omega r
          3. solves (H + lambda*I) dx = -b via sparse Cholesky (CHOLMOD)
          4. retracts: T_i <- T_i [+] dx_i, with theta wrapped
          5. accepts/rejects per LM rule, updates lambda, iterates
        """
        opt = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values)
        self.values = opt.optimize()

    def pose(self, key):
        p = self.values.atPose2(key)
        return np.array([p.x(), p.y(), p.theta()])


# =============================================================================
# SubmapStack: manages collection of submaps with two-active rotation
# =============================================================================
class SubmapStack:
    def __init__(self):
        self.submaps = []  # list of Submap, in creation order

    def all_active(self):
        return [s for s in self.submaps if not s.finalized]

    def matching_target(self):
        """Older active submap is the matching target -- has more accumulated data."""
        active = self.all_active()
        return active[0] if active else None

    def add_scan(self, sensor_pose_world, scan_local):
        """Insert into all active submaps. Returns list of submap IDs that received this scan.
        May trigger finalization of the older active submap.
        """
        active = self.all_active()
        # If no active submap, or the only active one is past half-full, start a new one
        if not active:
            new = Submap(sensor_pose_world)
            self.submaps.append(new)
            active = [new]
        elif len(active) == 1 and active[0].n >= SUBMAP_N // 2:
            new = Submap(sensor_pose_world)
            self.submaps.append(new)
            active.append(new)

        ids = []
        for sm in active:
            sm.insert(sensor_pose_world, scan_local)
            ids.append(sm.id)

        # Finalize the older active submap if it's full
        older = active[0]
        if older.n >= SUBMAP_N:
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
        # Map: submap.id -> graph_key for its origin
        self.submap_keys = {}
        # Map: graph_key for trajectory node -> (containing submap id, scan_local, sensor_pose_world)
        self.traj_meta = {}

    def _ensure_submap_in_graph(self, submap):
        """Make sure this submap has an origin node in the pose graph."""
        if submap.id not in self.submap_keys:
            key = self.graph.add_submap_origin(submap.origin)
            self.submap_keys[submap.id] = key
        return self.submap_keys[submap.id]

    def add_scan(self, scan_local, odom_delta):
        """Process one scan. Returns the (possibly post-optimization) world pose."""
        # 1. Motion prior in world frame
        prior_world = compose(self.last_pose, odom_delta)

        # 2. Match against current matching target
        target = self.stack.matching_target()
        if target is None or target.n < 2:
            # No data yet -- just accept the prior, low confidence
            matched_world = prior_world
            score = 0.0
        else:
            # Convert prior to submap-local frame
            prior_local = between(target.origin, prior_world)
            csm_local, score = csm(target, prior_local, scan_local)
            if score > 0.3:
                refined_local = refine(target, csm_local, prior_local, scan_local)
                matched_world = compose(target.origin, refined_local)
            else:
                # matcher unconfident -- trust the prior
                matched_world = prior_world

        # 3. Insert into submaps (may finalize the older submap)
        submap_ids = self.stack.add_scan(matched_world, scan_local)

        # 4. Add trajectory node and intra-submap edges
        node_key = self.graph.add_trajectory_node(matched_world)
        self.traj_meta[node_key] = {
            "submap_ids": list(submap_ids),
            "scan": scan_local.copy(),
            "sensor_pose_world": matched_world.copy(),
        }
        for sid in submap_ids:
            sm = next(s for s in self.stack.submaps if s.id == sid)
            sk = self._ensure_submap_in_graph(sm)
            rel = between(sm.origin, matched_world)
            self.graph.add_intra_submap(sk, node_key, rel)

        self.scans_since_optimize += 1

        # 5. Loop closure: every OPTIMIZE_EVERY scans, search for closures
        lc_added = False
        if self.scans_since_optimize >= OPTIMIZE_EVERY and node_key > LC_SKIP_NODES:
            lc_added = self._try_loop_closures(node_key, matched_world, scan_local)
            self.scans_since_optimize = 0

        if lc_added:
            self.graph.optimize()
            # refresh: pull updated pose for current node, update submap origins
            matched_world = self.graph.pose(node_key)
            for sid, sk in self.submap_keys.items():
                sm = next(s for s in self.stack.submaps if s.id == sid)
                sm.origin = self.graph.pose(sk)

        self.last_pose = matched_world
        return matched_world

    def _try_loop_closures(self, node_key, node_pose_world, scan_local):
        """For each finalized submap not adjacent to this node, run BBS.
        Add a loop-closure edge for each successful match.
        """
        added = False
        current_submap_ids = set(self.traj_meta[node_key]["submap_ids"])
        for sm in self.stack.submaps:
            if not sm.finalized:
                continue
            if sm.id in current_submap_ids:
                continue
            # spatial gating: skip submaps far from current pose
            d = np.linalg.norm(sm.origin[:2] - node_pose_world[:2])
            if d > BBS_WIN_XY:
                continue
            # BBS starting from current best estimate of the node in submap-local frame
            prior_local = between(sm.origin, node_pose_world)
            best, score = bbs(
                sm,
                scan_local,
                prior_local,
                win_xy=2.0,
                win_th=0.5,
                min_score=LC_MIN_SCORE,
            )
            if best is None:
                continue
            # Refine the BBS hit to sub-grid accuracy
            refined = refine(sm, best, prior_local, scan_local)
            self.graph.add_loop_closure(self.submap_keys[sm.id], node_key, refined)
            added = True
        return added

    def current_pose(self):
        return self.last_pose.copy()

    def snapshot(self):
        """Return a dict suitable for visualization."""
        return {
            "submaps": [
                (
                    s.id,
                    s.origin.copy(),
                    s.finalized,
                    s.log.numpy() if not s.finalized else s.prob.numpy(),
                )
                for s in self.stack.submaps
            ],
            "trajectory": [self.graph.pose(k) for k in sorted(self.traj_meta.keys())],
        }


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    # Minimal integration check: import + instantiate + run a few steps with
    # a simple synthetic scan. For real validation use test_e2e.py.
    print(f"Warp device: {DEV}")
    print(f"Submap size: {GRID_HW}x{GRID_HW} @ {RES}m/cell ({GRID_HW * RES}m)")
    print(
        f"CSM candidates: {(2 * int(WIN_XY / STEP_XY) + 1) ** 2 * (2 * int(WIN_TH / STEP_TH) + 1)}"
    )
    print(
        f"BBS pyramid: {BBS_LEVELS} levels (coarsest cell {RES * 2 ** (BBS_LEVELS - 1)}m)"
    )

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
        # crude scan from these wall points
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
