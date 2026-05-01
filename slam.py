import gtsam
import numpy as np
import warp as wp
from scipy.ndimage import map_coordinates, spline_filter
from scipy.optimize import least_squares

wp.init()
DEV = "cuda" if wp.is_cuda_available() else "cpu"

RES = 0.05
L_HIT, L_MISS = 0.85, -0.40
L_MIN, L_MAX = -2.2, 2.2
GRID_HW = 400

NUM_RANGE_DATA = 90
SUBMAP_TOTAL_SCANS = 2 * NUM_RANGE_DATA

MATCH_VOXEL_MAX_POINTS = 200
LC_VOXEL_MAX_POINTS = 100
VOXEL_MIN_SIZE = 0.025
VOXEL_MAX_SIZE = 0.5

MIN_MOTION_DISTANCE = 0.20
MIN_MOTION_ANGLE = np.deg2rad(1.0)
MAX_SCANS_BETWEEN_NODES = 100

USE_CSM = False
CSM_WIN_XY, CSM_STEP_XY = 0.15, 0.025
CSM_WIN_TH, CSM_STEP_TH = 0.17, 0.005
CSM_MIN_SCORE = 0.30
REFINE_ITERS = 20
REFINE_W_T = 10.0
REFINE_W_R = 40.0

BBS_LEVELS = 6
LC_MIN_SCORE = 0.55
LC_WIN_XY = 2.0
LC_WIN_TH = 0.5
LC_MAX_DISTANCE = 15.0

SIGMA_INTRA = np.array([0.025, 0.025, 0.005])
SIGMA_LC = np.array([0.05, 0.05, 0.02])
SIGMA_PRIOR = np.array([1e-3, 1e-3, 1e-3])
HUBER_K = 1.0

OPTIMIZE_EVERY = 30
LC_SKIP_NODES = 50
LC_RECENT_WINDOW = 50


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array(
        [a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])]
    )


def inverse(a):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]])


def between(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])


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
    i = wp.tid()
    p = pts[i]
    x0 = int(wp.floor((origin[0] - gx) / res))
    y0 = int(wp.floor((origin[1] - gy) / res))
    x1 = int(wp.floor((p[0] - gx) / res))
    y1 = int(wp.floor((p[1] - gy) / res))
    if x0 == x1 and y0 == y1:
        return
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
    i, j = wp.tid()
    s = seen[i, j]
    v = log[i, j]
    if s > 1.5:
        v += lhit
    elif s > 0.5:
        v += lmiss
    log[i, j] = wp.clamp(v, lo, hi)
    seen[i, j] = 0.0


@wp.kernel
def k_logodds_to_prob(
    lo: wp.array2d(dtype=wp.float32), pr: wp.array2d(dtype=wp.float32)
):
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
    it, iy, ix = wp.tid()
    th = pth + (float(it) - float(nth) / 2.0) * sth
    cx = px + (float(ix) - float(nx) / 2.0) * sxy
    cy = py + (float(iy) - float(ny) / 2.0) * sxy
    c = wp.cos(th)
    s = wp.sin(th)
    n = pts.shape[0]
    acc = float(0.0)
    for k in range(n):
        p = pts[k]
        wx = cx + c * p[0] - s * p[1]
        wy = cy + s * p[0] + c * p[1]
        gx_i = int(wp.floor((wx - gx) / res))
        gy_i = int(wp.floor((wy - gy) / res))
        if gx_i >= 0 and gx_i < W and gy_i >= 0 and gy_i < H:
            acc += grid[gy_i, gx_i]
    scores[it, iy, ix] = acc / float(n)


@wp.kernel
def k_pyramid_step(
    src: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    offset: int,
    H: int,
    W: int,
):
    i, j = wp.tid()
    m = src[i, j]
    if i + offset < H and src[i + offset, j] > m:
        m = src[i + offset, j]
    if j + offset < W and src[i, j + offset] > m:
        m = src[i, j + offset]
    if i + offset < H and j + offset < W and src[i + offset, j + offset] > m:
        m = src[i + offset, j + offset]
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
    k = wp.tid()
    cand = cands[k]
    c = wp.cos(cand[2])
    s = wp.sin(cand[2])
    n = pts.shape[0]
    acc = float(0.0)
    for i in range(n):
        p = pts[i]
        wx = cand[0] + c * p[0] - s * p[1]
        wy = cand[1] + s * p[0] + c * p[1]
        ix = int(wp.floor((wx - gx) / res))
        iy = int(wp.floor((wy - gy) / res))
        if ix >= 0 and ix < W and iy >= 0 and iy < H:
            acc += grid[iy, ix]
    scores[k] = acc / float(n)


def _voxel_filter(scan, voxel_size):
    if len(scan) == 0:
        return scan
    keys = np.floor(scan / voxel_size).astype(np.int64)
    packed = (keys[:, 0] << 32) | (keys[:, 1] & 0xFFFFFFFF)
    _, idx = np.unique(packed, return_index=True)
    return scan[idx]


def adaptive_voxel_filter(scan, max_points):
    if len(scan) <= max_points:
        return scan
    tight = _voxel_filter(scan, VOXEL_MIN_SIZE)
    if len(tight) <= max_points:
        return tight
    lo, hi = VOXEL_MIN_SIZE, VOXEL_MAX_SIZE
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        if len(_voxel_filter(scan, mid)) <= max_points:
            hi = mid
        else:
            lo = mid
    return _voxel_filter(scan, hi)


class MotionFilter:
    def __init__(self):
        self.last = None

    def is_similar(self, pose):
        if self.last is None:
            return False
        d = float(np.linalg.norm(pose[:2] - self.last[:2]))
        a = abs(wrap(pose[2] - self.last[2]))
        return d < MIN_MOTION_DISTANCE and a < MIN_MOTION_ANGLE

    def update(self, pose):
        self.last = pose.copy()


class Submap:
    def __init__(self, origin_pose, sid):
        self.id = sid
        self.origin = origin_pose.copy()
        self.gx = -GRID_HW * RES / 2.0
        self.gy = -GRID_HW * RES / 2.0
        self.log = wp.zeros((GRID_HW, GRID_HW), dtype=wp.float32, device=DEV)
        self.prob = wp.zeros_like(self.log)
        self.seen = wp.zeros_like(self.log)
        self.precomp = []
        self.n = 0
        self.finalized = False
        self._prob_dirty = True
        self._filtered_prob = None

    def insert(self, sensor_pose_world, scan_local):
        if self.finalized:
            raise RuntimeError("Cannot insert into finalized submap")
        if len(scan_local) == 0:
            return
        sensor_local = between(self.origin, sensor_pose_world)
        c, s = np.cos(sensor_local[2]), np.sin(sensor_local[2])
        scan_submap = scan_local @ np.array([[c, s], [-s, c]]) + sensor_local[:2]
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
        if self._prob_dirty:
            wp.launch(
                k_logodds_to_prob,
                dim=(GRID_HW, GRID_HW),
                inputs=[self.log, self.prob],
                device=DEV,
            )
            self._prob_dirty = False

    def finalize(self):
        if self.finalized:
            return
        self.refresh_prob()
        self.precomp = [self.prob]
        for h in range(1, BBS_LEVELS):
            nxt = wp.zeros((GRID_HW, GRID_HW), dtype=wp.float32, device=DEV)
            wp.launch(
                k_pyramid_step,
                dim=(GRID_HW, GRID_HW),
                inputs=[self.precomp[-1], nxt, 2 ** (h - 1), GRID_HW, GRID_HW],
                device=DEV,
            )
            self.precomp.append(nxt)
        self._filtered_prob = spline_filter(self.prob.numpy(), order=3, mode="constant")
        self.finalized = True
        self.log = None
        self.seen = None


def csm(submap, prior_local, scan):
    if len(scan) == 0:
        return prior_local.copy(), 0.0
    submap.refresh_prob()
    nx = int(2 * CSM_WIN_XY / CSM_STEP_XY) + 1
    nth = int(2 * CSM_WIN_TH / CSM_STEP_TH) + 1
    pts = wp.array(scan, dtype=wp.vec2, device=DEV)
    scores = wp.zeros((nth, nx, nx), dtype=wp.float32, device=DEV)
    wp.launch(
        k_csm,
        dim=(nth, nx, nx),
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
            nx,
            nth,
            CSM_STEP_XY,
            CSM_STEP_TH,
            scores,
        ],
        device=DEV,
    )
    s = scores.numpy()
    it, iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
    best = np.array(
        [
            prior_local[0] + (ix - nx // 2) * CSM_STEP_XY,
            prior_local[1] + (iy - nx // 2) * CSM_STEP_XY,
            wrap(prior_local[2] + (it - nth // 2) * CSM_STEP_TH),
        ]
    )
    return best, float(s[it, iy, ix])


def _make_sample_fn(submap):
    if submap._filtered_prob is not None:
        filtered = submap._filtered_prob
    else:
        submap.refresh_prob()
        filtered = spline_filter(submap.prob.numpy(), order=3, mode="constant")

    def sample(world_xy):
        fx = (world_xy[:, 0] - submap.gx) / RES - 0.5
        fy = (world_xy[:, 1] - submap.gy) / RES - 0.5
        return map_coordinates(
            filtered,
            np.vstack([fy, fx]),
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

    return sample


def refine(submap, init_pose_local, scan):
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
    if res.status <= 0:
        return init_pose_local.copy()
    out = res.x.copy()
    out[2] = wrap(out[2])
    return out


def _angular_step(scan):
    if len(scan) == 0:
        return 0.01
    d_max = float(np.linalg.norm(scan, axis=1).max())
    if d_max < 2 * RES:
        return 0.1
    return float(np.arccos(max(-1.0, 1.0 - RES * RES / (2.0 * d_max * d_max))))


def bbs(submap, scan, prior_local, win_xy, win_th, min_score=LC_MIN_SCORE):
    if not submap.finalized or len(scan) == 0:
        return None, 0.0
    L = BBS_LEVELS - 1
    coarsest_res = RES * (2**L)
    pts = wp.array(scan, dtype=wp.vec2, device=DEV)

    def score_at(lvl, cand_xyz):
        grid = submap.precomp[lvl]
        H, W = grid.shape
        cands = wp.array(cand_xyz.astype(np.float32), dtype=wp.vec3, device=DEV)
        scores = wp.zeros(len(cand_xyz), dtype=wp.float32, device=DEV)
        wp.launch(
            k_bbs_score,
            dim=len(cand_xyz),
            inputs=[pts, grid, submap.gx, submap.gy, RES, H, W, cands, scores],
            device=DEV,
        )
        return scores.numpy()

    dtheta = _angular_step(scan)
    thetas = np.linspace(
        prior_local[2] - win_th,
        prior_local[2] + win_th,
        int(np.ceil(2 * win_th / dtheta)) + 1,
    )
    half_n = int(np.ceil(win_xy / coarsest_res))
    offsets = (np.arange(2 * half_n + 1) - half_n) * coarsest_res
    TH, Y, X = np.meshgrid(
        thetas, prior_local[1] + offsets, prior_local[0] + offsets, indexing="ij"
    )
    root_xyz = np.column_stack([X.ravel(), Y.ravel(), wrap(TH.ravel())])
    if len(root_xyz) == 0:
        return None, 0.0

    root_scores = score_at(L, root_xyz)
    stack = [
        (
            float(root_scores[k]),
            float(root_xyz[k, 0]),
            float(root_xyz[k, 1]),
            float(root_xyz[k, 2]),
            L,
        )
        for k in np.argsort(root_scores)
    ]

    best_score = min_score
    best_pose = None
    x_lo, x_hi = prior_local[0] - win_xy, prior_local[0] + win_xy
    y_lo, y_hi = prior_local[1] - win_xy, prior_local[1] + win_xy

    while stack:
        score, x, y, th, lvl = stack.pop()
        if score < best_score:
            continue
        if lvl == 0:
            if x_lo <= x <= x_hi and y_lo <= y <= y_hi:
                best_score = score
                best_pose = np.array([x, y, th])
            continue
        finer = lvl - 1
        fr = RES * (2**finer)
        child_xyz = np.array(
            [[x, y, th], [x + fr, y, th], [x, y + fr, th], [x + fr, y + fr, th]]
        )
        c_scores = score_at(finer, child_xyz)
        for k in np.argsort(c_scores):
            if c_scores[k] >= best_score:
                stack.append(
                    (
                        float(c_scores[k]),
                        float(child_xyz[k, 0]),
                        float(child_xyz[k, 1]),
                        float(child_xyz[k, 2]),
                        finer,
                    )
                )

    if best_pose is None:
        return None, 0.0
    return best_pose, best_score


class PoseGraph:
    SUBMAP_ID_OFFSET = 1_000_000

    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self._next_traj = 0
        self._next_submap = 0
        self._intra_noise = gtsam.noiseModel.Diagonal.Sigmas(SIGMA_INTRA)
        self._lc_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(HUBER_K),
            gtsam.noiseModel.Diagonal.Sigmas(SIGMA_LC),
        )
        self._prior = gtsam.noiseModel.Diagonal.Sigmas(SIGMA_PRIOR)
        self._anchored = False

    def add_trajectory_node(self, pose):
        nid = self._next_traj
        self._next_traj += 1
        self.values.insert(nid, gtsam.Pose2(*pose))
        if not self._anchored:
            self.graph.add(gtsam.PriorFactorPose2(nid, gtsam.Pose2(*pose), self._prior))
            self._anchored = True
        return nid

    def add_submap_origin(self, pose):
        sid = self.SUBMAP_ID_OFFSET + self._next_submap
        self._next_submap += 1
        self.values.insert(sid, gtsam.Pose2(*pose))
        return sid

    def add_intra_submap(self, sk, nk, rel):
        self.graph.add(
            gtsam.BetweenFactorPose2(sk, nk, gtsam.Pose2(*rel), self._intra_noise)
        )

    def add_loop_closure(self, sk, nk, rel):
        self.graph.add(
            gtsam.BetweenFactorPose2(sk, nk, gtsam.Pose2(*rel), self._lc_noise)
        )

    def optimize(self):
        self.values = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.values
        ).optimize()

    def pose(self, key):
        p = self.values.atPose2(key)
        return np.array([p.x(), p.y(), p.theta()])


class SubmapStack:
    def __init__(self):
        self.submaps = []
        self._by_id = {}
        self._active = []
        self._finalized = []
        self._next_id = 0

    def by_id(self, sid):
        return self._by_id[sid]

    def matching_target(self):
        return self._active[0] if self._active else None

    def finalized_submaps(self):
        return self._finalized

    def add_scan(self, sensor_pose_world, scan_local):
        if not self._active or (
            len(self._active) == 1 and self._active[0].n >= NUM_RANGE_DATA
        ):
            sm = Submap(sensor_pose_world, self._next_id)
            self._next_id += 1
            self.submaps.append(sm)
            self._active.append(sm)
            self._by_id[sm.id] = sm
        ids = []
        for sm in self._active:
            sm.insert(sensor_pose_world, scan_local)
            ids.append(sm.id)
        if self._active[0].n >= SUBMAP_TOTAL_SCANS:
            older = self._active.pop(0)
            older.finalize()
            self._finalized.append(older)
        return ids


class Slam:
    MIN_SCANS_FOR_MATCHING = 5

    def __init__(self):
        self.stack = SubmapStack()
        self.graph = PoseGraph()
        self.motion_filter = MotionFilter()
        self.last_pose = np.zeros(3)
        self.scans_since_optimize = 0
        self._scans_since_node = 0
        self.submap_keys = {}
        self.traj_meta = {}
        self._lc_attempted = set()
        self._lc_found = set()

    def _ensure_submap_key(self, submap):
        if submap.id not in self.submap_keys:
            self.submap_keys[submap.id] = self.graph.add_submap_origin(submap.origin)
        return self.submap_keys[submap.id]

    def add_scan(self, scan_local, odom_delta):
        if len(scan_local) == 0:
            self.last_pose = compose(self.last_pose, odom_delta)
            return self.last_pose.copy()

        match_scan = adaptive_voxel_filter(scan_local, MATCH_VOXEL_MAX_POINTS)
        prior_world = compose(self.last_pose, odom_delta)
        target = self.stack.matching_target()

        if target is None or target.n < self.MIN_SCANS_FOR_MATCHING:
            matched_world = prior_world
        else:
            prior_local = between(target.origin, prior_world)
            init_local = prior_local
            if USE_CSM:
                csm_local, score = csm(target, prior_local, match_scan)
                if score > CSM_MIN_SCORE:
                    init_local = csm_local
            matched_world = compose(
                target.origin, refine(target, init_local, match_scan)
            )

        self.last_pose = matched_world
        self._scans_since_node += 1
        if (
            self.motion_filter.is_similar(matched_world)
            and self._scans_since_node < MAX_SCANS_BETWEEN_NODES
        ):
            return matched_world.copy()
        self._scans_since_node = 0
        self.motion_filter.update(matched_world)

        submap_ids = self.stack.add_scan(matched_world, scan_local)
        node_key = self.graph.add_trajectory_node(matched_world)
        self.traj_meta[node_key] = {
            "submap_ids": list(submap_ids),
            "scan": scan_local.copy(),
        }
        for sid in submap_ids:
            sm = self.stack.by_id(sid)
            self.graph.add_intra_submap(
                self._ensure_submap_key(sm), node_key, between(sm.origin, matched_world)
            )

        self.scans_since_optimize += 1
        if self.scans_since_optimize >= OPTIMIZE_EVERY and node_key > LC_SKIP_NODES:
            if self._sweep_loop_closures(node_key) > 0:
                self.graph.optimize()
                for sid, sk in self.submap_keys.items():
                    self.stack.by_id(sid).origin = self.graph.pose(sk)
                matched_world = self.graph.pose(node_key)
                self.last_pose = matched_world
                self._lc_attempted.clear()
            self.scans_since_optimize = 0

        return matched_world.copy()

    def _sweep_loop_closures(self, current_node_key):
        recent = sorted(self.traj_meta.keys())[-LC_RECENT_WINDOW:]
        if current_node_key not in recent:
            recent.append(current_node_key)
        added = 0
        for nk in recent:
            if nk <= LC_SKIP_NODES:
                continue
            node_pose = self.graph.pose(nk)
            lc_scan = adaptive_voxel_filter(
                self.traj_meta[nk]["scan"], LC_VOXEL_MAX_POINTS
            )
            node_submap_ids = set(self.traj_meta[nk]["submap_ids"])
            for sm in self.stack.finalized_submaps():
                if sm.id in node_submap_ids:
                    continue
                pair = (nk, sm.id)
                if pair in self._lc_found or pair in self._lc_attempted:
                    continue
                if np.linalg.norm(sm.origin[:2] - node_pose[:2]) > LC_MAX_DISTANCE:
                    self._lc_attempted.add(pair)
                    continue
                best, _ = bbs(
                    sm, lc_scan, between(sm.origin, node_pose), LC_WIN_XY, LC_WIN_TH
                )
                if best is None:
                    self._lc_attempted.add(pair)
                    continue
                self.graph.add_loop_closure(
                    self.submap_keys[sm.id], nk, refine(sm, best, lc_scan)
                )
                self._lc_found.add(pair)
                added += 1
        return added

    def current_pose(self):
        return self.last_pose.copy()
