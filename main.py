# pyright: reportIndexIssue=false, reportArgumentType=false
import gtsam
import numpy as np
import warp as wp
from scipy.ndimage import map_coordinates, spline_filter
from scipy.optimize import least_squares

RES = 0.05  # m / cell
L_HIT, L_MISS = 0.85, -0.40
L_MIN, L_MAX = -2.2, 2.2
GRID_HW = 400  # 20m x 20m submap

USE_CSM = False # off by default; enable if odometry is unreliable
CSM_WIN_XY, CSM_STEP_XY = 0.15, 0.025
CSM_WIN_TH, CSM_STEP_TH = 0.17, 0.005
CSM_MIN_SCORE = 0.30 # below this, fall back to odometry-only init for refinement

BBS_LEVELS = 6  # precomputed grid levels (covers 2^5 = 32 cells = 1.6m windowing)
LC_MIN_SCORE = 0.55
LC_WIN_XY = 2.0  # m, half-window for local LC search
LC_WIN_TH = 0.5  # rad, half-window for local LC search
LC_MAX_DISTANCE = 15.0  # m, spatial gating

wp.init()
DEV = "cuda" if wp.is_cuda_available() else "cpu"


def wrap(theta: np.float32):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def compose(a: np.ndarray, b: np.ndarray):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array(
        (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))
    )


def inverse(a: np.ndarray):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array((-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]))


def between(a: np.ndarray, b: np.ndarray):
    c, s = np.cos(a[2]), np.sin(a[2])
    delta = np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]])
    return np.array(
        (c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], wrap(delta[2]))
    )


@wp.kernel
def k_cast(
    pts: wp.array[wp.vec2],
    origin: wp.vec2,
    grid: wp.array2d,
    gx: wp.float,
    gy: wp.float,
    res: wp.float,
    l_hit: wp.float,
    l_miss: wp.float,
    H: wp.int,
    W: wp.int,
):
    i = wp.tid()
    p = pts[i]
    x0 = wp.int((origin[0] - gx) / res)
    y0 = wp.int((origin[1] - gy) / res)
    x1 = wp.int((p[0] - gx) / res)
    y1 = wp.int((p[1] - gy) / res)
    dx = x1 - x0
    dy = y1 - y0
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    dx *= sx
    dy *= sy
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
            wp.atomic_add(grid, y, x, l_miss)
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    if x1 >= 0 and x1 < W and y1 >= 0 and y1 < H:
        wp.atomic_add(grid, y1, x1, l_hit)


@wp.kernel
def k_clamp(grid: wp.array2d, lo: float, high: float):
    i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    grid[i, j] = wp.clamp(grid[i, j], lo, high)


@wp.kernel
def k_logodds_to_prob(lo: wp.array2d[wp.float], pr: wp.array2d[wp.float]):
    i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    pr[i, j] = 1.0 / (1.0 + wp.exp(-lo[i, j]))


@wp.kernel
def k_csm(
    pts: wp.array[wp.vec2],
    grid: wp.array2d[wp.float],
    gx: wp.float,
    gy: wp.float,
    res: wp.float,
    H: wp.int,
    W: wp.int,
    px: wp.float,
    py: wp.float,
    pth: wp.float,
    nx: wp.int,
    ny: wp.int,
    nth: wp.int,
    sxy: wp.float,
    sth: wp.float,
    scores: wp.array3d[wp.float],
):
    it, iy, ix = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    dth = (wp.float(it) - wp.float(nth) / 2.0) * sth
    dx = (wp.float(ix) - wp.float(nx) / 2.0) * sxy
    dy = (wp.float(iy) - wp.float(ny) / 2.0) * sxy
    th = pth + dth
    cx = px + dx
    cy = py + dy
    c, s = wp.cos(th), wp.sin(th)
    n = pts.shape[0]
    acc = 0.0
    for k in range(n):
        p = pts[k]
        wx = cx + c * p[0] - s * p[1]
        wy = cy + s * p[0] + c * p[1]
        fx = (wx - gx) / res - 0.5
        fy = (wy - gy) / res - 0.5
        x0 = wp.int(wp.floor(fx))
        y0 = wp.int(wp.floor(fy))
        tx = fx - wp.float(x0)
        ty = fy - wp.float(y0)
        if x0 >= 0 and x0 + 1 < W and y0 >= 0 and y0 + 1 < H:
            v = (grid[y0, x0] * (1.0 - tx) + grid[y0, x0 + 1] * tx) * (1.0 - ty) + (
                grid[y0 + 1, x0] * (1.0 - tx) + grid[y0 + 1, x0 + 1] * tx
            ) * ty
            acc += v
    scores[it, iy, ix] = acc / wp.float(n)


def k_pyramid_step(
    src: wp.array2d[wp.float],
    dst: wp.array2d[wp.float],
    offset: wp.int,
    H: wp.int,
    W: wp.int,
):
    i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    a = src[i, j]
    b = src[i + offset, j] if i + offset < H else 0.0
    c = src[i, j + offset] if j + offset < W else 0.0
    d = src[i + offset, j + offset] if i + offset < H and j + offset < W else 0.0
    m = wp.max(a, b)
    m = wp.max(m, c)
    m = wp.max(m, d)
    dst[i, j] = m


@wp.kernel
def k_bbs_score(
    pts: wp.array[wp.vec2],
    grid: wp.array2d[wp.float],
    gx: float,
    gy: float,
    res: float,
    H: int,
    W: int,
    cands: wp.array[wp.vec3],
    scores: wp.array[wp.float],
):
    k = wp.tid()
    cand = cands[k]
    cx = cand[0]
    cy = cand[1]
    th = cand[2]
    c, s = wp.cos(th), wp.sin(th)
    n = pts.shape[0]
    acc = 0.0
    for i in range(n):
        p = pts[i]
        wx = cx + c * p[0] - s * p[1]
        wy = cy + s * p[0] + c * p[1]
        ix = wp.int((wx - gx) / res)
        iy = wp.int((wy - gy) / res)
        if ix >= 0 and ix < W and iy >= 0 and iy < H:
            acc += grid[iy, ix]
    scores[k] = acc / wp.float(n)


class Submap:
    def __init__(self, origin_pose):
        self.id = Submap._next_id()
        Submap._next_id += 1
        self.origin = origin_pose.copy()
        self.gx = -GRID_HW * RES / 2.0
        self.gy = -GRID_HW * RES / 2.0
        self.log = wp.zeros((GRID_HW, GRID_HW), dtype=wp.float, device=DEV)
        self.prob = wp.zeros((GRID_HW, GRID_HW), dtype=wp.float, device=DEV)
        self.precomp = []
        self.scan = []
        self.n = 0
        self.finalized = False
        self._prob_dirty = True

    def insert(self, sensor_pose_world, scan_local):
        if self.finalized:
            raise RuntimeError("Cannot insert into finalized submap")
        if len(scan_local) == 0:
            return

        sensor_local = between(self.origin, sensor_pose_world)
        c, s = np.cos(sensor_local[2]), np.sin(sensor_local[2])
        R = np.array([[c, -s], [s, c]])
        scan_submap = scan_local @ R.T + sensor_local[:2]
        pts = wp.array(scan_submap, dtype=wp.vec2, device=DEV)
        wp.launch(
            k_cast,
            dim=len(scan_submap),
            inputs=[
                pts,
                wp.vec2(sensor_local[0], sensor_local[1]),
                self.log,
                self.gx,
                self.gy,
                RES,
                L_HIT,
                L_MISS,
                GRID_HW,
                GRID_HW,
            ],
        )
        wp.launch(
            k_clamp,
            dim=(GRID_HW, GRID_HW),
            inputs=[self.log, L_MIN, L_MAX],
            device=DEV,
        )
        self.scan.append(pts)
        self.n += 1
        self._prob_dirty = True

    def refresh_probs(self):
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
        self.refresh_probs()
        self.precomp = [self.prob]
        for h in range(1, BBS_LEVELS):
            prev = self.precomp[-1]
            offset = 2 ** (h - 1)
            nxt = wp.zeros((GRID_HW, GRID _HW), dtype=wp.float, device=DEVICE)
            wp.launch(
                k_pyramid_step,
                dim=(GRID_HW, GRID_HW),
                inputs=[prev, nxt, offset, GRID_HW, GRID_HW],
                device=DEV
            )
            self.precomp.append(nxt)
        self.finalized = True
        self.log = None

def csm(submap, prior_local, scan_local):
    if len(scan_local) == 0:
        return prior_local.copy(), 0.0
    submap.refresh_probs()
    nx = wp.int(2 * CSM_WIN_XY / CSM_STEP_XY) + 1
    ny = nx
    nth = wp.int(2 * CSM_WIN_TH / CSM_STEP_TH) + 1

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
            wp.float(prior_local[0]),
            wp.float(prior_local[1]),
            wp.float(prior_local[2]),
            nx,
            ny,
            nth,
            CSM_STEP_XY,
            CSM_STEP_TH,
            scores
        ]
    )
    s = scores.numpy()
    it, iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
    dth = (it - nth // 2) * CSM_STEP_TH
    dx = (ix - nx // 2) * CSM_STEP_XY
    dy = (iy - ny // 2) * CSM_STEP_XY
    best = np.array(
        (prior_local[0] + dx, prior_local[1] + dy, wrap(prior_local[2] + dth))
    )
    return best, s[it, iy, ix]

def _make_sample_fn(submap):
    submap.refresh_probs()
    prob = submap.prob.numpy()
    filtered = spline_filter(prob, mode="constant")
    def sample(world_xy):
        fx = (world_xy[:, 0] - submap.gx) / RES - 0.5
        fy = (world_xy[:, 1] - submap.gy) / RES - 0.5
        coords = np.vstack([fy, fx])
        return map_coordinates(
            filtered, coords, cval=0.5, prefilter=False
        )
    return sample

def refine(submap, init_pose_local, scan_local, prior_local):
    if len(scan_local) == 0:
        return init_pose_local.copy()
    sample = _make_sample_fn(submap)

    def residual(T):
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
        residuals, init_pose_local.copy(), method="lm",
        max_nfev=REFINE_ITRS * 4
    )
    out = res.x.copy()
    out[2] = wrap(out[2])
    return out

def _angular_step(scan_local, res):
    if len(scan_local) == 0:
        return 0.01
    d_max = float(np.linalg.norm(scan_local, aixs=1).max())
    if d_max < 2 * res:
        return 0.1
    cos_arg = max(-1.0, 1.0 - res * res / (2.0 * d_max * d_max))
    return float(np.arccos(cos_arg))

def _score_at_level(submap: Submap, level_idx, scan_local, candidates):
    grid = submap.precomp[level_idx]
    H, W = grid.shape
    cands = wp.array(candidates.astype(np.float32), dtype=wp.vec3, device=DEV)
    scores = wp.zeros(len(candidates), dtype=wp.float32, device=DEV)
    pts = wp.array(scan_local, dtype=wp.vec2, device=DEV)
    wp.launch(
        k_bbs_score,
        dim=len(candidates),
        inputs=[pts, grid, submap.gx, submap.gy, RES, H , W, cands, scores],
        device=DEV
    )
    return scores.numpy()

def bbs(submap, scan_local, prior_local, win_xy, win_th):
    if not submap.finalized or len(scan_local) == 0:
        return None, 0.0

    L = BBS_LEVELS - 1
    coarsest_res = RES * (2**L)

    dtheta = _angular_step(scan_local, RES)
    n_th = int(np.ceil(2 * win_th / dtheta)) + 1
    thetas = np.linspace(prior_local[2] - win_th, prior_local[2] + win_th, n_th)

    half_n = int(np.ceil(win_xy / coarsest_res))
    offsets = (np.arange(2 * half_n+1) - half_n) * coarsest_res
    xs = prior_local[0] + offsets
    ys = prior_local[1] + offsets

    root_cands = [(x, y, wrap(theta), L) for x in xs for y in ys for theta in thetas]
    if not root_cands:
        return None, 0.0
    cand_xyz = np.array([(c[0], c[1], c[2]) for c in root_cands])
    root_scores = _score_at_level(submap, L, scan_local, cand_xyz)

    indexed = sorted(zip(root_scores.tolist(), root_cands), key=lambda t: t[0])
    stack = list(indexed)

    best_score = LC_MIN_SCORE
    best_pose = None

    while stack:
        score, cand = stack.pop()
        if score < best_score:
            continue
        x, y, th, lvl = cand
        if lvl == 0:
            best_score = score
            best_pose = np.array([x, y, th])
            continue
        finer_lvl = lvl - 1
        finer_res = RES * (2**finer_lvl)
        children = [
            (x, y, th, finer_lvl),
            (x + finer_res, y, th, finer_lvl),
            (x, y + finer_res, th, finer_lvl),
            (x + finer_res, y + finer_res, th, finer_lvl)
        ]
        c_xyz = np.array([(cc[0], cc[1], cc[2]) for cc in  children])
        c_scores = _score_at_level(submap, finer_lvl, scan_local, c_xyz)
        for s, cc in sorted(zip(c_scores.tolist(), children), key=lambda t: t[0]):
            if s >= best_score:
                stack.append((s, cc))

    if best_pose is None:
        return None, 0.0
    return best_pose, best_score



def main():
    print("Hello from cs117-final!")


if __name__ == "__main__":
    main()
