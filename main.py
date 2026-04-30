import warp as wp


@wp.func
def wrap(theta: wp.float):
    return (theta + wp.pi) % (2 * wp.pi) - wp.pi


@wp.func
def compose(a: wp.vec3f, b: wp.vec3f):
    c, s = wp.cos(a.z), wp.sin(a.z)
    return wp.vec3(a.x + c * b.x - s * b.y, a.y + s * b.x + c * b.y, wrap(a.z + b.z))


@wp.func
def inverse(a: wp.vec3f):
    c, s = wp.cos(a.z), wp.sin(a.z)
    return wp.vec3(-c * a.x - s * a.y, s * a.x - c * a.y, -a.z)


@wp.func
def between(a: wp.vec3f, b: wp.vec3f):
    c, s = wp.cos(a.z), wp.sin(a.z)
    delta = wp.vec3(b.x - a.x, b.y - a.y, b.z - a.z)
    return wp.vec3(c * delta.x + s * delta.y, -s * delta.x + c * delta.y, wrap(delta.z))


@wp.kernel
def k_cast(
    pts: wp.array[wp.vec2],
    origin: wp.vec2,
    grid: wp.array2d,
    gx: float,
    gy: float,
    r: float,
    l_hit: float,
    l_miss: float,
    H: int,
    W: int,
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
    i, j = wp.tid()
    grid[i, j] = wp.clamp(grid[i, j], lo, high)


@wp.kernel
def k_logodds_to_prob(lo: wp.array2d[wp.float], pr: wp.array2d[wp.float]):
    i, j = wp.tid()
    pr[i, j] = 1.0 / (1.0 + wp.exp(-lo[i, j]))


@wp.kernel
def k_csm(
    pts: wp.array[wp.vec2f],
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
    scores: wp.array3d[wp.float32],
):
    it, iy, ix = wp.tid()
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
        wx = cx + c * p.x - s * p.y
        wy = cy + s * p.x + c * p.y
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
    i, j = wp.tid()
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
    pts: wp.array[wp.vec2f],
    grid: wp.array2d[wp.float],
    gx: float,
    gy: float,
    res: float,
    H: int,
    W: int,
    cands: wp.array[wp.vec3f],
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


class SubmapStack:
    def __init__(self):
        self.submaps = []

    def all_active(self):
        return [s for s in self.submaps if not s.finalized]

    def add_scan(self, sensor_pose_world, scan_local):
        active = self.all_active()


def main():
    print("Hello from cs117-final!")


if __name__ == "__main__":
    main()
