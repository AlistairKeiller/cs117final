import faulthandler

faulthandler.enable()

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, LaserScan
from tf2_ros import TransformBroadcaster

from slam import GRID_HW, RES, Slam, compose, wrap

NS = "/autodrive/roboracer_1"
WHEEL_R = 0.059
TICKS_PER_REV = 16 * 120
M_PER_TICK = 2 * np.pi * WHEEL_R / TICKS_PER_REV
LIDAR_X = 0.2733


def yaw(q):
    return np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class SlamNode(Node):
    def __init__(self):
        super().__init__("f1tenth_slam")
        self.slam = Slam()
        self.tf = TransformBroadcaster(self)
        self.pub = self.create_publisher(PoseStamped, "/slam_pose", 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", 1)
        self.yaw_curr = None
        self.l = self.r = 0.0
        self.yaw_prev = self.l_prev = self.r_prev = None
        self.create_subscription(
            Imu,
            f"{NS}/imu",
            lambda m: setattr(self, "yaw_curr", yaw(m.orientation)),
            50,
        )
        self.create_subscription(
            JointState,
            f"{NS}/left_encoder",
            lambda m: setattr(self, "l", float(m.position[0])),
            50,
        )
        self.create_subscription(
            JointState,
            f"{NS}/right_encoder",
            lambda m: setattr(self, "r", float(m.position[0])),
            50,
        )
        self.create_subscription(LaserScan, f"{NS}/lidar", self.on_scan, 10)
        self.create_timer(1.0, self.publish_map)

    def submap_world_origin(self, sm):
        if sm.finalized and sm.id in self.slam.submap_keys:
            return self.slam.graph.pose(self.slam.submap_keys[sm.id])
        return compose(self.slam._local_to_global, sm.origin)

    def on_scan(self, msg):
        if self.yaw_curr is None:
            return
        if self.yaw_prev is None:
            self.yaw_prev, self.l_prev, self.r_prev = self.yaw_curr, self.l, self.r
            return
        ds = 0.5 * M_PER_TICK * ((self.l - self.l_prev) + (self.r - self.r_prev))
        dth = wrap(self.yaw_curr - self.yaw_prev)
        self.yaw_prev, self.l_prev, self.r_prev = self.yaw_curr, self.l, self.r
        delta = np.array([ds * np.cos(dth / 2), ds * np.sin(dth / 2), dth])
        r = np.asarray(msg.ranges)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        m = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        scan = np.column_stack([r[m] * np.cos(a[m]) + LIDAR_X, r[m] * np.sin(a[m])])
        self.publish(self.slam.add_scan(scan, delta), msg.header.stamp)

    def publish(self, pose, stamp):
        pose = compose(self.slam._local_to_global, pose)
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        qz, qw = float(np.sin(th / 2)), float(np.cos(th / 2))
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = stamp, "map"
        ps.pose.position.x, ps.pose.position.y = x, y
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        self.pub.publish(ps)
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, "map", "base_link"
        t.transform.translation.x, t.transform.translation.y = x, y
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        self.tf.sendTransform(t)

    def publish_map(self):
        sms = self.slam.stack.submaps
        if not sms:
            return
        L = GRID_HW * RES
        cx, cy = [], []
        origins = [self.submap_world_origin(sm) for sm in sms]
        for sm, origin in zip(sms, origins):
            c, s = np.cos(origin[2]), np.sin(origin[2])
            for lx, ly in [
                (sm.gx, sm.gy),
                (sm.gx + L, sm.gy),
                (sm.gx, sm.gy + L),
                (sm.gx + L, sm.gy + L),
            ]:
                cx.append(origin[0] + c * lx - s * ly)
                cy.append(origin[1] + s * lx + c * ly)
        x0, y0 = min(cx) - RES, min(cy) - RES
        W = int(np.ceil((max(cx) - x0) / RES)) + 1
        H = int(np.ceil((max(cy) - y0) / RES)) + 1
        jj, ii = np.meshgrid(np.arange(W), np.arange(H))
        wx = x0 + (jj + 0.5) * RES
        wy = y0 + (ii + 0.5) * RES
        occ = np.full((H, W), -1, dtype=np.int8)
        for sm, origin in zip(sms, origins):
            sm.refresh_prob()
            p = sm.prob.numpy()
            c, s = np.cos(origin[2]), np.sin(origin[2])
            dx, dy = wx - origin[0], wy - origin[1]
            lj = ((c * dx + s * dy - sm.gx) / RES).astype(np.int32)
            li = ((-s * dx + c * dy - sm.gy) / RES).astype(np.int32)
            v = (li >= 0) & (li < GRID_HW) & (lj >= 0) & (lj < GRID_HW)
            if not v.any():
                continue
            pr = p[np.where(v, li, 0), np.where(v, lj, 0)]
            occ[v & (pr < 0.35) & (occ == -1)] = 0
            occ[v & (pr > 0.65)] = 100
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = RES
        msg.info.width, msg.info.height = W, H
        msg.info.origin.position.x = float(x0)
        msg.info.origin.position.y = float(y0)
        msg.info.origin.orientation.w = 1.0
        msg.data = occ.ravel().tolist()
        self.map_pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(SlamNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
