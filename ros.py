import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

from slam import Slam, between


def yaw(q):
    return np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class SlamNode(Node):
    def __init__(self):
        super().__init__("f1tenth_slam")
        self.slam = Slam()
        self.prev_odom = None
        self.tf = TransformBroadcaster(self)
        self.pub = self.create_publisher(PoseStamped, "/slam_pose", 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 50)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.odom = None

    def on_odom(self, msg):
        p = msg.pose.pose
        self.odom = np.array([p.position.x, p.position.y, yaw(p.orientation)])

    def on_scan(self, msg):
        if self.odom is None:
            return
        r = np.asarray(msg.ranges)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        m = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        scan = np.column_stack([r[m] * np.cos(a[m]), r[m] * np.sin(a[m])])
        delta = (
            between(self.prev_odom, self.odom)
            if self.prev_odom is not None
            else np.zeros(3)
        )
        self.prev_odom = self.odom
        self.publish(self.slam.add_scan(scan, delta), msg.header.stamp)

    def publish(self, pose, stamp):
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


def main():
    rclpy.init()
    rclpy.spin(SlamNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
