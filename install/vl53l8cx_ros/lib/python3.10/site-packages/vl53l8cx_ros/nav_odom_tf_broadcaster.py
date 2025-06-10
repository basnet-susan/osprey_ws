import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


def quaternion_multiply(q1, q2):
    # Quaternion multiplication (Hamilton product)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ]


class NavOdomTFBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_to_base_link_tf_broadcaster')

        # Create a TF broadcaster
        self.br = TransformBroadcaster(self)

        # Create a QoS profile matching PX4 publisher
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribe to PX4 vehicle odometry with correct QoS
        self.subscription = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_profile
        )

    def odom_callback(self, msg: VehicleOdometry):
        t = TransformStamped()

        # Use timestamp from PX4 message (in microseconds)
        t.header.stamp.sec = int(msg.timestamp / 1e6)
        t.header.stamp.nanosec = int((msg.timestamp % 1e6) * 1e3)
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        # Convert NED to ENU: x -> x, y -> -y, z -> -z
        x = float(msg.position[0])
        y = -float(msg.position[1])
        z = -float(msg.position[2])

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z

        # Convert quaternion from PX4 (FRD to NED) to ROS (FLU to ENU)
        qx, qy, qz, qw = msg.q
        q_px4_to_ros = [qx, -qy, -qz, qw]

        # Apply additional -90° rotation about X axis
        angle_rad = -np.pi / 2.0
        sin_a = np.sin(angle_rad / 2.0)
        cos_a = np.cos(angle_rad / 2.0)
        q_rot_x = [sin_a, 0.0, 0.0, cos_a]  # -90 deg about X

        q_final = quaternion_multiply(q_rot_x, q_px4_to_ros)

        t.transform.rotation.x = q_final[0]
        t.transform.rotation.y = q_final[1]
        t.transform.rotation.z = q_final[2]
        t.transform.rotation.w = q_final[3]

        # Broadcast the transform
        self.br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = NavOdomTFBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
