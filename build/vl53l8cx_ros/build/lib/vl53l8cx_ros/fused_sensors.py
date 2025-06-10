#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import math

class LaserScanToRays(Node):
    def __init__(self):
        super().__init__('scan_to_rviz_rays_node')
        self.sub = self.create_subscription(LaserScan, '/fused_obstacle_scan', self.callback, 10)
        self.pub = self.create_publisher(MarkerArray, '/fused_obstacle_rays', 10)
        self.frame_id = 'base_link'
        self.get_logger().info("LaserScan to RViz rays node started.")

    def callback(self, scan):
        marker_array = MarkerArray()

        angle = scan.angle_min
        index = 0

        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)

                marker = Marker()
                marker.header.frame_id = scan.header.frame_id or self.frame_id
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "scan_rays"
                marker.id = index
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                marker.scale.x = 0.02  # shaft diameter
                marker.scale.y = 0.05  # head diameter
                marker.scale.z = 0.05  # head length
                marker.color.a = 1.0
                marker.color.r = 0.2
                marker.color.g = 0.8
                marker.color.b = 0.2
                marker.lifetime.sec = 1

                from geometry_msgs.msg import Point
                p0 = Point(x=0.0, y=0.0, z=0.0)
                p1 = Point(x=x, y=y, z=0.0)
                marker.points = [p0, p1]

                marker_array.markers.append(marker)
                index += 1

            angle += scan.angle_increment

        self.pub.publish(marker_array)
        self.get_logger().info(f"Published {index} rays")

def main(args=None):
    rclpy.init(args=args)
    node = LaserScanToRays()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
