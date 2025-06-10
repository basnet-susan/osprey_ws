import rclpy
import numpy as np
import struct
import cv2
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import PointCloud2, PointField, Image
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_matrix


class VL53L8CXVisualization(Node):
    def __init__(self):
        super().__init__('vl53l8cx_visualization')

        # ROS2 Subscriber for sensor data
        self.subscription = self.create_subscription(
            Int32MultiArray, '/sensor_data', self.sensor_callback, 10)

        # Global publishers
        self.global_pointcloud_publisher = self.create_publisher(PointCloud2, "/point_cloud", 10)
        self.global_image_publisher = self.create_publisher(Image, "/depth_image", 10)

        # Per-sensor publishers (12 sensors)
        self.pointcloud_publishers = {
            f"/point_cloud_sensor_{i+1}": self.create_publisher(PointCloud2, f"/point_cloud_sensor_{i+1}", 10)
            for i in range(12)
        }
        self.image_publishers = {
            f"/depth_image_sensor_{i+1}": self.create_publisher(Image, f"/depth_image_sensor_{i+1}", 10)
            for i in range(12)
        }
        # Ensure the global depth image publisher is included
        self.image_publishers["/depth_image"] = self.create_publisher(Image, "/depth_image", 10)

        # TF2 Listener for transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Sensor setup
        self.num_sensors = 12
        self.zones_per_sensor = 64       # 8×8 = 64 zones per sensor
        self.fov = np.radians(65)        # 65° Field of View
        self.grid_size = 8               # 8×8 depth resolution
        self.pixel_angles = self.compute_pixel_angles()
        self.sensor_frames = [f"sensor_circ_{i+1}" for i in range(12)]
        self.bridge = CvBridge()

    def compute_pixel_angles(self):
        """Computes per-pixel (θ, φ) angles for each of the 8×8 zones in the sensor."""
        half_fov = self.fov / 2
        step = self.fov / (self.grid_size - 1)  # (65°)/(7) increments
        return [
            (-half_fov + i * step, -half_fov + j * step)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
        ]
        # The list will have 8×8 = 64 (θ, φ) pairs

    def sensor_callback(self, msg: Int32MultiArray):
        """Processes sensor data, publishing both per-sensor & global depth image & point cloud."""

        data_len = len(msg.data)
        # How many sensors’ worth of data did we actually get?
        available_sensors = data_len // self.zones_per_sensor

        if available_sensors < self.num_sensors:
            self.get_logger().warn(
                f"Expected {self.num_sensors * self.zones_per_sensor} data points, "
                f"but got {data_len}. Only processing {available_sensors} sensors."
            )

        # Only process as many sensors as fit fully into msg.data
        N = min(self.num_sensors, available_sensors)

        overall_points = []
        # Global depth image has 12 (max) sensors stacked; each has 8 rows
        # We’ll still allocate the full 12*8 rows, but only fill the first N*8 rows:
        global_depth_matrix = np.zeros((self.num_sensors * self.grid_size, self.grid_size))

        for sensor_idx in range(N):
            start = sensor_idx * self.zones_per_sensor
            end   = (sensor_idx + 1) * self.zones_per_sensor
            sensor_data = msg.data[start:end]  # should be exactly 64 values
            sensor_frame = self.sensor_frames[sensor_idx]
            sensor_points = []
            depth_matrix = np.zeros((self.grid_size, self.grid_size))

            try:
                # Check that the transform is available
                if not self.tf_buffer.can_transform(
                    "base_link", sensor_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                ):
                    self.get_logger().warn(f"TF lookup failed for {sensor_frame}: No transform found")
                    continue

                # Lookup transform from sensor_frame → base_link
                transform = self.tf_buffer.lookup_transform(
                    "base_link", sensor_frame, rclpy.time.Time()
                )

                tx = transform.transform.translation.x
                ty = transform.transform.translation.y
                tz = transform.transform.translation.z
                q = transform.transform.rotation
                rotation_matrix = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

                # Convert each of the 8×8 depth readings into a 3D point
                for idx, (theta, phi) in enumerate(self.pixel_angles):
                    # idx runs 0..63
                    i_row, j_col = divmod(idx, self.grid_size)
                    # sensor_data has 64 entries indexed 0..63
                    r = sensor_data[i_row * self.grid_size + j_col] / 1000.0  # mm → m

                    # record depth in mm for per-sensor image
                    depth_matrix[i_row, j_col] = r * 1000
                    # record depth in mm for global image (row offset: sensor_idx*8)
                    global_depth_matrix[sensor_idx * self.grid_size + i_row, j_col] = r * 1000

                    # Discard any invalid ranges
                    if r < 0.01 or r > 4.0:
                        continue

                    # Local 3D point in sensor frame: X = –r·tan(φ), Y = –r·tan(θ), Z = r
                    local_point = np.array([-r * np.tan(phi), -r * np.tan(theta), r])
                    world_point = rotation_matrix @ local_point + np.array([tx, ty, tz])
                    sensor_points.append((world_point[0], world_point[1], world_point[2]))

                # Publish this sensor’s point cloud (if any valid points)
                if sensor_points:
                    cloud_msg = self.create_point_cloud_msg(sensor_points)
                    self.pointcloud_publishers[f"/point_cloud_sensor_{sensor_idx+1}"].publish(cloud_msg)

                # Publish this sensor’s depth image (8×8)
                self.publish_depth_image(depth_matrix, f"/depth_image_sensor_{sensor_idx+1}")

                # Add to the global point list
                overall_points.extend(sensor_points)

            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for {sensor_frame}: {str(e)}")

        # Publish global point cloud
        if overall_points:
            cloud_msg = self.create_point_cloud_msg(overall_points)
            self.global_pointcloud_publisher.publish(cloud_msg)

        # Publish global depth image (12*8 = 96 rows × 8 cols)
        self.publish_depth_image(global_depth_matrix, "/depth_image")

    def create_point_cloud_msg(self, points):
        """Creates a ROS2 PointCloud2 message from a list of (x,y,z) tuples."""
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True

        buffer = []
        for (x, y, z) in points:
            buffer.extend(struct.pack("fff", x, y, z))

        msg.data = bytes(buffer)
        return msg

    def publish_depth_image(self, depth_matrix: np.ndarray, topic: str):
        """Publishes a depth matrix (in millimeters) as a color-mapped Image message."""
        # Normalize 0–255 for visualization
        depth_normalized = cv2.normalize(depth_matrix, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)

        # Apply a color map
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)

        image_msg = self.bridge.cv2_to_imgmsg(depth_colored, encoding="bgr8")
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "base_link"

        if topic in self.image_publishers:
            self.image_publishers[topic].publish(image_msg)
        else:
            self.get_logger().warn(f"Depth image topic '{topic}' not found in publishers.")


def main():
    rclpy.init()
    node = VL53L8CXVisualization()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
