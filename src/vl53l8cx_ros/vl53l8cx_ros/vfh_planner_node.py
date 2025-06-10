#!/usr/bin/env python3
"""
minimal_vfh_node.py

A minimal ROS 2 VFH+ node for PX4 + Gazebo, modified so that:
  1. We first compute φ_enu (absolute ENU bearing to goal).
  2. We check the raw LaserScan in exactly that direction:
       • Find the scan‐index whose angle is closest to φ_enu.
       • If its range is > distance_to_goal (i.e. no obstacle closer than the goal),
         we SHORT‐CIRCUIT and point straight at the target (φ_enu) immediately.
  3. Otherwise, fall back to standard VFH+‐histogram→valley logic (with NO left/right bias, only minimal absolute angular deviation).
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
import tf_transformations


class MinimalVFH(Node):
    def __init__(self):
        super().__init__('minimal_vfh')

        # --------------------------------------------------------------------
        # 1) PARAMETERS: goal in ENU (east, north)
        # --------------------------------------------------------------------
        self.declare_parameter('goal_x', 12.0)
        self.declare_parameter('goal_y', 0.0)
        gx = self.get_parameter('goal_x').value
        gy = self.get_parameter('goal_y').value
        self.goal = np.array([gx, gy])  # ENU coordinates of target

        # --------------------------------------------------------------------
        # 2) VFH+ SETTINGS
        # --------------------------------------------------------------------
        self.alpha      = 5.0                         # degrees per sector
        self.S          = int(360 / self.alpha)       # number of sectors (360/5 = 72)
        # Cost weights: make “stay near previous” small so goal‐bias wins quickly
        self.mu1, self.mu2, self.mu3 = 5.0, 1.5, 0.5
        self.robot_r    = 0.3                         # robot radius [m]
        self.margin     = 0.3                         # safety margin [m]
        self.tau_high   = 0.9                         # hysteresis high fraction
        self.tau_low    = 0.5                         # hysteresis low fraction

        # --------------------------------------------------------------------
        # 3) STATE VARIABLES
        # --------------------------------------------------------------------
        self.prev_b        = np.zeros(self.S, int)    # previous binary histogram
        self.prev_sector   = None                     # last selected sector index
        self.smoothed_yaw  = None                     # smoothed relative ENU radians
        self.yaw_alpha     = 0.2                      # smoothing factor
        self.commit_until  = 0.0                      # valley‐locking timer
        self.committed_sec = None                     # locked‐in sector index

        # --------------------------------------------------------------------
        # 4) CURRENT POSE IN ENU
        # --------------------------------------------------------------------
        self.x = 0.0
        self.y = 0.0
        self.psi_ned = 0.0            # raw PX4 yaw in NED [rad]
        self.yaw_enu = 0.0            # yaw in ENU [deg]

        # --------------------------------------------------------------------
        # 5) SUBSCRIPTIONS
        # --------------------------------------------------------------------
        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            LaserScan,
            '/fused_obstacle_scan',
            self.scan_cb,
            qos_profile_sensor_data
        )

        # --------------------------------------------------------------------
        # 6) PUBLISHERS
        # --------------------------------------------------------------------
        self.yaw_pub    = self.create_publisher(Float32, '/vfh_desired_yaw', 10)
        self.marker_pub = self.create_publisher(Marker,   '/vfh_arrow_marker', 10)

        self.get_logger().info(f"Minimal VFH+ node started, goal=({gx},{gy})")

    def odom_cb(self, msg: VehicleOdometry):
        """
        Update ENU position and yaw from PX4’s odometry.
        In most Gazebo+PX4 setups, VehicleOdometry.position is already ENU.
        """
        # Treat msg.position as ENU directly:
        self.x = msg.position[0]
        self.y = msg.position[1]

        # Extract quaternion (w, x, y, z) → compute NED yaw (psi_ned)
        q = msg.q  # [w, x, y, z]
        _, _, psi_ned = tf_transformations.euler_from_quaternion(
            [q[1], q[2], q[3], q[0]]
        )
        self.psi_ned = psi_ned % (2 * math.pi)

        # Convert NED yaw (0 = North, +CW) → ENU yaw (0 = East, +CCW)
        psi_enu = (math.pi / 2 - self.psi_ned) % (2 * math.pi)
        self.yaw_enu = math.degrees(psi_enu)

    def scan_cb(self, scan: LaserScan):
        """
        Main VFH+ callback:
         1. Compute φ_enu (absolute ENU bearing to goal) and relative sector tgt.
         2. Do a quick “line‐of‐sight” check on the raw LaserScan:
            • Find the scan‐index whose angle is closest to φ_enu.
            • If scan.ranges[bin] is greater than distance_to_goal (or inf),
              short‐circuit and aim directly at the target.
         3. Otherwise, run the normal VFH+ (histogram → binary hysteresis → mask →
            valley finding → cost pick WITHOUT left/right bias → smoothing → publish).
        """
        now = self.get_clock().now().nanoseconds * 1e-9

        # --------------------------------------------------------------------
        # 1) Compute φ_enu (absolute ENU bearing to goal) and tgt (relative sector)
        # --------------------------------------------------------------------
        dx, dy = self.goal - np.array([self.x, self.y])
        # φ_enu ∈ [0,360) with ENU convention: 0=East, 90=North, etc.
        φ_enu = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        # Relative sector index of φ_enu: subtract current yaw_enu, wrap, divide by α
        tgt = int(((φ_enu - self.yaw_enu) % 360.0) // self.alpha)
        if self.prev_sector is None:
            self.prev_sector = tgt

        # Compute straight‐line distance to goal
        dist_to_goal = math.hypot(dx, dy)

        # --------------------------------------------------------------------
        # 2) QUICK LINE‐OF‐SIGHT CHECK
        # --------------------------------------------------------------------
        # Find the index in 'scan.angles' closest to φ_enu (absolute scan angle).
        # First, convert φ_enu (ENU degrees) → laser frame degrees.
        # LaserScan angles (θ) run from scan.angle_min..scan.angle_max in radians,
        # and correspond to ENU bearings = (θ_degrees + current_yaw_enu) mod 360.
        #
        # We want to find the laser‐scan ray where (θ_degrees + yaw_enu) ≈ φ_enu.
        #
        # So define desired θ_deg_laser = (φ_enu - yaw_enu) mod 360.
        θ_deg_laser = (φ_enu - self.yaw_enu + 360.0) % 360.0
        # But scan.angle_min..max often cover only ±90° or ±120° around forward.
        # Convert to a value in [-180, +180] so we can compare directly to θ.
        if θ_deg_laser > 180.0:
            θ_deg_laser -= 360.0

        # Build an array of scan angles in degrees:
        angles_rad = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        angles_deg = np.degrees(angles_rad)

        # Find the index whose angles_deg is closest to θ_deg_laser
        idx = int(np.argmin(np.abs(angles_deg - θ_deg_laser)))
        range_at_idx = scan.ranges[idx]

        # If the laser reading at idx is invalid, treat it as “no obstacle” (infinite)
        if not math.isfinite(range_at_idx) or range_at_idx > scan.range_max:
            range_at_idx = math.inf

        # If that range is larger than (or equal to) our distance_to_goal,
        # it means “no obstacle between us and the goal.” Do a short‐circuit.
        if range_at_idx >= dist_to_goal:
            # Aim directly at φ_enu
            ψ_desired_enu = math.radians(φ_enu)

            # Convert absolute ENU → NED for PX4 offboard
            ψ_ned_cmd = (math.pi / 2 - ψ_desired_enu) % (2 * math.pi)
            self.yaw_pub.publish(Float32(data=float(ψ_ned_cmd)))

            # Publish arrow marker pointing at φ_enu in ENU
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns, m.id        = 'vfh', 0
            m.type, m.action = Marker.ARROW, Marker.ADD
            m.scale.x, m.scale.y, m.scale.z = 1.0, 0.1, 0.1
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
            m.pose.orientation.w = math.cos(ψ_desired_enu / 2.0)
            m.pose.orientation.z = math.sin(ψ_desired_enu / 2.0)
            self.marker_pub.publish(m)

            self.get_logger().info(
                f"[LOS-SHORTCUT] φ_enu={φ_enu:.1f}° | idx={idx} | "
                f"range={range_at_idx:.2f} m ≥ {dist_to_goal:.2f} m → "
                f"ψ_ned={math.degrees(ψ_ned_cmd):.1f}°"
            )
            return

        # --------------------------------------------------------------------
        # 3) STANDARD VFH+ HISTOGRAM
        # --------------------------------------------------------------------
        # Build primary histogram H[] and per-sector min-range d[]
        ranges = np.array(scan.ranges)
        H = np.zeros(self.S)
        d = np.full(self.S, np.inf)
        for r, θ_rad in zip(ranges, angles_rad):
            if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            m = max(scan.range_max - r, 0.0)
            raw_deg = (math.degrees(θ_rad) + 360.0) % 360.0
            k = int(raw_deg // self.alpha) % self.S
            H[k] += m
            d[k] = min(d[k], r)

        # Binary hysteresis on H → b[]
        b = np.zeros(self.S, int)
        for i in range(self.S):
            if   H[i] >= self.tau_high * scan.range_max: b[i] = 1
            elif H[i] <= self.tau_low  * scan.range_max: b[i] = 0
            else:                                        b[i] = self.prev_b[i]
        self.prev_b = b.copy()

        # Inflate obstacles by (robot_r + margin) → mask[]
        mask = b.copy()
        for i in np.where(b == 1)[0]:
            dist = d[i] if math.isfinite(d[i]) else scan.range_max
            dist = max(dist, 1e-3)
            β = math.degrees(math.asin(min(1.0, (self.robot_r + self.margin) / dist)))
            w = int(math.ceil(β / self.alpha))
            for off in range(-w, w + 1):
                mask[(i + off) % self.S] = 1

        # Find valleys (contiguous runs of mask==0)
        valleys = []
        i = 0
        while i < self.S:
            if mask[i] == 0:
                s = i
                while i < self.S and mask[i] == 0:
                    i += 1
                valleys.append((s, i - 1))
            i += 1

        # Handle wrap-around valley spanning end→start
        if valleys and mask[0] == 0 and mask[-1] == 0:
            sl, el = valleys[-1]
            s0, e0 = valleys[0]
            valleys[0] = (sl, e0 + self.S)
            valleys.pop()

        # Extract candidate sectors from each valley
        cands = []
        for (s, e) in valleys:
            width = e - s + 1
            if width < 3:
                cands.append((s + e) // 2)
            else:
                cands.append(s + 1)
                cands.append(e - 1)
            if s <= tgt <= e:
                cands.append(tgt)

        # Cost pick WITHOUT any left/right bias: just choose minimal absolute deviation
        def signed_angle(sec: int) -> float:
            raw = (sec * self.alpha + self.alpha / 2.0) % 360.0
            return raw if raw <= 180.0 else raw - 360.0

        if cands:
            best = min(cands, key=lambda c: abs(signed_angle(c)))
        else:
            best = tgt

        # Hysteresis on valley selection (5 s lock)
        if self.committed_sec is None or now > self.commit_until:
            self.committed_sec = best
            self.commit_until = now + 5.0
        sec = self.committed_sec
        self.prev_sector = sec

        # Convert sector → relative ENU offset, then smooth
        ψ_enu_rel = math.radians(sec * self.alpha + self.alpha / 2.0)
        if self.smoothed_yaw is None:
            self.smoothed_yaw = ψ_enu_rel
        else:
            self.smoothed_yaw = (
                self.yaw_alpha * ψ_enu_rel + (1.0 - self.yaw_alpha) * self.smoothed_yaw
            )

        # Form absolute ENU heading by adding current yaw_enu
        ψ_desired_enu = (math.radians(self.yaw_enu) + self.smoothed_yaw) % (2 * math.pi)

        # Convert absolute ENU → NED for PX4 offboard
        ψ_ned_cmd = (math.pi / 2 - ψ_desired_enu) % (2 * math.pi)
        self.yaw_pub.publish(Float32(data=float(ψ_ned_cmd)))

        # Publish arrow marker showing the ABSOLUTE ENU heading
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns, m.id        = 'vfh', 0
        m.type, m.action = Marker.ARROW, Marker.ADD
        m.scale.x, m.scale.y, m.scale.z = 1.0, 0.1, 0.1
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
        m.pose.orientation.w = math.cos(ψ_desired_enu / 2.0)
        m.pose.orientation.z = math.sin(ψ_desired_enu / 2.0)
        self.marker_pub.publish(m)

        self.get_logger().info(
            f"φ_enu={φ_enu:.1f}°, tgt={tgt}, sec={sec}, ψ_rel={math.degrees(self.smoothed_yaw):.1f}°, "
            f"ψ_enu_abs={math.degrees(ψ_desired_enu):.1f}° → ψ_ned={math.degrees(ψ_ned_cmd):.1f}°"
        )


def main():
    rclpy.init()
    node = MinimalVFH()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
