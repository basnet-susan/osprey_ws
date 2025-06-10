#!/usr/bin/env python3
"""
vfh_plus_live_smoothed.py

Adds heading-rate limiting and smoothing to reduce oscillation
when the robot passes obstacles.

Based on:
  “VFH+: Reliable Obstacle Avoidance for Fast Mobile Robots”
  I. Ulrich & J. Borenstein, 1998
"""

import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def simulate_scan(robot_pose, obstacles,
                  max_range=4.0, res_deg=0.5, step=0.02):
    """
    Same as before, but max_range is now a parameter.
    """
    angles_deg = np.arange(0, 360, res_deg)
    ranges = np.full_like(angles_deg, max_range, dtype=float)
    x0, y0, _ = robot_pose

    for i, a in enumerate(np.deg2rad(angles_deg)):
        for d in np.arange(0, max_range, step):
            x, y = x0 + d*np.cos(a), y0 + d*np.sin(a)
            if any((x-ox)**2 + (y-oy)**2 <= r**2
                   for ox, oy, r in obstacles):
                ranges[i] = d
                break

    return angles_deg, ranges


def vfh_plus(angles_deg, ranges,
             robot_heading_deg, prev_heading_deg,
             max_range=4.0,
             a_deg=5.0, robot_radius=0.2, safety_dist=0.05,
             open_frac=0.25, close_frac=0.5,
             mu1=8.0, mu2=2.0, mu3=1.0,
             s_min_sectors=16):
    """
    VFH+ that scales with max_range.
      • open_frac, close_frac are fractions of max_range to open/close sectors
      • everything else as before
    """
    S = int(360 // a_deg)
    H = np.zeros(S)

    # 1) primary histogram scaled by max_range
    for ang, rng in zip(angles_deg, ranges):
        k = int(ang // a_deg) % S
        H[k] += max(0.0, max_range - rng)

    # width compensation (unchanged logic)
    delta_sector = a_deg / (angles_deg[1] - angles_deg[0])
    for k in np.nonzero(H > 0)[0]:
        idx0 = int(k * delta_sector)
        idx1 = int((k + 1) * delta_sector)
        d = np.min(ranges[idx0:idx1] + 1e-6)
        widen_deg = np.degrees(np.arcsin(min(1, (robot_radius + safety_dist)/d)))
        w = int(np.ceil(widen_deg / a_deg))
        for dk in range(-w, w+1):
            H[(k+dk) % S] += H[k]

    # 2) binary hysteresis with thresholds scaled to max_range
    thresh_open  = open_frac  * max_range
    thresh_close = close_frac * max_range

    B = np.zeros(S, dtype=int)
    for i in range(S):
        if   H[i] > thresh_close: B[i] = 1
        elif H[i] < thresh_open:  B[i] = 0
        else:                     B[i] = B[i-1]
    M = B.copy()

    # 3) find free valleys (same as before)
    valleys, i = [], 0
    while i < S:
        if M[i] == 0:
            s = i
            while i < S and M[i] == 0:
                i += 1
            valleys.append((s, i-1))
        i += 1
    # wrap‐around
    if valleys and M[0]==0 and M[-1]==0:
        sl, el = valleys[-1]
        s0, e0 = valleys[0]
        valleys[0] = (sl, e0+S)
        valleys.pop()

    # 4) build candidates
    candidates = []
    for s, e in valleys:
        w = e - s + 1
        if w < s_min_sectors:
            candidates.append((s+e)/2)
        else:
            candidates += [s + s_min_sectors/2,
                           e - s_min_sectors/2]

    # constrain to ±90°
    target_sec = (robot_heading_deg % 360) / a_deg
    max_rel    = 90.0 / a_deg
    candidates = [c for c in candidates
                  if min(abs(c-target_sec), S-abs(c-target_sec)) <= max_rel]

    # cost‐based selection
    def angdiff(a, b):
        d = abs(a-b) % S
        return min(d, S-d)

    best, best_cost = None, float('inf')
    for c in candidates:
        cost = (mu1 * angdiff(c, target_sec)
              + mu2 * angdiff(c, robot_heading_deg/a_deg)
              + mu3 * angdiff(c, prev_heading_deg/a_deg))
        if cost < best_cost:
            best_cost, best = cost, c

    if best is None:
        # no good valley: fallback to pointing at goal
        return np.deg2rad(robot_heading_deg)
    return np.deg2rad(best * a_deg)


# ---------------- Live plotting setup ----------------
def setup_plots(obstacles, target):
    plt.ion()
    fig, (ax_env, ax_dist) = plt.subplots(1, 2, figsize=(12, 6))
    # environment
    ax_env.set_aspect('equal')
    for ox, oy, r in obstacles:
        ax_env.add_patch(plt.Circle((ox, oy), r, color='gray', alpha=0.5))
    tx, ty = target
    ax_env.plot(tx, ty, 'rX', markersize=12)
    path_line, = ax_env.plot([], [], '-b')
    robot_patch = plt.Polygon([[0,0],[0,0],[0,0]], color='blue', alpha=0.8)
    ax_env.add_patch(robot_patch)
    ax_env.set_title('Environment'); ax_env.grid(True)
    # distance
    ax_dist.set_title('Distance to Goal')
    ax_dist.set_xlabel('Time [s]'); ax_dist.set_ylabel('Distance [m]')
    dist_line, = ax_dist.plot([], [], '-g')
    ax_dist.grid(True)
    plt.tight_layout()
    return fig, ax_env, ax_dist, path_line, robot_patch, dist_line

def update_robot_patch(patch, pose):
    x, y, theta = pose
    tri = np.array([[ 0.3,  0.2],
                    [ 0.3, -0.2],
                    [-0.2,  0.0]])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    pts = (R @ tri.T).T + np.array([x, y])
    patch.set_xy(pts)

# ---------------- Main simulation ----------------
def main():
    obstacles = [(3.0,0.2,0.2),(3.8,0.2,0.2),(4.5,0.2,0.2),(3.0,1.2,0.2),(1.0,1.5,0.3),(-1.5,2.0,0.7),(0.0,-2.5,0.4)]
    target = (4.5, -0.75)
    start_pose = np.array([0.0, 0.0, 0.0])
    prev_heading = start_pose[2]
    v, dt = 0.5, 0.1  # m/s, s
    max_range = 2.0    # max range of the laser scanner

    # damping parameters
    MAX_TURN_DEG_PER_SEC = 360.0          # max turn rate
    SMOOTHING_ALPHA       = 0.2          # exponential smoothing factor

    fig, ax_env, ax_dist, path_line, robot_patch, dist_line = \
        setup_plots(obstacles, target)

    pose = start_pose.copy()
    sim_time = 0.0
    path = [pose[:2].copy()]
    times, dists = [0.0], [np.linalg.norm(pose[:2] - target)]

    for step in range(1, 1001):
        dist_to_goal = np.linalg.norm(pose[:2] - target)
        if dist_to_goal < 0.1:
            print(f"Goal reached at t={sim_time:.1f}s")
            break

        angles, ranges = simulate_scan(pose, obstacles)
        raw_heading = np.arctan2(target[1]-pose[1], target[0]-pose[0])
        vfh_heading = vfh_plus(angles, ranges,
                  np.rad2deg(raw_heading),
                  np.rad2deg(prev_heading),
                  max_range=max_range,
                  open_frac=0.1,    # tweak these if needed
                  close_frac=0.2,
                  mu1=8.0, mu2=2.0, mu3=1.0)

        # 1) Limit turn rate
        max_delta = np.deg2rad(MAX_TURN_DEG_PER_SEC * dt)
        delta = (vfh_heading - pose[2] + np.pi) % (2*np.pi) - np.pi
        delta = np.clip(delta, -max_delta, max_delta)
        limited_heading = pose[2] + delta

        # 2) Exponential smoothing
        new_heading = (SMOOTHING_ALPHA * limited_heading +
                       (1-SMOOTHING_ALPHA) * pose[2])

        prev_heading, pose[2] = pose[2], new_heading

        # motion
        pose[0] += v * np.cos(pose[2]) * dt
        pose[1] += v * np.sin(pose[2]) * dt
        sim_time += dt

        # record & update plots
        path.append(pose[:2].copy())
        times.append(sim_time)
        dists.append(dist_to_goal)

        path_arr = np.array(path)
        path_line.set_data(path_arr[:,0], path_arr[:,1])
        update_robot_patch(robot_patch, pose)
        ax_env.relim(); ax_env.autoscale_view()

        dist_line.set_data(times, dists)
        ax_dist.set_xlim(0, max(times)+1)
        ax_dist.set_ylim(0, max(dists)+0.5)

        fig.canvas.draw(); fig.canvas.flush_events()
        plt.pause(0.001)

    plt.ioff(); plt.show()

if __name__ == "__main__":
    main()
