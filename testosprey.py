#!/usr/bin/env python3
"""
vfh_plus_escape_fixed.py

2D LIDAR sim + VFH+ steering, with:
 - ±180° sector window
 - instant adoption of limited turn (no over-damping)
 - corrected cost terms
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def simulate_scan_vl53(robot_pose, obstacles,
                       max_range=4.0, sensor_count=8,
                       zones_per_sensor=8, sensor_fov_deg=60.0,
                       step=0.02):
    x0, y0, _ = robot_pose
    sensor_orients = np.linspace(0,360,sensor_count,endpoint=False)
    half_fov = sensor_fov_deg/2.0
    beam_angles = []
    for orient in sensor_orients:
        for z in range(zones_per_sensor):
            rel = -half_fov + (z+0.5)*(sensor_fov_deg/zones_per_sensor)
            beam_angles.append((orient+rel)%360.0)
    angles_deg = np.array(beam_angles)

    ranges = np.full_like(angles_deg, max_range, dtype=float)
    for i, ang in enumerate(angles_deg):
        a = np.deg2rad(ang)
        for d in np.arange(0, max_range, step):
            x = x0 + d*np.cos(a)
            y = y0 + d*np.sin(a)
            hit = False
            for obs in obstacles:
                if obs['type']=='cylinder':
                    cx,cy = obs['center']; r=obs['radius']
                    if (x-cx)**2 + (y-cy)**2 <= r**2:
                        hit = True; break
                else:
                    cx,cy = obs['center']; sx,sy = obs['size']
                    if abs(x-cx)<=sx/2 and abs(y-cy)<=sy/2:
                        hit = True; break
            if hit:
                ranges[i] = d
                break
    return angles_deg, ranges

def vfh_plus(angles_deg, ranges,
             robot_heading_deg, prev_heading_deg,
             max_range=4.0, a_deg=5.0,
             robot_radius=0.2, safety_dist=0.5,
             open_frac=0.25, close_frac=0.5,
             mu1=5.0, mu2=2.0, mu3=2.0,
             s_min_sectors=16):
    S = int(360//a_deg)
    H = np.zeros(S)

    # 1) build primary histogram
    for ang, rng in zip(angles_deg, ranges):
        k = int(ang//a_deg) % S
        H[k] += max(0.0, max_range - rng)

    # 2) width compensation
    res = angles_deg[1] - angles_deg[0]
    delta = a_deg/res
    for k in np.nonzero(H>0)[0]:
        i0,i1 = int(k*delta), int((k+1)*delta)
        d = (ranges[i0] if i1<=i0 else np.min(ranges[i0:i1])) + 1e-6
        widen = np.degrees(np.arcsin(min(1,(robot_radius+safety_dist)/d)))
        w = int(np.ceil(widen/a_deg))
        for off in range(-w,w+1):
            H[(k+off)%S] += H[k]

    # 3) binary hysteresis
    B = np.zeros(S,int)
    to,tc = open_frac*max_range, close_frac*max_range
    for i in range(S):
        if   H[i]>tc: B[i]=1
        elif H[i]<to: B[i]=0
        else:         B[i]=B[i-1]
    M = B.copy()

    # 4) find valleys
    valleys=[]; i=0
    while i<S:
        if M[i]==0:
            s=i
            while i<S and M[i]==0: i+=1
            valleys.append((s,i-1))
        i+=1
    # wraparound merge
    if valleys and M[0]==0 and M[-1]==0:
        sl,el=valleys[-1]; s0,e0=valleys[0]
        valleys[0] = (sl, e0+S)
        valleys.pop()

    # 5) candidate directions
    cands=[]
    for s,e in valleys:
        w=e-s+1
        if w<s_min_sectors:
            cands.append((s+e)/2)
        else:
            cands += [s+s_min_sectors/2, e-s_min_sectors/2]

    # ---- relaxed ±180° around goal ----
    tgt_sector = (robot_heading_deg%360.0)/a_deg
    max_sectors = 90.0/a_deg
    cands = [c for c in cands
             if min(abs(c-tgt_sector), S-abs(c-tgt_sector)) <= max_sectors]

    # 6) cost‐based pick
    def sector_diff(a,b):
        d = abs(a-b)%S
        return min(d, S-d)

    best=None; best_cost=1e9
    cur_sec = tgt_sector
    last_sec = prev_heading_deg / a_deg

    for c in cands:
        cost = ( mu1*sector_diff(c, tgt_sector)
               + mu2*sector_diff(c, cur_sec)
               + mu3*sector_diff(c, last_sec) )
        if cost<best_cost:
            best_cost, best = cost, c

    if best is None:
        return np.deg2rad(robot_heading_deg)
    return np.deg2rad(best*a_deg)

def setup_plots(obstacles, target):
    plt.ion()
    fig, (ax_env, ax_dist, ax_yaw) = plt.subplots(1,3,figsize=(18,6))
    ax_env.set_aspect('equal')
    for obs in obstacles:
        if obs['type']=='cylinder':
            c = plt.Circle(obs['center'], obs['radius'], color='gray', alpha=0.5)
            ax_env.add_patch(c)
        else:
            cx,cy=obs['center']; sx,sy=obs['size']
            r = plt.Rectangle((cx-sx/2,cy-sy/2),sx,sy,color='gray',alpha=0.5)
            ax_env.add_patch(r)
    ax_env.plot(*target,'rX',markersize=12)
    path_line, = ax_env.plot([],[],'-b')
    robot_tri  = plt.Polygon([[0,0],[0,0],[0,0]], color='blue', alpha=0.8)
    ax_env.add_patch(robot_tri)
    beam_col = LineCollection([], colors='orange', linewidths=0.7, alpha=0.6)
    ax_env.add_collection(beam_col)
    arrow_artist = None
    ax_env.set_title('Env (with beams + VFH arrow)'); ax_env.grid(True)

    ax_dist.set_title('Distance to Goal')
    ax_dist.set_xlabel('Time [s]'); ax_dist.set_ylabel('m')
    dist_line, = ax_dist.plot([],[],'-g'); ax_dist.grid(True)

    ax_yaw.set_title('Yaw (rad)')
    ax_yaw.set_xlabel('Time [s]'); ax_yaw.set_ylabel('rad')
    yaw_line, = ax_yaw.plot([],[],'-m'); ax_yaw.grid(True)

    plt.tight_layout()
    return (fig, ax_env, ax_dist, ax_yaw,
            path_line, robot_tri, beam_col,
            dist_line, yaw_line, arrow_artist)

def update_robot_patch(patch, pose):
    x,y,theta = pose
    tri = np.array([[0.3,0.2],[0.3,-0.2],[-0.2,0.0]])
    R = np.array([[np.cos(theta),-np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    pts = (R@tri.T).T + np.array([x,y])
    patch.set_xy(pts)

def main():
    obstacles = [
      {'type':'box',      'center':(0, 15), 'size':(30,1)},
      {'type':'box',      'center':(0, -5), 'size':(30,1)},
      {'type':'cylinder', 'center':(0, 8),  'radius':0.5},
      {'type':'cylinder', 'center':(5, 0),  'radius':0.5},
      {'type':'cylinder', 'center':(3, 2),  'radius':0.5}
    ]
    target = (10.0, 0.0)
    pose   = np.array([0.0, 0.0, 0.0])
    prev_hd = pose[2]

    v, dt      = 0.5, 0.1
    MAX_TURN   = 360.0  # deg/s

    (fig, ax_env, ax_dist, ax_yaw,
     path_line, robot_tri, beam_col,
     dist_line, yaw_line, arrow_artist) = setup_plots(obstacles,target)

    sim_t  = 0.0
    path   = [pose[:2].copy()]
    times  = [0.0]
    dists  = [np.linalg.norm(pose[:2]-target)]
    yaws   = [pose[2]]

    for _ in range(1000):
        dist_goal = np.linalg.norm(pose[:2]-target)
        if dist_goal<0.1:
            print(f"Reached at t={sim_t:.1f}s")
            break

        angles,ranges = simulate_scan_vl53(pose,obstacles)

        # draw beams
        x0,y0,_ = pose; segs=[]
        for ang,rng in zip(angles,ranges):
            a = np.deg2rad(ang)
            segs.append([(x0,y0),(x0+rng*np.cos(a),y0+rng*np.sin(a))])
        beam_col.set_segments(segs)

        # VFH+ heading
        raw_hd = np.arctan2(target[1]-pose[1], target[0]-pose[0])
        pose[2] = raw_hd
        prev_hd = raw_hd
        vfh_hd = vfh_plus(angles, ranges,
                          np.rad2deg(raw_hd),
                          np.rad2deg(prev_hd))

        # draw arrow
        if arrow_artist: arrow_artist.remove()
        arrow_artist = ax_env.arrow(
            x0, y0,
            0.9*np.cos(vfh_hd),
            0.9*np.sin(vfh_hd),
            head_width=0.2, head_length=0.2,
            fc='red', ec='red')

        # rate‐limit and instant turn (no SMOOTH)
        max_d = np.deg2rad(MAX_TURN*dt)
        delta = ((vfh_hd - pose[2] + np.pi) % (2*np.pi)) - np.pi
        delta = np.clip(delta, -max_d, max_d)
        new_hd = pose[2] + delta
        prev_hd = pose[2]
        pose[2] = new_hd

        # move forward
        pose[0] += v*np.cos(pose[2])*dt
        pose[1] += v*np.sin(pose[2])*dt
        sim_t += dt

        # record & redraw
        path.append(pose[:2].copy()); times.append(sim_t)
        dists.append(dist_goal); yaws.append(pose[2])

        arr = np.array(path)
        path_line.set_data(arr[:,0],arr[:,1])
        update_robot_patch(robot_tri,pose)
        ax_env.relim(); ax_env.autoscale_view()

        dist_line.set_data(times,dists)
        ax_dist.set_xlim(0,max(times)+1)
        ax_dist.set_ylim(0,max(dists)+0.5)

        yaw_line.set_data(times,yaws)
        ax_yaw.set_xlim(0,max(times)+1)
        ax_yaw.set_ylim(min(yaws)-0.1,max(yaws)+0.1)

        fig.canvas.draw(); fig.canvas.flush_events()
        plt.pause(0.001)

    plt.ioff(); plt.show()

if __name__=="__main__":
    main()
