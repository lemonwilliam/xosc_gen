# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 15:17:13 2025

@author: dianel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


def read_csv():
    path = '../data/processed/inD/trajectory/world/12_2250_2550.csv'
    data = pd.read_csv(path)
    row = find_representative_points(data, 24)
    print(row)

def curvature_from_xy(x, y, time):
    dx = np.gradient(x, time)
    dy = np.gradient(y, time)
    ddx = np.gradient(dx, time)
    ddy = np.gradient(dy, time)

    epsilon = 1e-8
    denom = (dx**2 + dy**2)**1.5 + epsilon        
    curvature = np.abs(dx * ddy - dy * ddx) / denom
    return curvature


def split_segments(indices):
    segments = []
    current_segment = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            current_segment.append(indices[i])
        else:
            segments.append(current_segment)
            current_segment = [indices[i]]
    segments.append(current_segment)
    return segments

def find_representative_points_each_segment(df, track_id, turn_thresh=0.1, seg_thresh=0.0):
    df_track = df[df['trackId'] == track_id].copy().reset_index(drop=True)
    x = df_track['world_x'].values
    y = df_track['world_y'].values
    time = df_track['time'].values

    curvature = curvature_from_xy(x, y, time)
    tthresh = np.max(curvature) * turn_thresh
    sthresh = np.max(curvature) * seg_thresh
    turning_idx = np.where(curvature >= tthresh)[0]

    if len(turning_idx) == 0:
        print("No turning points found.")
        return None

    segments = split_segments(turning_idx)

    rep_points_idx = []
    peak = []

    for seg in segments:
        start_idx = seg[0]
        end_idx = seg[-1]
        seg_curv = curvature[seg]
        local_max_indices = argrelextrema(seg_curv, np.greater)[0]
        peak_indices = [seg[i] for i in local_max_indices if seg_curv[i] >= sthresh]
        if len(peak_indices) == 0:
            continue
        
        rep_points_idx.append(start_idx)
        rep_points_idx.append(end_idx)
        rep_points_idx.extend(peak_indices)
        peak.extend((peak_indices))
        
    rep_points_idx.append(0)
    rep_points_idx.append(len(df_track) - 1)
    rep_points_idx = sorted(list(set(rep_points_idx)))
    
    if len(rep_points_idx) < 5:
        total_len = len(df_track)
        q1 = total_len // 4
        median = total_len // 2
        q3 = (3 * total_len) // 4
        rep_points_idx.extend([q1, median, q3])
        rep_points_idx = sorted(set(rep_points_idx))
    
    rep_points = df_track.iloc[rep_points_idx].copy()
    rep_points['curvature'] = curvature[rep_points_idx]
   
    peak = sorted(list(set(peak)))
    
    # debug graph
    if __debug__:
        plt.figure(figsize=(10,6))
        plt.plot(x, y, label='Path', linewidth=2)
        plt.scatter(x[turning_idx], y[turning_idx], color='orange', label='Turning Points', s=20)
        plt.scatter(x[rep_points_idx], y[rep_points_idx], color='red', label='Representative Points', s=40)
        for i, idx in enumerate(rep_points_idx):
            plt.text(x[idx], y[idx], f'{idx}', fontsize=10, color='red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Track {track_id} Representative Points Per Turning Segment')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
    
        plt.figure(figsize=(10, 4))
        plt.plot(curvature, label='Curvature', linewidth=2)
        plt.scatter(peak, curvature[peak], color='red', label='peak', zorder=3, s=20)
        plt.scatter(rep_points_idx, curvature[rep_points_idx], color='orange', label='Representative Points', zorder=2, s=40)
        for i, idx in enumerate(rep_points_idx):
            plt.text(idx, curvature[idx], f'{idx}', fontsize=10, color='red')
        plt.xlabel('Row Index')
        plt.ylabel('Curvature')
        plt.title(f'Track {track_id} Curvature vs Row Index')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return rep_points

def find_representative_points(data, trackID, plot=False, turn_thresh=0.1, seg_thresh=0.3):
    '''
    Analyze the trajectory of the specified track ID to identify representative points 
    in each turning segment. These include the segment's start/end points and points 
    of maximum curvature.

    Parameters
    ----------
    data : pandas.DataFrame
        Input world-coordinate trajectory data. Must include columns: "trackId", "x", "y", and "time".
    trackID : int
        The ID of the trajectory to analyze.
    plot : bool, optional
        Whether to plot the path and curvature visualization. Default is False.
    turn_thresh : float, optional
        Relative threshold (percentage of max curvature) used to detect turning points. Default is 0.1.
    seg_thresh : float, optional
        Relative threshold to detect multiple peaks in one segment. Default is 0.3.

    Returns
    -------
    list
        Indices (row numbers) of the representative points within the given track.
    '''

    rep_points = find_representative_points_each_segment(
        df=data,
        track_id=trackID,
        turn_thresh=turn_thresh,
        seg_thresh=seg_thresh
    )

    if rep_points is not None:
        return rep_points.index.tolist()
    else:
        return []

if __name__ == '__main__':
    read_csv()

        
