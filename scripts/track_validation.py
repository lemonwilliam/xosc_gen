# -*- coding: utf-8 -*-
"""
Created on Fri May  2 05:45:14 2025

@author: dianel
"""
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class PointData:
    def __init__(self, timestamp=0, x=0, y=0, heading=0, velocity=0):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = velocity
    
class Track:
    def __init__(self, ID=0, isstatic=True, timescope=20, timestep=0.2):
        self.ID = ID
        self.track = []
        self.isstatic = isstatic
        self.timescope = timescope
        self.timestep = timestep
    
    def remove_redundancy(self, ori):
        if self.timescope <= ori.timescope:
            return
        min_redundancy_idx = int(ori.timescope/ori.timestep) + 1
        for i in range(min_redundancy_idx, len(self.track)):
            if self.track[i].velocity != 0:
                min_redundancy_idx = i + 1
        self.track = self.track[:min_redundancy_idx]
        self.timescope = self.track[-1].timestamp
        
        

class TrackSet:
    def __init__(self):
        self.tracks = []
        self.track_id = list()
    
    def readTrack(self, path):
        with open(path) as f:
            csvReader = csv.DictReader(f)
            
            iid = -1
            for row in csvReader:
                tid = int(row['trackId'])
                timestep, x, y, h, v = float(row['time']), float(row['x']), float(row['y']), float(row['heading']), float(row['velocity'])    
                if tid not in self.track_id :
                    iid += 1
                    self.track_id.append(tid)
                    self.tracks.append(Track(ID=iid))

                self.tracks[iid].track.append(
                    PointData(timestep, x, y, h, v)                     
                )
                if v!= 0.0 and self.tracks[iid].isstatic:
                    self.tracks[iid].isstatic = False
    
                
        self.track_id = [i for i in range(0, len(self.track_id))]
        for i in self.track_id:
            track = self.tracks[i]
            track.timescope = track.track[-1].timestamp
            if track.timescope != 0:
                track.timestep = track.track[1].timestamp - track.track[0].timestamp


class TrackValidation:
    def __init__(self):
        pass
    
    def compute_direction_vectors(self, track):
        pts = np.array([(p.x, p.y) for p in track.track])
        directions = pts[1:] - pts[:-1]
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.clip(norms, 1e-8, None)
        return directions
    
    
    def cosine_similarity_vectors(self, vecs1, vecs2):
        L = min(len(vecs1), len(vecs2))
        dot = np.sum(vecs1[:L] * vecs2[:L], axis=1)
        similarity = np.mean(dot)
        return similarity
    
    
    def pos_similarity(self, track0, track1):
        v0 = self.compute_direction_vectors(track0)
        v1 = self.compute_direction_vectors(track1)
        sim = self.cosine_similarity_vectors(v0, v1)
        return max(0.0, sim)  # clip 負值變 0
        
    
    def heading_similarity(self, track0, track1):
        h0 = [p.heading for p in track0.track]
        h1 = [p.heading for p in track1.track]
        min_len = min(len(h0), len(h1))
        diffs = [
            abs(np.arctan2(np.sin(a - b), np.cos(a - b)))
            for a, b in zip(h0[:min_len], h1[:min_len])
        ]
        return 1.0 - np.mean(diffs) / np.pi if diffs else 0.0
    
    def velocity_similarity(self, track0, track1):
        v0 = [p.velocity for p in track0.track]
        v1 = [p.velocity for p in track1.track]
        min_len = min(len(v0), len(v1))
        diffs = [abs(a - b) for a, b in zip(v0[:min_len], v1[:min_len])]
        vmax = max(max(v0, default=1.0), max(v1, default=1.0))  # 避免除以 0
        return 1.0 - np.mean(diffs) / vmax if diffs else 0.0
    
    def combined_similarity(self, track0, track1, alpha=0.6, beta=0.2, gamma=0.2):
        pos_sim = self.pos_similarity(track0, track1)
        head_sim = self.heading_similarity(track0, track1)
        vel_sim = self.velocity_similarity(track0, track1)
        print(f'position score:{pos_sim:.2f}, headind score:{head_sim:.2f}, velocity score:{vel_sim:.2f}')
        return alpha * pos_sim + beta * head_sim + gamma * vel_sim
         
    def plot_tracks(self, track_ori: Track, track_gen: Track):
        def extract_points(track):
            xs = [p.x for p in track.track]
            ys = [p.y for p in track.track]
            headings = [p.heading for p in track.track]
            return np.array(xs), np.array(ys), np.array(headings)
    
        x_ori, y_ori, h_ori = extract_points(track_ori)
        x_gen, y_gen, h_gen = extract_points(track_gen)
    
        plt.figure(figsize=(10, 8))
        
        # 原始路徑 (紅色)
        plt.plot(x_ori, y_ori, 'r-', label='Original Track')
        plt.quiver(x_ori, y_ori, np.cos(h_ori), np.sin(h_ori), 
                   color='red', angles='xy', scale_units='xy', scale=1, width=0.003)
    
        # 生成路徑 (藍色)
        plt.plot(x_gen, y_gen, 'b-', label='Generated Track')
        plt.quiver(x_gen, y_gen, np.cos(h_gen), np.sin(h_gen), 
                   color='blue', angles='xy', scale_units='xy', scale=1, width=0.003)
    
        plt.title(f"Track {track_ori.ID} vs Generated")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()
                
                
    def validation(self, tracks_ori, tracks_gen, alpha=0.6, beta=0.2, gamma=0.2): #請確保兩個timestep一致
        if len(tracks_ori.track_id) != len(tracks_gen.track_id):
            print("please check every tracks are correspond\n")
            return -1
        scores = []
        move_unsync = 0
        data_num = len(tracks_ori.track_id)
    
        for to, tg in zip(tracks_ori.tracks, tracks_gen.tracks):
            if to.isstatic != tg.isstatic:
                move_unsync += 1
            elif not to.isstatic:
                tg.remove_redundancy(to)
                sim = self.combined_similarity(to, tg, alpha=alpha, beta=beta, gamma=gamma)
                scores.append(sim)
                self.plot_tracks(to, tg)
                print (f"the similiar score of track {tg.ID} is {sim:.2f}\n")
    
        if scores:
            score = np.mean(scores)  - move_unsync / data_num
        
        print (f"the similiar score of this 2 group of tracks are {score:.2f}, and {move_unsync} track(s) is/are move->static or static->move\n\n")
        return score

  

track0 = TrackSet()
track0.readTrack("12_2300_2549.csv")
track1 = TrackSet()
track1.readTrack('12_2300_2549_gen.csv')
tv = TrackValidation()
score = tv.validation(track0, track1)
track0 = TrackSet()
track0.readTrack("14_0_299.csv")
track1 = TrackSet()
track1.readTrack('14_0_299_gen.csv')
tv = TrackValidation()
score = tv.validation(track0, track1)
