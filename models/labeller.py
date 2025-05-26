import yaml
import pandas as pd
import numpy as np
from copy import deepcopy

# 1. Define a marker class that inherits from list
class FlowList(list):
    pass

# 2. Define a representer for this FlowList
def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


class Labeller:
    def __init__(self, meta_path: str, map_path: str, gt_trajectory_path: str):

        yaml.add_representer(FlowList, flow_list_representer)

        # 1) load metadata
        with open(meta_path, 'r') as f:
            self.metadata = yaml.safe_load(f)
        # 2) load map
        with open(map_path, 'r') as f:
            self.map_data = yaml.safe_load(f)['junctions']
        # 3) load trajectory
        self.lane_df = pd.read_csv(gt_trajectory_path)

        # 4) initialize scenario dict
        self.scenario = {
            'description': "Placeholder desc",
            'author': "William",
            'dataset': self.metadata['dataset'],
            'location': self.metadata['location'],
            'duration': self.metadata['duration'],
            'agents': []
        }

        for agent in self.metadata['agents']:
            tid = agent['track_id']
            agent_traj = self.lane_df[self.lane_df.trackId == tid].reset_index(drop=True)
            init_row = agent_traj.loc[0]
            self.scenario['agents'].append({
                'track_id': agent['track_id'],
                'type': agent['class'],
                'enter_simulation_time': init_row.time,
                'initial_position': FlowList([int(init_row.road_id), int(init_row.lane_id), init_row.lane_offset, init_row.s, init_row.heading]),
                'initial_speed': init_row.velocity,
                'actions': []
            })

    def _find_route_actions_for_track(self, track_id):
        """
        Returns a list of dicts, each with:
         - 'type': decision type
         - 'entry_idx', 'exit_idx': integer row positions in self.lane_df
        """
        df = self.lane_df[self.lane_df.trackId == track_id].reset_index(drop=True)
        actions = []

        for junction in self.map_data.values():
            # build set of all road IDs in this junction (both entry & exit)
            jroads = set()
            for conn in junction['connections']:
                jroads.add(int(conn['Connection_road']))

            # find when the agent crosses into/out of that set
            inside     = df['road_id'].isin(jroads)
            entry_idxs = df.index[(~inside.shift(1, fill_value=False)) & (inside)].tolist()
            exit_idxs  = df.index[(inside) & (~inside.shift(-1, fill_value=False))].tolist()

            for ent, ext in zip(entry_idxs, exit_idxs):
                if ent < ext:
                    h_in  = df.at[ent, 'heading']
                    h_out = df.at[ext, 'heading']
                    delta = ((h_out - h_in + 180) % 360) - 180

                    if abs(delta) <= 40:
                        typ = 'go_straight'
                    elif delta < -40:
                        typ = 'turn_right'
                    else:
                        typ = 'turn_left'

                    actions.append({
                        'type':      typ,
                        'entry_idx': ent,
                        'exit_idx':  ext
                    })

        return actions


    def _is_link_legal(self, entry_pt, exit_pt):
        """
        entry_pt, exit_pt: [road_id, lane_id, ...]
        Returns True if any connection in any junction matches this pair.
        """
        e_road, e_lane = entry_pt[0], entry_pt[1]
        x_road, x_lane = exit_pt[0],  exit_pt[1]
        for junction in self.map_data.values():
            for conn in junction['connections']:
                if (conn['Entry_road'] == e_road and
                    conn['Exit_road']  == x_road and
                    e_lane  in conn['Entry_lanes'] and
                    x_lane  in conn['Exit_lanes']):
                    return True
                elif (conn['Connection_road'] == e_road and
                      conn['Exit_road']  == x_road and
                      x_lane  in conn['Exit_lanes']):
                    return True
                elif (conn['Connection_road'] == x_road and
                      conn['Entry_road'] == e_road and
                      e_lane  in conn['Entry_lanes']):
                    return True
        return False


    def label_longitudinal_actions(self):
        """
        Fill each agent’s actions list with speed_up, slow_down, cruise entries
        before route‐decisions get prepended later.
        """
        for agent in self.scenario["agents"]:
            tid = agent['track_id']
            df = self.lane_df[self.lane_df.trackId == tid].reset_index(drop=True)
            times = df['time'].values
            vels  = df['velocity'].values

            # compute accel between frames
            dt = np.diff(times)
            dv = np.diff(vels)
            acc = dv / dt

            # label each frame:
            #   acc >  0.05 → speed_up
            #   acc < -0.05 → slow_down
            #  -0.05 <= acc <= 0.05 → cruise
            conditions = [
                acc >  0.05,
                acc < -0.05
            ]
            choices = ['speed_up', 'slow_down']
            labels = np.select(conditions, choices, default='cruise')

            long_actions = []
            start = 0
            for i in range(1, len(labels)):
                if labels[i] != labels[start] or i == len(labels)-1:
                    typ = labels[start]
                    duration = times[i] - times[start]
                    if typ in ('speed_up','slow_down') and duration > 0.5:
                        long_actions.append({
                            'type': typ,
                            'attributes': {
                                'start_time': round(float(times[start]),3),
                                'target_speed': round(float(vels[i]),3),
                                'duration': round(duration,3)
                            }
                        })
                    start = i

            # assign (will be prepended later)
            agent['actions'] = long_actions + agent.get('actions', [])


    def label_lateral_actions(self):
        """
        Append to each agent’s actions list any lane_change entries.
        """
        for agent in self.scenario["agents"]:
            tid = agent['track_id']
            df = self.lane_df[self.lane_df.trackId == tid].reset_index(drop=True)
            times = df['time'].values
            roads = df['road_id'].values
            lanes = df['lane_id'].values

            lat_actions = []
            # detect indices where lane_id jumps by +-1
            jumps = np.where(np.abs(np.diff(lanes)) == 1)[0]
            for idx in jumps:
                # back up to when stable
                start = idx
                current_road = roads[idx]
                if roads[idx] != roads[idx+1]:
                    continue
                while start>0 and lanes[start] == lanes[start-1] and roads[start] == current_road:
                    start -= 1
                # forward to settle
                end = idx+1
                while end+1 < len(lanes) and lanes[end] == lanes[end+1] and roads[end] == current_road:
                    end += 1

                duration = times[end] - times[start]
                if 1.0 <= duration <= 3.0:
                    new_lane = int(lanes[end])
                    lat_actions.append({
                        'type': 'lane_change',
                        'attributes': {
                            'start_time': round(float(times[start]),3),
                            'target_lane': new_lane,
                            'duration': round(duration,3)
                        }
                    })
                elif duration > 3.0:
                    new_lane = int(lanes[end])
                    lat_actions.append({
                        'type': 'lane_change',
                        'attributes': {
                            'start_time': round(float(times[idx])-1.5,3),
                            'target_lane': new_lane,
                            'duration': 3.0
                        }
                    })

            # merge with whatever actions already exist (longitudinal or route will come)
            agent['actions'] = lat_actions + agent.get('actions', [])


    def label_route_decisions(self):
        """
        Prepend each agent's route-decision actions, marking legality.
        """
        for agent in self.scenario["agents"]:
            tid = agent['track_id']
            lane_df_track = self.lane_df[self.lane_df.trackId == tid].reset_index(drop=True)

            actions = []
            for ra in self._find_route_actions_for_track(tid):
                ent, ext = ra['entry_idx'], ra['exit_idx']
                # 1) pick non-intersection entry/exit:
                #    entry_pt = row just before entering
                #    exit_pt  = row just after leaving
                entry_row = max(ent-1, 0)
                exit_row  = min(ext+1, len(lane_df_track)-1)

                ep = lane_df_track.loc[entry_row, ['road_id','lane_id','lane_offset','s','heading']].tolist()
                xp = lane_df_track.loc[exit_row, ['road_id','lane_id','lane_offset','s','heading']].tolist()

                # cast ids to int
                entry_pt = FlowList([int(ep[0]), int(ep[1]), ep[2], ep[3], ep[4]])
                exit_pt  = FlowList([int(xp[0]), int(xp[1]), xp[2], xp[3], xp[4]])

                start_time = float(lane_df_track.loc[entry_row, 'time'])
                end_time = float(lane_df_track.loc[exit_row, 'time'])
                legal = self._is_link_legal(entry_pt, exit_pt)

                if legal:
                    actions.append({
                        'type': ra['type'],
                        'attributes': {
                            'start_time':   start_time,
                            'end_time':     end_time,
                            'legal':        True,
                            'entry_point':  entry_pt,
                            'exit_point':   exit_pt
                        }
                    })
                else:
                    # build fallback trajectory of 5 points
                    indices = np.linspace(entry_row, min(exit_row+5, len(lane_df_track)-1), 5, dtype=int)
                    trajectory = []
                    for idx in indices:
                        row = lane_df_track.loc[idx, ['road_id','lane_id','lane_offset','s','heading']].tolist()
                        pt = FlowList([int(row[0]), int(row[1]), row[2], row[3], row[4]])
                        trajectory.append(pt)
                    actions.append({
                        'type': ra['type'],
                        'attributes': {
                            'start_time': start_time,
                            'end_time':   end_time,
                            'legal':      False,
                            'trajectory': trajectory
                        }
                    })

            # prepend these new actions
            agent['actions'] = actions + agent.get('actions', [])


    def label(self):
        """
        Run all three labelers in sequence and then sort each agent’s actions
        chronologically by their start_time.
        """

        self.label_longitudinal_actions()
        self.label_lateral_actions()
        self.label_route_decisions()

        # Sort each agent’s actions by start_time
        for agent in self.scenario["agents"]:
            # only keep actions that have a start_time attribute
            agent["actions"].sort(
                key=lambda act: act["attributes"].get("start_time", 0.0)
            )


    def normalize(self, obj):
        """
        Recursively walk obj and convert:
        - numpy scalars to native Python scalars
        - numpy arrays to lists
        - leave everything else alone
        """
        if isinstance(obj, FlowList):
            return FlowList(self.normalize(x) for x in obj)
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [self.normalize(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.normalize(v) for k, v in obj.items()}
        return obj           


    def save(self, output_path: str):
        clean = self.normalize(self.scenario)
        with open(output_path, 'w') as f:
            yaml.dump({"scenario": clean}, f, sort_keys=False, default_flow_style=False, indent=2)