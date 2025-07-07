import pandas as pd
import yaml
import numpy as np
from shapely.geometry import LineString

class ScenarioDescriber:
    def __init__(self, scenario_yaml_path, trajectory_csv_path, map_yaml_path):
        with open(scenario_yaml_path, "r") as f:
            self.agents = yaml.safe_load(f)["scenario"]["agents"]     
        self.road_order = self._load_map_description(map_yaml_path)
        # Note: We assume the trajectory CSV is sorted by time for each agent.
        # It must also contain a 'speed' column in m/s.
        self.trajectory = pd.read_csv(trajectory_csv_path)
        self.agent_dict = self._build_agent_dict()
        self.agent_classifications = {}
        
    def _load_map_description(self, map_path):
        with open(map_path, "r") as f:
            road_order = yaml.safe_load(f)["Roads"]
        return road_order

    def _get_agent_by_id(self, track_id):
        for agent in self.agents:
            if agent["track_id"] == track_id:
                return agent
        return None

    def _build_agent_dict(self):
        agent_dict = {}
        for agent in self.agents:
            track_id = agent["track_id"]
            action = self._get_route_action(agent)
            if action:
                entry_time = action["attributes"]["start_time"]
                exit_time = action["attributes"]["end_time"]
            else:
                entry_time = exit_time = None

            agent_dict[track_id] = {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "type": agent["type"]
            }
        return agent_dict

    def _get_route_action(self, agent):
        for action in agent.get("actions", []):
            if action["type"] in ["go_straight", "turn_left", "turn_right"]:
                return action
        return None

    def _ms_to_kmh(self, speed_ms):
        """Converts speed from meters per second to kilometers per hour."""
        return speed_ms * 3.6

    def _get_speed_at_time(self, tid, t):
        """
        Gets the speed of an agent at a specific time from the trajectory data.
        Returns the speed in m/s.
        """
        # Find the last recorded data point at or just before the given time 't'
        segment = self.trajectory[(self.trajectory.trackId == tid) & (self.trajectory.time <= t)]
        if segment.empty:
            # If no data point is found (e.g., time 't' is before the agent appears),
            # return None.
            return None
        # Return the speed from the last row of the filtered segment
        return segment.iloc[-1]["velocity"]

    def _trajectory_linestring(self, df, tid, t_start, t_end):
        seg = df[(df.trackId == tid) & (df.time >= t_start) & (df.time <= t_end)]
        if len(seg) < 2:
            return None
        coords = seg[["world_x", "world_y"]].values
        return LineString(coords)

    def _classify_agents(self, ego_id):
        classification = {
            "Ego": [ego_id],
            "Key": [],
            "Affected": [],
            "Unrelated": []
        }
        ego_entry = self.agent_dict[ego_id]["entry_time"]
        ego_exit = self.agent_dict[ego_id]["exit_time"]

        if ego_entry is None or ego_exit is None:
            for tid in self.agent_dict:
                if tid != ego_id:
                    classification["Unrelated"].append(tid)
            return classification

        ego_line = self._trajectory_linestring(self.trajectory, ego_id, ego_entry, ego_exit)

        for tid, meta in self.agent_dict.items():
            if tid == ego_id:
                continue

            a_entry = meta["entry_time"]
            a_exit = meta["exit_time"]

            if a_entry is None or a_exit is None:
                classification["Unrelated"].append(tid)
                continue

            overlap = not (a_exit < ego_entry or a_entry > ego_exit)
            if not overlap:
                classification["Unrelated"].append(tid)
                continue

            a_line = self._trajectory_linestring(self.trajectory, tid, a_entry, a_exit)
            if a_line is None or not ego_line.intersects(a_line):
                classification["Unrelated"].append(tid)
                continue

            if a_exit < ego_exit:
                classification["Key"].append(tid)
            else:
                classification["Affected"].append(tid)
        return classification
    
    def _acceleration_magnitude(self, value):
        abs_val = abs(value)
        if abs_val < 1.0:
            return "Slightly"
        elif abs_val < 2.0:
            return "Moderately"
        else:
            return "Significantly"

    def _initial_description(self, ego_id):
        ego = self._get_agent_by_id(ego_id)
        t0 = ego["enter_simulation_time"]
        road, lane = ego["initial_position"][:2]
        agent_type = self.agent_dict[ego_id]["type"]
        
        # Get initial speed, convert to km/h, and add to sentence.
        initial_speed_ms = ego.get("initial_speed", 0.0)
        initial_speed_kmh = self._ms_to_kmh(initial_speed_ms)
        speed_phrase = f"at {initial_speed_kmh:.1f} km/h"

        # Use .lower() to match example output ("car 20:")
        header = f"{agent_type.lower()} {ego_id}:"

        if road in self.road_order:
            return f"{header}\n- Enters the scenario at t={t0:.2f}, starting from road {road}, lane {lane} {speed_phrase}."
        else:
            return f"{header}\n- Enters the scenario at t={t0:.2f}, starting inside the intersection {speed_phrase}."

    def _timeline_description(self, ego_id, classification):
        ego = self._get_agent_by_id(ego_id)
        events = []
        ego_actions = ego.get("actions", [])
        route_action = self._get_route_action(ego)

        if route_action:
            attr = route_action["attributes"]
            t_entry = attr["start_time"]
            t_exit = attr["end_time"]
            if "trajectory" in attr:
                entry_road, entry_lane = attr["trajectory"][0][:2]
                exit_road, exit_lane = attr["trajectory"][-1][:2]     
            else:
                entry_road, entry_lane = attr["entry_point"][:2]
                exit_road, exit_lane = attr["exit_point"][:2]
            movement = route_action["type"].replace("_", " ")
            
            if exit_road in self.road_order:
                if entry_road in self.road_order:
                    sentence = f"- Enters the intersection at t={t_entry:.2f}, {movement} from Road {entry_road}, Lane {entry_lane} towards Road {exit_road}, Lane {exit_lane}."
                else:
                    sentence = f"- {movement} towards Road {exit_road}, Lane {exit_lane}."
            else:
                if entry_road in self.road_order:
                    entry_idx = self.road_order.index(entry_road)
                    offset = {"go straight": 2, "turn right": 1, "turn left": 3}.get(movement, 0)
                    dest_idx = (entry_idx + offset) % len(self.road_order)
                    estimated_exit_road = self.road_order[dest_idx]
                    sentence = f"- Enters the intersection at t={t_entry:.2f}, {movement} from Road {entry_road}, Lane {entry_lane} towards Road {estimated_exit_road}."
                else:
                    sentence = f"- {movement} inside the intersection"

            events.append((t_entry, sentence))
            if exit_road in self.road_order:
                events.append((t_exit, f"- Leaves the intersection at t={t_exit:.2f}."))

        for action in ego_actions:
            t_start = action["attributes"]["start_time"]
            act_type = action["type"]

            if act_type == "slow_down" or act_type == "speed_up":
                # --- MODIFICATION START ---
                acc = action["attributes"].get("acceleration", 0.0)
                adverb = self._acceleration_magnitude(acc)
                target_speed_ms = action["attributes"].get("target_speed")
                duration = action["attributes"].get("duration")

                # Build the time phrase first
                if duration is not None:
                    t_end = t_start + duration
                    time_phrase = f"from t={t_start:.2f} to t={t_end:.2f}"
                else:
                    time_phrase = f"at t={t_start:.2f}" # Fallback if no duration

                speed_before_ms = self._get_speed_at_time(ego_id, t_start)
                verb = "slows down" if act_type == "slow_down" else "speeds up"

                if speed_before_ms is None or target_speed_ms is None:
                    # Fallback to a simpler description if speed data is incomplete
                    line = f"- {adverb} {verb} {time_phrase}."
                else:
                    # Build the full, detailed description
                    speed_before_kmh = self._ms_to_kmh(speed_before_ms)
                    speed_after_kmh = self._ms_to_kmh(target_speed_ms)
                    stop_phrase = " till stopped" if target_speed_ms < 0.5 else ""

                    line = (f"- {adverb} {verb} {time_phrase}, from {speed_before_kmh:.1f} km/h "
                            f"to {speed_after_kmh:.1f} km/h{stop_phrase}.")
                # --- MODIFICATION END ---
                events.append((t_start, line))

            elif act_type == "lane_change":
                direction = action["attributes"]["direction"]
                line = f"- Changes lane to the {direction} at t={t_start:.2f}."
                events.append((t_start, line))

        exit_time = ego["exit_simulation_time"]
        events.append((exit_time, f"- Exits the scenario at t={exit_time:.2f}."))
        sorted_lines = [line for _, line in sorted(events)]
        return sorted_lines
    
    def generate_description(self, ego_id):
        classification = self._classify_agents(ego_id)
        self.agent_classifications[ego_id] = classification
        
        # _initial_description now returns the header and the first line
        initial_lines = self._initial_description(ego_id).split('\n')
        
        lines = initial_lines
        lines += self._timeline_description(ego_id, classification)
        return "\n".join(filter(None, lines))