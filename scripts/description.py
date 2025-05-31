import pandas as pd
import yaml
import numpy as np
from shapely.geometry import LineString


class ScenarioDescriber:
    def __init__(self, scenario_yaml_path, trajectory_csv_path, map_yaml_path):
        self.agents = self._load_yaml(scenario_yaml_path)["scenario"]["agents"]
        self.road_id_to_name, self.road_order = self._load_map_description(map_yaml_path)
        self.trajectory = pd.read_csv(trajectory_csv_path)
        self.agent_dict = self._build_agent_dict()
        self.agent_classifications = {}

    def _load_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
        
    def _load_map_description(self, map_path):
        with open(map_path, "r") as f:
            road_map = yaml.safe_load(f)["Roads"]
        id_to_name = {v: k for k, v in road_map.items()}
        ordered_names = list(road_map.keys())  # counter-clockwise
        return id_to_name, ordered_names

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
                "type": agent["type"]  # <- store type here
            }
        return agent_dict

    def _get_route_action(self, agent):
        for action in agent.get("actions", []):
            if action["type"] in ["go_straight", "turn_left", "turn_right"]:
                return action
        return None

    def _trajectory_linestring(self, df, tid, t_start, t_end):
        seg = df[(df.trackId == tid) & (df.time >= t_start) & (df.time <= t_end)]
        if len(seg) < 2:
            return None
        coords = seg[["x", "y"]].values
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

    def _initial_description(self, ego_id):
        ego = self._get_agent_by_id(ego_id)
        t0 = ego["enter_simulation_time"]
        road = ego["initial_position"][0]
        agent_type = self.agent_dict[ego_id]["type"]

        if road in self.road_id_to_name:
            road_name = self.road_id_to_name[road]
            return f"{agent_type} {ego_id}:\n- Enters the scenario at t={t0:.2f}, starting from {road_name}."
        else:
            return f"{agent_type} {ego_id}:\n- Enters the scenario at t={t0:.2f}, starting inside the intersection."


    def _timeline_description(self, ego_id, classification):
        ego = self._get_agent_by_id(ego_id)
        events = []
        key_agents = classification["Key"]
        ego_actions = ego.get("actions", [])
        route_action = self._get_route_action(ego)
        intersection_end = self.agent_dict[ego_id]["exit_time"]

        # Insert intersection entry and exit as events if applicable
        if route_action:
            attr = route_action["attributes"]
            t_entry = attr["start_time"]
            t_exit = attr["end_time"]
            entry_road, entry_lane = attr["entry_point"][:2]
            exit_road, exit_lane = attr["exit_point"][:2]
            movement = route_action["type"].replace("_", " ")
            
            if exit_road in self.road_id_to_name:
                dest_road_name = self.road_id_to_name[exit_road]
                sentence = f"- Enters the intersection at t={t_entry:.2f}, {movement} towards {dest_road_name}."
            else:
                current_name = self.road_id_to_name[entry_road]
                idx = self.road_order.index(current_name)
                offset = {"go straight": 2, "turn right": 1, "turn left": 3}.get(movement, 0)
                dest_idx = (idx + offset) % len(self.road_order)
                dest_road_name = self.road_order[dest_idx]
                sentence = f"- Enters the intersection at t={t_entry:.2f}, {movement} towards {dest_road_name}."

            events.append((t_entry, sentence))
            events.append((t_exit, f"- Leaves the intersection at t={t_exit:.2f}."))

        # Track previous action state for reasoning
        previous_action_type = None
        previous_causes = []

        for i, action in enumerate(ego_actions):
            t_start = action["attributes"]["start_time"]
            act_type = action["type"]

            if act_type == "slow_down":
                t_end = None
                for a in ego_actions[i+1:]:
                    if a["type"] == "speed_up":
                        t_end = a["attributes"]["start_time"]
                        break
                causes = []
                for aid in key_agents:
                    a_entry = self.agent_dict[aid]["entry_time"]
                    a_exit = self.agent_dict[aid]["exit_time"]
                    if t_end and not (a_exit < t_start or a_entry > t_end):
                        causes.append(aid)
                if t_start < intersection_end:
                    if causes:
                        listed = ", ".join(f"{self.agent_dict[c]['type']} {c}" for c in causes)
                        line = f"- Starts slowing down at t={t_start:.2f} to look out for {listed}."
                    else:
                        line = f"- Starts slowing down at t={t_start:.2f} for cautious driving."
                else:
                    line = f"- Starts slowing down at t={t_start:.2f} for cautious driving."
                events.append((t_start, line))
                previous_action_type = "slow_down"
                previous_causes = causes

            elif act_type == "speed_up":
                if previous_action_type == "slow_down" and previous_causes:
                    latest = max(previous_causes, key=lambda x: self.agent_dict[x]["exit_time"])
                    atype = self.agent_dict[latest]["type"]
                    line = f"- Starts speeding up at t={t_start:.2f} since {atype} {latest} has passed the intersection."
                else:
                    line = f"- Starts speeding up at t={t_start:.2f} since the path is clear."
                events.append((t_start, line))
                previous_action_type = "speed_up"

        # Sort all events chronologically
        sorted_lines = [line for _, line in sorted(events)]
        return sorted_lines
    
    def generate_description(self, ego_id):
        classification = self._classify_agents(ego_id)
        self.agent_classifications[ego_id] = classification
        lines = [
            self._initial_description(ego_id)
        ]
        lines += self._timeline_description(ego_id, classification)
        return "\n".join(filter(None, lines))

