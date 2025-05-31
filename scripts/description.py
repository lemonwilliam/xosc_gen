import pandas as pd
import yaml
import numpy as np
from shapely.geometry import LineString


class ScenarioDescriber:
    def __init__(self, scenario_yaml_path, trajectory_csv_path):
        self.agents = self._load_yaml(scenario_yaml_path)["scenario"]["agents"]
        self.trajectory = pd.read_csv(trajectory_csv_path)
        self.agent_dict = self._build_agent_dict()
        self.agent_classifications = {}

    def _load_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

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
        road, lane = ego["initial_position"][:2]
        agent_type = self.agent_dict[ego_id]["type"]
        return f"{agent_type} {ego_id}:\n- Enters the scenario at t={t0:.2f}, starting from Road {road}, lane {lane}."

    def _intersection_description(self, ego_id):
        ego = self._get_agent_by_id(ego_id)
        action = self._get_route_action(ego)
        if not action:
            return ""
        attr = action["attributes"]
        t_entry = attr["start_time"]
        t_exit = attr["end_time"]
        exit_road, exit_lane = attr["exit_point"][:2]
        return f"- Enters the intersection at t={t_entry:.2f}, {action['type'].replace('_', ' ')} towards Road {exit_road}, lane {exit_lane}. Leaves the intersection at t={t_exit:.2f}."

    def _action_description(self, ego_id, classification):
        lines = []
        ego = self._get_agent_by_id(ego_id)
        key_agents = classification["Key"]
        ego_actions = ego.get("actions", [])
        intersection_end = self.agent_dict[ego_id]["exit_time"]
        previous_action_type = None
        previous_action_start = None
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
                        lines.append(f"- Starts slowing down at t={t_start:.2f} to look out for {listed}.")
                    else:
                        lines.append(f"- Starts slowing down at t={t_start:.2f} for cautious driving.")
                else:
                    lines.append(f"- Starts slowing down at t={t_start:.2f} for cautious driving.")
                previous_action_type = "slow_down"
                previous_action_start = t_start
                previous_causes = causes

            elif act_type == "speed_up":
                if previous_action_type == "slow_down" and previous_causes:
                    latest = max(previous_causes, key=lambda x: self.agent_dict[x]["exit_time"])
                    atype = self.agent_dict[latest]["type"]
                    lines.append(f"- Starts speeding up at t={t_start:.2f} since {atype} {latest} has passed the intersection.")
                else:
                    lines.append(f"- Starts speeding up at t={t_start:.2f} since the path is clear.")
                previous_action_type = "speed_up"

        return lines

    def generate_description(self, ego_id):
        classification = self._classify_agents(ego_id)
        self.agent_classifications[ego_id] = classification
        lines = [
            self._initial_description(ego_id),
            self._intersection_description(ego_id)
        ]
        lines += self._action_description(ego_id, classification)
        return "\n".join(filter(None, lines))
