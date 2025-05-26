import pandas as pd
import yaml
import numpy as np
from shapely.geometry import LineString


class ScenarioDescriber:
    def __init__(self, scenario_yaml_path, trajectory_csv_path, ego_id):
        self.ego_id = ego_id
        self.scenario = self._load_yaml(scenario_yaml_path)
        self.trajectory = pd.read_csv(trajectory_csv_path)
        self.agent_dict = self._build_agent_dict()
        self.ego_agent = self._get_agent_by_id(ego_id)
        self.classification = self._classify_agents()

    def _load_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _get_agent_by_id(self, track_id):
        for agent in self.scenario["scenario"]["agents"]:
            if agent["track_id"] == track_id:
                return agent
        return None

    def _build_agent_dict(self):
        agent_dict = {}
        for agent in self.scenario["scenario"]["agents"]:
            track_id = agent["track_id"]
            action = self._get_route_action(agent)
            if action:
                entry_time = action["attributes"]["start_time"]
                exit_time = action["attributes"]["end_time"]
            else:
                entry_time = exit_time = None
            agent_dict[track_id] = {
                "entry_time": entry_time,
                "exit_time": exit_time
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

    def _classify_agents(self):
        classification = {
            "Ego": [self.ego_id],
            "Key": [],
            "Affected": [],
            "Unrelated": []
        }

        ego_entry = self.agent_dict[self.ego_id]["entry_time"]
        ego_exit = self.agent_dict[self.ego_id]["exit_time"]

        if ego_entry is None or ego_exit is None:
            for tid in self.agent_dict:
                if tid != self.ego_id:
                    classification["Unrelated"].append(tid)
            return classification

        ego_line = self._trajectory_linestring(self.trajectory, self.ego_id, ego_entry, ego_exit)

        for tid, meta in self.agent_dict.items():
            if tid == self.ego_id:
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

    def _initial_description(self):
        init = self.ego_agent
        t0 = init["enter_simulation_time"]
        road, lane = init["initial_position"][:2]
        return f"Ego agent (car {self.ego_id}):\n- Enters the scenario at t={t0:.2f}, starting at Road {road}, lane {lane}."

    def _intersection_description(self):
        action = self._get_route_action(self.ego_agent)
        if not action:
            return ""
        attr = action["attributes"]
        t_entry = attr["start_time"]
        t_exit = attr["end_time"]
        exit_road, exit_lane = attr["exit_point"][:2]
        return f"- Enters the intersection at t={t_entry:.2f}, {action['type'].replace('_', ' ')} towards Road {exit_road}, lane {exit_lane}. Leaves the intersection at t={t_exit:.2f}."

    def _action_description(self):
        lines = []
        key_agents = self.classification["Key"]
        ego_actions = self.ego_agent.get("actions", [])
        intersection_end = self.agent_dict[self.ego_id]["exit_time"]
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
                        listed = ", ".join(f"car {c}" for c in causes)
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
                    lines.append(f"- Starts speeding up at t={t_start:.2f} since car {latest} has passed the intersection.")
                else:
                    lines.append(f"- Starts speeding up at t={t_start:.2f} since the path is clear.")
                previous_action_type = "speed_up"

        return lines

    def _affected_description(self):
        desc = []
        for tid in self.classification["Affected"]:
            route = self._get_route_action(self._get_agent_by_id(tid))
            if route:
                desc.append(f"- car {tid} {route['type'].replace('_', ' ')} after ego agent has passed the intersection.")
        return desc

    def generate_description(self):
        lines = [
            self._initial_description(),
            self._intersection_description()
        ]
        lines += self._action_description()
        lines += self._affected_description()
        return "\n".join(filter(None, lines))
