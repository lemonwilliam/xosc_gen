import pandas as pd
import yaml
import numpy as np
from shapely.geometry import LineString


class ScenarioDescriber:
    def __init__(self, scenario_yaml_path, trajectory_csv_path, map_yaml_path):
        with open(scenario_yaml_path, "r") as f:
            self.agents = yaml.safe_load(f)["scenario"]["agents"]     
        self.road_order = self._load_map_description(map_yaml_path)
        self.trajectory = pd.read_csv(trajectory_csv_path)
        self.agent_dict = self._build_agent_dict()
        self.agent_classifications = {}
        
    def _load_map_description(self, map_path):
        with open(map_path, "r") as f:
            road_map = yaml.safe_load(f)["Roads"]
        road_order = list(road_map.values())  # Order of the roads, counter-clockwise
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

        if road in self.road_order:
            return f"{agent_type} {ego_id}:\n- Enters the scenario at t={t0:.2f}, starting from road {road}, lane {lane}."
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
            if "trajectory" in attr:
                entry_road, entry_lane = attr["trajectory"][0][:2]
                exit_road, exit_lane = attr["trajectory"][-1][:2]     
            else:
                entry_road, entry_lane = attr["entry_point"][:2]
                exit_road, exit_lane = attr["exit_point"][:2]
            movement = route_action["type"].replace("_", " ")
            
            if exit_road in self.road_order:
                sentence = f"- Enters the intersection at t={t_entry:.2f}, {movement} from Road {entry_road}, Lane {entry_lane} towards Road {exit_road}, Lane {exit_lane}."
            else:
                entry_idx = self.road_order.index(entry_road)
                offset = {"go straight": 2, "turn right": 1, "turn left": 3}.get(movement, 0)
                dest_idx = (entry_idx + offset) % len(self.road_order)
                exit_road = self.road_order[dest_idx]
                sentence = f"- Enters the intersection at t={t_entry:.2f}, {movement} from Road {entry_road}, Lane {entry_lane} towards {exit_road}."

            events.append((t_entry, sentence))
            events.append((t_exit, f"- Leaves the intersection at t={t_exit:.2f}."))

        # Track previous action state for reasoning
        previous_action_type = None
        previous_causes = []

        for i, action in enumerate(ego_actions):
            t_start = action["attributes"]["start_time"]
            act_type = action["type"]

            if act_type == "slow_down":

                # Decide action magnitude according to acceleration
                acc = action["attributes"].get("acceleration")
                adverb = self._acceleration_magnitude(acc)

                # Check if the agents slows down till stopped
                target_speed = action["attributes"].get("target_speed")
                stop_phrase = " till stopped" if target_speed < 0.5 else ""

                # Set the end time of the slow down action as the next time it starts speeding up
                t_end = t_start + action["attributes"]["duration"]
                for a in ego_actions[i+1:]:
                    if a["type"] == "speed_up":
                        t_end = a["attributes"]["start_time"]
                        break

                # 
                causes = []
                for aid in key_agents:
                    a_entry = self.agent_dict[aid]["entry_time"]
                    a_exit = self.agent_dict[aid]["exit_time"]
                    if t_end and not (a_exit < t_start or a_entry > t_end):
                        causes.append(aid)

                if intersection_end:
                    if t_start < intersection_end:
                        if causes:
                            listed = ", ".join(f"{self.agent_dict[c]['type']} {c}" for c in causes)
                            line = f"- {adverb} slows down at t={t_start:.2f}{stop_phrase} to look out for {listed}."
                        else:
                            line = f"- {adverb} slows down at t={t_start:.2f}{stop_phrase} for cautious driving."
                    else:
                        line = f"- {adverb} slows down at t={t_start:.2f}{stop_phrase}."
                else:
                    line = f"- {adverb} slows down at t={t_start:.2f}{stop_phrase}."

                # Append event then set previous action
                events.append((t_start, line))
                previous_action_type = "slow_down"
                previous_causes = causes

            elif act_type == "speed_up":
                acc = action["attributes"].get("acceleration", 0.0)
                adverb = self._acceleration_magnitude(acc)
                if previous_action_type == "slow_down" and previous_causes:
                    latest = max(previous_causes, key=lambda x: self.agent_dict[x]["exit_time"])
                    atype = self.agent_dict[latest]["type"]
                    line = f"- {adverb} speeds up at t={t_start:.2f} since {atype} {latest} has passed the intersection."
                else:
                    line = f"- {adverb} speeds up at t={t_start:.2f}."
                events.append((t_start, line))
                previous_action_type = "speed_up"

            elif act_type == "lane_change":
                direction = action["attributes"]["direction"]
                line = f"- Changes lane to the {direction}."
                events.append((t_start, line))

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

