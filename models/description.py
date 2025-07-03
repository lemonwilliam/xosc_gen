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

        if road in self.road_order:
            return f"{agent_type} {ego_id}:\n- Enters the scenario at t={t0:.2f}, starting from road {road}, lane {lane}."
        else:
            return f"{agent_type} {ego_id}:\n- Enters the scenario at t={t0:.2f}, starting inside the intersection."


    def _timeline_description(self, ego_id, classification):
        """
        Generates a simplified, action-focused timeline of events for the ego agent.
        Intentions and reasons for actions are omitted.
        """
        ego = self._get_agent_by_id(ego_id)
        events = []
        ego_actions = ego.get("actions", [])
        route_action = self._get_route_action(ego)

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

        # Process other actions like slow_down, speed_up, lane_change
        for action in ego_actions:
            t_start = action["attributes"]["start_time"]
            act_type = action["type"]

            if act_type == "slow_down":
                # Get action magnitude and check if it's a full stop
                acc = action["attributes"].get("acceleration")
                adverb = self._acceleration_magnitude(acc)
                target_speed = action["attributes"].get("target_speed")
                stop_phrase = " till stopped" if target_speed < 0.5 else ""

                # Create a simple description of the action without the reason
                line = f"- {adverb} slows down at t={t_start:.2f}{stop_phrase}."
                events.append((t_start, line))

            elif act_type == "speed_up":
                acc = action["attributes"].get("acceleration", 0.0)
                adverb = self._acceleration_magnitude(acc)

                # Create a simple description of the action without the reason
                line = f"- {adverb} speeds up at t={t_start:.2f}."
                events.append((t_start, line))

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

