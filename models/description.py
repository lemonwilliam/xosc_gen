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

    def _initial_description(self, agent_id):
        agent = self._get_agent_by_id(agent_id)
        t0 = agent["enter_simulation_time"]
        road, lane = agent["initial_position"][:2]
        agent_type = self.agent_dict[agent_id]["type"]

        initial_speed_ms = agent.get("initial_speed", 0.0)
        initial_speed_kmh = self._ms_to_kmh(initial_speed_ms)
        speed_phrase = f"{initial_speed_kmh:.1f} km/h"

        # Determine if the agent is heading towards or away from the intersection
        heading_phrase = ""
        if road in self.road_order:
            # If agent starts on a road and has a maneuver action, it must be heading towards the intersection
            if self._get_route_action(agent):
                heading_phrase = ", heading towards the intersection"
            else:
                heading_phrase = ", heading away from the intersection"
        else:
            heading_phrase = ""

        # Generate the header (this part is from the previous change)
        if hasattr(self, 'main_ego_id') and agent_id == self.main_ego_id:
            header = f"Ego ({agent_type.lower()} {agent_id}):"
        else:
            header = f"{agent_type.lower()} {agent_id}:"
        
        # 2. Reformat the sentence with the timestamp at the beginning
        if road in self.road_order:
            description_line = f"t={t0:.2f}: Enters the scenario from road {road}, lane {lane} at {speed_phrase}{heading_phrase}."
        else:
            description_line = f"t={t0:.2f}: Enters the scenario inside the intersection at {speed_phrase}{heading_phrase}."

        return f"{header}\n{description_line}"

    def _timeline_description(self, agent_id, classification):

        agent = self._get_agent_by_id(agent_id)
        events = []
        agent_actions = agent.get("actions", [])
        route_action = self._get_route_action(agent)

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

            # Intersection entry sentence
            if exit_road in self.road_order:
                if entry_road in self.road_order:
                    sentence = f"t={t_entry:.2f}: Enters the intersection, {movement} from Road {entry_road}, Lane {entry_lane} towards Road {exit_road}."
                else:
                    sentence = f"t={t_entry:.2f}: {movement.capitalize()} towards Road {exit_road}."
            else:
                if entry_road in self.road_order:
                    entry_idx = self.road_order.index(entry_road)
                    offset = {"go straight": 2, "turn right": 1, "turn left": 3}.get(movement, 0)
                    dest_idx = (entry_idx + offset) % len(self.road_order)
                    estimated_exit_road = self.road_order[dest_idx]
                    sentence = f"t={t_entry:.2f}: Enters the intersection, {movement} from Road {entry_road}, Lane {entry_lane} towards Road {estimated_exit_road}."
                else:
                    sentence = f"t={t_entry:.2f}: {movement.capitalize()} inside the intersection."

            events.append((t_entry, sentence))

            # Intersection exit sentence
            if exit_road in self.road_order:
                events.append((t_exit, f"t={t_exit:.2f}: Leaves the intersection from Road {exit_road}, Lane {exit_lane}."))

        for action in agent_actions:
            t_start = action["attributes"]["start_time"]
            act_type = action["type"]

            if act_type == "slow_down" or act_type == "speed_up":
                acc = action["attributes"].get("acceleration", 0.0)
                adverb = self._acceleration_magnitude(acc)
                target_speed_ms = action["attributes"].get("target_speed")
                duration = action["attributes"].get("duration")

                # Build the time prefix (e.g., "t=1.23~4.56:")
                if duration is not None and duration > 0:
                    t_end = t_start + duration
                    time_prefix = f"t={t_start:.2f}~{t_end:.2f}:"
                else:
                    time_prefix = f"t={t_start:.2f}:"

                speed_before_ms = self._get_speed_at_time(agent_id, t_start)
                verb = "slows down" if act_type == "slow_down" else "speeds up"

                if speed_before_ms is None or target_speed_ms is None:
                    line = f"{time_prefix} {adverb} {verb}."
                else:
                    speed_before_kmh = self._ms_to_kmh(speed_before_ms)
                    speed_after_kmh = self._ms_to_kmh(target_speed_ms)
                    stop_phrase = " till stopped" if target_speed_ms < 0.5 else ""
                    line = (f"{time_prefix} {adverb} {verb} from {speed_before_kmh:.1f} km/h "
                            f"to {speed_after_kmh:.1f} km/h{stop_phrase}.")
                events.append((t_start, line))

            elif act_type == "lane_change":
                direction = action["attributes"]["direction"]
                line = f"t={t_start:.2f}: Changes lane to the {direction}."
                events.append((t_start, line))

        exit_time = agent["exit_simulation_time"]
        events.append((exit_time, f"t={exit_time:.2f}: Exits the scenario."))
        sorted_lines = [line for _, line in sorted(events)]
        return sorted_lines
    
    def _generate_single_agent_description(self, agent_id):
        """Generates the full text description for a single agent."""

        classification = self._classify_agents(agent_id)
        self.agent_classifications[agent_id] = classification
        
        initial_lines = self._initial_description(agent_id).split('\n')
        
        lines = initial_lines
        lines += self._timeline_description(agent_id, classification)
        return "\n".join(filter(None, lines))
    
    def generate_description(self, ego_id):
        """
        Generates a complete report for all relevant agents in the scenario,
        with the specified ego agent listed first and given a special header.
        """
        # Store the main ego ID so other methods can identify it
        self.main_ego_id = ego_id
        all_descriptions = []
        
        # Identify all agents that need a description.
        relevant_agent_ids = []
        for agent in self.agents:
            if agent["type"] == "pedestrian":
                continue
            if agent["initial_speed"] == 0.0 and agent["actions"] == []:
                continue
            relevant_agent_ids.append(agent["track_id"])

        # Generate the description for the ego agent first.
        if ego_id in relevant_agent_ids:
            ego_desc = self._generate_single_agent_description(ego_id)
            all_descriptions.append(ego_desc)
        else:
            print(f"Warning: Designated ego_id {ego_id} is not a relevant agent. It will not be described.")
            
        # Generate descriptions for all other relevant agents.
        for agent_id in relevant_agent_ids:
            if agent_id == ego_id:
                continue      
            other_desc = self._generate_single_agent_description(agent_id)
            all_descriptions.append(other_desc)
            
        # Join all descriptions into a single string and return, clean up the stored ego ID as a good practice
        del self.main_ego_id
        return "\n\n".join(all_descriptions)
    