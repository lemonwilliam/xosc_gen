import math
import os
import argparse
import xml.etree.ElementTree as ET
import yaml
import json
import itertools
import numpy as np
import csv

class NoTupleDumper(yaml.SafeDumper):
    def represent_tuple(self, data):
        return self.represent_list(data)

# --- Core XODR Parsing and Data Extraction Functions ---
def compute_junction_connections(junction_elem: ET.Element, roads_dict: dict[int, ET.Element]) -> list:
    """Extracts connection details from a <junction> element."""
    conns = []
    for conn_elem in junction_elem.findall("connection"):
        conn_road_id = int(conn_elem.get("connectingRoad"))
        entry_road_id = int(conn_elem.get("incomingRoad"))
        contact_point_val = conn_elem.get("contactPoint") # Get contactPoint

        lane_links_pairs = [] 
        for ll_elem in conn_elem.findall("laneLink"):
            lane_links_pairs.append(
                (int(ll_elem.get("from")), int(ll_elem.get("to")))
            )

        conn_road_elem = roads_dict.get(conn_road_id)
        exit_arterial_road_id = None
        if conn_road_elem:
            link_elem = conn_road_elem.find("link")
            if link_elem:
                pred_elem = link_elem.find("predecessor")
                succ_elem = link_elem.find("successor")
                # This logic for exit_arterial_road_id was already correct based on contact_point
                if contact_point_val == "start": 
                    if succ_elem is not None and succ_elem.get("elementType") == "road":
                        exit_arterial_road_id = int(succ_elem.get("elementId"))
                elif contact_point_val == "end": 
                    if pred_elem is not None and pred_elem.get("elementType") == "road":
                        exit_arterial_road_id = int(pred_elem.get("elementId"))
        
        conns.append({
            "Connection_road": conn_road_id,
            "Entry_road": entry_road_id,
            "Lane_links": lane_links_pairs,
            "Exit_road": exit_arterial_road_id,
            "Contact_point_on_connecting_road": contact_point_val, # Store contactPoint
            # Keep these for compatibility if other parts use them, though extract_maneuvers will prioritize Lane_links
            "Entry_lanes": [p[0] for p in lane_links_pairs],
            "Connecting_road_lanes": [p[1] for p in lane_links_pairs]
        })
    return conns

def get_all_junctions_data(root_elem: ET.Element, roads_dict: dict[int, ET.Element]) -> dict:
    """Parses all junctions in the XODR."""
    junctions_map_data = {}
    for junction_elem in root_elem.findall("junction"):
        jid = junction_elem.get("id")
        connections = compute_junction_connections(junction_elem, roads_dict)
        junctions_map_data[f"Junction {jid}"] = {"connections": connections}
    return junctions_map_data

def normalize_angle(angle: float) -> float:
    """Normalize an angle (in radians) to [0, 2π)."""
    return angle % (2.0 * math.pi)

def extract_road_heading(road_elem: ET.Element, contact_type: str) -> float:
    """Extracts road heading in radians as it leaves the junction."""
    plan_view = road_elem.find("planView")
    if not plan_view: raise ValueError(f"Road ID={road_elem.get('id')} has no <planView>.")
    geometry = plan_view.find("geometry")
    if not geometry: raise ValueError(f"Road ID={road_elem.get('id')} has no <geometry> in <planView>.")
    hdg = float(geometry.get("hdg", 0.0))
    return normalize_angle(hdg + math.pi if contact_type == "successor" else hdg)

def collect_arterial_roads(root_elem: ET.Element, target_junction_id: str) -> dict[str, str]:
    """Identifies 4 arterial roads connected to the target junction."""
    connected_roads = {}
    for road_elem in root_elem.findall("road"):
        road_id_str = road_elem.get("id")
        link = road_elem.find("link")
        if not link: continue
        pred = link.find("predecessor")
        if pred is not None and pred.get("elementType") == "junction" and pred.get("elementId") == target_junction_id:
            connected_roads[road_id_str] = "predecessor"
            continue
        succ = link.find("successor")
        if succ is not None and succ.get("elementType") == "junction" and succ.get("elementId") == target_junction_id:
            connected_roads[road_id_str] = "successor"
    if len(connected_roads) != 4:
        raise RuntimeError(f"Junction {target_junction_id} expects 4 connected roads, found {len(connected_roads)}.")
    return connected_roads

def get_ordered_roads_and_ccw_indices(
    arterial_roads_by_type: dict[str, str], 
    roads_dict: dict[int, ET.Element]
) -> tuple[list[int], dict[int, int]]:
    """Orders arterial roads CCW starting from the one closest to 270 degrees."""
    angle_list = []
    for road_id_str, contact_type in arterial_roads_by_type.items():
        road_elem = roads_dict.get(int(road_id_str))
        if not road_elem: raise RuntimeError(f"Road ID {road_id_str} not found in roads_dict.")
        angle = extract_road_heading(road_elem, contact_type)
        angle_list.append((road_id_str, angle))
    
    angle_list.sort(key=lambda item: item[1])
    target_angle = 1.5 * math.pi
    diffs = [abs(normalize_angle(ang - target_angle + math.pi) - math.pi) for _, ang in angle_list]
    
    if not angle_list: return [], {}
    start_idx = min(range(len(angle_list)), key=lambda i: diffs[i])
    
    rotated_angle_list = angle_list[start_idx:] + angle_list[:start_idx]
    ordered_ids = [int(r_id_str) for r_id_str, _ in rotated_angle_list]
    id_to_ccw_idx = {int(r_id_str): i for i, (r_id_str, _) in enumerate(rotated_angle_list)}
    return ordered_ids, id_to_ccw_idx

def get_road_lane_details(road_elem: ET.Element, contact_type: str) -> dict:
    """Extracts detailed lane ID lists for a given arterial road."""
    details = {'driving_towards_ids': set(), 'driving_away_ids': set(),
               'biking_towards_ids': set(), 'biking_away_ids': set(),
               'bidirectional_ids': set()}
    if not road_elem or not contact_type: # Should not happen with prior checks
        for key in details: details[key] = []
        details['all_towards_lanes_ids'] = []
        return details

    for ls_elem in road_elem.findall(".//laneSection"):
        for side_elem in [ls_elem.find("left"), ls_elem.find("right")]:
            if not side_elem: continue
            for lane_elem in side_elem.findall("lane"):
                lane_id, lane_type = int(lane_elem.get("id")), lane_elem.get("type")
                if lane_type not in ["driving", "biking", "bidirectional"]: continue
                if lane_type == "bidirectional":
                    details['bidirectional_ids'].add(lane_id)
                    continue
                
                vl_elem = lane_elem.find(".//userData/vectorLane")
                travel_dir = vl_elem.get("travelDir") if vl_elem else None
                is_eff_forward = (travel_dir == "forward") or (travel_dir is None and lane_id < 0)
                is_towards_junc = (contact_type == "predecessor" and not is_eff_forward) or \
                                  (contact_type == "successor" and is_eff_forward)
                
                category_key = f"{lane_type}_{'towards' if is_towards_junc else 'away'}_ids"
                details[category_key].add(lane_id)

    for key in details: details[key] = sorted(list(details[key]))
    details['all_towards_lanes_ids'] = sorted(list(set(details['driving_towards_ids']) | set(details['biking_towards_ids'])))
    return details

def get_turn_type(entry_road_id: int, exit_road_id: int, road_id_to_ccw_index: dict[int,int]) -> str | None:
    """Determines turn type (straight, left, right, U-turn)."""
    if entry_road_id not in road_id_to_ccw_index or exit_road_id not in road_id_to_ccw_index:
        return None
    idx_entry, idx_exit = road_id_to_ccw_index[entry_road_id], road_id_to_ccw_index[exit_road_id]
    diff = (idx_exit - idx_entry + 4) % 4
    if entry_road_id == exit_road_id: return "U-turn"
    if diff == 1: return "right"
    if diff == 2: return "straight"
    if diff == 3: return "left"
    return None

def extract_maneuvers(
    target_junction_connections: list, 
    road_id_to_ccw_index: dict[int, int],
    road_lane_details_map: dict[int, dict], 
    roads_dict: dict[int, ET.Element], 
    arterial_road_ids: list[int]
) -> list:
    """Extracts all valid maneuvers through the junction, correctly determining exit lanes."""
    maneuvers = []
    for entry_road_id in arterial_road_ids:
        road_details = road_lane_details_map.get(entry_road_id, {})
        lanes_towards_junc = road_details.get('all_towards_lanes_ids', [])
        
        for entry_lane_id in lanes_towards_junc:
            for conn_info in target_junction_connections:
                if conn_info["Entry_road"] == entry_road_id:
                    arterial_exit_road_id = conn_info["Exit_road"]
                    if arterial_exit_road_id is None or arterial_exit_road_id not in arterial_road_ids:
                        continue 
                    
                    turn_type_str = get_turn_type(entry_road_id, arterial_exit_road_id, road_id_to_ccw_index)
                    if not turn_type_str: continue

                    lane_on_conn_road = None
                    if "Lane_links" in conn_info:
                        for from_lane, to_lane in conn_info["Lane_links"]:
                            if from_lane == entry_lane_id:
                                lane_on_conn_road = to_lane
                                break
                    
                    if lane_on_conn_road is None:
                        # This specific entry_lane_id is not part of this conn_info's path
                        continue 

                    final_lane_on_arterial_exit = None # Initialize
                    contact_point = conn_info.get("Contact_point_on_connecting_road")
                    conn_road_elem = roads_dict.get(conn_info["Connection_road"])

                    if conn_road_elem and contact_point:
                        found_final_link = False
                        for ls_cr in conn_road_elem.findall('.//laneSection'):
                            for side_cr in [ls_cr.find('left'), ls_cr.find('right')]:
                                if not side_cr: continue
                                for lane_cr_elem in side_cr.findall('lane'):
                                    if int(lane_cr_elem.get('id')) == lane_on_conn_road:
                                        link_cr_elem = lane_cr_elem.find('link')
                                        if link_cr_elem:
                                            # Determine which link (predecessor or successor) of the connecting road's lane
                                            # leads to the arterial exit road.
                                            if contact_point == "start": # Traffic flows towards successor of connecting road's lane
                                                link_target_elem = link_cr_elem.find('successor')
                                            elif contact_point == "end": # Traffic flows towards predecessor of connecting road's lane
                                                link_target_elem = link_cr_elem.find('predecessor')
                                            else: # Should not happen if contactPoint is always start/end
                                                link_target_elem = None
                                            
                                            if link_target_elem is not None and link_target_elem.get('id') is not None:
                                                final_lane_on_arterial_exit = int(link_target_elem.get('id'))
                                                found_final_link = True
                                                break 
                                if found_final_link: break
                            if found_final_link: break
                    
                    if final_lane_on_arterial_exit is None:
                        # If logic failed to find it, default or skip. For now, let's use a placeholder.
                        # print(f"Warning: Could not trace final exit lane for maneuver: {entry_road_id}:{entry_lane_id} -> CR{conn_info['Connection_road']}:{lane_on_conn_road}")
                        final_lane_on_arterial_exit = -999 # Indicates an issue in tracing

                    maneuvers.append({
                        "entry_road": entry_road_id,
                        "entry_lane": entry_lane_id,
                        "turn_type": turn_type_str,
                        "exit_road": arterial_exit_road_id,
                        "exit_lane": final_lane_on_arterial_exit, 
                        "connecting_road_id": conn_info["Connection_road"],
                        "lane_on_connecting_road": lane_on_conn_road
                    })
    return maneuvers


# --- Geometry Helper Functions (Simplified for brevity, assuming previous detailed versions) ---

def on_segment(p, q, r):
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and \
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def do_segments_intersect(p1, q1, p2, q2):
    o1, o2 = orientation(p1, q1, p2), orientation(p1, q1, q2)
    o3, o4 = orientation(p2, q2, p1), orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True
    return False

# --- New Function to Load Lane Paths from Visualizer's CSV ---
_parsed_csv_lane_data = {} # Cache for CSV data: {(road_id_str, lane_id_str): [(x,y), ...]}

def load_lane_paths_from_csv(csv_file_path: str):
    """
    Parses the visualizer's CSV file and stores lane centerline points.
    This should be called once.
    """
    global _parsed_csv_lane_data
    if _parsed_csv_lane_data: # Already loaded
        return

    try:
        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f, skipinitialspace=True)
            current_lane_key = None
            for row in reader:
                if not row: continue
                if row[0] == 'lane':
                    road_id_str = row[1]
                    # lane_section_idx = row[2] # Not strictly needed for path lookup by road/lane ID
                    lane_id_str_csv = row[3]
                    # lane_type_csv = row[4] if len(row) > 4 else "driving" # visualizer's type
                    
                    # We are interested in paths of connecting roads' lanes
                    current_lane_key = (road_id_str, lane_id_str_csv)
                    if current_lane_key not in _parsed_csv_lane_data:
                        _parsed_csv_lane_data[current_lane_key] = []
                elif current_lane_key and len(row) >= 2:
                    try:
                        x, y = float(row[0]), float(row[1])
                        _parsed_csv_lane_data[current_lane_key].append((x, y))
                    except ValueError:
                        # print(f"Warning: Skipping malformed coordinate in CSV: {row}")
                        pass
    except FileNotFoundError:
        print(f"Error: Visualizer CSV file not found at {csv_file_path}. Crossing detection will be limited.")
    except Exception as e:
        print(f"Error reading visualizer CSV {csv_file_path}: {e}")

def get_lane_segments_from_csv_data(road_id: int, lane_id_on_road: int) -> list[tuple[tuple[float,float], tuple[float,float]]]:
    """
    Retrieves lane path segments from the pre-parsed CSV data.
    `lane_id_on_road` is the ID of the lane on the specified `road_id` (typically a connectingRoad).
    """
    global _parsed_csv_lane_data
    lane_key = (str(road_id), str(lane_id_on_road)) # CSV uses string IDs
    
    points = _parsed_csv_lane_data.get(lane_key, [])
    if len(points) < 2:
        # print(f"Warning: Not enough points in CSV for lane path: road {road_id}, lane {lane_id_on_road}")
        return []
    
    segments = []
    for i in range(len(points) - 1):
        segments.append((points[i], points[i+1]))
    return segments


# --- generate_text_description (Assumed to be the refined version from previous response) ---
def generate_text_description(
    ordered_arterial_road_ids: list[int],
    road_lane_details_map: dict[int, dict],
    valid_maneuvers: list[dict]
) -> str:
    text_parts = ["Interpretation Rules for the Map Diagram:",
                  "- Vehicles heading towards the intersection drive on the right side of the road.", 
                  "- Colored zones represent drivable lanes of the surrounding roads of the intersection.",
                  "- Red boxed numbers represent Road IDs.", 
                  "- Black boxed numbers represent lane ids.",
                  "- The blue curved lines in the intersection represent all possible legal routes through the intersection.",
                  "- For a given maneuver (e.g., 'from Road X to Road Y'), assume the vehicle follows the most direct and conventional blue curved path.",
                  "- When asked if paths cross, carefully trace the two described maneuvers using these blue lines and determine if they overlap in the central intersection area.",
                  "\nRoads:"]

    for r_id in ordered_arterial_road_ids:
        details = road_lane_details_map.get(r_id, {})
        desc_clauses = []
        
        dir_lanes = []
        if details.get('driving_towards_ids'): dir_lanes.append(f"{len(details['driving_towards_ids'])} driving lane(s) (id: {', '.join(map(str, details['driving_towards_ids']))})")
        if details.get('biking_towards_ids'): dir_lanes.append(f"{len(details['biking_towards_ids'])} biking lane(s) (id: {', '.join(map(str, details['biking_towards_ids']))})")
        if dir_lanes: desc_clauses.append(f"{', '.join(dir_lanes)} going towards the intersection")

        dir_lanes = []
        if details.get('driving_away_ids'): dir_lanes.append(f"{len(details['driving_away_ids'])} driving lane(s) (id: {', '.join(map(str, details['driving_away_ids']))})")
        if details.get('biking_away_ids'): dir_lanes.append(f"{len(details['biking_away_ids'])} biking lane(s) (id: {', '.join(map(str, details['biking_away_ids']))})")
        if dir_lanes: desc_clauses.append(f"{', '.join(dir_lanes)} leaving the intersection")

        if details.get('bidirectional_ids'): desc_clauses.append(f"{len(details['bidirectional_ids'])} bidirectional lane(s) (id: {', '.join(map(str, details['bidirectional_ids']))})")
        
        text_parts.append(f"- Road ID {r_id} has {', '.join(desc_clauses)}" if desc_clauses else f"- Road ID {r_id} has no relevant lanes.")

    text_parts.append("\nIntersection permissions:")
    perms_by_entry = {}
    for m in valid_maneuvers:
        key = (m["entry_road"], m["entry_lane"])
        if key not in perms_by_entry: perms_by_entry[key] = []
        perms_by_entry[key].append(f"{m['turn_type']} to Road ID {m['exit_road']} lane {m['exit_lane']}")
    
    for (entry_r, entry_l), perm_list in sorted(perms_by_entry.items()):
        perm_list.sort()
        text_parts.append(f"- Road ID {entry_r} lane {entry_l} can {', or '.join(perm_list)}")
        
    return "\n".join(text_parts)


# --- generate_questions_json (Modified for CSV-based crossing) ---
def generate_questions_json(
    ordered_arterial_road_ids: list[int],
    road_lane_details_map: dict[int, dict],
    valid_maneuvers: list[dict], # This now contains detailed maneuver info
    roads_dict: dict[int, ET.Element] # For checking sidewalk type of connecting roads
) -> dict:
    questions = {"Basic structure": [], "Road relations": [], "Crossing routes": []}

    # Basic structure
    questions["Basic structure"].append({"question": "How many branching roads does the intersection have?", "answer": str(len(ordered_arterial_road_ids))})
    for r_id in ordered_arterial_road_ids:
        details = road_lane_details_map.get(r_id, {})
        questions["Basic structure"].append({"question": f"How many driving lanes does Road {r_id} have going towards the intersection?", "answer": str(len(details.get('driving_towards_ids',[])))})
        questions["Basic structure"].append({"question": f"How many driving lanes does Road {r_id} have going away from the intersection?", "answer": str(len(details.get('driving_away_ids',[])))})

    # Road relations
    for r_id in ordered_arterial_road_ids:
        # Go straight
        s_man = next((m for m in valid_maneuvers if m["entry_road"] == r_id and m["turn_type"] == "straight"), None)
        ans = str(s_man["exit_road"]) if s_man else "This maneuver is not directly possible."
        questions["Road relations"].append({"question": f"If a vehicle enters the intersection from Road {r_id} then goes straight, what is the ID of the Road it arrives?", "answer": ans}) # Corrected typo "intesection"
        
        # Turn Left
        l_man = next((m for m in valid_maneuvers if m["entry_road"] == r_id and m["turn_type"] == "left"), None)
        l_ans = str(l_man["exit_road"]) if l_man else "This maneuver is not directly possible."
        questions["Road relations"].append({
            "question": f"If a vehicle enters the intersection from Road {r_id} then turns left, what is the ID of the Road it arrives?",
            "answer": l_ans
        })

        # Turn Right
        r_man = next((m for m in valid_maneuvers if m["entry_road"] == r_id and m["turn_type"] == "right"), None)
        r_ans = str(r_man["exit_road"]) if r_man else "This maneuver is not directly possible."
        questions["Road relations"].append({
            "question": f"If a vehicle enters the intersection from Road {r_id} then turns right, what is the ID of the Road it arrives?",
            "answer": r_ans
        })

    if valid_maneuvers:
        # Pick an example maneuver for lane-specific permission questions
        example_maneuver_for_lane_q = None
        for m_ex in valid_maneuvers: # Try to find a maneuver that allows a right turn for a more interesting "Yes"
            if any(vm for vm in valid_maneuvers if vm["entry_road"] == m_ex["entry_road"] and vm["entry_lane"] == m_ex["entry_lane"] and vm["turn_type"] == "right"):
                example_maneuver_for_lane_q = m_ex
                break
        if not example_maneuver_for_lane_q: # Fallback to the first maneuver if none found
            example_maneuver_for_lane_q = valid_maneuvers[0]

        er, el = example_maneuver_for_lane_q["entry_road"], example_maneuver_for_lane_q["entry_lane"]
        
        has_r_turn_lane_specific = any(m for m in valid_maneuvers if m["entry_road"]==er and m["entry_lane"]==el and m["turn_type"]=="right")
        questions["Road relations"].append({"question": f"If a vehicle enters from Road {er} lane {el}, is it allowed to turn right? Answer with Yes / No.", "answer": "Yes" if has_r_turn_lane_specific else "No"})
        
        has_l_turn_lane_specific = any(m for m in valid_maneuvers if m["entry_road"]==er and m["entry_lane"]==el and m["turn_type"]=="left")
        questions["Road relations"].append({"question": f"If a vehicle enters from Road {er} lane {el}, is it allowed to turn left? Answer with Yes / No.", "answer": "Yes" if has_l_turn_lane_specific else "No"})

    
    # Crossing routes (CSV-based)
    non_sidewalk_maneuvers = []
    for m in valid_maneuvers:
        conn_road_elem = roads_dict.get(m["connecting_road_id"]) 
        if conn_road_elem:
            is_sidewalk_path = any(l.get('type') == 'sidewalk' for l in conn_road_elem.findall('.//lane'))
            if not is_sidewalk_path:
                non_sidewalk_maneuvers.append(m)
        else:
            non_sidewalk_maneuvers.append(m) 

    # Remove the debug print for m in non_sidewalk_maneuvers:
    # for m in non_sidewalk_maneuvers:
    #     print(m)

    generated_q_keys = set()
    for m1, m2 in itertools.combinations(non_sidewalk_maneuvers, 2):
        if m1["entry_road"] == m2["entry_road"] or \
           m1["exit_road"] == m2["exit_road"] or \
           m1["connecting_road_id"] == m2["connecting_road_id"]:
            continue

        # --- MODIFIED QUESTION TEXT LOGIC ---
        def get_turn_action_phrase(turn_type_str):
            if turn_type_str == 'left': return "turns left"
            if turn_type_str == 'right': return "turns right"
            if turn_type_str == 'straight': return "goes straight"
            if turn_type_str == 'U-turn': return "makes a U-turn"
            return f"moves ({turn_type_str})" # Fallback

        turn_action_m1 = get_turn_action_phrase(m1['turn_type'])
        turn_action_m2 = get_turn_action_phrase(m2['turn_type'])

        q_text = (f"Consider the blue curved lines as potential paths. "
                  f"If one vehicle {turn_action_m1} from Road {m1['entry_road']} to exit onto Road {m1['exit_road']}, "
                  f"and another vehicle {turn_action_m2} from Road {m2['entry_road']} to exit onto Road {m2['exit_road']}, "
                  f"will their primary paths (blue lines) through the intersection cross?  Answer with Yes / No.")
        # --- END OF MODIFIED QUESTION TEXT LOGIC ---
        
        # Canonical key generation now includes turn types to differentiate questions
        # for the same road pair but different maneuvers.
        route1_key_elements = (str(m1['entry_road']), str(m1['exit_road']), m1['turn_type'])
        route2_key_elements = (str(m2['entry_road']), str(m2['exit_road']), m2['turn_type'])
        # Sort individual route elements and then sort the pair of routes
        canonical_q_key = tuple(sorted((route1_key_elements, route2_key_elements)))


        if canonical_q_key in generated_q_keys: continue
        generated_q_keys.add(canonical_q_key)

        path1_segments = get_lane_segments_from_csv_data(m1["connecting_road_id"], m1["lane_on_connecting_road"])
        path2_segments = get_lane_segments_from_csv_data(m2["connecting_road_id"], m2["lane_on_connecting_road"])

        if not path1_segments or not path2_segments:
            questions["Crossing routes"].append({"question": q_text, "answer": "No"}) 
            continue

        intersects = any(do_segments_intersect(s1p1, s1q1, s2p2, s2q2)
                         for s1p1, s1q1 in path1_segments for s2p2, s2q2 in path2_segments)
        questions["Crossing routes"].append({"question": q_text, "answer": "Yes" if intersects else "No"})
        
    return questions

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="XODR Map Preprocessor")
    parser.add_argument("--dataset", "-d", default="inD", help="Dataset name")
    parser.add_argument("--map_id", "-m", default="01_bendplatz", help="Map ID")
    parser.add_argument("--junction", "-j", default=None, help="Target Junction ID")
    parser.add_argument("--viz_csv", "-v", help="Path to the visualizer's preprocessed CSV track file for crossing detection.") # New argument
    args = parser.parse_args()

    base_dir = "./data"
    xodr_path = os.path.join(base_dir, "raw", args.dataset, "maps", "opendrive", f"{args.map_id}.xodr")
    processed_dir = os.path.join(base_dir, "processed", args.dataset, "map")
    os.makedirs(processed_dir, exist_ok=True)
    
    yaml_out_path = os.path.join(processed_dir, f"{args.map_id}.yaml")
    txt_out_path = os.path.join(processed_dir, f"{args.map_id}_description.txt")
    json_q_path = os.path.join(processed_dir, f"{args.map_id}_questions.json")

    if not os.path.exists(xodr_path):
        print(f"Error: XODR file not found at {xodr_path}"); return
        
    tree = ET.parse(xodr_path)
    root_elem = tree.getroot()
    roads_dict = {int(r.get("id")): r for r in root_elem.findall("road")}

    target_junction_id_str = args.junction
    if not target_junction_id_str:
        first_junction = root_elem.find(".//junction")
        if not first_junction: raise ValueError("No <junction> in XODR."); 
        target_junction_id_str = first_junction.get("id")
    
    target_junction_elem = root_elem.find(f".//junction[@id='{target_junction_id_str}']")
    if not target_junction_elem: raise ValueError(f"Junction {target_junction_id_str} not found.")

    # Load CSV data if path provided
    if args.viz_csv:
        if os.path.exists(args.viz_csv):
            print(f"Loading lane paths from visualizer CSV: {args.viz_csv}")
            load_lane_paths_from_csv(args.viz_csv)
        else:
            print(f"Warning: Visualizer CSV path provided but file not found: {args.viz_csv}. Crossing detection will be limited.")
    else:
        print("Warning: No visualizer CSV path provided (--viz_csv). Crossing detection accuracy might be limited or based on heuristics if CSV data is unavailable.")


    all_junctions_map_data = get_all_junctions_data(root_elem, roads_dict) # Ensure this is the refined function
    arterial_roads_info = collect_arterial_roads(root_elem, target_junction_id_str)
    
    ordered_arterial_ids, road_id_to_ccw_idx = get_ordered_roads_and_ccw_indices(arterial_roads_info, roads_dict)
    if not ordered_arterial_ids:
        print("Warning: Could not order arterial roads."); return

    road_lane_details_map = {
        r_id: get_road_lane_details(roads_dict[r_id], arterial_roads_info[str(r_id)]) # Ensure refined get_road_lane_details
        for r_id in ordered_arterial_ids
    }
    
    target_junc_conns = all_junctions_map_data.get(f"Junction {target_junction_id_str}", {}).get("connections", [])
    valid_maneuvers = extract_maneuvers(target_junc_conns, road_id_to_ccw_idx, # Ensure refined extract_maneuvers
                                        road_lane_details_map, roads_dict, ordered_arterial_ids)

    # Outputs
    simplified_junctions_data_for_yaml = {}
    for junction_key, junction_content in all_junctions_map_data.items():
        simplified_connections = []
        for m in valid_maneuvers:
            simplified_connections.append({
                "Connection_road": m["connecting_road_id"],
                "Entry_road": m["entry_road"],
                "Entry_lane": m["entry_lane"],
                "Exit_road": m["exit_road"],
                "Exit_lane": m["exit_lane"] # Using the traced final exit lane
            })
        simplified_junctions_data_for_yaml[junction_key] = {"connections": simplified_connections}
    output_yaml_data = { # Renamed from output_dict_yaml to avoid confusion
        "Roads": ordered_arterial_ids,
        "Junctions": simplified_junctions_data_for_yaml # Use the new simplified data
    }

    with open(yaml_out_path, "w") as f:
        yaml.dump(output_yaml_data, f, Dumper=NoTupleDumper, sort_keys=False, default_flow_style=False)
    print(f"YAML output generated: {yaml_out_path}")

    text_desc = generate_text_description(ordered_arterial_ids, road_lane_details_map, valid_maneuvers)
    with open(txt_out_path, "w") as f: f.write(text_desc)
    print(f"Text description: {txt_out_path}")

    questions_data = generate_questions_json(ordered_arterial_ids, road_lane_details_map,
                                             valid_maneuvers, roads_dict)
    with open(json_q_path, "w") as f: json.dump(questions_data, f, indent=4)
    print(f"JSON questions: {json_q_path}")

if __name__ == "__main__":
    main()