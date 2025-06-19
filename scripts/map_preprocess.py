import math
import os
import argparse
import xml.etree.ElementTree as ET
import yaml


def compute_junction_connections(junction: ET.Element, roads: dict) -> list:
    """
    Given a <junction> element and a dict of <road> elements by id,
    return a list of connection dicts:
      - Connection_road: (int) the ID of the connectingRoad
      - Entry_road: (int) the incomingRoad ID from the connection
      - Entry_lanes: list[int] of laneLink “from” IDs
      - Exit_road: (int | None) the ID of the road on the far side of connectingRoad, if it's a road. Can be None.
      - Exit_lanes: list[int] of laneLink “to” IDs
    """
    conns = []
    for conn_elem in junction.findall("connection"):
        conn_road_id = int(conn_elem.get("connectingRoad"))
        entry_road = int(conn_elem.get("incomingRoad"))

        entry_lanes = []
        exit_lanes = []
        for ll in conn_elem.findall("laneLink"):
            entry_lanes.append(int(ll.get("from")))
            exit_lanes.append(int(ll.get("to")))

        road_elem = roads.get(conn_road_id) # Get <road> element for connectingRoad
        exit_road_id = None

        if road_elem is not None:
            link_elem = road_elem.find("link")
            if link_elem is not None:
                pred_elem = link_elem.find("predecessor")
                succ_elem = link_elem.find("successor")

                pred_road_id_val = None
                if pred_elem is not None and pred_elem.get("elementType") == "road":
                    pred_road_id_val = int(pred_elem.get("elementId"))

                succ_road_id_val = None
                if succ_elem is not None and succ_elem.get("elementType") == "road":
                    succ_road_id_val = int(succ_elem.get("elementId"))

                contact_point = conn_elem.get("contactPoint")

                if contact_point == "start":
                    exit_road_id = succ_road_id_val
                elif contact_point == "end":
                    exit_road_id = pred_road_id_val
                # If contactPoint is missing or invalid, exit_road_id remains None, which is handled later.

        conns.append({
            "Connection_road": conn_road_id,
            "Entry_road": entry_road,
            "Entry_lanes": entry_lanes,
            "Exit_road": exit_road_id,
            "Exit_lanes": exit_lanes
        })
    return conns


def get_all_junctions(xodr_file: str) -> dict:
    tree = ET.parse(xodr_file)
    root = tree.getroot()
    roads = {int(r.get("id")): r for r in root.findall("road")}
    junctions_data = {}
    for junction_elem in root.findall("junction"):
        jid = junction_elem.get("id")
        conns = compute_junction_connections(junction_elem, roads)
        junctions_data[f"Junction {jid}"] = {"connections": conns}
    return junctions_data


def normalize_angle(angle: float) -> float:
    two_pi = 2.0 * math.pi
    return angle % two_pi


def extract_road_heading(road_elem: ET.Element, contact_type: str) -> float:
    plan_view = road_elem.find("planView")
    if plan_view is None:
        raise ValueError(f"Road ID={road_elem.get('id')} has no <planView>.")
    geom = plan_view.find("geometry")
    if geom is None:
        raise ValueError(f"Road ID={road_elem.get('id')} missing <geometry> under <planView>.")
    hdg = float(geom.get("hdg", "0.0"))
    raw = hdg + math.pi if contact_type == "successor" else hdg
    return normalize_angle(raw)


def collect_connected_roads(tree: ET.ElementTree, junction: ET.Element) -> dict[str, str]:
    """
    Find the four <road> IDs whose <link> references this junction.
    Returns a dict mapping road_id (str) -> contact_type ("predecessor" or "successor").
    """
    root = tree.getroot()
    jid = junction.get("id")
    connected = {}
    for road in root.findall("road"):
        rid = road.get("id")
        link = road.find("link")
        if link is None: continue
        pred = link.find("predecessor")
        if pred is not None and pred.get("elementType") == "junction" and pred.get("elementId") == jid:
            connected[rid] = "predecessor"
            continue
        succ = link.find("successor")
        if succ is not None and succ.get("elementType") == "junction" and succ.get("elementId") == jid:
            connected[rid] = "successor"
    if len(connected) != 4:
        raise RuntimeError(f"Expected 4 roads connected to junction ID={jid}, but found {len(connected)}.")
    return connected


def get_ordered_roads_and_ccw_indices(connected_roads_by_type: dict[str, str], tree: ET.ElementTree) -> tuple[list[int], dict[int, int]]:
    """
    Computes each road’s heading and orders them counter-clockwise (CCW)
    starting from the one closest to 3π/2 (270°).

    Args:
        connected_roads_by_type: Dict mapping road_id (str) -> contact_type ("predecessor" or "successor").
        tree: Parsed XML tree.

    Returns:
        A tuple containing:
        - ordered_road_ids_list: List of road IDs (int) in CCW order.
        - road_id_to_ccw_index: Dict mapping road ID (int) to its CCW index (0-3).
    """
    root = tree.getroot()
    angle_list = [] # Stores (road_id_str, angle)
    for rid_str, contact in connected_roads_by_type.items():
        road_elem = root.find(f".//road[@id='{rid_str}']")
        if road_elem is None:
            raise RuntimeError(f"Connected road ID={rid_str} not found in <road> elements.")
        angle = extract_road_heading(road_elem, contact)
        angle_list.append((rid_str, angle))

    # Sort by angle ascending (0..2π)
    angle_list.sort(key=lambda x: x[1])

    # Find which entry is “bottom” (closest to 3π/2)
    target_angle = 1.5 * math.pi  # 270° in radians
    # Calculate angular difference, handling wrap-around
    diffs = [abs(((ang - target_angle + math.pi) % (2 * math.pi)) - math.pi) for (_, ang) in angle_list]

    if not angle_list: # Should not happen if collect_connected_roads works
        return [], {}

    start_idx = min(range(len(angle_list)), key=lambda i: diffs[i])

    # Rotate so that the "bottom-most" road becomes index 0
    rotated_angle_list = angle_list[start_idx:] + angle_list[:start_idx]

    ordered_road_ids_list = [int(rid_str) for rid_str, _ in rotated_angle_list]
    road_id_to_ccw_index = {int(rid_str): i for i, (rid_str, _) in enumerate(rotated_angle_list)}

    return ordered_road_ids_list, road_id_to_ccw_index


def generate_text_description(ordered_road_ids: list[int],
                              road_id_to_ccw_index: dict[int, int],
                              all_junctions_data: dict, tree: ET.ElementTree,
                              target_junction_id_str: str,
                              road_contact_types: dict[str, str]) -> str:
    text_parts = []
    root = tree.getroot()

    text_parts.append("Roads:")
    # road_lane_details: road_id_int -> {'all_towards_lanes': [ids]} for permissions
    # Also stores detailed breakdowns for the road description string.
    road_lane_details = {}

    for road_id_int in ordered_road_ids:
        road_id_str = str(road_id_int)
        road_elem = root.find(f".//road[@id='{road_id_str}']")
        if road_elem is None:
            text_parts.append(f"- Road ID {road_id_int}: Error - Road element not found.")
            continue
        contact_type = road_contact_types.get(road_id_str)
        if contact_type is None:
            text_parts.append(f"- Road ID {road_id_int}: Error - Contact type not found.")
            continue

        driving_towards, driving_away = [], []
        biking_towards, biking_away = [], []
        bidirectional_lanes_list = []
        all_towards_lanes_for_permissions = []

        for lane_section in road_elem.findall(".//laneSection"):
            for side_elem in [lane_section.find("left"), lane_section.find("right")]:
                if side_elem is None: continue
                for lane_elem in side_elem.findall("lane"):
                    lane_id = int(lane_elem.get("id"))
                    lane_type_attr = lane_elem.get("type")

                    if lane_type_attr not in ["driving", "biking", "bidirectional"]:
                        continue

                    if lane_type_attr == "bidirectional":
                        bidirectional_lanes_list.append(lane_id)
                    else: # driving or biking
                        vec_lane_elem = lane_elem.find(".//userData/vectorLane")
                        travel_dir_attr = vec_lane_elem.get("travelDir") if vec_lane_elem is not None else None
                        
                        effective_forward_travel = False
                        if travel_dir_attr == "forward": effective_forward_travel = True
                        elif travel_dir_attr == "backward": effective_forward_travel = False
                        else: effective_forward_travel = (lane_id < 0)

                        is_towards_junction = False
                        if contact_type == "predecessor":
                            is_towards_junction = not effective_forward_travel
                        else: # successor
                            is_towards_junction = effective_forward_travel

                        if is_towards_junction:
                            all_towards_lanes_for_permissions.append(lane_id)
                            if lane_type_attr == "driving": driving_towards.append(lane_id)
                            elif lane_type_attr == "biking": biking_towards.append(lane_id)
                        else:
                            if lane_type_attr == "driving": driving_away.append(lane_id)
                            elif lane_type_attr == "biking": biking_away.append(lane_id)
        
        driving_towards = sorted(list(set(driving_towards)))
        driving_away = sorted(list(set(driving_away)))
        biking_towards = sorted(list(set(biking_towards)))
        biking_away = sorted(list(set(biking_away)))
        bidirectional_lanes_list = sorted(list(set(bidirectional_lanes_list)))
        all_towards_lanes_for_permissions = sorted(list(set(all_towards_lanes_for_permissions)))

        road_lane_details[road_id_int] = {
            'all_towards_lanes': all_towards_lanes_for_permissions
        }
        
        road_desc_parts = []
        towards_clauses = []
        if driving_towards: towards_clauses.append(f"{len(driving_towards)} driving lane(s) (id: {', '.join(map(str, driving_towards))})")
        if biking_towards: towards_clauses.append(f"{len(biking_towards)} biking lane(s) (id: {', '.join(map(str, biking_towards))})")
        if towards_clauses: road_desc_parts.append(f"{', '.join(towards_clauses)} going towards the intersection")

        away_clauses = []
        if driving_away: away_clauses.append(f"{len(driving_away)} driving lane(s) (id: {', '.join(map(str, driving_away))})")
        if biking_away: away_clauses.append(f"{len(biking_away)} biking lane(s) (id: {', '.join(map(str, biking_away))})")
        if away_clauses: road_desc_parts.append(f"{', '.join(away_clauses)} leaving the intersection")
        
        if bidirectional_lanes_list: road_desc_parts.append(f"{len(bidirectional_lanes_list)} bidirectional lane(s) (id: {', '.join(map(str, bidirectional_lanes_list))})")

        if not road_desc_parts:
            text_parts.append(f"- Road ID {road_id_int} has no relevant driving, biking, or bidirectional lanes.")
        else:
            text_parts.append(f"- Road ID {road_id_int} has {', '.join(road_desc_parts)}")

    text_parts.append("\nIntersection permissions:")
    target_junction_key = f"Junction {target_junction_id_str}"
    if target_junction_key not in all_junctions_data:
        text_parts.append(f"  Error: Junction {target_junction_id_str} data not found for permissions.")
        return "\n".join(text_parts)

    target_connections = all_junctions_data[target_junction_key]["connections"]

    for entry_road_id_int in ordered_road_ids: # Iterate in CCW order
        if entry_road_id_int not in road_lane_details or not road_lane_details[entry_road_id_int]['all_towards_lanes']:
            continue
        
        for entry_lane_id in road_lane_details[entry_road_id_int]['all_towards_lanes']:
            permissions_for_lane = []
            for conn_info in target_connections:
                if conn_info["Entry_road"] == entry_road_id_int and entry_lane_id in conn_info["Entry_lanes"]:
                    try:
                        lane_idx_in_conn = conn_info["Entry_lanes"].index(entry_lane_id)
                        exit_road_id_int = conn_info["Exit_road"]

                        if exit_road_id_int is None or exit_road_id_int not in road_id_to_ccw_index: # Check if exit road is one of the 4
                            continue 

                        exit_lane_id = conn_info["Exit_lanes"][lane_idx_in_conn]

                        idx_entry = road_id_to_ccw_index[entry_road_id_int]
                        idx_exit = road_id_to_ccw_index[exit_road_id_int]
                        diff = (idx_exit - idx_entry + 4) % 4 

                        turn_action_str = ""
                        if entry_road_id_int == exit_road_id_int : turn_action_str = "make a U-turn" # diff will be 0
                        elif diff == 1: turn_action_str = "turn right"
                        elif diff == 2: turn_action_str = "go straight"
                        elif diff == 3: turn_action_str = "turn left"
                        
                        if turn_action_str:
                            permissions_for_lane.append(
                                f"{turn_action_str} to Road ID {exit_road_id_int} lane {exit_lane_id}")
                    except ValueError: pass # entry_lane_id not in this conn_info["Entry_lanes"]
                    except IndexError: pass # Mismatch between Entry_lanes and Exit_lanes (should not happen)
            
            if permissions_for_lane:
                permissions_for_lane.sort() # Sort for consistent output
                text_parts.append(
                    f"- Road ID {entry_road_id_int} lane {entry_lane_id} can {', or '.join(permissions_for_lane)}")
    return "\n".join(text_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Extract junction connections and order roads around an intersection."
    )
    parser.add_argument("--dataset", "-d", default="inD", help="Drone dataset to process")
    parser.add_argument("--map_id", "-m", default="01_bendplatz", help="Map ID to process")
    parser.add_argument("--junction", "-j", default=None, help="Junction ID. Uses first <junction> if omitted.")
    args = parser.parse_args()

    xodr_path = f"./data/raw/{args.dataset}/maps/opendrive/{args.map_id}.xodr"
    processed_dir = f"./data/processed/{args.dataset}/map/"
    os.makedirs(processed_dir, exist_ok=True)
    yaml_output_path = os.path.join(processed_dir, f"{args.map_id}.yaml")
    txt_output_path = os.path.join(processed_dir, f"{args.map_id}_description.txt")

    all_junctions_data = get_all_junctions(xodr_path)
    tree = ET.parse(xodr_path)
    root = tree.getroot()
    junction_elem = None
    target_junction_id_str = None

    if args.junction is not None:
        target_junction_id_str = args.junction
        junction_elem = root.find(f".//junction[@id='{target_junction_id_str}']")
        if junction_elem is None:
            raise ValueError(f"Junction ID={target_junction_id_str} not found in XODR.")
    else:
        junction_elem = root.find(".//junction")
        if junction_elem is None:
            raise ValueError("No <junction> found in XODR.")
        target_junction_id_str = junction_elem.get("id")

    # road_contact_types: dict {"road_id_str": "predecessor" or "successor"}
    road_contact_types = collect_connected_roads(tree, junction_elem)
    
    # ordered_road_ids_list: List of road IDs (int) in CCW order.
    # road_id_to_ccw_index: dict mapping road ID (int) to its CCW index (0-3).
    ordered_road_ids_list, road_id_to_ccw_index = get_ordered_roads_and_ccw_indices(road_contact_types, tree)

    output_dict_yaml = {
        "Roads": ordered_road_ids_list, # List of road IDs in CCW order
        "Junctions": all_junctions_data
    }

    with open(yaml_output_path, "w") as f:
        yaml.dump(output_dict_yaml, f, sort_keys=False, default_flow_style=False)
    print(f"YAML output generated at: {yaml_output_path}")

    description_text = generate_text_description(
        ordered_road_ids_list,
        road_id_to_ccw_index,
        all_junctions_data,
        tree,
        target_junction_id_str,
        road_contact_types
    )
    with open(txt_output_path, "w") as f:
        f.write(description_text)
    print(f"Text description generated at: {txt_output_path}")

if __name__ == "__main__":
    main()