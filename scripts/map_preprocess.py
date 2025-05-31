"""
map_preprocess.py

This script parses an OpenDRIVE (.xodr) file containing a single cross‐intersection,
extracts each junction’s detailed connections (road‐to‐road lane mappings),
and labels the four roads around the main intersection as “bottom road”, “right road”,
“top road”, and “left road” in counterclockwise order.
"""

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
      - Exit_road: (int) the road on the far side of connectingRoad (predecessor or successor ≠ incomingRoad)
      - Exit_lanes: list[int] of laneLink “to” IDs
    """
    conns = []
    for conn in junction.findall("connection"):
        conn_road_id = int(conn.get("connectingRoad"))
        entry_road   = int(conn.get("incomingRoad"))

        # gather all laneLink “from” and “to”
        entry_lanes = []
        exit_lanes  = []
        for ll in conn.findall("laneLink"):
            entry_lanes.append(int(ll.get("from")))
            exit_lanes.append(int(ll.get("to")))

        # find that connectingRoad’s <road> element
        road_elem = roads[conn_road_id]
        link_elem = road_elem.find("link")
        # The <link> has both predecessor and successor; one of them is the junction,
        # the other is the “opposite” road end. We want the road ID ≠ entry_road.
        pred = link_elem.find("predecessor")
        succ = link_elem.find("successor")
        pred_id = int(pred.get("elementId")) if pred is not None else None
        succ_id = int(succ.get("elementId")) if succ is not None else None

        if pred_id == entry_road:
            exit_road = succ_id
        elif succ_id == entry_road:
            exit_road = pred_id
        else:
            # Fallback—if neither matches, just pick the successor
            exit_road = succ_id

        conns.append({
            "Connection_road": conn_road_id,
            "Entry_road":      entry_road,
            "Entry_lanes":     entry_lanes,
            "Exit_road":       exit_road,
            "Exit_lanes":      exit_lanes
        })

    return conns


def get_all_junctions(xodr_file: str) -> dict:
    """
    Parse the .xodr file and return a dict:
      junctions:
        Junction <id>:
          connections: [ { Connection_road, Entry_road, Entry_lanes, Exit_road, Exit_lanes }, ... ]
    """
    tree = ET.parse(xodr_file)
    root = tree.getroot()

    # Build a lookup of <road> elements by their integer ID
    roads = { int(r.get("id")): r for r in root.findall("road") }

    junctions = {}
    for junction in root.findall("junction"):
        jid = junction.get("id")
        conns = compute_junction_connections(junction, roads)
        junctions[f"Junction {jid}"] = {"connections": conns}

    return junctions


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle (in radians) to [0, 2π).
    """
    two_pi = 2.0 * math.pi
    return angle % two_pi


def extract_road_heading(road_elem: ET.Element, contact_type: str) -> float:
    """
    Given a <road> element and contact_type ("predecessor" or "successor"),
    extract the heading (in radians) of the road “as it leaves the junction.”

    - If contact_type == "predecessor", the junction is at the road’s start (s=0),
      so we use the <geometry hdg> directly.
    - If contact_type == "successor", the junction is at the road’s far end (s=length),
      so add π to hdg to reverse the vector.

    Returns a normalized angle in [0, 2π).
    """
    plan_view = road_elem.find("planView")
    if plan_view is None:
        raise ValueError(f"Road ID={road_elem.get('id')} has no <planView>.")

    geom = plan_view.find("geometry")
    if geom is None:
        raise ValueError(f"Road ID={road_elem.get('id')} missing <geometry> under <planView>.")

    hdg = float(geom.get("hdg", "0.0"))
    if contact_type == "predecessor":
        raw = hdg
    else:  # "successor"
        raw = hdg + math.pi

    return normalize_angle(raw)


def collect_connected_roads(tree: ET.ElementTree, junction: ET.Element) -> dict:
    """
    Find the four <road> IDs whose <link> references this junction (as predecessor or successor).
    Returns a dict mapping road_id (str) -> contact_type ("predecessor" or "successor").
    Raises RuntimeError if not exactly 4 roads are connected.
    """
    root = tree.getroot()
    jid = junction.get("id")
    connected = {}

    for road in root.findall("road"):
        rid = road.get("id")
        link = road.find("link")
        if link is None:
            continue

        pred = link.find("predecessor")
        if pred is not None and pred.get("elementType") == "junction" and pred.get("elementId") == jid:
            connected[rid] = "predecessor"
            continue

        succ = link.find("successor")
        if succ is not None and succ.get("elementType") == "junction" and succ.get("elementId") == jid:
            connected[rid] = "successor"

    if len(connected) != 4:
        raise RuntimeError(f"Expected 4 roads connected to junction ID={jid}, but found {len(connected)}.")

    return connected  # e.g. { "0": "predecessor", "1": "successor", ... }


def label_four_roads(connected: dict, tree: ET.ElementTree) -> dict:
    """
    Given a mapping of 4 road IDs -> contact_type, compute each road’s heading
    (in [0, 2π)) and assign labels in CCW order:

      “bottom road”  → angle closest to 3π/2 (270°)
      “right road”   → next CCW (angle > bottom)
      “top road”     → next CCW
      “left road”    → next CCW

    Returns:
      {
        "bottom road": "<road_id>",
        "right road":  "<road_id>",
        "top road":    "<road_id>",
        "left road":   "<road_id>"
      }
    """
    root = tree.getroot()
    angle_list = []

    for rid, contact in connected.items():
        # find the <road> element with attribute id=rid
        road_elem = root.find(f".//road[@id='{rid}']")
        if road_elem is None:
            raise RuntimeError(f"Connected road ID={rid} not found in <road> elements.")
        angle = extract_road_heading(road_elem, contact)
        angle_list.append((rid, angle))

    # Sort by angle ascending (0..2π)
    angle_list.sort(key=lambda x: x[1])

    # Find which entry is “bottom” (closest to 3π/2)
    target = 1.5 * math.pi  # 270° in radians
    diffs = [abs(((ang - target + math.pi) % (2 * math.pi)) - math.pi) for (_, ang) in angle_list]
    bottom_idx = int(min(range(4), key=lambda i: diffs[i]))

    # Rotate so that bottom_idx becomes index 0
    rotated = angle_list[bottom_idx:] + angle_list[:bottom_idx]

    labels = ["Bottom Road", "Right Road", "Top Road", "Left Road"]
    result = {labels[i]: int(rotated[i][0]) for i in range(4)}
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract junction connections and label the four roads around the intersection."
    )
    parser.add_argument(
        "--dataset", 
        "-d", 
        default="inD", 
        help="Drone dataset to process, e.g. inD, exiD, etc."
    )
    parser.add_argument(
        "--map_id", 
        "-m", 
        default="01_bendplatz", 
        help="Map ID to process, e.g. 01_bendplatz, 02_frankenburg, etc."
    )
    parser.add_argument(
        "--junction", 
        "-j", 
        default=None,
        help="Junction ID to target. If omitted, uses the first <junction>."
    )
    args = parser.parse_args()

    xodr_path = f"./data/raw/{args.dataset}/maps/opendrive/{args.map_id}.xodr"
    output_path = f"./data/processed/{args.dataset}/map/{args.map_id}.yaml"

    # 1. Parse the file and build “junctions” dictionary
    all_junctions = get_all_junctions(xodr_path)

    # 2. Also load the XML tree to compute headings
    tree = ET.parse(xodr_path)
    junction_elem = None

    if args.junction is not None:
        # find <junction id="...">
        junction_elem = tree.getroot().find(f".//junction[@id='{args.junction}']")
        if junction_elem is None:
            raise ValueError(f"Junction ID={args.junction} not found in XODR.")
    else:
        # pick the first junction
        junction_elem = tree.getroot().find(".//junction")
        if junction_elem is None:
            raise ValueError("No <junction> found in XODR.")

    # 3. Find the four roads connected to that junction
    connected_roads = collect_connected_roads(tree, junction_elem)

    # 4. Label them as bottom/right/top/left
    labeled_roads = label_four_roads(connected_roads, tree)

    # 5. Combine everything into one output dict
    output_dict = {
        "Roads": labeled_roads,
        "Junctions": all_junctions    
    }

    # 6. Dump to YAML (file or stdout)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(output_dict, f, sort_keys=False, default_flow_style=False)
    print(yaml.dump(output_dict, sort_keys=False, default_flow_style=False))


if __name__ == "__main__":
    main()
