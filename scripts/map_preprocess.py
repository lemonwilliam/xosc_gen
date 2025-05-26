import xml.etree.ElementTree as ET
import os
import argparse
import yaml

def compute_junction_connections(junction, roads):
    """
    Given a <junction> element and a dict of <road> elements by id,
    return a list of connection dicts:
      - id: connectingRoad id
      - Entry_road: incomingRoad
      - Entry_lanes: list of laneLink @from
      - Exit_road: the other end of the connectingRoad (predecessor or successor ≠ incomingRoad)
      - Exit_lanes: list of laneLink @to
    """
    connections = []
    for conn in junction.findall('connection'):
        conn_road_id = int(conn.get('connectingRoad'))
        entry_road   = int(conn.get('incomingRoad'))

        # collect laneLinks
        entry_lanes = []
        exit_lanes  = []
        for ll in conn.findall('laneLink'):
            entry_lanes.append(int(ll.get('from')))
            exit_lanes.append(int(ll.get('to')))

        # find the road element for connectingRoad
        road_elem = roads[conn_road_id]
        link      = road_elem.find('link')
        pred = link.find('predecessor').get('elementId')
        succ = link.find('successor').get('elementId')
        pred = int(pred)
        succ = int(succ)

        # determine exit road: the one ≠ entry_road
        if entry_road == pred:
            exit_road = succ
        elif entry_road == succ:
            exit_road = pred
        else:
            # fallback if neither matches
            exit_road = succ

        connections.append({
            'Connection_road':   conn_road_id,
            'Entry_road':   entry_road,
            'Entry_lanes':  entry_lanes,
            'Exit_road':    exit_road,
            'Exit_lanes':   exit_lanes
        })
    return connections

def get_all_junctions(xodr_file):
    """
    Parse the .xodr file and return a dict:
      junctions:
        Junction <id>:
          connections: [ { ... }, ... ]
    """
    tree = ET.parse(xodr_file)
    root = tree.getroot()

    # build lookup of road elements by id
    roads = { int(r.get('id')): r for r in root.findall('road') }

    junctions = {}
    for junction in root.findall('junction'):
        jid = junction.get('id')
        conns = compute_junction_connections(junction, roads)
        junctions[f"Junction {jid}"] = {'connections': conns}

    return junctions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing map data from drone datasets")
    parser.add_argument(
        "--dataset", 
        "-d", 
        type=str, 
        default="inD", 
        help="Drone dataset to process, e.g. inD, exiD, etc."
    )
    parser.add_argument(
        "--map_id", 
        "-m", 
        type=str, 
        default="01_bendplatz", 
        help="Map ID to process, e.g. 01_bendplatz, 02_frankenburg, etc."
    )
    args = parser.parse_args()

    all_junctions = get_all_junctions(f"./data/raw/{args.dataset}/maps/opendrive/{args.map_id}.xodr")

    # write to YAML
    output_path = f"./data/processed/{args.dataset}/map/{args.map_id}.yaml"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump({"junctions": all_junctions}, f, sort_keys=False, default_flow_style=False)