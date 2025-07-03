import pandas as pd
import numpy as np
import os
import argparse
import yaml
from world_to_lane import CoordProjector

def get_offset(recordingMeta_csv):
    """
    Return a list of trackIds involved in the time interval [t1, t2].
    """
    # 1. Load the CSV (assuming it's tab-separated; adjust 'sep' if needed)
    df = pd.read_csv(recordingMeta_csv)

    x_offset = df['xUtmOrigin'][0]
    y_offset = df['yUtmOrigin'][0]

    return x_offset, y_offset

def find_involved_tracks(tracksMeta_csv, t1, t2):
    """
    Return a list of trackIds that are present in the time interval [t1, t2]
    and have more than 50 overlapping frames.
    """
    df = pd.read_csv(tracksMeta_csv)

    # Compute start and end of overlap
    overlap_start = np.maximum(df['initialFrame'], t1)
    overlap_end = np.minimum(df['finalFrame'], t2)

    # Compute overlap length in frames
    overlap_frames = (overlap_end - overlap_start + 1).clip(lower=0)

    # Apply both conditions
    condition = (overlap_frames > 50)

    involved_tracks = df.loc[condition, 'trackId'].unique()

    return np.array(involved_tracks)


def filter_trajectory(trajectory_csv, overlapping_track_ids, t1, t2, x_offset, y_offset, xodr_path):
    """
    Filters trajectory CSV by time and agent IDs, projects world coordinates to lane coordinates,
    and computes total velocity. Returns a processed DataFrame.
    """
    # Load trajectory data
    df_frames = pd.read_csv(trajectory_csv)
    condition = (
        df_frames["trackId"].isin(overlapping_track_ids)
        & (df_frames["frame"] >= t1)
        & (df_frames["frame"] <= t2)
        & (df_frames["frame"] % 2 == 0)
    )
    df_overlap = df_frames.loc[condition].copy()

    # Create simulation time column
    df_overlap["time"] = 0.04 * (df_overlap["frame"] - t1)

    # Initialize RoadManager
    projector = CoordProjector(xodr_path)

    # Project each row’s (xCenter, yCenter) to road/lane coordinates
    road_ids = []
    lane_ids = []
    lane_offsets = []
    s_coords = []
    for _, row in df_overlap.iterrows():
        road_id, lane_id, lane_offset, s = projector.coord_project(row["xCenter"]+x_offset, row["yCenter"]+y_offset)
        road_ids.append(road_id)
        lane_ids.append(lane_id)
        lane_offsets.append(round(lane_offset, 3))
        s_coords.append(round(s, 3))

    # Compute velocity magnitude from longitudinal and lateral components
    velocities = np.sqrt(df_overlap["lonVelocity"] ** 2 + df_overlap["latVelocity"] ** 2).round(3)

    # Assemble final DataFrame
    df_output = pd.DataFrame({
        "trackId": df_overlap["trackId"],
        "time": df_overlap["time"].round(2),
        "world_x": df_overlap["xCenter"].round(3),
        "world_y": df_overlap["yCenter"].round(3),
        "road_id": road_ids,
        "lane_id": lane_ids,
        "lane_offset": lane_offsets,
        "s": s_coords,
        "heading": df_overlap["heading"].round(3),
        "velocity": velocities
    })

    return df_output


def export_meta_to_yaml(tracks_meta_path, overlapping_track_ids, args, loc, offsets, output_path="scenario_metadata.yaml"):
    """
    Extract agent metadata and scenario-wide metadata, and write to a YAML file.
    """
    df_frames = pd.read_csv(tracks_meta_path)
    condition = df_frames["trackId"].isin(overlapping_track_ids)
    df_overlap = df_frames.loc[condition].copy()

    df_relevant = df_overlap[["trackId", "width", "length", "class"]]

    scenario_dict = {
        "dataset": args.dataset,
        "location": loc,
        "x_offset": float(offsets[0]),
        "y_offset": float(offsets[1]),
        "duration": (args.end_time - args.start_time + 25) / 25,
        "agents": []
    }

    for _, row in df_relevant.iterrows():
        scenario_dict["agents"].append({
            "track_id": int(row["trackId"]),
            "class": str(row["class"]),
            "width": float(row["width"]),
            "length": float(row["length"])
        })

    with open(output_path, "w") as f:
        yaml.dump(scenario_dict, f, sort_keys=False)
    
    print(f"Metadata written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing data from the intersection drone dataset")
    parser.add_argument(
        "--dataset", 
        "-d", 
        type=str, 
        default="inD", 
        help="Drone dataset to process, e.g. inD, exiD, etc."
    )
    parser.add_argument(
        "--scenario_id", 
        "-s", 
        type=str, 
        default="14", 
        help="Scenario ID to process, e.g. 00, 01, etc."
    )
    parser.add_argument(
        "--start_time", 
        "-st", 
        type=int, 
        default="0", 
        help="Start timestamp of the interval"
    )
    parser.add_argument(
        "--end_time", 
        "-et", 
        type=int, 
        default="300", 
        help="End timestamp of the interval"
    )
    args = parser.parse_args()

    recordingMeta_path = f"./data/raw/{args.dataset}/data/{args.scenario_id}_recordingMeta.csv"
    tracksMeta_path = f"./data/raw/{args.dataset}/data/{args.scenario_id}_tracksMeta.csv"
    trajectory_path = f"./data/raw/{args.dataset}/data/{args.scenario_id}_tracks.csv"

    scenario_meta = pd.read_csv(recordingMeta_path)
    map_dict = {
        'inD':{
            1: "01_bendplatz",
            2: "02_frankenburg",
            3: "03_heckstrasse",
            4: "04_aseag"
        }
    }
    loc = map_dict[args.dataset][scenario_meta['locationId'][0]]
    map_path = f"./data/raw/{args.dataset}/maps/opendrive/{loc}.xodr"

    # 1. Define your time interval
    t1, t2 = args.start_time, args.end_time

    # 2. Acquire offset
    x_offset, y_offset = get_offset(recordingMeta_path)
    #print(x_offset, y_offset)

    # 2. Find the trackIds overlapping [t1, t2]
    overlapping_ids = find_involved_tracks(tracksMeta_path, t1, t2)

    # 3. Filter and preprocess the frame-level trajectory CSV, write to respective directories
    df_output = filter_trajectory(trajectory_path, overlapping_ids, t1, t2, x_offset, y_offset, map_path)
    output_path = f"./data/processed/{args.dataset}/trajectory/{args.scenario_id}_{t1}_{t2}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_output.to_csv(output_path, index=False)
    print(f"Overlapping track frames saved to {output_path}")
    
    # 4. Write the relevant slices of metadata to another YAML file
    output_meta_path = f"./data/processed/{args.dataset}/metadata/{args.scenario_id}_{t1}_{t2}.yaml"
    os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)
    export_meta_to_yaml(tracksMeta_path, overlapping_ids, args, loc, [x_offset, y_offset], output_meta_path)

    
