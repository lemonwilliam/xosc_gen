import argparse
import os
import pandas as pd
import yaml
import subprocess
import sys
from PyQt5.QtWidgets import (
    QApplication
)

from models.labeller import Labeller
from models.description import ScenarioDescriber
from models.scenario_interpretation import SceneInterpretation
from models.file_generation import FileGeneration
from scripts.esmini_process import EsminiSimulator
from scripts.scoring import Scorer
from scripts.visualization import Visualization
from scripts.gif_play import GifPlayer

from models.reflection import Reflection


def main(args):
    """
    Main function to load essential traffic rules, map comprehension guidelines, and scenario data,
    interact with GPT-4o using RAG, and generate scenario descriptions.
    """

    full_id = f"{args.scenario_id}_{args.start_time}_{args.end_time}"
    
    # Step 0: Preprocess and load data
    subprocess.run([
        "python", "scripts/drone_dataset_preprocess.py",
        "-d", args.dataset,
        "-s", args.scenario_id,
        "-st", f"{args.start_time}",
        "-et", f"{args.end_time}"
    ], check=True)

    '''
    subprocess.run([
        "python", "data/raw/drone-dataset-tools/src/record_segment.py",
        "--dataset", args.dataset,
        "--recording", args.scenario_id,
        "--start_frame", f"{args.start_time}",
        "--end_frame", f"{args.end_time}",
        "--output_gif_path", f"data/processed/{args.dataset}/gif/{full_id}.gif",
        "--playback_speed", "2"
    ], check=True)
    '''

    # Load scenario metadata
    metadata_path = f"data/processed/{args.dataset}/metadata/{full_id}.yaml"
    try:
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading tracks metadata: {e}")
        return
    
    # Load ground truth tracks
    gt_trajectory_path = f"data/processed/{args.dataset}/trajectory/{full_id}.csv"
    try:
        gt_trajectory = pd.read_csv(gt_trajectory_path)
    except Exception as e:
        print(f"Error loading ground truth tracks: {e}")
        return
    
    # Load map information
    loc = metadata['location']
    map_intersection_path = f"data/processed/{args.dataset}/map/{loc}.yaml"
    try:
        with open(map_intersection_path, "r") as f:
            map_intersection = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading map info: {e}")
        return
    
    
    # Step 1: Label individual agent actions using raw trajectory
    print("\n🔹 Step 1: Label individual actions")
    scenario_yaml_path = f"results/{args.dataset}/yaml/{full_id}.yaml"
    labeller = Labeller(
        meta_path=metadata_path,
        map_yaml_path=map_intersection_path,
        gt_trajectory_path=gt_trajectory_path
    )
    labeller.label()
    labeller.save(scenario_yaml_path)
    print(f"✅ Individual vehicle actions added to YAML file\n")


    # Step 2: Create scenario descriptions for all relevant agents
    print("\n🔹 Step 2: Create natural language description for agent actions")
    describer = ScenarioDescriber(
        scenario_yaml_path=scenario_yaml_path,
        trajectory_csv_path=gt_trajectory_path,
        map_yaml_path=map_intersection_path
    )

    full_scenario_description = describer.generate_description(ego_id=args.ego_id)

    behavior_log_path = f"results/{args.dataset}/description/{full_id}.txt"
    os.makedirs(os.path.dirname(behavior_log_path), exist_ok=True)
    with open(behavior_log_path, "w") as f:
        f.write(full_scenario_description)


    # # Step 3: Use LLM to acquire trigger conditions for actions
    # print("\n🔹 Step 3: Use LLM to acquire trigger conditions for actions")
    # #  Initialize the Interpreter Engine
    # try:
    #     interpreter = SceneInterpretation(model="gemini-2.5-pro")
    # except Exception as e:
    #     print(f"FATAL: Failed to initialize MapInterpretation engine: {e}")
    #     exit(1)

    # # Run the End-to-End Pipeline
    # # Call the main pipeline method on the instance
    # interactions_yaml_path = f"results/{args.dataset}/yaml/{full_id}_inter.yaml"
    # success = interpreter.run_analysis_pipeline(
    #     map_location = loc,
    #     agent_actions_path=behavior_log_path,
    #     output_yaml_path=interactions_yaml_path
    # )

    # # Report Final Status
    # if success:
    #     print("\n✅ Pipeline completed successfully.")
    # else:
    #     print("\n❌ Pipeline failed. Please check the logs above for errors.")

    # interpreter.cleanup_session(session_id=loc)


    interactions_yaml_path = f"results/{args.dataset}/yaml/{full_id}_inter.yaml"
    
    # Step 4: Generate initial OpenSCENARIO file
    print("\n🔹 Step 4: XOSC File Generation")
    filegen_model = FileGeneration()
    output_xosc_paths = [f"results/{args.dataset}/xosc/{full_id}_gen.xosc", f"esmini/resources/xosc/{args.dataset}/{full_id}_gen.xosc"]
    with open(scenario_yaml_path, "r") as f:
        agent_dict = yaml.safe_load(f)
    with open(interactions_yaml_path, "r") as f:
        interaction_dict = yaml.safe_load(f)
        ids = set()
        for interaction in interaction_dict.get("Interactions", []):
            ids.add(interaction["agent"])
            ids.add(interaction["interacts_with"])
        relevent_agents = sorted(list(ids))
    filegen_model.parse_scenario_description(
        agent_dict = agent_dict,
        interaction_dict = interaction_dict,
        gt_trajectory_path = gt_trajectory_path, 
        agent_categories = describer.agent_classifications[args.ego_id], 
        output_paths = output_xosc_paths
    )
    for op in output_xosc_paths:
        filegen_model.parameterize(op, op, relevent_agents)
    print(f"✅ Initial OpenSCENARIO file generated\n")


    # Step 5: Scoring
    print("\n🔹 Step 5: Scoring")
    esminiRunner = EsminiSimulator()
    scorer = Scorer()

    gt_ids = gt_trajectory["trackId"].unique().tolist()
    id_mapping = dict(zip(list(range(len(gt_ids))), gt_ids))

    esminiRunner.run(
        xosc_path = output_xosc_paths[1],
        record_path = f"./results/{args.dataset}/trajectory/{full_id}_gen.dat",
        x_offset = metadata.get("x_offset"),
        y_offset = metadata.get("y_offset"),
        track_id_mapping = id_mapping
    )

    fde_score = scorer.compute_fde(
        gt_csv_path = f"./data/processed/{args.dataset}/trajectory/{full_id}.csv",
        gen_csv_path = f"./results/{args.dataset}/trajectory/{full_id}_gen.csv"
    )
    print("FDE score:", fde_score)

    vis = Visualization(
        gt_csv_path=f"./data/processed/{args.dataset}/trajectory/{full_id}.csv", 
        gen_csv_path=f"./results/{args.dataset}/trajectory/{full_id}_gen.csv"
    )
    vis.interactive_view()

    '''
    app = QApplication(sys.argv)
    player = GifPlayer(
        f"./data/processed/{args.dataset}/gif/{full_id}.gif",
        "./esmini/bin/esmini",
        output_xosc_paths[1]
    )
    player.show()
    sys.exit(app.exec_())
    '''


    '''
    # Step 5–7: Reflection loop for refinement
    max_attempts = 3
    threshold = 0.85
    attempt = 0
    current_output_path = initial_output_path
    result_record_path = f"output/{args.scenario_id}_result.dat"
    refined_output_path = f"output/{args.scenario_id}_refined.xosc"

    while attempt < max_attempts:
        print(f"\n🔁 Attempt {attempt + 1} — Evaluating Scenario")

        # Run esmini and compute score
        reflection_model.run_esmini_and_convert(current_output_path, result_record_path)
        result_csv_path = result_record_path.replace(".dat", ".csv")
        score_result = reflection_model.compute_similarity(trajectory_path, result_csv_path)

        print("Similarity Score:", score_result["similarity_score"])

        if score_result["similarity_score"] >= threshold:
            print("✅ Scenario meets the similarity threshold. Final scenario file:", current_output_path)
            break

        print("⚠ Similarity score below threshold. Requesting improvement suggestions...")
        suggestions = reflection_model.suggest_improvements_with_gpt(trajectory_path, result_csv_path)
        print("✍ Suggestions:\n", suggestions)

        print("🔧 Refining scenario with GPT...")
        with open(current_output_path, "r") as f:
            original_xosc = f.read()

        refined_xosc = filegen_model.refine_openscenario_with_gpt(original_xosc, suggestions)
        with open(refined_output_path, "w") as f:
            f.write(refined_xosc)

        current_output_path = refined_output_path
        attempt += 1

    if attempt == max_attempts and score_result["similarity_score"] < threshold:
        print("❌ Maximum attempts reached. The final scenario may still need improvement.")'
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Scenario Analysis using GPT")
    parser.add_argument(
        "--dataset", 
        "-d", 
        type=str, 
        default="inD", 
        help="Dataset source name"
    )
    parser.add_argument(
        "--scenario_id", 
        "-s", 
        type=str, 
        default="08", 
        help="Scenario ID to process, e.g. 00, 01, etc."
    )
    parser.add_argument(
        "--start_time", 
        "-st", 
        type=int, 
        default=1250, 
        help="Start timestamp of the interval"
    )
    parser.add_argument(
        "--end_time", 
        "-et", 
        type=int, 
        default=1600, 
        help="End timestamp of the interval"
    )
    parser.add_argument(
        "--ego_id", 
        "-e", 
        type=int, 
        help="Ego Vehicle ID in the scenario")
    args = parser.parse_args()
    main(args)
