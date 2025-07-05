import argparse
import os
import pandas as pd
import yaml

from models.labeller import Labeller
from models.description import ScenarioDescriber
from models.scenario_interpretation import SceneInterpretation
from models.file_generation import FileGeneration
from scripts.esmini_process import EsminiSimulator
from scripts.scoring import Scorer
from scripts.visualization import Visualization

from models.reflection import Reflection


def main(args):
    """
    Main function to load essential traffic rules, map comprehension guidelines, and scenario data,
    interact with GPT-4o using RAG, and generate scenario descriptions.
    """

    '''
    Define input data paths and load
    '''

    # Load scenario metadata
    metadata_path = f"data/processed/{args.dataset}/metadata/{args.scenario_id}.yaml"
    try:
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading tracks metadata: {e}")
        return
    
    # Load ground truth tracks
    gt_trajectory_path = f"data/processed/{args.dataset}/trajectory/{args.scenario_id}.csv"
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
    scenario_yaml_path = f"results/{args.dataset}/yaml/{args.scenario_id}.yaml"
    labeller = Labeller(
        meta_path=metadata_path,
        map_yaml_path=map_intersection_path,
        gt_trajectory_path=gt_trajectory_path
    )
    labeller.label()
    labeller.save(scenario_yaml_path)
    print(f"✅ Route secision actions added to YAML file\n")


    # Step 2: Create scenario descriptions for all relevant agents
    scenario_yaml = yaml.safe_load(open(scenario_yaml_path))
    all_agents = scenario_yaml["scenario"]["agents"]
    descriptions = []

    describer = ScenarioDescriber(
        scenario_yaml_path = scenario_yaml_path,
        trajectory_csv_path = gt_trajectory_path,
        map_yaml_path = map_intersection_path
    )

    for agent in all_agents:
        tid = agent["track_id"]
        if agent["type"] == "pedestrian":
            continue
        if agent["initial_speed"] == 0.0 and agent["actions"] == []:
            continue

        description = describer.generate_description(ego_id=tid)
        descriptions.append(description)

    behavior_log_path = f"results/{args.dataset}/description/{args.scenario_id}.txt"
    os.makedirs(os.path.dirname(behavior_log_path), exist_ok=True)
    with open(behavior_log_path, "w") as f:
        f.write("\n\n".join(descriptions))

    
    # Step 2.5: Use OpenAI API to acquire trigger conditions for actions
    #  Initialize the Interpreter Engine
    try:
        interpreter = SceneInterpretation(model="gpt-4o")
    except Exception as e:
        print(f"FATAL: Failed to initialize MapInterpretation engine: {e}")
        exit(1)

    # Run the End-to-End Pipeline
    # Call the main pipeline method on the instance
    interactions_yaml_path = f"results/{args.dataset}/yaml/{args.scenario_id}_inter.yaml"
    success = interpreter.run_analysis_pipeline(
        map_location = loc,
        agent_actions_path=behavior_log_path,
        output_yaml_path=interactions_yaml_path
    )

    # Report Final Status
    if success:
        print("\n✅ Pipeline completed successfully.")
    else:
        print("\n❌ Pipeline failed. Please check the logs above for errors.")

    '''
    
    # Step 3: Generate initial OpenSCENARIO file
    print("\n🔹 Step 3: XOSC File Generation")
    filegen_model = FileGeneration()
    output_xosc_paths = [f"results/{args.dataset}/xosc/{args.scenario_id}_gen.xosc", f"esmini/resources/xosc/{args.dataset}/{args.scenario_id}_gen.xosc"]
    with open(scenario_yaml_path, "r") as f:
        agent_dict = yaml.safe_load(f)
    with open(interactions_yaml_path, "r") as f:
        interaction_dict = yaml.safe_load(f)
    filegen_model.parse_scenario_description(
        agent_dict = agent_dict,
        interaction_dict = interaction_dict,
        gt_trajectory_path = gt_trajectory_path, 
        agent_categories = describer.agent_classifications[args.ego_id], 
        output_paths = output_xosc_paths
    )
    for op in output_xosc_paths:
        filegen_model.parameterize(op, op)
    print(f"✅ Initial OpenSCENARIO file generated\n")


    # Step 4: Scoring
    print("\n🔹 Step 4: Scoring")
    esminiRunner = EsminiSimulator()
    scorer = Scorer()

    # Load UTM offset from recording metadata
    x_offset = metadata.get("x_offset")
    y_offset = metadata.get("y_offset")

    gt_ids = gt_trajectory["trackId"].unique().tolist()
    id_mapping = dict(zip(list(range(len(gt_ids))), gt_ids))

    esminiRunner.run(
        xosc_path = output_xosc_paths[1],
        record_path = f"./results/{args.dataset}/trajectory/{args.scenario_id}_gen.dat",
        x_offset = x_offset,
        y_offset = y_offset,
        track_id_mapping = id_mapping
    )

    fde_score = scorer.compute_fde(
        gt_csv_path = f"./data/processed/{args.dataset}/trajectory/world/{args.scenario_id}.csv",
        gen_csv_path = f"./results/{args.dataset}/trajectory/{args.scenario_id}_gen.csv"
    )
    print("FDE score:", fde_score)

    vis = Visualization(
        gt_csv_path=f"./data/processed/{args.dataset}/trajectory/world/{args.scenario_id}.csv", 
        gen_csv_path=f"./results/{args.dataset}/trajectory/{args.scenario_id}_gen.csv"
    )
    vis.interactive_view()

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
        default="14_0_299", 
        help="Scenario ID to process")
    parser.add_argument(
        "--ego_id", 
        "-e", 
        type=int, 
        help="Ego Vehicle ID in the scenario")
    args = parser.parse_args()
    main(args)
