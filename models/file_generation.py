import yaml
import os
import numpy as np
import pandas as pd
from scenariogeneration import xosc, prettyprint
import xml.etree.ElementTree as ET
from collections import defaultdict
from xml.dom import minidom
from geomdl import knotvector, BSpline, operations
from geomdl.visualization import VisMPL


class ScenarioInfo:
    def __init__(self):
        self.agents = None
        self.agentNames = []
        self.interactions = None
    

class FileGeneration:
    def __init__(self, api_key=None):
        '''
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either directly or through the 'OPENAI_API_KEY' environment variable.")
        self.client = openai.OpenAI(api_key=self.api_key)
        '''
        
    def __parameter_def(self):
        unrelated_vehicle = xosc.Parameter(name="UnrelatedVehicle",parameter_type="string",value="car_white")
        key_vehicle = xosc.Parameter(name="KeyVehicle",parameter_type="string",value="car_red")
        affected_vehicle = xosc.Parameter(name="AffectedVehicle",parameter_type="string",value="car_blue")
        ego_vehicle = xosc.Parameter(name="EgoVehicle",parameter_type="string",value="car_yellow")
        truck = xosc.Parameter(name="Truck",parameter_type="string",value="truck_yellow")
        bicycle = xosc.Parameter(name="Bicycle",parameter_type="string",value="bicycle")
        pedestrian = xosc.Parameter(name="Pedestrian",parameter_type="string",value="pedestrian_adult")
        paraList = [unrelated_vehicle, key_vehicle, affected_vehicle, ego_vehicle, truck, bicycle, pedestrian]
        parameter_declarations = xosc.ParameterDeclarations()
        for i in paraList:
            parameter_declarations.add_parameter(i)
        return parameter_declarations
    
    def __entites_def(self, scenario, agent_categories):
        entities = xosc.Entities()
        # scale_prop = xosc.Property(name="scale", value="0.5") # create a “scale” property (0.5 = half size)
        # mode_prop  = xosc.Property(name="scaleMode", value="None") # disable any automatic BB→model or model→BB resizing
        scenario.agentNames = []
        for agent in scenario.agents:
            scenario.agentNames.append(f"Agent{agent['track_id']}")
            if agent["type"] == "car":
                if agent['track_id'] in agent_categories['Ego']:
                    agentObject = xosc.CatalogReference(catalogname="VehicleCatalog", entryname="$EgoVehicle")
                    entities.add_scenario_object(name = scenario.agentNames[-1], entityobject=agentObject)
                elif agent['track_id'] in agent_categories['Key']:
                    agentObject = xosc.CatalogReference(catalogname="VehicleCatalog", entryname="$KeyVehicle")
                    entities.add_scenario_object(name = scenario.agentNames[-1], entityobject=agentObject)
                elif agent['track_id'] in agent_categories['Affected']:
                    agentObject = xosc.CatalogReference(catalogname="VehicleCatalog", entryname="$AffectedVehicle")
                    entities.add_scenario_object(name = scenario.agentNames[-1], entityobject=agentObject)
                else:
                    agentObject = xosc.CatalogReference(catalogname="VehicleCatalog", entryname="$UnrelatedVehicle")
                    entities.add_scenario_object(name = scenario.agentNames[-1], entityobject=agentObject)
            elif agent["type"] == "bicycle":
                agentObject = xosc.CatalogReference(catalogname="VehicleCatalog", entryname="$Bicycle")
                entities.add_scenario_object(name = scenario.agentNames[-1], entityobject=agentObject)
            elif agent["type"] == "pedestrian":
                agentObject = xosc.CatalogReference(catalogname="PedestrianCatalog", entryname="$Pedestrian")
                entities.add_scenario_object(name = scenario.agentNames[-1], entityobject=agentObject)
            elif agent["type"] == "truck_bus":
                agentObject = xosc.CatalogReference(catalogname="VehicleCatalog", entryname="$Truck")
                entities.add_scenario_object(name = scenario.agentNames[-1], entityobject=agentObject)
                
        return entities
    
    def __story_init(self, scenario, traj_df):
        init = xosc.Init()
        # Spawn agents if they are in the scenario initially
        for i, agent in enumerate(scenario.agents):

            tid = agent["track_id"]
            agent_traj = traj_df[traj_df.trackId == tid].reset_index(drop=True)
            init_row = agent_traj.loc[0]

            if init_row.time == 0:                
                init.add_init_action(
                    scenario.agentNames[i],
                    xosc.TeleportAction(
                        xosc.LanePosition(
                            init_row.s, 
                            init_row.lane_offset, 
                            int(init_row.lane_id), 
                            int(init_row.road_id), 
                            xosc.Orientation(h=np.deg2rad(init_row.heading))
                        )
                    )
                )
                init.add_init_action(
                    scenario.agentNames[i],
                    xosc.AbsoluteSpeedAction(init_row.velocity, xosc.TransitionDynamics(xosc.DynamicsShapes.step, xosc.DynamicsDimension.time, 0))
                )

        return init
    
    def __relative_speed(self, df, track_id_a, track_id_b, time_target):
        """
        Compute the relative speed between two agents at a given time.
        
        Parameters:
            df (pd.DataFrame): The full trajectory dataframe.
            track_id_a (int): Track ID of the first vehicle.
            track_id_b (int): Track ID of the second vehicle.
            time_target (float): The time at which to compute relative speed.
        
        Returns:
            float: The relative speed in m/s. Returns None if data is missing.
        """
        # Filter rows for each vehicle at the specified time
        row_a = df[(df["trackId"] == track_id_a) & (df["time"] == time_target)]
        print(row_a)
        row_b = df[(df["trackId"] == track_id_b) & (df["time"] == time_target)]
        print(row_b)
        
        if row_a.empty or row_b.empty:
            print(f"Missing data at time {time_target} for one of the trackIds.")
            return None

        # Extract values
        heading_a = np.deg2rad(row_a.iloc[0]["heading"])
        speed_a = row_a.iloc[0]["velocity"]
        vx_a = speed_a * np.cos(heading_a)
        vy_a = speed_a * np.sin(heading_a)

        heading_b = np.deg2rad(row_b.iloc[0]["heading"])
        speed_b = row_b.iloc[0]["velocity"]
        vx_b = speed_b * np.cos(heading_b)
        vy_b = speed_b * np.sin(heading_b)

        # Compute relative velocity vector and its magnitude
        rel_vx = vx_a - vx_b
        rel_vy = vy_a - vy_b
        relative_speed = np.sqrt(rel_vx**2 + rel_vy**2)

        return relative_speed
    
    def __distance(self, df, track_id_a, track_id_b, time_target):
        """
        Compute the Euclidean distance between two agents at a given time.

        Parameters:
            df (pd.DataFrame): The full trajectory dataframe.
            track_id_a (int): Track ID of the first vehicle.
            track_id_b (int): Track ID of the second vehicle.
            time_target (float): The time at which to compute the distance.

        Returns:
            float: Euclidean distance in meters. Returns None if data is missing.
        """
        row_a = df[(df["trackId"] == track_id_a) & (df["time"] == time_target)]
        print(row_a)
        row_b = df[(df["trackId"] == track_id_b) & (df["time"] == time_target)]
        print(row_b)

        if row_a.empty or row_b.empty:
            print(f"Missing data at time {time_target} for one of the trackIds.")
            return None

        x_a, y_a = row_a.iloc[0]["world_x"], row_a.iloc[0]["world_y"]
        x_b, y_b = row_b.iloc[0]["world_x"], row_b.iloc[0]["world_y"]

        return np.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)
    
    def __time_headway(self, df, follower_id, leader_id, time_target):
        """
        Compute the time headway between two vehicles using distance and relative speed.

        Parameters:
            df (pd.DataFrame): The full trajectory dataframe.
            follower_id (int): Track ID of the following vehicle.
            leader_id (int): Track ID of the leading vehicle.
            time_target (float): Timestamp to evaluate.

        Returns:
            float: Time headway in seconds. Returns None if not computable or relative speed is zero or negative.
        """
        distance = self.__distance(df, follower_id, leader_id, time_target)
        rel_speed = self.__relative_speed(df, follower_id, leader_id, time_target)

        if distance is None or rel_speed is None:
            return None
        if rel_speed <= 0:
            return float('inf')  # Vehicles are not approaching
        return distance / rel_speed
    
    def __get_condition(self, track_id, action_attrs, all_interactions, raw_trajectory):

        rules = {
            "lessThan": xosc.Rule.lessThan,
            "equalTo": xosc.Rule.equalTo,
            "greaterThan": xosc.Rule.greaterThan
        }

        for interaction in all_interactions:
            a = interaction.get("agent")
            t = interaction.get("timestamp")
            if a == track_id and t == action_attrs["start_time"]:
                e = interaction.get("details", {}).get("interacts_with")
                r = interaction.get("details", {}).get("rule")
                v = interaction.get("details", {}).get("speed")
                d = interaction.get("details", {}).get("duration")                   
                if interaction["trigger"] == "Time Headway Condition":
                    condition = xosc.TimeHeadwayCondition(
                        entity = f"Agent{e}",
                        value = self.__time_headway(raw_trajectory, a, e, t),
                        rule = rules[r],
                        alongroute = False,
                        freespace = False,
                        distance_type = xosc.RelativeDistanceType.euclidianDistance,
                        coordinate_system = xosc.CoordinateSystem.entity
                    )
                    return condition, "TimeHeadwayCondition"
                elif interaction["trigger"] == "Speed Condition":
                    condition = xosc.SpeedCondition(
                        value = v, 
                        rule = rules[r]
                    )
                    return condition, "SpeedCondition"
                elif interaction["trigger"] == "Relative Speed Condition":
                    condition = xosc.RelativeSpeedCondition(
                        value = self.__relative_speed(raw_trajectory, a, e, t), 
                        rule = rules[r], 
                        entity = f"Agent{e}"
                    )
                    return condition, "RelativeSpeedCondition"
                elif interaction["trigger"] == "Relative Distance Condition":
                    condition = xosc.RelativeDistanceCondition(
                        value = self.__distance(raw_trajectory, a, e, t),
                        rule = rules[r],
                        dist_type = xosc.RelativeDistanceType.cartesianDistance,
                        entity = f"Agent{e}",
                        alongroute = False,
                        freespace = False,
                        routing_algorithm = None
                    )
                    return condition, "RelativeDistanceCondition"
                elif interaction["trigger"] == "Stand Still Condition":
                    condition = xosc.StandStillCondition(
                        duration = d
                    )
                    return condition, "StoryboardElementStateCondition"
                else:
                    print("Non existent condition type")
                    break
                
        condition = xosc.SimulationTimeCondition(
            value=action_attrs["start_time"],
            rule=xosc.Rule.greaterThan
        )

        return condition, "SimulationTimeCondition"
    
    def __write_story(self, scenario, traj_df, duration):
        init = self.__story_init(scenario, traj_df)
        speed_ctrl = ['speed_up', 'slow_down', 'reverse']
        land_ctrl = ['lane_change']
        road_ctrl = ['go_straight', 'turn_right', 'turn_left', 'follow']

        sb = xosc.StoryBoard(
            init,
            xosc.ValueTrigger(
                name="StoryStop",
                delay=0,
                conditionedge=xosc.ConditionEdge.rising,
                valuecondition=xosc.SimulationTimeCondition(value=duration, rule="greaterThan"),
                triggeringpoint="stop"
            )
        )

        story = xosc.Story(name="UnifiedStory")
        act = xosc.Act(name="UnifiedAct")   
        

        for i, agent in enumerate(scenario.agents):

            tid = agent["track_id"]
            agent_traj = traj_df[traj_df.trackId == tid].reset_index(drop=True)
            init_row = agent_traj.loc[0]
            end_row = agent_traj.loc[len(agent_traj)-1]

            maneuver = xosc.Maneuver("Maneuver_{}".format(tid))
            valid_maneuver = False

            speed_event_count = 0
            lateral_event_count = 0
            route_event_count = 0
            assign_trajectory = False

            ## ➡️ First: Initialize agents that did not get teleported in Init
            if init_row.time != 0:
                spawn_event = xosc.Event(
                    f"Spawn_{scenario.agentNames[i]}",
                    xosc.Priority.parallel,
                    maxexecution=1
                )

                # 1) Spawn the entity
                spawn_event.add_action(
                    "SpawnAgent", 
                    xosc.AddEntityAction(
                        entityref = scenario.agentNames[i],
                        position = xosc.LanePosition(
                            init_row.s, 
                            init_row.lane_offset, 
                            int(init_row.lane_id), 
                            int(init_row.road_id), 
                            xosc.Orientation(h=np.deg2rad(init_row.heading))
                        )
                    )
                )
                # 2) Immediately set its speed
                spawn_event.add_action(
                    "SetSpeed", 
                    xosc.AbsoluteSpeedAction(
                        speed = init_row.velocity, 
                        transition_dynamics = xosc.TransitionDynamics(xosc.DynamicsShapes.step, xosc.DynamicsDimension.time, 0)
                    )
                )

                # 3) Finally, trigger the entire block at the right time
                spawn_event.add_trigger(
                    xosc.ValueTrigger(
                        name="SpawnTrigger",
                        delay=0,
                        conditionedge=xosc.ConditionEdge.rising,
                        valuecondition=xosc.SimulationTimeCondition(
                            value=init_row.time,
                            rule="greaterThan"
                        )
                    )
                )
                maneuver.add_event(spawn_event)
                valid_maneuver = True

            ## ➡️ Then: Add Events for each Action agent performed
            for j, action in enumerate(agent["actions"]):

                action_type = action["type"]
                attrs = action["attributes"]

                valid_action = False
                valid_event = False

                ## ➡️ First determine the Action of the event

                # Longitudinal action
                if action_type in speed_ctrl:
                    event = xosc.Event(
                        f"{scenario.agentNames[i]}_SpeedEvent{speed_event_count}",
                        xosc.Priority.parallel,
                        maxexecution=1
                    )
                    speed_event_count += 1
                    dynamics = xosc.TransitionDynamics("linear", "time", attrs["duration"])
                    event.add_action(
                        action_type,
                        xosc.AbsoluteSpeedAction(attrs["target_speed"], dynamics)
                    )
                    valid_action = True
                # Lateral Action
                elif action_type in land_ctrl:
                    event = xosc.Event(
                        f"{scenario.agentNames[i]}_LateralEvent{lateral_event_count}",
                        xosc.Priority.parallel,
                        maxexecution=1
                    )
                    lateral_event_count += 1
                    lane_id = attrs['target_lane']
                    dynamics = xosc.TransitionDynamics(xosc.DynamicsShapes.cubic, "time", 2.0)
                    event.add_action(
                        action_type,
                        xosc.AbsoluteLaneChangeAction(lane_id, dynamics)
                    )
                    valid_action = True
                # Route Decision Action
                elif action_type in road_ctrl:
                    event = xosc.Event(
                        f"{scenario.agentNames[i]}_RouteEvent{route_event_count}",
                        xosc.Priority.parallel,
                        maxexecution=1
                    )
                    route_event_count += 1
                    if attrs['legal']:
                        route = xosc.Route("Agent_{}_route".format(tid), False)
                        sp, ep = attrs['entry_point'], attrs['exit_point']
                        route_start = xosc.LanePosition(sp[3], sp[2], sp[1], sp[0], xosc.Orientation(h=np.deg2rad(sp[4])))
                        route_end = xosc.LanePosition(ep[3], ep[2], ep[1], ep[0], xosc.Orientation(h=np.deg2rad(ep[4])))
                        route.add_waypoint(route_start, xosc.RouteStrategy().shortest)
                        route.add_waypoint(route_end, xosc.RouteStrategy().shortest)
                        event.add_action(
                            "SetRoute",
                            xosc.AssignRouteAction(route)
                        )                           
                    else:
                        # Add control points to Nurbs and set knots
                        order = 3
                        curve  = BSpline.Curve()
                        curve.degree = order - 1
                        representpts = []
                       
                        for wp in attrs['trajectory']:
                            representpts.append([wp[5], wp[6]])
                        curve.ctrlpts = representpts
                        # Compute and add knots based on number of control points and degree
                        curve.knotvector = knotvector.generate(curve.degree, curve.ctrlpts_size)
                        operations.refine_knotvector(curve, [1])
                        
                        '''
                        if __debug__:
                            curve.vis = VisMPL.VisCurve2D()
                            curve.render()
                            print("curve point:", curve.ctrlpts)
                            print("curve knot:", curve.knotvector)
                        '''
                            
                        nurbs = xosc.Nurbs(order=order) # Create Nurbs objects
                        for cc in curve.ctrlpts:
                            nurbs.add_control_point(xosc.ControlPoint(xosc.WorldPosition(x=cc[0], y=cc[1])))
                        nurbs.add_knots(curve.knotvector)

                        # Create Trajectory Object and assign nurbs
                        traj = xosc.Trajectory("Agent_{}_trajectory".format(tid), False)
                        traj.add_shape(nurbs)
                        event.add_action(
                            action_type,
                            xosc.FollowTrajectoryAction(traj, following_mode="position")
                        )

                    valid_action = True                   
                else:
                    print(action_type, 'is not valid')
                    continue

                if valid_action:
                    # Reach timestamp
                    if action_type in speed_ctrl or action_type in land_ctrl:
                        condition, condition_type = self.__get_condition(tid, attrs, scenario.interactions, traj_df)
                        if condition_type == "SimulationTimeCondition":
                            event.add_trigger(
                                xosc.ValueTrigger(
                                    name=condition_type,
                                    delay=0,
                                    conditionedge=xosc.ConditionEdge.rising,
                                    valuecondition=condition
                                )
                            )
                        else:
                            event.add_trigger(
                                xosc.EntityTrigger(
                                    name=condition_type,
                                    delay=0,
                                    conditionedge=xosc.ConditionEdge.rising,
                                    entitycondition=condition,
                                    triggerentity=scenario.agentNames[i],
                                    triggeringrule="any"
                                )
                            )
                        valid_event = True
                    # Reach position
                    if action_type in road_ctrl:
                        if attrs['legal']:
                            sp = attrs["entry_point"]
                            start_position = xosc.LanePosition(sp[3], sp[2], sp[1], sp[0])
                            positionCondition = xosc.ReachPositionCondition(
                                tolerance = 3.0,
                                position = start_position
                            )
                            event.add_trigger(
                                xosc.EntityTrigger(
                                    name="ReachPositionCondition",
                                    delay=0,
                                    conditionedge=xosc.ConditionEdge.rising,
                                    entitycondition=positionCondition,
                                    triggerentity=scenario.agentNames[i],
                                    triggeringrule="any"
                                )
                            )   
                                
                        else:
                            assign_trajectory = True
                            timeCondition = xosc.SimulationTimeCondition(
                                value=attrs["start_time"],
                                rule="greaterThan"
                            )
                            event.add_trigger(
                                xosc.ValueTrigger(
                                    name="SimulationTimeCondition",
                                    delay=0,
                                    conditionedge=xosc.ConditionEdge.none,
                                    valuecondition=timeCondition
                                )
                            )
                           
                        
                        valid_event = True
                    if valid_event:
                        maneuver.add_event(event)
                        valid_maneuver = True
            
            if assign_trajectory:
                despawn_event = xosc.Event(
                        f"Despawn_{scenario.agentNames[i]}_Event",
                        xosc.Priority.parallel,
                        maxexecution=1
                )
                despawn_event.add_action(
                    f"Despawn_{scenario.agentNames[i]}_Action",
                    xosc.DeleteEntityAction(entityref = scenario.agentNames[i])
                )
                despawn_event.add_trigger(
                    xosc.ValueTrigger(
                        name = "StoryboardElementStateCondition",
                        delay = 0.2,
                        conditionedge = xosc.ConditionEdge.rising,
                        valuecondition = xosc.StoryboardElementStateCondition(
                            element="event",
                            reference =  f"{scenario.agentNames[i]}_RouteEvent{route_event_count-1}",
                            state = xosc.StoryboardElementState.completeState
                        )
                    )
                )
                maneuver.add_event(despawn_event)
                valid_maneuver = True
            elif end_row.time < duration:
                despawn_event = xosc.Event(
                        f"Despawn_{scenario.agentNames[i]}_Event",
                        xosc.Priority.parallel,
                        maxexecution=1
                )
                despawn_event.add_action(
                    f"Despawn_{scenario.agentNames[i]}_Action",
                    xosc.DeleteEntityAction(entityref = scenario.agentNames[i])
                )
                despawn_event.add_trigger(
                    xosc.ValueTrigger(
                        name = "SimulationTimeCondition",
                        delay = 0.2,
                        conditionedge = xosc.ConditionEdge.rising,
                        valuecondition = xosc.SimulationTimeCondition(
                            value = end_row.time,
                            rule = "greaterThan"
                        )
                    )
                )
                '''
                despawn_event.add_trigger(
                    xosc.EntityTrigger(
                        name="ReachPositionCondition",
                        delay=0.5,
                        conditionedge=xosc.ConditionEdge.rising,
                        entitycondition = xosc.ReachPositionCondition(
                            tolerance = 3.0,
                            position = xosc.LanePosition(                          
                                end_row.s, 
                                end_row.lane_offset, 
                                int(end_row.lane_id), 
                                int(end_row.road_id)
                            )
                        ),
                        triggerentity=agents.agentNames[i],
                        triggeringrule="any"
                    )
                )
                '''
                maneuver.add_event(despawn_event)
                valid_maneuver = True

            if valid_maneuver:
                mg = xosc.ManeuverGroup("MG_{}".format(agent["track_id"]), maxexecution=1)
                mg.add_actor(scenario.agentNames[i])
                mg.add_maneuver(maneuver)
                act.add_maneuver_group(mg)
        
        story.add_act(act)
        sb = sb.add_story(story)
        return sb
    

    
    def parse_scenario_description(self, agent_dict, interaction_dict, gt_trajectory_path, agent_categories, output_paths):
        #init/catalog
        scenario = ScenarioInfo()
        scenario.agents = agent_dict["scenario"]["agents"]
        scenario.interactions = interaction_dict["Interactions"]
        author = agent_dict.get("scenario", {}).get("author", "Unknown")
        road = xosc.RoadNetwork(roadfile="../../xodr/{}/{}.xodr".format(agent_dict["scenario"]["dataset"], agent_dict["scenario"]["location"]))
        catalog = xosc.Catalog()
        catalog.add_catalog("VehicleCatalog", "../Catalogs/Vehicles")
        catalog.add_catalog("PedestrianCatalog", "../Catalogs/Pedestrians")
        catalog.add_catalog("ControllerCatalog", "../Catalogs/Controllers")
        
        parameter_declarations = self.__parameter_def()       
        entities = self.__entites_def(scenario, agent_categories)

        traj_df = pd.read_csv(gt_trajectory_path)
        sb = self.__write_story(scenario, traj_df, agent_dict["scenario"]["duration"])

        scenario = xosc.Scenario(
            name = "GeneratedScenario",
            author = author,
            parameters = parameter_declarations,
            entities = entities,
            storyboard = sb,
            roadnetwork = road,
            catalog = catalog,
            osc_minor_version=0
        )

        for path in output_paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            scenario.write_xml(path)

        return
    
    
    
    def parameterize(self,path_in, path_out):
        
        tree = ET.parse(path_in)
        root = tree.getroot()
        param_counter = defaultdict(int)
        param_map = {}
        
        # Find ParameterDeclarations block or create it
        param_decls = root.find(".//ParameterDeclarations")
        if param_decls is None:
            storyboard = root.find(".//Storyboard")
            param_decls = ET.Element("ParameterDeclarations")
            root.insert(list(root).index(storyboard), param_decls)

        #event
        for event in root.findall(".//Event"):
            agent_name = event.attrib.get("name", "UnknownAgent")
            for action in event.findall(".//PrivateAction"):
                action_type = "unknown"
                if action.find(".//SpeedAction") is not None:
                    action_type = "speedaction"
                    
                    # Duration
                    for dyn, ats in zip(action.findall(".//SpeedActionDynamics"), action.findall(".//AbsoluteTargetSpeed")):
                        dynval = dyn.attrib.get("value")
                        atsval = ats.attrib.get("value")
                        if (dynval is None) or (atsval is None) or (float(dynval)==0):
                            continue
                        #duration
                        param_name = f"{agent_name}Duration{param_counter[(agent_name, action_type)]}"
                        dyn.attrib["value"] = f"${param_name}"
                        param_map[param_name] = dynval
                        
                        #targetspeed
                        param_name = f"{agent_name}TargetSpeed{param_counter[(agent_name, action_type)]}"
                        ats.attrib["value"] = f"${param_name}"
                        param_map[param_name] = atsval
                        param_counter[(agent_name, action_type)] += 1
        
                if action.find(".//LaneChangeAction") is not None:
                    action_type = "lanechangeaction"
                    
                    # Duration
                    for dyn in action.findall(".//LaneChangeActionDynamics"):
                        val = dyn.attrib.get("value")
                        if (val is None) or (float(val)==0):
                            continue
                        param_name = f"{agent_name}Duration{param_counter[(agent_name, action_type)]}"
                        dyn.attrib["value"] = f"${param_name}"
                        param_map[param_name] = val
                        param_counter[(agent_name, action_type)] += 1
           
        # Add parameters
        for name, val in param_map.items():
            param_elem = ET.Element("ParameterDeclaration", {
                "name": name,
                "parameterType": "double",
                "value": str(val)
            })
            param_decls.append(param_elem)
        
        xml_str = ET.tostring(root, encoding="utf-8")
        pretty = minidom.parseString(xml_str).toprettyxml(indent="    ")
        out = "\n".join([line for line in pretty.split("\n") if line.strip() != ""])
        with open(path_out, "w", encoding="utf-8") as f:
            f.write(out)