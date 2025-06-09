import yaml
import os
import numpy as np
import pandas as pd
from scenariogeneration import xosc, prettyprint
import xml.etree.ElementTree as ET
from collections import defaultdict
from xml.dom import minidom
from geomdl import knotvector


class Agents:
    def __init__(self):
        self.agents = None
        self.agentNames = []
    

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
    
    def __entites_def(self, agents, agent_categories):
        entities = xosc.Entities()
        # scale_prop = xosc.Property(name="scale", value="0.5") # create a “scale” property (0.5 = half size)
        # mode_prop  = xosc.Property(name="scaleMode", value="None") # disable any automatic BB→model or model→BB resizing
        agents.agentNames = []
        for agent in agents.agents:
            agents.agentNames.append(f"Agent{agent['track_id']}")
            if agent["type"] == "car":
                if agent['track_id'] in agent_categories['Ego']:
                    agentObject = xosc.CatalogReference(catalogname="VehicleCatalog", entryname="$EgoVehicle")
                    entities.add_scenario_object(name = agents.agentNames[-1], entityobject=agentObject)
                elif agent['track_id'] in agent_categories['Key']:
                    agentObject = xosc.CatalogReference(catalogname="VehicleCatalog", entryname="$KeyVehicle")
                    entities.add_scenario_object(name = agents.agentNames[-1], entityobject=agentObject)
                elif agent['track_id'] in agent_categories['Affected']:
                    agentObject = xosc.CatalogReference(catalogname="VehicleCatalog", entryname="$AffectedVehicle")
                    entities.add_scenario_object(name = agents.agentNames[-1], entityobject=agentObject)
                else:
                    agentObject = xosc.CatalogReference(catalogname="VehicleCatalog", entryname="$UnrelatedVehicle")
                    entities.add_scenario_object(name = agents.agentNames[-1], entityobject=agentObject)
            elif agent["type"] == "bicycle":
                agentObject = xosc.CatalogReference(catalogname="VehicleCatalog", entryname="$Bicycle")
                entities.add_scenario_object(name = agents.agentNames[-1], entityobject=agentObject)
            elif agent["type"] == "pedestrian":
                agentObject = xosc.CatalogReference(catalogname="PedestrianCatalog", entryname="$Pedestrian")
                entities.add_scenario_object(name = agents.agentNames[-1], entityobject=agentObject)
            elif agent["type"] == "truck_bus":
                agentObject = xosc.CatalogReference(catalogname="VehicleCatalog", entryname="$Truck")
                entities.add_scenario_object(name = agents.agentNames[-1], entityobject=agentObject)
                
        return entities
    
    def __story_init(self, agents, traj_df):
        init = xosc.Init()
        # Spawn agents if they are in the scenario initially
        for i, agent in enumerate(agents.agents):

            tid = agent["track_id"]
            agent_traj = traj_df[traj_df.trackId == tid].reset_index(drop=True)
            init_row = agent_traj.loc[0]

            if init_row.time == 0:                
                init.add_init_action(
                    agents.agentNames[i],
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
                    agents.agentNames[i],
                    xosc.AbsoluteSpeedAction(init_row.velocity, xosc.TransitionDynamics(xosc.DynamicsShapes.step, xosc.DynamicsDimension.time, 0))
                )

        return init
    
    def __write_story(self, agents, agent_categories, traj_df, duration):
        init = self.__story_init(agents, traj_df)
        speed_ctrl = ['speed_up', 'slow_down', 'reverse']
        land_ctrl = ['lane_change']
        road_ctrl = ['go_straight', 'turn_right', 'turn_left']

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

        for i, agent in enumerate(agents.agents):

            tid = agent["track_id"]
            agent_traj = traj_df[traj_df.trackId == tid].reset_index(drop=True)
            init_row = agent_traj.loc[0]
            end_row = agent_traj.loc[len(agent_traj)-1]

            maneuver = xosc.Maneuver("Maneuver_{}".format(tid))
            valid_maneuver = False

            speed_event_count = 0
            lateral_event_count = 0
            route_event_count = 0

            ## ➡️ First: Initialize agents that did not get teleported in Init
            if init_row.time != 0:
                spawn_event = xosc.Event(
                    f"Spawn_{agents.agentNames[i]}",
                    xosc.Priority.parallel,
                    maxexecution=1
                )

                # 1) Spawn the entity
                spawn_event.add_action(
                    "SpawnAgent", 
                    xosc.AddEntityAction(
                        entityref = agents.agentNames[i],
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
                        f"{agents.agentNames[i]}_SpeedEvent{speed_event_count}",
                        xosc.Priority.parallel,
                        maxexecution=3
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
                        f"{agents.agentNames[i]}_LateralEvent{lateral_event_count}",
                        xosc.Priority.parallel,
                        maxexecution=3
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
                        f"{agents.agentNames[i]}_RouteEvent{route_event_count}",
                        xosc.Priority.parallel,
                        maxexecution=3
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
                        ctrl_point_num = 0
                        nurbs = xosc.Nurbs(order=order) # Create Nurbs objects
                        for wp in attrs['trajectory']:
                            nurbs.add_control_point(xosc.ControlPoint(xosc.LanePosition(wp[3], wp[2], wp[1], wp[0])))
                            ctrl_point_num += 1
                        
                        
                        # Compute and add knots based on number of control points and degree
                        knots = knotvector.generate(order-1, ctrl_point_num)
                        nurbs.add_knots(knots)

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
                        if tid in agent_categories["Ego"]:
                            timeCondition = xosc.SimulationTimeCondition(
                                value=attrs["start_time"],
                                rule="greaterThan"
                            )
                            event.add_trigger(
                                xosc.ValueTrigger(
                                    name="SimulationTimeCondition",
                                    delay=0,
                                    conditionedge=xosc.ConditionEdge.rising,
                                    valuecondition=timeCondition
                                )
                            )
                        else:
                            timeCondition = xosc.SimulationTimeCondition(
                                value=attrs["start_time"],
                                rule="greaterThan"
                            )
                            event.add_trigger(
                                xosc.ValueTrigger(
                                    name="SimulationTimeCondition",
                                    delay=0,
                                    conditionedge=xosc.ConditionEdge.rising,
                                    valuecondition=timeCondition
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
                                    triggerentity=agents.agentNames[i],
                                    triggeringrule="any"
                                )
                            )   
                                
                        else:
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
                    '''
                    elif condition_type == "DistanceCondition":
                        entityCondition = xosc.DistanceCondition(
                            value=condition_attrs["distance"],
                            freespace=False,
                            alongRoute=False,
                            rule=condition_attrs["rule"]
                        )
                        event.add_trigger(
                            xosc.EntityTrigger(
                                name=condition_type,
                                delay=0,
                                conditionedge=xosc.ConditionEdge.rising,
                                entitycondition=entityCondition,
                                triggerentity=agents.agentNames[int(condition_attrs["target_agent"])],
                                triggeringrule="any"
                            )
                        )
                        valid_event = True
                    '''
                    if valid_event:
                        maneuver.add_event(event)
                        valid_maneuver = True
            
            if end_row.time < duration:
                despawn_event = xosc.Event(
                        f"Despawn_{agents.agentNames[i]}_Event",
                        xosc.Priority.parallel,
                        maxexecution=1
                )
                despawn_event.add_action(
                    f"Despawn_{agents.agentNames[i]}_Action",
                    xosc.DeleteEntityAction(entityref = agents.agentNames[i])
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
                mg.add_actor(agents.agentNames[i])
                mg.add_maneuver(maneuver)
                act.add_maneuver_group(mg)
        
        story.add_act(act)
        sb = sb.add_story(story)
        return sb
    

    
    def parse_scenario_description(self, scenario_dict, gt_trajectory_path, agent_categories, output_paths):
        #init/catalog
        agents = Agents()
        agents.agents = scenario_dict["scenario"]["agents"]
        description = scenario_dict.get("scenario", {}).get("description", "No description")
        author = scenario_dict.get("scenario", {}).get("author", "Unknown")
        road = xosc.RoadNetwork(roadfile="../../xodr/{}/{}.xodr".format(scenario_dict["scenario"]["dataset"], scenario_dict["scenario"]["location"]))
        catalog = xosc.Catalog()
        catalog.add_catalog("VehicleCatalog", "../Catalogs/Vehicles")
        catalog.add_catalog("PedestrianCatalog", "../Catalogs/Pedestrians")
        catalog.add_catalog("ControllerCatalog", "../Catalogs/Controllers")
        
        parameter_declarations = self.__parameter_def()       
        entities = self.__entites_def(agents, agent_categories)

        traj_df = pd.read_csv(gt_trajectory_path)
        sb = self.__write_story(agents, agent_categories, traj_df, scenario_dict["scenario"]["duration"])

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