TASK_DESCRIPTION = """Your task is to ingest the following data and break down what occurred during the robot episode into hierarchical, granular subtasks. Each subtask should be a sequential step that occurred during the robot episode. Identify the start step and end step of each subtask. The Hierarchy should capture major intent changes of the robot, where each sub-task within a hierarchy is a low level, granular motion primitive.

Use chain of thought to break down the data into subtasks.

Report the list of subtasks as a tuple with at the beginning of your response:
(<start_step>, <end_step>, <subtask_description>, <hiearchy>)

As an example:
```subtask_decomposition = [(0, 9, "move to above object", 1.1), 
                            (10, 15, <subtask_description>", 1.2>), 
                            (16, 50, <subtask_description>, 2.1)]
```
The 'hierarchy' major numbers are the subtask and the minor numbers are the subsubtask further down in the hierarchy.

Then explain your thoughts.
"""

TXT_DATA_DESCRIPTION = (
    lambda columns: f"""The data captures a simulated episode of a robot end effector manipulating an environment. Each entry in the schema contains the action for the robot and an observation. The schema is composed of {columns}. The action column contains 6 DoF torque values for the robot arm and the 7th entry being gripper state. Each observation is x-y-z position values."""
)

FRAME_DATA_DESCRIPTION = (
    """The frames are the associated observation images for each step."""
)

env2description = {
    "lift": "a single arm robot lifting a can",
    "open_door": "a single arm robot opening a door",
    "picknplace": "a single arm robot picking and placing a can",
    "stack": "a single arm robot stacking a cube on top of another cube",
    "transport": "two robot arms with one robot transporting an object to the other robot",
    "tool_hang": "a single arm robot hanging arbitrary tools on fixture",
}

ENV_DESCRIPTION = (
    lambda env_name: f"""The environment is a titled {env_name} and consists {env2description[env_name]}."""
)
