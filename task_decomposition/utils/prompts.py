TASK_DESCRIPTION = """Your task is to ingest the following data and break down what occurred during the robot episode into granular subtasks. Each subtask should be a sequential step that occurred during the robot episode. Identify the start step and end step of each subtask.

Use chain of thought to break down the data into subtasks.

In your response, only report the list of subtasks as a tuple. As an example, only report this:
```
subtask_decomposition = [(0, 9, "move to above object"), 
                            (<start_step>, <end_step>, <subtask_description>), 
                            (<start_step>, <end_step>, <subtask_description>)]
```
"""

TXT_DATA_DESCRIPTION = (
    lambda columns: f"""The data captures a simulated episode of a robot end effector manipulating an environment. Each entry in the schema contains the action for the robot and an observation. The schema is composed of {columns}. The action column contains 6 DoF torque values for the robot arm and the 7th entry being gripper state. Each observation is x-y-z position values."""
)

FRAME_DATA_DESCRIPTION = (
    """The frames are the associated observation images for each step."""
)

env2description = {
    "Lift": "a single arm robot lifting a can",
    "Door": "a single arm robot opening a door",
    "PickPlace": "a single arm robot picking and placing a can",
    "Stack": "a single arm robot stacking a cube on top of another cube",
    "ToolHang": "a single arm robot hanging arbitrary tools on fixture",
    "NutAssemblySquare": "a single arm robot assembling nuts on a fixture",
}

ENV_DESCRIPTION = (
    lambda env_name: f"""The environment is a titled {env_name} and consists {env2description[env_name]}. Do not assume the environment is a success, in other words, the robot may not have completed the task."""
)
