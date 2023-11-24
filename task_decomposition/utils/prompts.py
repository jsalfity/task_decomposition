TASK_DESCRIPTION = """Your task is to ingest the following data and break down what occurred during the robot episode into hierarchical, granular subtasks. Each subtask should be a sequential step that occurred during the robot episode. Identify the start step and end step of each subtask. Focus on hierarchy with repeated lower level primitives.

Use chain of thought to break down the data into subtasks.

Report the list of subtasks as a dictionary with at the beginning of your response:
```subtask_decomposition = [
{1.1: {'start_step: <>, 'end_step': <>, subtask: '<low level primitive>'},
1.2: {'start_step: <>, 'end_step': <>, subtask: '<low level primitive>'},
2.1: {'start_step: <>, 'end_step': <>, subtask: '<low level primitive>'},
...
}]
```
The major numbers are the subtask and the minor numbers are the subsubtask.

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
