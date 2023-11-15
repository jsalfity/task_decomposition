# TASK_DESCRIPTION = """Your task is to ingest the following data and break down what occured during the robot episode into granular subtasks. Each subtask should be a sequential step that occured during the robot episode. You should identify the start and end step of each subtask. Create as many granular substasks as you see in the data.

# Use chain of thought to break down the data into subtasks.

# Report the list of subtasks in the following format, a list of dictionaries, at the beginning of your response:
# ```
# [{'start_step': <>, 'end_step': <>, 'subtask': <name of subtask>}, {'start_step': <>, 'end_step': <>, 'subtask': <name of subtask>}, {'start_step': <>, 'end_step': <>, 'subtask': <name of subtask>}, ...]
# ```

# Then explain your thoughts.
# """

TASK_DESCRIPTION = """Your task is to ingest the following data and break down what occurred during the robot episode into hierarchical granular subtasks. Each subtask should be a sequential step that occurred during the robot episode. You should identify the start step and end step of each subtask. Create as many granular subtasks as you see in the data.

Use chain of thought to break down the data into subtasks.

Report the list of subtasks as a dictionary with at the beginning of your response:
```
{1: <subtask_name>,
1.1: {'start_step: 1, 'end_step': 10, subtask: '<low level primitive>'},
1.2: {'start_step: 11, 'end_step': 20, subtask: '<low level primitive>'},
2: <subtask_name>,
2.1: {'start_step: 21, 'end_step': 30, subtask: '<low level primitive>'},
...
}
```
Then explain your thoughts.
"""

DATA_DESCRIPTION = """The data captures a simulated episode of a robot end effector manipulating an environment. Each entry in the schema contains the action for the robot and an observation."""

SCHEMA_DESCRIPTION = (
    lambda columns: f"The schema is composed of {columns}. The action column contains 6 DoF torque values for the robot arm and the 7th entry being gripper state. Each observation is x-y-z position values."
)
