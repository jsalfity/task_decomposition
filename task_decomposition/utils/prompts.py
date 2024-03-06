STACK_INCONTEXT = """Here is an example:
step	action	robot0_eef_pos	cubeA_pos	cubeB_pos	gripper_to_cubeA	gripper_to_cubeB	cubeA_to_cubeB
0	[2.41, 1.75, 0.0, 0.0, 0.0, 0.0, -1.0]	[-0.08, 0.01, 1.01]	[0.14, 0.17, 0.82]	[0.0, 0.0, 0.82]	[0.23, 0.16, -0.19]	[0.09, -0.01, -0.19]	[-0.14, -0.17, 0.0]
...
12	[0.21, 0.05, 0.0, 0.0, 0.0, 0.0, -1.0]	[0.13, 0.17, 1.01]	[0.14, 0.17, 0.82]	[0.0, 0.0, 0.82]	[0.01, 0.0, -0.19]	[-0.13, -0.17, -0.18]	[-0.14, -0.17, 0.01]
13	[0.13, 0.03, 0.0, 0.0, 0.0, 0.0, -1.0]	[0.14, 0.17, 1.01]	[0.14, 0.17, 0.82]	[0.0, 0.0, 0.82]	[0.01, 0.0, -0.19]	[-0.13, -0.17, -0.18]	[-0.14, -0.17, 0.01]
14	[0.08, 0.01, -1.88, 0.0, 0.0, 0.0, -1.0]	[0.14, 0.17, 0.99]	[0.14, 0.17, 0.82]	[0.0, 0.0, 0.82]	[0.0, 0.0, -0.17]	[-0.14, -0.
...
26	[0.01, -0.01, -0.1, 0.0, 0.0, 0.0, -1.0]	[0.14, 0.17, 0.83]	[0.14, 0.17, 0.82]	[0.0, 0.0, 0.82]	[0.0, -0.0, -0.01]	[-0.14, -0.17, -0.0]	[-0.14, -0.17, 0.01]
27	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]	[0.14, 0.17, 0.83]	[0.14, 0.17, 0.82]	[0.0, 0.0, 0.82]	[-0.0, -0.0, -0.01]	[-0.14, -0.17, -0.0]	[-0.14, -0.17, 0.01]
28	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]	[0.14, 0.17, 0.83]	[0.14, 0.17, 0.82]	[0.0, 0.0, 0.82]	[-0.0, -0.0, -0.01]	[-0.14, -0.17, -0.
...
40	[0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]	[0.14, 0.18, 0.9]	[0.14, 0.18, 0.88]	[0.0, 0.0, 0.82]	[-0.0, 0.0, -0.01]	[-0.14, -0.17, -0.07]	[-0.14, -0.18, -0.06]
41	[0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]	[0.14, 0.18, 0.9]	[0.14, 0.18, 0.89]	[0.0, 0.0, 0.82]	[-0.0, 0.0, -0.01]	[-0.14, -0.17, -0.08]	[-0.14, -0.18, -0.06]
42	[0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]	[0.14, 0.18, 0.91]	[0.14, 0.18, 0.89]	[0.0, 0.0, 0.82]	[-0.0, 0.0, -0.01]	[-0.14, -0.17, -0.08]	[-0.14, -0.18, -0.07]
43	[-1.39, -1.75, 2.5, 0.0, 0.0, 0.0, 1.0]	[0.13, 0.16, 0.92]	[0.13, 0.16, 0.91]	[0.0, 0.0, 0.82]	[-0.0, 0.0, -0.01]	[-0.13, -0.16, 
...
53	[-0.02, -0.15, 0.0, 0.0, 0.0, 0.0, 1.0]	[0.0, 0.01, 0.93]	[0.0, 0.01, 0.92]	[0.0, 0.0, 0.82]	[0.0, 0.0, -0.01]	[0.0, -0.01, -0.1]	[0.0, -0.01, -0.09]
54	[0.0, -0.1, -0.41, 0.0, 0.0, 0.0, 1.0]	[0.0, 0.01, 0.92]	[0.0, 0.01, 0.91]	[0.0, 0.0, 0.82]	[0.0, 0.0, -0.01]	[0.0, -0.01, -0.1]	[0.0, -0.01, -0.08]
...
61	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]	[0.0, 0.0, 0.89]	[0.0, 0.0, 0.88]	[0.0, 0.0, 0.82]	[0.0, 0.0, -0.01]	[0.0, 0.0, -0.07]	[0.0, 0.0, -0.06]
62	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]	[0.0, 0.0, 0.89]	[0.0, 0.0, 0.87]	[0.0, 0.0, 0.82]	[0.0, 0.0, -0.03]	[0.0, 0.0, -0.07]	[0.0, 0.0, -0.04]
63	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]	[0.0, 0.0, 0.89]	[0.0, 0.0, 0.87]	[0.0, 0.0, 0.82]	[0.0, 0.0, -0.02]	[0.0, 0.0, -0.07]	[0.0, 0.0, -0.04]
...
67	[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]	[-0.0, 0.0, 0.97]	[0.0, 0.0, 0.87]	[0.0, 0.0, 0.82]	[0.01, -0.0, -0.11]	[0.01, -0.0, -0.15]	[0.0, 0.0, -0.04]

subtask_decomposition = [(0, 13, 'Move to above Cube A'), (14, 26, 'Move directly down to Cube A'), (27, 28, 'Grasp Cube A'), (29, 42, 'Vertically pick up Cube A'), (43, 53, 'Align Cube A with Cube B'), (54, 59, 'Move Cube A vertically down to Cube B'), (60, 63, 'Release Cube A onto Cube B'), (64, 67, 'Return Home')]
"""

LIFT_INCONTEXT = """Here is an example:
Data = 
step	action	robot0_eef_pos	cube_pos	gripper_to_cube_pos
0	[0.13, 0.17, -1.81, 0.0, 0.0, 0.0, -1.0]	[-0.11, 0.0, 0.99]	[-0.11, 0.02, 0.82]	[-0.01, -0.01, 0.17]
...
10	[0.08, -0.0, -0.14, 0.0, 0.0, 0.0, -1.0]	[-0.11, 0.02, 0.83]	[-0.11, 0.02, 0.82]	[-0.01, 0.0, 0.01]
11	[0.08, -0.01, -0.09, 0.0, 0.0, 0.0, -1.0]	[-0.11, 0.02, 0.83]	[-0.11, 0.02, 0.82]	[-0.01, 0.0, 0.01]
12	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]	[-0.11, 0.02, 0.83]	[-0.11, 0.02, 0.82]	[-0.01, 0.0, 0.01]
13	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]	[-0.11, 0.02, 0.83]	[-0.11, 0.02, 0.82]	[-0.01, 0.0, 0.01]
...
26	[0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]	[-0.12, 0.02, 0.9]	[-0.11, 0.02, 0.89]	[-0.01, 0.0, 0.01]
27	[0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]	[-0.12, 0.02, 0.9]	[-0.11, 0.02, 0.9]	[-0.01, 0.0, 0.01]

subtask_decomposition = [(0, 11, 'Move to cube'), (12, 13, 'Grasp Cube'), (14, 27, 'Lift Cube')]
LLM Subtasks: [(0, 9, 'move end effector closer to cube'), (10, 11, 'maintain end effector position above cube'), (12, 13, 'change gripper state to open'), (14, 27, 'lift cube upwards')]

The real data is below"""

DOOR_INCONTEXT = """Here is an example:
step	action	robot0_eef_pos	door_pos	handle_pos	door_to_eef_pos	handle_to_eef_pos
0	[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]	[-0.1, 0.01, 1.02]	[-0.2, -0.34, 1.1]	[-0.15, -0.25, 1.08]	[-0.1, -0.35, 0.08]	[-0.04, -0.25, 0.05]
...
6	[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]	[-0.12, 0.01, 1.16]	[-0.2, -0.34, 1.1]	[-0.15, -0.25, 1.08]	[-0.09, -0.35, -0.06]	[-0.03, -0.25, -0.08]
...
12	[-0.07, -1.21, 0.0, 0.0, 0.0, 0.0, -1.0]	[-0.14, -0.12, 1.19]	[-0.2, -0.34, 1.1]	[-0.15, -0.25, 1.08]	[-0.06, -0.22, -0.09]	[-0.0, -0.12, -0.11]
13	[-0.04, -0.98, 0.0, 0.0, 0.0, 0.0, -1.0]	[-0.14, -0.15, 1.18]	[-0.2, -0.34, 1.1]	[-0.15, -0.25, 1.08]	[-0.06, -0.2, -0.08]	[-0.0, -0.1, -0.11]
14	[-0.01, -0.75, 0.0, 0.0, 0.0, 0.0, -1.0]	[-0.15, -0.16, 1.18]	[-0.2, -0.34, 1.1]	[-0.15, -0.25, 1.08]	[-0.06, -0.18, -0.08]	[0.0, -0.08, -0.11]
....
28	[0.13, -0.06, -0.79, 0.0, 0.0, 0.0, -1.0]	[-0.14, -0.22, 1.03]	[-0.2, -0.34, 1.1]	[-0.12, -0.25, 1.0]	[-0.06, -0.12, 0.07]	[0.02, -0.03, -0.03]
29	[0.18, -0.06, -0.77, 0.0, 0.0, 0.0, -1.0]	[-0.13, -0.22, 1.01]	[-0.2, -0.34, 1.1]	[-0.11, -0.25, 0.99]	[-0.07, -0.12, 0.09]	[0.02, -0.03, -0.02]
...
...
79	[0.36, 0.15, 0.16, 0.0, 0.0, 0.0, -1.0]	[-0.13, -0.01, 0.94]	[-0.22, -0.24, 1.1]	[-0.09, -0.1, 0.95]	[-0.09, -0.23, 0.16]	[0.04, -0.08, 0.02]

subtask_decomposition = [(0,6, "Align manipulator height with Door"), (7, 20, "Get closer to Door"), (21, 40, "Turn Door handle"), (41, 79, "Open Door")]
"""

PICKPLACE_INCONTEXT = """Here is an example:
step	action	robot0_eef_pos	Can_pos	Can_to_robot0_eef_pos
0	[1.49, -1.65, 0.0, 0.0, 0.0, 0.0, -1.0]	[-0.03, -0.1, 1.0]	[0.11, -0.25, 0.86]	[0.12, 0.15, 0.15]
...
11	[0.07, -0.06, -1.21, 0.0, 0.0, 0.0, -1.0]	[0.1, -0.25, 0.98]	[0.11, -0.25, 0.86]	[-0.01, 0.01, 0.12]
12	[0.04, -0.05, -1.05, 0.0, 0.0, 0.0, -1.0]	[0.1, -0.25, 0.96]	[0.11, -0.25, 0.86]	[-0.01, 0.01, 0.1]
...
118	[-0.43, 0.21, -0.55, 0.0, 0.0, 0.0, -1.0]	[0.15, -0.28, 0.93]	[0.11, -0.25, 0.86]	[-0.05, -0.02, 0.06]
119	[-0.43, 0.21, -0.55, 0.0, 0.0, 0.0, -1.0]	[0.15, -0.28, 0.93]	[0.11, -0.25, 0.86]	[-0.05, -0.02, 0.06]

subtask_decomposition = [(0, 9, "move to above object"),(11,119, "Move down to can")]
"""

TASK_DESCRIPTION = """Your task is to ingest the following data and break down what occurred during the robot episode into granular subtasks. Each subtask should be a sequential step that occurred during the robot episode. Identify the start step and end step of each subtask.
To identify changes in subtasks, look for significant changes in the robot's action and the environment's state.
Your response will be evaluated by a metric which computes the similarity between your subtask_decomposition and a ground truth subtask decomposition based on temporal and semantic similarity, so pay particular attention to the granularity and accuracy of your subtask decomposition.

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
