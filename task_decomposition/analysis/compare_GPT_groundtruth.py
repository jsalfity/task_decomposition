from task_decomposition.utils.plotting import visualize_trajectory_decompositions


def calculate_iou(A, B, option):
    """
    Calculate the Intersection over Union (IoU) between two trajectory decompositions.

    Args:
    A (list of dictionaries): The first trajectory decomposition.
    B (list of dictionaries): The second trajectory decomposition.

    Returns:
    float: The IoU value between the two decompositions.
    """
    if option == "steps":
        set_a = set((e["start_step"], e["end_step"]) for e in A)
        set_b = set((e["start_step"], e["end_step"]) for e in B)
    else:
        raise NotImplementedError

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    iou = intersection / union if union != 0 else 0.0
    return iou


##### Stack ####
ground_truth_subtask_task = [
    {"start_step": 0, "end_step": 7, "subtask": "Move to above Cube A"},
    {"start_step": 8, "end_step": 19, "subtask": "Moving directly down to Cube A"},
    {"start_step": 20, "end_step": 21, "subtask": "Grasping Cube A"},
    {"start_step": 22, "end_step": 35, "subtask": "Vertically picking up Cube A"},
    {"start_step": 36, "end_step": 43, "subtask": "Aligning Cube A with Cube B"},
    {
        "start_step": 44,
        "end_step": 49,
        "subtask": "Moving Cube A vertically down to Cube B",
    },
    {"start_step": 50, "end_step": 53, "subtask": "Releasing Cube A onto Cube B"},
    {"start_step": 54, "end_step": 79, "subtask": "Returning Home"},
]

GPT_subtask_list = [
    {"start_step": 0, "end_step": 10, "subtask": "Move end effector towards Cube A"},
    {"start_step": 11, "end_step": 20, "subtask": "Lower end effector to reach Cube A"},
    {"start_step": 20, "end_step": 20, "subtask": "Close gripper to grip Cube A"},
    {"start_step": 21, "end_step": 35, "subtask": "Raise end effector with Cube A"},
    {
        "start_step": 36,
        "end_step": 50,
        "subtask": "Move end effector with Cube A towards Cube B",
    },
    {"start_step": 50, "end_step": 50, "subtask": "Open gripper to release Cube A"},
    {"start_step": 51, "end_step": 79, "subtask": "Move end effector to home position"},
]

###### Open Door ####
# ground_truth_subtask_task = [
#     {"start_step": 0, "end_step": 6, "subtask": "Align manipulator height with Door"},
#     {"start_step": 7, "end_step": 19, "subtask": "Get closer to Door"},
#     {"start_step": 20, "end_step": 41, "subtask": "Turn Door handle"},
#     {"start_step": 42, "end_step": 74, "subtask": "Open Door"},
# ]
# GPT_subtask_list = [
#     {
#         "start_step": 0,
#         "end_step": 6,
#         "subtask": "Maintain gripper state while approaching the door",
#     },
#     {
#         "start_step": 7,
#         "end_step": 19,
#         "subtask": "Adjust end effector position and orientation near the door",
#     },
#     {
#         "start_step": 20,
#         "end_step": 41,
#         "subtask": "Manipulate door handle with varying torque and gripper state",
#     },
#     {
#         "start_step": 42,
#         "end_step": 73,
#         "subtask": "Reposition to pull door open while maintaining gripper state",
#     },
#     {
#         "start_step": 74,
#         "end_step": 74,
#         "subtask": "End of task with the door opened and gripper state maintained",
#     },
# ]

###### Lift ####
# ground_truth_subtask_task = [
#     {"start_step": 0, "end_step": 12, "subtask": "Move to above cube"},
#     {"start_step": 13, "end_step": 14, "subtask": "Grasp Cube"},
#     {"start_step": 15, "end_step": 39, "subtask": "Lift Cube"},
# ]

# GPT_subtask_list = [
#     {"start_step": 0, "end_step": 10, "subtask": "Move end effector towards the cube"},
#     {
#         "start_step": 11,
#         "end_step": 13,
#         "subtask": "Position end effector above the cube",
#     },
#     {"start_step": 13, "end_step": 14, "subtask": "Activate gripper to grasp the cube"},
#     {"start_step": 15, "end_step": 20, "subtask": "Lift the cube upwards"},
#     {"start_step": 21, "end_step": 39, "subtask": "Hold the cube steady in the air"},
# ]

print(calculate_iou(A=ground_truth_subtask_task, B=GPT_subtask_list, option="steps"))
visualize_trajectory_decompositions(
    actual=ground_truth_subtask_task,
    predicted=GPT_subtask_list,
    title="Overlap between Trajectory Decompositions -- Open Door Env",
)
