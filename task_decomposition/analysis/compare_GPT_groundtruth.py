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


print(calculate_iou(A=ground_truth_subtask_task, B=GPT_subtask_list, option="steps"))
visualize_trajectory_decompositions(
    actual=ground_truth_subtask_task,
    predicted=GPT_subtask_list,
    title="Overlap between Trajectory Decompositions -- Open Door Env",
)
