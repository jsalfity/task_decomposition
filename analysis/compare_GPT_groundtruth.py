import pandas as pd
from typing import List

from task_decomposition.paths import DATA_PATH
from task_decomposition.utils.logging import format_df_to_subtask_list

gt_filename = "/open_door_gt.txt"


def compare_task_decomposition(actual: List, estimated: List) -> float:
    """
    compare these two lists

    Assumptions:
        - lists are the same size, there occur the same number of subtasks

    Ideas:
        - distance between word embeddings, i.e. naming of subtasks
        -
    """
    w1 = 0.1
    w2 = 0.9
    total = []
    for task_actual, task_expected in zip(actual, estimated):
        error_step = abs(task_actual[0] - task_expected[0])
        error_stage = abs(task_actual[1] - task_actual[1])
        total.append(w1 * error_step + w2 * error_stage)
    return sum(total)


def calculate_task_decomposition_similarity(actual: List, estimated: List) -> float:

    # check if both lists are the same length

    step_tolerance = 3  # Adjust the tolerance as needed
    subtask_match_weight = 0.4  # Adjust the weight based on your preference
    step_match_weight = 0.3  # Adjust the weight based on your preference

    # Initialize a matrix to store similarity scores between entries in list1 and list2
    similarity_matrix = [[0] * len(estimated) for _ in range(len(actual))]

    # Calculate similarity scores for each pair of entries
    for i, task_actual in enumerate(actual):
        for j, task_estimated in enumerate(estimated):
            # Extract subtask, start step, and end step from each entry
            subtask1 = task_actual["subtask"]
            subtask2 = task_estimated["subtask"]
            start_step1 = task_actual["start_step"]
            start_step2 = task_estimated["start_step"]
            end_step1 = task_actual["end_step"]
            end_step2 = task_estimated["end_step"]

            # Calculate subtask similarity (1 if identical, 0 otherwise)
            subtask_similarity = 1 if subtask1 == subtask2 else 0

            # Calculate step similarity based on a defined tolerance
            step_similarity = 1 - min(
                1,
                abs(start_step1 - start_step2)
                + abs(end_step1 - end_step2) / step_tolerance,
            )

            # Calculate the overall similarity score for this pair of entries
            entry_similarity = (subtask_similarity * subtask_match_weight) + (
                step_similarity * step_match_weight
            )

            similarity_matrix[i][j] = entry_similarity

    # Calculate the overall similarity between the two lists
    total_similarity = 0

    for i in range(len(actual)):
        best_match_score = max(similarity_matrix[i]) if any(similarity_matrix[i]) else 0
        total_similarity += best_match_score

    # Normalize the total similarity by dividing by the number of entries in list1
    if len(actual) > 0:
        normalized_similarity = total_similarity / len(actual)
    else:
        normalized_similarity = 0

    return normalized_similarity


fake_gpt_subtask_list = [
    {"start_step": 0, "end_step": 6, "subtask": 0},
    {"start_step": 7, "end_step": 19, "subtask": 1},
    {"start_step": 20, "end_step": 45, "subtask": 2},
    {"start_step": 46, "end_step": 74, "subtask": 3},
]

gpt_subtask_list = [
    {"start_step": 0, "end_step": 7, "subtask": "Manipulating the door"},
    {"start_step": 8, "end_step": 32, "subtask": "Moving the door"},
    {"start_step": 33, "end_step": 74, "subtask": "Adjusting the door"},
]

gt_df = pd.read_csv(DATA_PATH + gt_filename, delimiter="\t")
actual_subtask_list = format_df_to_subtask_list(gt_df)
print(
    calculate_task_decomposition_similarity(
        actual=actual_subtask_list, estimated=fake_gpt_subtask_list
    )
)

# similarity_score = calculate_task_decomposition_similarity(list1, list2)
# print(f"Overall List Similarity Score: {similarity_score * 100:.2f}%")
