# %%
import ast
import json
import numpy as np
from pprint import pprint
from typing import Union

from task_decomposition.utils.plotting import visualize_trajectory_decompositions
from task_decomposition.paths import DATA_GT_TXT_PATH, GPT_OUTPUT_PATH

from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

ROUND_DIGITS = 4

START_IDX = 0
END_IDX = 1
SUBTASK_NAME_IDX = 2


# %%
def extract_subtask_from_groundtruth_file(filepath: str) -> list:
    """
    This function extracts the subtask from the groundtruth file
    The groundtruth file is a txt file with the following format:
    step	subtask	stage
    0	Align manipulator height with Door	0
    1	Align manipulator height with Door	0
    2	Align manipulator height with Door	0
    3	Align manipulator height with Door	0
    4	Align manipulator height with Door	0
    5	Align manipulator height with Door	0
    6	Align manipulator height with Door	0
    7	Get closer to Door	1
    8	Get closer to Door	1
    9	Get closer to Door	1
    10	Get closer to Door	1
    11	Get closer to Door	1
    12	Get closer to Door	1
    13	Get closer to Door	1
    14	Get closer to Door	1
    15	Get closer to Door	1
    16	Get closer to Door	1
    17	Get closer to Door	1
    18	Get closer to Door	1
    ...

    The function returns a list of subtasks that specifies the start and end step of each subtask
    [(<start_step>, <end_step>, <subtask_name>), ...]
    """
    _step_idx = 0
    _subtask_idx = 1
    subtask_decomposition = []
    with open(filepath, "r") as f:
        # read and remove the header
        lines = f.readlines()
        lines = lines[1:] if lines[0].startswith("step") else lines

        # set initial values
        current_subtask = lines[0].split("\t")[_subtask_idx]
        start_step = int(lines[0].split("\t")[_step_idx])

        for idx, line in enumerate(lines):
            subtask = line.split("\t")[_subtask_idx]
            if subtask != current_subtask:
                # log values
                end_step = int(line.split("\t")[_step_idx]) - 1
                subtask_decomposition.append((start_step, end_step, current_subtask))

                # reset values
                start_step = int(line.split("\t")[_step_idx])
                current_subtask = subtask
            elif idx == len(lines) - 1:
                end_step = int(line.split("\t")[_step_idx])
                subtask_decomposition.append((start_step, end_step, current_subtask))

            # is there a corner case here where we are missing the last element should there be a change?

    return subtask_decomposition


def extract_subtask_from_gpt_output_file(filepath: str) -> list:
    """
    This function extracts the subtask from the gpt output file.
    The gpt output file is a json, with the field "response" containing the output of the gpt model.
    """
    # read the json file and load as a dictionary
    with open(filepath, "r") as f:
        data = json.load(f)

    response = data["response"]
    start = response.find("subtask_decomposition = [") + len(
        "subtask_decomposition = ["
    )
    end = response.find("]", start)
    list_str = response[start:end]

    # Converting string representation of list to actual Python list
    subtask_decomposition = ast.literal_eval("[" + list_str + "]")
    return subtask_decomposition


# %%
def intersection(subtask_A: tuple, subtask_B: tuple) -> bool:
    """
    This function checks if two subtasks intersect.
    """
    a1, a2 = subtask_A[START_IDX], subtask_A[END_IDX]
    b1, b2 = subtask_B[START_IDX], subtask_B[END_IDX]

    return a1 <= b2 and b1 <= a2


def get_IOU(subtask_A: tuple, subtask_B: tuple) -> float:
    a1, a2 = subtask_A[START_IDX], subtask_A[END_IDX]
    b1, b2 = subtask_B[START_IDX], subtask_B[END_IDX]

    # Calculate the intersection
    intersection = max(0, min(a2, b2) - max(a1, b1))

    # Calculate the union
    union = max(a2, b2) - min(a1, b1)

    # Avoid division by zero
    if union == 0:
        return 0

    # Calculate the IoU
    iou = intersection / union

    return round(iou, ROUND_DIGITS)


def bert_encode(text):
    """
    Encode the text using BERT
    """
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)[0].detach().numpy()


def semantic_distance(A: str, B: str) -> float:
    """
    Compare the similarity between two descriptions using BERT embeddings
    """
    embedding1 = bert_encode(A)
    embedding2 = bert_encode(B)
    similarity = 1 - cosine(embedding1, embedding2)
    return round(similarity, ROUND_DIGITS)


def subtask_similarity(
    subtask_decomp_A: list, subtask_decomp_B: list, DEBUG: bool = True
) -> dict:
    """
    This function computes the similarity between two subtask decompositions.
    """
    assert len(subtask_decomp_A) > 0 and len(subtask_decomp_B) > 0
    # assert subtask_decomp_A[0][START_IDX] == subtask_decomp_B[0][START_IDX]
    # assert subtask_decomp_A[-1][END_IDX] == subtask_decomp_B[-1][END_IDX]

    TEMPORAL_WEIGHT = 0.5
    SEMANTIC_WEIGHT = 0.5

    # TODO: assert subtask_decomp_A and subtask_decomp_B are non-overlapping
    N = subtask_decomp_A[-1][END_IDX] + 1  # Assuming the last index is end index
    temporal_score = 0
    semantic_score = 0
    for subtask_a in subtask_decomp_A:
        for subtask_b in subtask_decomp_B:
            if intersection(subtask_a, subtask_b):
                IOU = get_IOU(subtask_a, subtask_b)
                _temporal = IOU
                _semantic = (
                    semantic_distance(
                        subtask_a[SUBTASK_NAME_IDX], subtask_b[SUBTASK_NAME_IDX]
                    )
                    * IOU
                )
                # Apply weight based on intersection length relative to total task length
                temporal_weight = (
                    max(subtask_a[END_IDX], subtask_b[END_IDX])
                    - min(subtask_a[START_IDX], subtask_b[START_IDX])
                    + 1
                ) / N
                # semantic_weight = (
                #     min(subtask_a[END_IDX], subtask_b[END_IDX])
                #     - max(subtask_a[START_IDX], subtask_b[START_IDX])
                #     + 1
                # ) / N
                temporal_score += _temporal * temporal_weight
                semantic_score += (
                    _semantic * temporal_weight
                )  # or without IOU and just semantic_weight

    score = {}
    score["temporal"] = temporal_score
    score["semantic"] = semantic_score
    score["total"] = TEMPORAL_WEIGHT * temporal_score + SEMANTIC_WEIGHT * semantic_score
    return score


# %%
def test():
    filepath = DATA_GT_TXT_PATH + "/Lift_20240213-110117_5_gt.txt"
    subtask_decomposition = extract_subtask_from_groundtruth_file(filepath)
    print(subtask_decomposition)
    filepath = GPT_OUTPUT_PATH + "/Lift_20240213-110117_5.json"
    gpt_subtask_decomposition = extract_subtask_from_gpt_output_file(filepath)
    print(gpt_subtask_decomposition)
    print(" ")

    score1 = subtask_similarity(subtask_decomposition, gpt_subtask_decomposition)
    print(f"Score1: {score1}")

    score2 = subtask_similarity(gpt_subtask_decomposition, subtask_decomposition)
    print(f"Score2: {score2}")


if __name__ == "__main__":
    test()
