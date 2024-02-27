import ast
import json
import numpy as np


import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

from task_decomposition.paths import ROBOT_TRAJ_GROUNDTRUTH_DATA_PATH, LLM_OUTPUT_PATH
from task_decomposition.constants import (
    START_STEP_IDX,
    END_STEP_IDX,
    DESCRIPTION_IDX,
    TEMPORAL_WEIGHT,
    SEMANTIC_WEIGHT,
    USE_MODULE_URL,
)


Bert_Tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
Bert = BertModel.from_pretrained("bert-base-uncased")
UniversalSentenceEncoder = hub.load(USE_MODULE_URL)


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


def extract_subtask_from_LLM_output_file(filepath: str, llm_model: str) -> list:
    """
    This function extracts the subtask from the LLM output file.
    The LLM output file is a json, with the field "response" containing the output of the LLM model.
    """
    # read the json file and load as a dictionary
    with open(filepath, "r") as f:
        data = json.load(f)

    if (
        llm_model == "gpt-4-vision-preview"
        or llm_model == "gpt-4-1106-preview"
        or llm_model == "gemini-pro"
        or llm_model == "gemini-pro-vision"
    ):
        response = data["response"]
        if "subtask_decomposition" not in response:
            return []
        start = response.find("subtask_decomposition = [") + len(
            "subtask_decomposition = ["
        )
        end = response.find("]", start)
        list_str = response[start:end]

        # Converting string representation of list to actual Python list
        try:
            subtask_decomposition = ast.literal_eval("[" + list_str + "]")
        except:
            subtask_decomposition = []

    elif llm_model == "video-llava":
        subtask_decomposition = data["subtask_decomposition"]

    else:
        raise NotImplementedError

    return subtask_decomposition


def intersection(subtask_A: tuple, subtask_B: tuple) -> bool:
    """
    This function checks if two subtasks intersect.
    """
    a1, a2 = subtask_A[START_STEP_IDX], subtask_A[END_STEP_IDX]
    b1, b2 = subtask_B[START_STEP_IDX], subtask_B[END_STEP_IDX]

    return a1 <= b2 and b1 <= a2


def get_IOU(subtask_A: tuple, subtask_B: tuple) -> np.float64:
    a1, a2 = subtask_A[START_STEP_IDX], subtask_A[END_STEP_IDX]
    b1, b2 = subtask_B[START_STEP_IDX], subtask_B[END_STEP_IDX]

    # Calculate the intersection
    intersection = max(0, min(a2, b2) - max(a1, b1))

    # Calculate the union
    union = max(a2, b2) - min(a1, b1)

    # Avoid division by zero
    if union == 0:
        return 0

    # Calculate the IoU
    iou = intersection / union

    assert iou >= 0 and iou <= 1, f"{iou} is not in [0, 1] range."
    return iou


def get_BERT_distance(A: str, B: str) -> np.float64:
    """
    Compare the similarity between two descriptions using BERT Encodings
    """

    def _bert_encode(text):
        """
        Encode the text using BERT
        """
        inputs = Bert_Tokenizer(text, return_tensors="pt")
        outputs = Bert(**inputs)
        return outputs.last_hidden_state.mean(dim=1)[0].detach().numpy()

    ## BERT model
    embedding1 = _bert_encode(A)
    embedding2 = _bert_encode(B)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity


def get_USE_distance(A: str, B: str) -> np.float64:
    """
    Compare the similarity between two descriptions using USE Encodings
    """
    ## USE model
    embedding1 = UniversalSentenceEncoder([A])
    embedding2 = UniversalSentenceEncoder([B])
    normalized_tensor1 = tf.nn.l2_normalize(embedding1, axis=-1)
    normalized_tensor2 = tf.nn.l2_normalize(embedding2, axis=-1)
    cosine_similarity = tf.reduce_sum(
        tf.multiply(normalized_tensor1, normalized_tensor2), axis=-1
    )
    cosine_similarity = cosine_similarity.numpy()[0]
    similarity = (
        cosine_similarity + 1
    ) / 2  # map cosine similarity from [-1, 1] to [0, 1]
    return similarity


def get_semantic_distance(A: str, B: str) -> np.float64:
    """
    Compare the similarity between two descriptions using USE Encodings
    """
    # similarity = get_BERT_distance(A, B)
    similarity = get_USE_distance(A, B)
    assert similarity >= 0 and similarity <= 1, f"{similarity} is not in [0, 1] range."
    return similarity


def is_valid_subtask(subtask: tuple) -> bool:
    """
    This function checks if a subtask is valid
    """
    return (
        type(subtask[START_STEP_IDX]) is int
        and type(subtask[END_STEP_IDX]) is int
        and subtask[END_STEP_IDX] >= subtask[START_STEP_IDX]
        and subtask[START_STEP_IDX] >= 0
        and subtask[END_STEP_IDX] >= 0
        and type(subtask[DESCRIPTION_IDX]) is str
    )


def get_subtask_similarity(subtask_decomp_A: list, subtask_decomp_B: list) -> dict:
    """
    This function computes the similarity between two subtask decompositions.
    INPUT:
    subtask_decomp_A: list of tuples
    subtask_decomp_B: list of tuples

    OUTPUT:
    score: dict
        - temporal: np.float64
        - semantic: np.float64
        - total: np.float64
    """
    # Error Checking
    assert len(subtask_decomp_A) > 0 and len(subtask_decomp_B) > 0
    for subtask in subtask_decomp_A + subtask_decomp_B:  # Check both lists
        assert is_valid_subtask(subtask), "Invalid subtask: start index after end index"

    temporal_scores = np.array([], dtype=np.float64)
    semantic_scores = np.array([], dtype=np.float64)
    interval_weights = np.array([], dtype=np.float64)

    _FINAL_STEP_A = subtask_decomp_A[-1][END_STEP_IDX]
    _FINAL_STEP_B = subtask_decomp_B[-1][END_STEP_IDX]
    _MAX_LENGTH = max(_FINAL_STEP_A, _FINAL_STEP_B)

    for subtask_a in subtask_decomp_A:
        for subtask_b in subtask_decomp_B:
            if intersection(subtask_a, subtask_b):
                _IOU = get_IOU(subtask_a, subtask_b)
                _SD = get_semantic_distance(
                    subtask_a[DESCRIPTION_IDX], subtask_b[DESCRIPTION_IDX]
                )
                # Small window length normalized over entire trajectory length
                _INTERVAL_WEIGHT = (
                    min(subtask_a[END_STEP_IDX], subtask_b[END_STEP_IDX])
                    - max(subtask_a[START_STEP_IDX], subtask_b[START_STEP_IDX])
                    + 1
                ) / _MAX_LENGTH

                temporal_scores = np.append(temporal_scores, _IOU)
                semantic_scores = np.append(semantic_scores, _SD)
                interval_weights = np.append(interval_weights, _INTERVAL_WEIGHT)

    ## Assert everything is the same size
    assert len(temporal_scores) == len(semantic_scores) == len(interval_weights)

    score = {}
    sum_interval_weights = np.sum(interval_weights)
    score["temporal"] = np.dot(temporal_scores, interval_weights) / sum_interval_weights
    score["semantic"] = np.dot(semantic_scores, interval_weights) / sum_interval_weights
    score["total"] = (
        TEMPORAL_WEIGHT * score["temporal"] + SEMANTIC_WEIGHT * score["semantic"]
    )
    return score


def test():
    env_name = "Lift"
    filepath = (
        ROBOT_TRAJ_GROUNDTRUTH_DATA_PATH(env_name) + "/Lift_20240213-110117_5_gt.txt"
    )
    subtask_decomposition = extract_subtask_from_groundtruth_file(filepath)
    print(subtask_decomposition)
    filepath = LLM_OUTPUT_PATH + "/Lift_20240213-110117_5.json"
    gpt_subtask_decomposition = extract_subtask_from_gpt_output_file(filepath)
    print(gpt_subtask_decomposition)
    print(" ")

    score1 = get_subtask_similarity(subtask_decomposition, gpt_subtask_decomposition)
    print(f"Score1: {score1}")

    score2 = get_subtask_similarity(gpt_subtask_decomposition, subtask_decomposition)
    print(f"Score2: {score2}")


if __name__ == "__main__":
    test()
