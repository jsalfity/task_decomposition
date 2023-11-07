import json
import csv
import pandas as pd
from typing import List

import robosuite as suite

from task_decomposition.paths import GIF_PATH, DATA_PATH


def make_env(config):
    return suite.make(**config)


# from https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
# def save_frames_as_gif(frames, path=GIF_PATH, filename="hex123.gif"):

CSV_SCHEMA = [
    "step",
    "robot0_eef_pos",
    "cube_pos",
    "gripper_to_cube_pos",
    "action",
    "grasp",
    "reward",
]


def pack_data(actual_obs, expected_obs):
    """
    Only get the data we want to record
    """
    return [actual_obs[o] for o in expected_obs]


def log_data(data, filename):
    data_json = json.dumps(data)
    with open(DATA_PATH + "/" + filename, "a") as f:
        f.write(data_json + "\n")


def log_csv_data(data, filename):
    with open(DATA_PATH + "/" + filename, "a") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(data)


def save_df_to_csv(df: pd.DataFrame, filename):
    """
    Dump pandas dataframe to file
    """
    df.to_csv(DATA_PATH + "/" + filename, index=False)


def save_df_to_txt(df: pd.DataFrame, filename):
    """
    Dump pandas dataframe to file
    """
    df.to_csv(DATA_PATH + "/" + filename, sep="\t", index=False)


def save_groundtruth_to_txt(df: pd.DataFrame, filename):
    """
    Dump pandas dataframe to file
    Groundtruth file should have:
        step:    subtask:
    """
    df.to_csv(DATA_PATH + "/" + filename, sep="\t", index=False)


def format_df_to_subtask_list(df: pd.DataFrame) -> List:
    """
    Format the groundtruth dataframe into a List of subtasks.
    The incoming dataframe will rows representing a step and subtask label

    The returned list will have a compact representation of the task decomposition:
    subtask_list = [
        {'start_step': 0, 'end_step': 6, 'subtask': 0}
        {'start_step': 7, 'end_step': 19, 'subtask': 1}
        {'start_step': 20, 'end_step': 45, 'subtask': 2}
        {'start_step': 46, 'end_step': 74, 'subtask': 3}]
    ]
    """
    subtask_list = []
    n_steps = df.shape[0]

    # Append first subtasktask
    current_subtask = df.loc[0].subtask
    subtask_list.append({"start_step": 0, "subtask": current_subtask})
    n_subtasks = 1

    for step in range(n_steps):
        if current_subtask != df.loc[step]["subtask"]:
            # Finish the last subtask
            subtask_list[n_subtasks - 1]["end_step"] = step - 1

            # Change the current subtask
            current_subtask = df.loc[step]["subtask"]

            # Start a new subtask
            subtask_list.append({"start_step": step, "subtask": current_subtask})
            n_subtasks += 1

    subtask_list[n_subtasks - 1]["end_step"] = step
    return subtask_list
