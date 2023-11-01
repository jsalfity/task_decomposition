import json
import csv
import pandas as pd

import robosuite as suite

# from matplotlib import animation
# import matplotlib.pyplot as plt

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
    df.to_csv(DATA_PATH + "/" + filename)


def save_df_to_txt(df: pd.DataFrame, filename):
    """
    Dump pandas dataframe to file
    """
    df.to_csv(DATA_PATH + "/" + filename, sep="\t")
