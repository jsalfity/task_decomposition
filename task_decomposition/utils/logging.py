import cv2
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import imageio

import robosuite as suite

from task_decomposition.paths import DATA_TXT_PATH, DATA_FRAMES_PATH

FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_env(config):
    return suite.make(**config)


def save_df_to_txt(df: pd.DataFrame, filename):
    """
    Dump pandas dataframe to file
    """
    filename = filename + ".txt" if filename.split(".")[-1] != "txt" else None
    df.to_csv(DATA_TXT_PATH + "/" + filename, sep="\t", index=False)


def save_video_fn(frames: List, filename: str):
    """
    Save a video of the frames
    """
    full_filename = DATA_FRAMES_PATH + "/" + filename + "_annotated.mp4"
    video_writer = imageio.get_writer(full_filename, fps=20)
    for idx, frame in enumerate(frames):
        # annotate videos with step number
        frame = frame.astype(np.uint8)
        cv2.putText(
            frame, f"step number: {idx}", (10, 30), FONT, 1, (0, 0, 255), 2, cv2.LINE_AA
        )

        video_writer.append_data(frame)

    video_writer.close()
    return
