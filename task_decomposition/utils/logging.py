import cv2
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import imageio

import robosuite as suite

from task_decomposition.paths import (
    DATA_GT_TXT_PATH,
    DATA_RAW_TXT_PATH,
    DATA_VIDEOS_PATH,
    CUSTOM_GPT_OUTPUT_PATH,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_env(config):
    return suite.make(**config)


def save_df_to_txt(df: pd.DataFrame, filename, kind):
    """
    Dump pandas dataframe to file
    """
    if kind == "raw":
        savepath = DATA_RAW_TXT_PATH
    elif kind == "gt":
        savepath = DATA_GT_TXT_PATH
    else:
        raise ValueError("Category must be 'raw' or 'gt'")

    filename = filename + ".txt" if filename.split(".")[-1] != "txt" else None
    df.to_csv(savepath + "/" + filename, sep="\t", index=False)


def save_video_fn(frames: List, filename: str):
    """
    Save video of the frames and annotate each frame with step number.
    """
    full_filename = DATA_VIDEOS_PATH + "/" + filename + ".mp4"
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


def get_annotation(idx: int, filename: str):
    """
    get the annotation for each idx in the video according to the GPT response
    GPT responses are formatted:
    [
        (0, 12, "approach the cube", 1.1),
        (13, 13, "activate gripper", 1.2),
        (14, 14, "maintain gripper activation", 1.3),
        (15, 20, "lift the cube slightly", 2.1),
        (21, 38, "hold the cube steady", 2.2)
    ]
    """
    response = CUSTOM_GPT_OUTPUT_PATH[filename]
    for start, end, annotation, stage in response:
        if idx >= start and idx <= end:
            return annotation
    return ""


def gpt_annotate_video_fn(frames: List, filename: str):
    """
    Save a video of the frames
    """
    full_filename = DATA_VIDEOS_PATH + "/" + filename + "_GPT_annotated.mp4"
    video_writer = imageio.get_writer(full_filename, fps=20)
    for idx, frame in enumerate(frames):
        # annotate videos with step number
        frame = frame.astype(np.uint8)
        cv2.putText(
            frame,
            f"step number: {idx}",
            (10, 30),
            FONT,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # annotate videos with GPT response
        annotation = get_annotation(idx, filename)
        cv2.putText(frame, annotation, (10, 60), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        video_writer.append_data(frame)

    video_writer.close()
    return
