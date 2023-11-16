import json
import csv
import cv2
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import numpy as np

import robosuite as suite

from task_decomposition.paths import DATA_TXT_PATH, DATA_FRAMES_PATH


def make_env(config):
    return suite.make(**config)


def save_df_to_txt(df: pd.DataFrame, filename):
    """
    Dump pandas dataframe to file
    """
    filename = filename + ".txt" if filename.split(".")[-1] != "txt" else None
    df.to_csv(DATA_TXT_PATH + "/" + filename, sep="\t", index=False)


def save_frames_fn(frames: List, filename: str):
    """
    Write the step number on images and save to folder
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
    """
    for idx, frame in enumerate(frames):
        # Add text to the image
        frame = frame.astype(np.uint8)
        cv2.putText(
            img=frame,
            text=f"step number: {idx}",
            org=(0, 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            # thickness=2,
        )
        plt.imsave(f"{DATA_FRAMES_PATH}/{filename}/{idx}.png", frame)
    return
