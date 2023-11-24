import os
import openai
import csv
import pandas as pd
from typing import Union
import cv2
import base64

from task_decomposition.paths import DATA_TXT_PATH, DATA_FRAMES_PATH, DATA_PATH
from task_decomposition.utils.plotting import encode_image


openai.api_key = os.getenv("OPENAI_API_KEY")

from task_decomposition.utils.prompts import (
    TASK_DESCRIPTION,
    TXT_DATA_DESCRIPTION,
    FRAME_DATA_DESCRIPTION,
    ENV_DESCRIPTION,
)

MAX_TOKENS = 2000


def get_completion(prompt: str, model: str) -> Union[dict, str]:
    """"""
    messages = [{"role": "user", "content": prompt}]

    API_response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0, max_tokens=MAX_TOKENS
    )

    response = API_response.choices[0].message["content"]
    usage = API_response.usage
    return response, usage


def get_data_for_prompt(config: dict) -> Union[pd.DataFrame, str]:
    """
    Read a csv or text file and return the data frame and text
    """
    full_filename = DATA_TXT_PATH + "/" + config["txt_filename"]

    if full_filename.split(".")[-1] == "txt":
        df = pd.read_csv(full_filename, delimiter="\t", encoding="utf-8")
        with open(full_filename, "r") as f:
            text = f.read()

    elif full_filename.split(".")[-1] == "csv":
        with open(full_filename, "r") as file:
            csv_reader = csv.reader(file)
            text = "\n".join(["\t".join(row) for row in csv_reader])
        df = pd.read_csv(full_filename, encoding="utf-8")

    else:
        raise NotImplementedError

    # return every Nth row
    txt_step = config["txt_step"] if config["txt_step"] is not None else 1
    df = df.iloc[::txt_step, :]
    text = "\n".join(text.split("\n")[::txt_step])

    return df, text


def get_prompt(config: dict) -> str:
    """ """
    PROMPT = f"""{TASK_DESCRIPTION} + {ENV_DESCRIPTION}\n"""

    if not config["use_txt"] and not config["use_images"] and not config["use_video"]:
        raise ValueError("Must use at least one of txt, frames, or video.")

    if config["use_images"] and config["use_video"]:
        raise ValueError("Cannot use both images and video.")

    if config["use_txt"]:
        data_df, data_text = get_data_for_prompt(config)
        columns = str(data_df.columns.to_list())
        PROMPT += TXT_DATA_DESCRIPTION(columns)
        PROMPT += data_text

    if config["use_images"]:
        encoded_frames = [
            encode_image(os.path.join(DATA_FRAMES_PATH, config["frames"], file))
            for file in sorted(
                os.listdir(os.path.join(DATA_FRAMES_PATH, config["frames"]))
            )
        ]
        # Need to add a lambda function in the prompt messages
        # https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding
        PROMPT = [
            PROMPT + FRAME_DATA_DESCRIPTION,
            *map(
                lambda x: {"image": x, "resize": config["resize_frames"]},
                encoded_frames[0 :: config["frame_step"]],
            ),
        ]

    if config["use_video"]:
        video = cv2.VideoCapture(os.path.join(DATA_PATH, config["video_filename"]))
        base64Frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()

        # Need to add a lambda function in the prompt messages
        # https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding
        PROMPT = [
            PROMPT + FRAME_DATA_DESCRIPTION,
            *map(
                lambda x: {"image": x, "resize": 480},
                base64Frames[0 :: config["frame_step"]],
            ),
        ]

    return PROMPT
