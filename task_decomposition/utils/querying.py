import os
import csv
from typing import Union
import pandas as pd
import cv2
import base64

import openai
import google.generativeai as genai

from task_decomposition.paths import DATA_RAW_TXT_PATH, DATA_VIDEOS_PATH


openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

from task_decomposition.utils.prompts import (
    TASK_DESCRIPTION,
    TXT_DATA_DESCRIPTION,
    FRAME_DATA_DESCRIPTION,
    ENV_DESCRIPTION,
)

MAX_TOKENS = 20000


def get_completion(prompt: str, llm_model: str) -> Union[dict, str]:
    """"""
    if llm_model == "gpt-4-vision-preview":
        messages = [{"role": "user", "content": prompt}]

        API_response = openai.ChatCompletion.create(
            model=llm_model, messages=messages, temperature=0, max_tokens=MAX_TOKENS
        )

        response = API_response.choices[0].message["content"]
        usage = API_response.usage

    elif llm_model == "gemini-pro":
        model = genai.GenerativeModel(llm_model)
        response = model.generate_content(prompt)
        usage = {}

    elif llm_model == "gemini-pro-vision":
        response = "Not implemented"
        usage = {}

    return response, usage


def get_data_for_prompt(config: dict) -> Union[pd.DataFrame, str]:
    """
    Read a csv or text file and return the data frame and text
    """
    full_filename = DATA_RAW_TXT_PATH + "/" + config["txt_filename"]

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

    # return a subset of the data frame and txt file
    idx_step = config["txt_step"] if config["txt_step"] is not False else 1
    start_idx = config["start_txt_idx"] if config["start_txt_idx"] is not False else 0
    if config["end_txt_idx"] is not None or config["end_txt_idx"] > df.shape[0]:
        end_idx = -1

    df = df.iloc[start_idx:end_idx:idx_step]
    text = "\n".join(text.split("\n")[start_idx:end_idx:idx_step])

    return df, text


def get_prompt(config: dict) -> str:
    """ """
    PROMPT = f"""{TASK_DESCRIPTION} + {ENV_DESCRIPTION(config['env_name'])}\n"""

    if not config["use_txt"] and not config["use_images"] and not config["use_video"]:
        raise ValueError("Must use at least one of txt, frames, or video.")

    # modify the frame step to be the same as the txt step
    frame_step = config["frame_step"] if config["frame_step"] is not False else 1
    start_frame = config["start_frame"] if config["start_frame"] is not False else 0
    end_frame = config["start_frame"] if config["end_frame"] is not False else -1

    if config["use_txt"]:
        data_df, data_text = get_data_for_prompt(config)
        columns = str(data_df.columns.to_list())
        PROMPT += TXT_DATA_DESCRIPTION(columns)
        PROMPT += data_text

    if config["use_video"]:
        video = cv2.VideoCapture(
            os.path.join(DATA_VIDEOS_PATH, config["video_filename"])
        )
        base64Frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()

        # slices the frames
        base64Frames = base64Frames[start_frame:end_frame:frame_step]
        # Need to add a lambda function in the prompt messages
        # https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding
        PROMPT = [
            PROMPT + FRAME_DATA_DESCRIPTION,
            *map(
                lambda x: {"image": x, "resize": config["resize_frames"]},
                base64Frames,
            ),
        ]

    return PROMPT
