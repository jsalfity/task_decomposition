import os
import openai
import csv
import pandas as pd
from typing import Union

from task_decomposition.paths import DATA_TXT_PATH, DATA_FRAMES_PATH
from task_decomposition.utils.plotting import encode_image


openai.api_key = os.getenv("OPENAI_API_KEY")

from task_decomposition.utils.prompts import (
    TASK_DESCRIPTION,
    TXT_DATA_DESCRIPTION,
    FRAME_DATA_DESCRIPTION,
)


def get_completion(prompt: str, model: str) -> Union[dict, str]:
    """"""
    messages = [{"role": "user", "content": prompt}]

    API_response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0, max_tokens=1000
    )

    response = API_response.choices[0].message["content"]
    usage = API_response.usage
    return response, usage


def get_data_for_prompt(filename: str) -> Union[pd.DataFrame, str]:
    """
    Read a csv or text file and return the data frame and text
    """
    full_filename = DATA_TXT_PATH + "/" + filename

    if filename.split(".")[-1] == "txt":
        df = pd.read_csv(full_filename, delimiter="\t", encoding="utf-8")
        with open(full_filename, "r") as f:
            text = f.read()

    elif filename.split(".")[-1] == "csv":
        with open(full_filename, "r") as file:
            csv_reader = csv.reader(file)
            text = "\n".join(["\t".join(row) for row in csv_reader])
        df = pd.read_csv(full_filename, encoding="utf-8")

    else:
        raise NotImplementedError

    return df, text


def get_prompt(config: dict) -> str:
    """ """
    PROMPT = f"""{TASK_DESCRIPTION}\n"""

    if config["use_txt"]:
        data_df, data_text = get_data_for_prompt(filename=config["txt_filename"])
        columns = str(data_df.columns.to_list())
        PROMPT += TXT_DATA_DESCRIPTION(columns)
        PROMPT += data_text

    if config["use_frames"]:
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

    return PROMPT
