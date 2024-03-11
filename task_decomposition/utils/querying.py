import os
import csv
from typing import Union
import pandas as pd
import cv2
import base64

# import openai
# import google.generativeai as genai
import vertexai # Gemini Pro Vision
from vertexai.generative_models import GenerativeModel as vertexai_GenerativeModel
from vertexai.generative_models import Part

from task_decomposition.paths import ROBOT_TRAJ_TEXT_PATH, ROBOT_TRAJ_VIDEO_PATH, GCLOUD_URI
from task_decomposition.constants import GPT_MAX_RESPONSE_TOKENS

# openai.api_key = os.getenv("OPENAI_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
vertexai.init(project="gen-lang-client-0368774908", location='us-central1')

from task_decomposition.utils.prompts import (
    TASK_DESCRIPTION,
    TXT_DATA_DESCRIPTION,
    FRAME_DATA_DESCRIPTION,
    ENV_DESCRIPTION,
    STACK_INCONTEXT,
    LIFT_INCONTEXT,
    DOOR_INCONTEXT,
    PICKPLACE_INCONTEXT,
)


def _get_runid_filename(config: dict, kind: str) -> str:
    env_name = config["env_name"]
    runid = config["robot_traj_runid"]
    if kind == "textual":
        return ROBOT_TRAJ_TEXT_PATH(env_name) + "/" + runid + ".txt"
    elif kind == "video":
        return ROBOT_TRAJ_VIDEO_PATH(env_name) + "/" + runid + ".mp4"
    else:
        raise ValueError("Kind must be 'textual' or 'video'")


def get_completion(prompt: str, llm_model: str) -> Union[dict, str]:
    """"""
    if llm_model == "gpt-4-vision-preview":
        messages = [{"role": "user", "content": prompt}]

        API_response = openai.ChatCompletion.create(
            model=llm_model,
            messages=messages,
            temperature=0,
            max_tokens=GPT_MAX_RESPONSE_TOKENS,
        )

        response = API_response.choices[0].message["content"]
        usage = API_response.usage

    elif llm_model == "gemini-pro":
        model = genai.GenerativeModel(llm_model)
        response = model.generate_content(prompt)
        response = response.parts[0].text
        usage = {}

    elif llm_model == "gemini-pro-vision":
        model = vertexai_GenerativeModel("gemini-pro-vision")
        response = model.generate_content(prompt)
        response = response.candidates[0].content.parts[0].text
        usage = {}
    else:
        raise NotImplementedError

    return response, usage


def get_textual_data_for_prompt(config: dict) -> Union[pd.DataFrame, str]:
    """
    Read a csv or text file and return the data frame and text
    """
    full_filename = _get_runid_filename(config, kind="textual")

    df = pd.read_csv(full_filename, delimiter="\t", encoding="utf-8")
    with open(full_filename, "r") as f:
        text = f.read()

    # return a subset of the data frame and txt file
    idx_step = config["txt_step"] if config["txt_step"] is not False else 1
    start_idx = config["start_txt_idx"] if config["start_txt_idx"] is not False else 0
    if config["end_txt_idx"] is not None or config["end_txt_idx"] > df.shape[0]:
        end_idx = -1

    df = df.iloc[start_idx:end_idx:idx_step]
    text = "\n".join(text.split("\n")[start_idx:end_idx:idx_step])

    return df, text


def get_incontext(config: dict) -> str:
    if config["env_name"] == "Stack":
        return STACK_INCONTEXT
    elif config["env_name"] == "Lift":
        return LIFT_INCONTEXT
    elif config["env_name"] == "Door":
        return DOOR_INCONTEXT
    elif config["env_name"] == "PickPlace":
        return PICKPLACE_INCONTEXT
    else:
        raise ValueError("env_name must be one of 'Stack', 'Lift', 'Door', 'PickPlace'")


def get_prompt(config: dict) -> str:
    """ """
    PROMPT = f"""{TASK_DESCRIPTION} + {ENV_DESCRIPTION(config['env_name'])}\n"""
    if config["in_context"]:
        PROMPT += get_incontext(config)

    if not config["textual_input"] and not config["video_input"]:
        raise ValueError("Must use at least one of txt, frames, or video.")

    # modify the frame step to be the same as the txt step
    frame_step = config["frame_step"] if config["frame_step"] is not False else 1
    start_frame = config["start_frame"] if config["start_frame"] is not False else 0
    end_frame = config["start_frame"] if config["end_frame"] is not False else -1

    if config["textual_input"]:
        data_df, data_text = get_textual_data_for_prompt(config)
        columns = str(data_df.columns.to_list())
        PROMPT += TXT_DATA_DESCRIPTION(columns)
        PROMPT += data_text

    if config["video_input"]:
        video_path = _get_runid_filename(config, kind="video")
        video = cv2.VideoCapture(video_path)
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
        if config["llm_model"] == "gpt-4-vision-preview":
            # Need to add a lambda function in the prompt messages
            # https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding
            PROMPT = [
                PROMPT + FRAME_DATA_DESCRIPTION,
                *map(
                    lambda x: {"image": x, "resize": config["resize_frames"]},
                    base64Frames,
                ),
            ]
        elif config["llm_model"] == "gemini-pro-vision":
            ## This is to upload a video, but its not working
            # base64video = base64.b64encode(open(video_path, "rb").read()).decode(
            #     "utf-8"
            # )
            # PROMPT = {
            #     "parts": [
            #         {
            #             "text": PROMPT + FRAME_DATA_DESCRIPTION,
            #             "inline_data": {
            #                 "mime_type": "video/mp4",
            #                 "data": base64video,
            #             },
            #         }
            #     ],
            # }

            # This is to upload images, but only accepting 16 at a time.
            # N_FRAMES_ACCEPTED = 16
            # PROMPT = {
            #     "parts": [{"text": PROMPT + FRAME_DATA_DESCRIPTION}]
            #     + [
            #         {"inline_data": {"mime_type": "image/jpeg", "data": frame_data}}
            #         for frame_data in base64Frames
            #     ]
            # }
            gcloud_filepath = GCLOUD_URI + config['env_name']+ "/" + config['video_filename']
            PROMPT = [
                Part.from_text(PROMPT + FRAME_DATA_DESCRIPTION),
                # Part.from_uri(uri=gcloud_filepath, mime_type="video/mp4")
                # Part.from_uri(uri="gs://task_decomposition_data/IMG_2788.jpeg", mime_type="image/jpeg")
                Part.from_uri(uri="gs://task_decomposition_data/Door_20240213-183948_0.mp4", mime_type="video/mp4")
            ]

    return PROMPT
