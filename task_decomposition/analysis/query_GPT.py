import os
from pprint import pprint
from collections import defaultdict
from datetime import datetime, timedelta
import json
import yaml
import tqdm
from time import sleep

from task_decomposition.utils.offload_cost import calculate_usage_cost
from task_decomposition.utils.querying import get_completion, get_prompt
from tqdm import tqdm
from task_decomposition.paths import (
    CUSTOM_GPT_OUTPUT_PATH,
    GPT_QUERY_CONFIG_YAML,
    DATA_RAW_TXT_PATH,
    DATA_VIDEOS_PATH,
)


def sleep_with_progress(n):
    for i in tqdm(range(n)):
        sleep(1)


def run_decomposition(config: dict):
    """ """
    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M")

    prompt = get_prompt(config)
    print(f"[{timestamp}] Querying GPT...")
    pprint(config["demo_id"])
    response, usage = get_completion(prompt, model=config["gpt_model"])
    usage_cost = calculate_usage_cost(gpt_model=config["gpt_model"], usage=usage)

    data = {
        "timestamp": timestamp,
        **config,
        "usage": usage,
        "usage_cost": f"${usage_cost}",
        "response": response,
    }

    # append to json file
    if config["save_response"]:
        with open(CUSTOM_GPT_OUTPUT_PATH(config["demo_id"]), "a") as f:
            f.write("\n" + json.dumps(data) + "\n")

    print(f"GPT Response: ")
    print("==========================")
    print(response)
    print("==========================")
    print(f"usage: {usage}")
    print(f"This request costs: ${usage_cost}")
    if config["save_response"]:
        print("Saved to:" + CUSTOM_GPT_OUTPUT_PATH(config["demo_id"]))
    return response, usage


WAITTIME = 60


def main():
    with open(GPT_QUERY_CONFIG_YAML, "r") as file:
        config = yaml.safe_load(file)

    config = defaultdict(lambda: False, config if config is not None else {})

    # Multiple queries means we want to run the same query on multiple files
    if config["multiple_queries"]:
        # list all files with with the same prefix in DATA_TXT and DATA_VIDEOS
        txt_files = os.listdir(DATA_RAW_TXT_PATH)
        env_name_txt_files = [f for f in txt_files if config["env_name"] in f]
        video_files = os.listdir(DATA_VIDEOS_PATH)
        env_name_video_files = [f for f in video_files if config["env_name"] in f]

        assert len(env_name_txt_files) == len(env_name_video_files)
        env_name_video_files.sort(), env_name_txt_files.sort()

        # For mulitiple files, we want to run the same query on all files
        last_API_call_timestamp = datetime.now() - timedelta(seconds=1000)
        for txt_file, video_file in zip(env_name_txt_files, env_name_video_files):
            assert txt_file.split(".")[0] == video_file.split(".")[0]
            config["txt_filename"] = txt_file
            config["video_filename"] = video_file
            config["demo_id"] = txt_file.split(".")[0]

            # check if we need to wait before making the next API call
            timetowait = (
                WAITTIME - (datetime.now() - last_API_call_timestamp).total_seconds()
            )
            timetowait = int(timetowait) + 1
            if timetowait < WAITTIME:
                print(f"Waiting {timetowait} seconds before making the next API call.")
                sleep_with_progress(timetowait)

            last_API_call_timestamp = datetime.now()
            try:
                response, usage = run_decomposition(config)
            except Exception as e:
                input("Something went wrong. Press enter to continue.")
                print(f"{e}")

    # Single query means we want to run the query on a single file
    else:
        txt_file = config["txt_filename"]
        video_file = config["video_filename"]
        if config["use_txt_file"] and config["use_video_file"]:
            assert txt_file.split(".")[0] == video_file.split(".")[0]
        config["demo_id"] = video_file.split(".")[0]
        try:
            response, usage = run_decomposition(config=config)
        except Exception as e:
            print(f"{e}")


if __name__ == "__main__":
    main()
