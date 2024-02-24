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
    LLM_QUERY_CONFIG_YAML,
    CUSTOM_LLM_OUTPUT_PATH,
    DATA_RAW_TXT_PATH,
    DATA_VIDEOS_PATH,
)

WAITTIME = 2  # [seconds] the waittime to make the next API call


def sleep_with_progress(n):
    for i in tqdm(range(n)):
        sleep(1)


def run_decomposition(config: dict, verbose=False):
    """ """
    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M")

    prompt = get_prompt(config)
    print(f"[{timestamp}] Querying GPT...")
    pprint(config["demo_id"])
    response, usage = get_completion(prompt, llm_model=config["llm_model"])
    if usage != {}:
        usage_cost = calculate_usage_cost(gpt_model=config["llm_model"], usage=usage)

    data = {
        "timestamp": timestamp,
        **config,
        "usage": usage if usage is not {} else None,
        "usage_cost": None,  # f"${usage_cost}" if usage is not {} else None,
        "response": response.parts[0].text,
    }

    # append to json file
    if config["save_response"]:
        savepath = CUSTOM_LLM_OUTPUT_PATH(
            config["llm_model"], config["env_name"], config["demo_id"]
        )
        if config["use_txt"] and config["use_video"]:
            savepath = savepath + "/textvideo"
        with open(savepath, "a") as f:
            f.write("\n" + json.dumps(data) + "\n")
        print("Saved to:" + savepath)

    if verbose:
        print(f"Response: ")
        print("==========================")
        print(response)
        print("==========================")
        print(f"usage: {usage}")
        print(f"This request costs: ${usage_cost}")

    return response, usage


def main():
    with open(LLM_QUERY_CONFIG_YAML, "r") as file:
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
        failed_calls = []
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
                print(f"Error calling {config['txt_filename']}. Skipping...")
                print(f"{e}")
                failed_calls.append(config["txt_filename"])

        if len(failed_calls) > 0:
            print("The following files failed to call the API:")
            pprint(failed_calls)

    # Single query means we want to run the query on a single file
    else:
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
