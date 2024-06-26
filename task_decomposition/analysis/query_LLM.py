import os
from pprint import pprint
from collections import defaultdict
from datetime import datetime, timedelta
import json
import yaml
from tqdm import tqdm
from time import sleep

from task_decomposition.utils.offload_cost import calculate_usage_cost
from task_decomposition.utils.querying import get_completion, get_prompt
from task_decomposition.paths import (
    LLM_QUERY_CONFIG_YAML,
    ROBOT_TRAJ_TEXT_PATH,
    ROBOT_TRAJ_VIDEO_PATH,
    LLM_OUTPUT_PATH,
)
from task_decomposition.constants import (
    GPT_MAX_TOKENS_PER_MINUE,
    GPT_MAX_RESPONSE_TOKENS,
    WAITTIME,
)


def sleep_with_progress(n):
    for i in tqdm(range(n)):
        sleep(1)


def _get_input_mode(config: dict) -> str:
    if config["textual_input"] and config["video_input"]:
        return "textualvideo"
    elif config["textual_input"]:
        return "textual"
    elif config["video_input"]:
        return "video"
    else:
        raise ValueError("One of textual_input or video_input must be True")


def run_decomposition(config: dict, verbose=False):
    """ """
    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M")

    prompt = get_prompt(config)
    print(f"[{timestamp}] Querying {config['llm_model']}...")
    print(config["robot_traj_runid"])

    ######################
    #### Call the LLM ####
    response, usage = get_completion(prompt, llm_model=config["llm_model"])
    ######################

    usage_cost = calculate_usage_cost(llm_model=config["llm_model"], usage=usage)

    data = {
        "timestamp": timestamp,
        **config,
        "usage": usage,
        "usage_cost": usage_cost,
        "response": response,
    }

    # append to json file
    if config["save_response"]:
        input_mode = _get_input_mode(config)
        savepath = LLM_OUTPUT_PATH(
            llm_model=config["llm_model"],
            input_mode=input_mode,
            env_name=config["env_name"],
            in_context=config["in_context"],
        )
        savepath = savepath + "/" + config["robot_traj_runid"] + ".json"
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

        # Get all files in the directory
        textual_files = os.listdir(ROBOT_TRAJ_TEXT_PATH(config["env_name"]))
        video_files = os.listdir(ROBOT_TRAJ_VIDEO_PATH(config["env_name"]))

        # Remove .git* files
        textual_files = [f for f in textual_files if not f.startswith(".git")]
        video_files = [f for f in video_files if not f.startswith(".git")]

        assert len(textual_files) == len(video_files)
        textual_files.sort(), video_files.sort()

        # For mulitiple files, we want to run the same query on all files
        # Some will fail, so keep track of them
        failed_calls = []
        last_API_call_timestamp = datetime.now() - timedelta(seconds=1000)
        last_prompt_tokens = 5000  # upper bound guess
        tokens_used_this_minute = 0
        for txt_file, video_file in zip(textual_files, video_files):
            assert (
                txt_file.split(".")[0] == video_file.split(".")[0]
            ), f"Inconsistent run_ids, txt runid: {txt_file}, video runid: {video_file}"
            config["txt_filename"] = txt_file if config["textual_input"] else False
            config["video_filename"] = video_file if config["video_input"] else False
            config["robot_traj_runid"] = txt_file.split(".")[0]

            # We cannot go over the GPT_MAX_TOKENS_PER_MINUE
            if config["llm_model"] == "gpt-4-vision-preview" and (
                tokens_used_this_minute + last_prompt_tokens + GPT_MAX_RESPONSE_TOKENS
                > GPT_MAX_TOKENS_PER_MINUE
            ):
                print(
                    f"Used {tokens_used_this_minute} tokens this minute. Waiting for the next minute to start."
                )
                sleep_with_progress(WAITTIME)
                tokens_used_this_minute = 0
            # # check if we need to wait before making the next API call
            # timetowait = (
            #     WAITTIME - (datetime.now() - last_API_call_timestamp).total_seconds()
            # )
            # timetowait = int(timetowait) + 1
            # if timetowait < WAITTIME:
            #     print(f"Waiting {timetowait} seconds before making the next API call.")
            #     sleep_with_progress(timetowait)

            # last_API_call_timestamp = datetime.now()
            try:
                response, usage = run_decomposition(config)
                if usage != {}:
                    tokens_used_this_minute += usage["total_tokens"]
                    last_prompt_tokens = usage["prompt_tokens"]
            except Exception as e:
                print(f"Error calling {config['robot_traj_runid']}. Skipping...")
                print(f"{e}")
                failed_calls.append(config["robot_traj_runid"])

        if len(failed_calls) > 0:
            print("The following files failed to call the API:")
            pprint(failed_calls)

    # Single query means we want to run the query on a single file
    else:
        video_file = config["video_filename"]
        if config["textual_input_file"] and config["video_input_file"]:
            assert txt_file.split(".")[0] == video_file.split(".")[0]
        config["robot_traj_runid"] = video_file.split(".")[0]
        try:
            response, usage = run_decomposition(config=config)
        except Exception as e:
            print(f"{e}")


if __name__ == "__main__":
    main()
