from pprint import pprint
from collections import defaultdict
from datetime import datetime
import json
import yaml

from task_decomposition.utils.offload_cost import calculate_usage_cost
from task_decomposition.utils.querying import get_completion, get_prompt
from task_decomposition.paths import CUSTOM_GPT_OUTPUT_PATH, GPT_QUERY_CONFIG_YAML


def run_decomposition(config: dict):
    """ """
    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M")

    prompt = get_prompt(config)
    print(f"[{timestamp}] Querying GPT...")
    pprint(dict(config))
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
        with open(CUSTOM_GPT_OUTPUT_PATH(config["env_name"]), "a") as f:
            f.write("\n" + json.dumps(data) + "\n")

    print(f"GPT Response: ")
    print("==========================")
    print(response)
    print("==========================")
    print(f"usage: {usage}")
    print(f"This request costs: ${usage_cost}")
    print("Saved to:" + CUSTOM_GPT_OUTPUT_PATH(config["env_name"]))
    return response, usage


def main():
    with open(GPT_QUERY_CONFIG_YAML, "r") as file:
        config = yaml.safe_load(file)

    config = defaultdict(lambda: False, config if config is not None else {})

    if config["multiple_queries"]:
        N_TXT_STEPS = config["N_TXT_STEPS"]
        N_FRAME_STEPS = config["N_FRAME_STEPS"]

        for n_txt, n_frame in zip(range(N_TXT_STEPS), range(N_FRAME_STEPS)):
            config["start_txt_idx"] = n_txt * N_TXT_STEPS // config["n_queries"] + 1
            config["end_txt_idx"] = (n_txt + 1) * N_TXT_STEPS // config["n_queries"]
            config["start_frame"] = n_frame * N_FRAME_STEPS // config["n_queries"] + 1
            config["end_frame"] = (n_frame + 1) * N_FRAME_STEPS // config["n_queries"]

            try:
                response, usage = run_decomposition(config=config)
            except Exception as e:
                print(f"{e}")

            input("Press Enter to continue...")

    else:
        response, usage = run_decomposition(config=config)


if __name__ == "__main__":
    main()
