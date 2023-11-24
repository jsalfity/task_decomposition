from pprint import pprint
from collections import defaultdict
from datetime import datetime
import json
import yaml

from task_decomposition.utils.offload_cost import calculate_usage_cost
from task_decomposition.utils.querying import get_completion, get_prompt
from task_decomposition.paths import GPT_OUTPUT_PATH, GPT_QUERY_CONFIG_YAML


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
        with open(GPT_OUTPUT_PATH, "a") as f:
            f.write("\n" + json.dumps(data) + "\n")

    print(f"GPT Response: ")
    print("==========================")
    print(response)
    print("==========================")
    print(f"This request costs: ${usage_cost}")
    print(f"Saved to {GPT_OUTPUT_PATH}")
    return response


def main():
    with open(GPT_QUERY_CONFIG_YAML, "r") as file:
        config = yaml.safe_load(file)

    config = defaultdict(lambda: False, config if config is not None else {})

    run_decomposition(config=config)


if __name__ == "__main__":
    main()
