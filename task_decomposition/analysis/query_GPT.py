import argparse
from datetime import datetime
import json

from task_decomposition.utils.offload_cost import usage_cost
from task_decomposition.utils.querying import get_completion, get_prompt
from task_decomposition.paths import GPT_OUTPUT_PATH, GPT_MODEL


def _setup_parser():
    """Set up Python's ArgumentParser with params"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--filename", type=str, default="open_door.txt")
    parser.add_argument("--save", type=int, default=1)
    return parser


def run_decomposition(filename: str, save: bool = True):
    """ """
    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M")
    prompt = get_prompt(filename=filename)
    print(f"[{timestamp}] Querying GPT...\n")
    response, usage = get_completion(prompt)

    data = {
        "timestamp": timestamp,
        "datafile": filename,
        "model": GPT_MODEL,
        "usage": usage,
        "usage_cost": f"${usage_cost(gpt_model=GPT_MODEL, usage=usage)}",
        "response": response,
    }

    # append to json file
    if save:
        with open(GPT_OUTPUT_PATH, "a") as f:
            f.write("\n" + json.dumps(data) + "\n")

    print(f"GPT Response: ")
    print(response)
    print("\n")
    print(f"Saved to {GPT_OUTPUT_PATH}")
    return response


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    run_decomposition(filename=args.filename, save=args.save)


if __name__ == "__main__":
    main()
