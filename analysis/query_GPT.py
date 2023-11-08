import argparse
from datetime import datetime
import json

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
    prompt = get_prompt(filename=filename)
    response = get_completion(prompt)

    # append to json file
    if save:
        with open(GPT_OUTPUT_PATH, "a") as f:
            f.write(
                "\n"
                + json.dumps(
                    {
                        "timestamp": datetime.now().strftime("%d-%m-%Y_%H:%M"),
                        "datafile": filename,
                        "model": GPT_MODEL,
                        "response": response,
                    }
                )
                + "\n"
            )
    print(response)

    return response


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    run_decomposition(filename=args.filename, save=args.save)


if __name__ == "__main__":
    main()
