import argparse

from task_decomposition.utils.querying import get_completion, get_prompt


def _setup_parser():
    """Set up Python's ArgumentParser with params"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--filename", type=str, default="open_door.txt")
    return parser


def run_decomposition(filename: str):
    """ """
    prompt = get_prompt(filename=filename)
    response = get_completion(prompt)
    print(response)

    return response


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    run_decomposition(filename=args.filename)


if __name__ == "__main__":
    main()
