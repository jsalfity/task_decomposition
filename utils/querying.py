import os
import openai
import csv
import pandas as pd
from typing import Union

from task_decomposition.paths import DATA_PATH, GPT_MODEL

openai.api_key = os.getenv("OPENAI_API_KEY")


TASK_DESCRIPTION = """Your task is to ingest the following data and break down what occured during the robot episode into granular subtasks. Each subtask should be a sequential step that occured during the robot episode. You should identify the start and end step of each subtask. Create as many granular substasks as you see and report the list of subtasks in the following format:
```
[{'start_step': <>, 'end_step': <>, 'subtask': <name of subtask>}, {'start_step': <>, 'end_step': <>, 'subtask': <name of subtask>}, {'start_step': <>, 'end_step': <>, 'subtask': <name of subtask>}, ...]
```

Use chain of thought to break down the data into subtasks and explain your thoughts.
"""

DATA_DESCRIPTION = """The data captures a simulated episode of a robot end effector manipulating an environment. Each entry in the schema contains an observation."""

SCHEMA_DESCRIPTION = lambda columns: f"The schema is composed of {columns}."


def get_completion(prompt, model=GPT_MODEL):
    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


def get_data_for_prompt(filename: str) -> Union[pd.DataFrame, str]:
    """
    Read a csv or text file and return the data frame and text
    """
    full_filename = DATA_PATH + "/" + filename

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


def get_prompt(filename: str) -> str:
    """ """

    data_df, data_text = get_data_for_prompt(filename=filename)
    columns = str(data_df.columns.to_list())

    return f"""{TASK_DESCRIPTION}\n
    {DATA_DESCRIPTION}\n
    {SCHEMA_DESCRIPTION(columns)}\n
    Data: {data_text}
    """
