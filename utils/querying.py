import os
import openai
import csv
import pandas as pd
from typing import Union

from task_decomposition.paths import DATA_PATH, GPT_MODEL

openai.api_key = os.getenv("OPENAI_API_KEY")

from utils.prompts import (
    TASK_DESCRIPTION,
    DATA_DESCRIPTION,
    SCHEMA_DESCRIPTION,
)


def get_completion(prompt, model=GPT_MODEL) -> Union[dict, str]:
    messages = [{"role": "user", "content": prompt}]

    API_response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    response = API_response.choices[0].message["content"]
    usage = API_response.usage
    return response, usage


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
    Data to analyze: {data_text}
    """
