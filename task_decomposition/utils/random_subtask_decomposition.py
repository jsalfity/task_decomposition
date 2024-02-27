import os
import random

from task_decomposition.constants import POSSIBLE_SUBTASKS, SHAKESPEARE_SUBTASKS
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def ask_gemini_pro(prompt: str):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    response = response.parts[0].text
    print(response)


def get_random_subtask_decomposition(N: int):
    """
    This function returns a list of n random subtasks from the POSSIBLE_SUBTASKS list.
    Each subtask is a tuple with (START_STEP_IDX, END_STEP_IDX, DESCRIPTION_IDX)
    """
    # create a subtask decomposition, with random start and end steps,
    # until we've reached the length limit
    if POSSIBLE_SUBTASKS == []:
        assert f"POSSIBLE_SUBTASKS in task_decomposition/constants.py is empty"

    subtask_decomposition = []
    i = 0
    while i <= N:
        start = i
        end = random.randint(start + 1, N - 1) if i < N - 1 else N
        subtask = (start, end, random.choice(POSSIBLE_SUBTASKS))
        subtask_decomposition.append(subtask)
        i = end + 1

    return subtask_decomposition


if __name__ == "__main__":
    prompt = "Generate 20 unique subtask descriptions for a robot manipulation task. Report back with `possible_subtasks = [...]`"
    ask_gemini_pro(prompt)
