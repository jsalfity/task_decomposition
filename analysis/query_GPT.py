# %%
import os
import json
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
TOKEN_LIMIT = 3800

# %%
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def read_csv(filename):
    with open(filename, "r") as f:
        return f.read()

# def convert_json_to_str(filename):
#     with open(filename, "r") as f:
#         data = json.load(f)
#     return "\n".join([str(d) for d in data])

def read_file_to_string(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data
# %%
DATA_FOLDER = "/home/js222238/dev/robosuite/data"
# DATA_FILE = "task_breakdown_sparse_schema.csv"
DATA_FILE = "task_breakdown3.json"

# %%
## READ DATA
# episode_data = read_csv(os.path.join(DATA_FOLDER, DATA_FILE))
episode_data = read_file_to_string(os.path.join(DATA_FOLDER, DATA_FILE))

# %%
## GENERATE PROMPT
schema = "{{step, robot0_eef_pos, cube_pos, gripper_to_cube_pos, action, grasp, reward}}"


task_description = f"""
Your task is to ingest the following data and describe what occurred during a robot episode.
Categorize the data into subtasks. Create substasks and their corresponding labels based on what you think is occuring during each subtask. Use the following format:
```
Subtask: <subtask name>
Description: <description of subtask>
Steps: [<the first step number of the subtask>, <the final step number of the subtask>]
```
"""

data_description = """
The data captures a simulated episode of a robot end effector manipulating an environment. Each entry in the schema contains an observation. There is missing data because the step numbers are only when there is a change in action. Note that you need to incorporate the 'previous_output' into your description of the current step.
"""

schema_description = f"""
Each observation has the schema: {schema}
"""

previous_output = "No previous output, this is the first step."

def get_prompt(data_chunk: str, previous_output: str) -> str:
    return f"""{task_description}\n
    {data_description}\n
    {schema_description}\n
    previous_output: {previous_output}\n
    Data: {data_chunk}
    """

# %%
def parse_data(data):
    "Parse the csv file into data chunks smaller than the token limit"
    data_chunks = []
    data_chunk = ""
    for line in data.split("\n"):
        if len(data_chunk) + len(line) < TOKEN_LIMIT:
            data_chunk += line + "\n"
        else:
            data_chunks.append(data_chunk)
            data_chunk = ""
    return data_chunks

# %%
data_chunks = parse_data(episode_data)

# %%
for data_chunk in data_chunks:
    prompt = get_prompt(data_chunk, previous_output)
    previous_output = get_completion(prompt)
    print(previous_output)

# %%
response = get_completion(prompt)

# %%
print(response)

# %%
response=get_completion("are you aware of what just in the previuos message?")
print(response)

# %%



