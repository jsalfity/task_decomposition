# Task Decomposition
Repo for testing out the task decomposition project

## Prerequisites
Download the mujoco binaries from [here](https://github.com/google-deepmind/mujoco/releases).
Place in `~/.mujoco/mujoco<>/` folder. Install mujoco via pip
```sh
pip install mujoco
```

Install [`robosuite`](https://robosuite.ai/docs/installation.html):
```sh
pip install robosuite
```

Install this package
```sh
pip install -e .
```

## Running robosuite simulation and generating data
### Running robosuite based simulations with a predefined state machine
Configure which environment to run in the `/scripts/demo_config.yaml`. 
Currently we have 4 environments: "Stack", "Lift", "Door", "PickPlace".
Follow the uncommented lines in `/scripts/demo_config.yaml` to set the correct fields.
Run the data generation script

```sh
python scripts/run_demo.py
```

### Running downloaded demo_v141.hdf5 files and generating data
Go to the robomimic site to download data: https://robomimic.github.io/docs/datasets/robomimic_v0.1.html.
(Note that this currently only seems to work with Safari broswer).
Place the downloaded hdf5 files in the respective `data/robomimic` folder.
Run the data generation script `/scripts/record_robomimic_data.py` with command line args that specificy the path to the `demo_v141.hdf5` file, the number of demos to run, and whether to `save_txt` or `save_video`.
The script will automatically extract the specific `env_name` and place the text and videos in the respective `data/txt` or `data/video` folders.

Example:
```sh
python scripts/record_robomimic_data.py --dataset path/to/robomimic/demo_v141.hdf5 --num_demos 1 --save_txt 1 --save_video 1
```

(Hacky) To modify the columns recorded in the txt file, modify the `get_data_to_record(env_name: str)` and `def query_sim_for_data(env, desired_obs):` functions.

## Querying an LLM
(Assuming you have set up OpenAI and generativeai python packages and set the API keys as environment variables, i.e. `OPENAI_API_KEY` and `GENERATIVEAI_API_KEY`)

The configuration file for the LLM is in `config/query_LLM_config.yaml`.
The following are options for the LLM model:
- `gpt-4-vision-preview`
- `gpt-4-1106-preview`
- `gemini-pro`
- `gemini-pro-vision`

The following are options for the environment:
- `Stack`
- `Lift`
- `Door`
- `PickPlace`
- `ToolHang`
- `NutAssemblySquare`

The following are options for input modalities to include in the LLM prompt query, which can be used in combination with each other:
- `use_txt` 
- `use_video`

Set the configuration file for the LLM and the environment to query in `config/query_LLM_config.yaml`.

To run the LLM, run the following command:
```sh
python analysis/query_LLM.py
```

## Comparison between GPT output and groundtruth data (WIP)
See `analysis/compare_GPT_groundtruth.ipynb` to generate plots show in the report.
