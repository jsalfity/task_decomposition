# Task Decomposition
This repository is the code base for the IROS 2024 submission: [Temporal and Semantic Evaluation Metrics for Foundation Models in Post-Hoc Analysis of Robotic Sub-tasks](https://arxiv.org/abs/2403.17238) by Jonathan Salfity, Selma Wanna, Minkyu Choi, and Mitch Pryor. The corresponding author is Jonathan Salfity (j [dot] salfity [at] utexas [dot] edu).

The code base is divided into the following sections:
- Data Generation through Robosuite simulations and Finite State machine (FSM) implementation is in  (`scripts/`). The data is stored in `data/` as `.txt` and/or `.mp4` files upon generation, depending on the config file. For data used in the original paper submission, contact j [dot] salfity [at] utexas [dot] edu for access.
- Querying a Foundation Model (FM) for sub-task decomposition is in `(analysis/query_LLM.py)`
- Analysis of the FM output, comparison with groundtruth data, plot and table generation is in `(analysis/main_metrics_calculations.ipynb)`
- The main metrics (temporal and semantic) calculations are in `analysis/comparisons.py`, specifically the `get_subtask_similarity` function.

Supporting functions including API call, prompt building, in-context learning examples, and random baseline implementation are found in `/utils`.

## Prerequisites
Install this package
```sh
pip install -e .
```

### To run the robosuite simulations
Download the mujoco binaries from [here](https://github.com/google-deepmind/mujoco/releases).
Place in `~/.mujoco/mujoco<>/` folder. Install mujoco via pip
```sh
pip install mujoco
```

Install [`robosuite`](https://robosuite.ai/docs/installation.html):
```sh
pip install robosuite
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
(This data is not used in the IROS paper)
Go to the robomimic site to download data: https://robomimic.github.io/docs/datasets/robomimic_v0.1.html.
(Note that this currently only seems to work with Safari broswer).
Place the downloaded hdf5 files in the respective `data/robomimic` folder.
Run the data generation script `/scripts/record_robomimic_data.py` with command line args that specificy the path to the `demo_v141.hdf5` file, the number of demos to run, and whether to `save_txt` or `save_video`.
The script will automatically extract the specific `env_name` and place the text and videos in the respective `data/txt` or `data/video` folders.

Example:
```sh
python scripts/record_robomimic_data.py --dataset path/to/robomimic/demo_v141.hdf5 --num_demos 1 --save_txt 1 --save_video 1
```

## Querying an FM
Assuming you have set up OpenAI and generativeai python packages and set the API keys as environment variables, i.e. `OPENAI_API_KEY` and `GOOGLE_API_KEY`.

The configuration file for the LLM is in `config/query_LLM_config.yaml`.
The following are options for the FM model:
- `gpt-4-vision-preview`
- `gpt-4-1106-preview`
- `gemini-pro`
- `gemini-pro-vision` (Not in this repo, called via Google Cloud Vertix AI API)

All states in each environment are in R^3 and represent the x-y-z position of the object in the environment. All actions are in R^7, using the Robosuites `OSC_POSE` controller.
The following are options for the environment:
- `Door`:
  - States: `robot0_eef_pos`, `door_pos`, `handle_pos`, `door_to_eef_pos`, `handle_to_eef_pos`.
- `Lift`
  - States: `robot0_eef_pos`, `cube_pos`, `gripper_to_cube`.
- `PickPlace`
  - States: `robot0_eef_pos`, `Can_pos`, `Can_to_robot0_eef_pos`.
- `Stack`
  - States: `robot0_eef_pos`, `cubeA_pos`, `cubeB_pos`, `gripper_to_cubeA`, `gripper_to_cubeB`, `cubeA_to_cubeB`.


The following are options for input modalities and in-context learning examples to include in the LLM prompt query, which can be used in combination with each other:
- `textual_input`: (True or False) 
- `video_input`: (True or False)
- `in_context`: (True or False)

To run the LLM, run the following command:
```sh
python analysis/query_LLM.py
```

## Comparison between FM output and groundtruth data
See `analysis/main_metrics_calculations.ipynb` to generate plots show in the paper.