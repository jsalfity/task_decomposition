# Task Decomposition
Repo for testing out the task decomposition project

## Prerequisites
Download the mujoco binaries from [here](https://github.com/google-deepmind/mujoco/releases).
Place in `~/.mujoco/mujoco<>/` folder. Install mujoco via pip
```
pip install mujoco
```

Install [`robosuite`](https://robosuite.ai/docs/installation.html):
```
pip install robosuite
```

Install this package
```
pip install -e .
```

## Running robosuite simulation and generating data
Running scripts is in `/scripts` folder.
```py
python scripts/run_open_door.py
```
Check appropriate command line arguments for saving data.

## Analysis using GPT
To run a call to GPT, set the config in `analysis/gpt_query_config.yaml` and run
```py
python analysis/query_GPT.py
```

## Comparison between GPT output and groundtruth data
See `analysis/compare_GPT_groundtruth.ipynb` to generate plots show in the report.
