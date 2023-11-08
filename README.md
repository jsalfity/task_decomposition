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

## Analysis using GPT
To run a call to GPT:
```py
python analysis/query_GPT.py
```

## Comparison between GPT output and groundtruth data
Not Yet Implemented 