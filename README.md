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

## Running
Running scripts is in `/scripts` folder.
```py
python scripts/run_open_door.py
```

To run a call to GPT (Not fully implemented yet):
```py
python analysis/query_GPT.py
```
