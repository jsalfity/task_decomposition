import os

cd = os.path.dirname(os.path.realpath(__file__))

DEMO_CONFIG_YAML = cd + "/scripts/demo_config.yaml"
DATA_PATH = cd + "/data"
ROBOT_TRAJ_GROUNDTRUTH_DATA_PATH = (
    lambda env_name: cd + f"/data/robot_traj_texts/groundtruths/{env_name}"
)
ROBOT_TRAJ_TEXT_PATH = (
    lambda env_name: cd + f"/data/robot_traj_texts/statescontrols/{env_name}"
)
ROBOT_TRAJ_VIDEO_PATH = lambda env_name: cd + f"/data/robot_traj_videos/{env_name}"

LLM_QUERY_CONFIG_YAML = cd + "/analysis/query_LLM_config.yaml"
LLM_OUTPUT_PATH = (
    lambda llm_model, input_mode, env_name: cd
    + f"/data/{llm_model}_outputs/{input_mode}/{env_name}"
)
