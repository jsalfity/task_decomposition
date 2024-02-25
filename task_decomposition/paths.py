import os

cd = os.path.dirname(os.path.realpath(__file__))

DEMO_CONFIG_YAML = cd + "/scripts/demo_config.yaml"
DATA_PATH = cd + "/data"
DATA_GT_TXT_PATH = lambda env_name: cd + f"/data/txt/groundtruths/{env_name}"
DATA_RAW_TXT_PATH = cd + "/data/txt/raw"
DATA_VIDEOS_PATH = cd + "/data/videos"

# GPT_QUERY_CONFIG_YAML = cd + "/analysis/gpt_query_config.yaml"
# GPT_OUTPUT_PATH = cd + "/data/gpt_outputs"
# CUSTOM_GPT_OUTPUT_PATH = lambda env, runid: cd + f"/data/gpt_outputs/{env}/{runid}.json"

LLM_QUERY_CONFIG_YAML = cd + "/analysis/query_LLM_config.yaml"
LLM_OUTPUT_PATH = (
    lambda llm_model, input_mode, env_name: cd
    + f"/data/{llm_model}_outputs/{input_mode}/{env_name}"
)
CUSTOM_LLM_OUTPUT_PATH = (
    lambda llm, env, runid: cd + f"/data/{llm}_outputs/{env}/{runid}.json"
)
