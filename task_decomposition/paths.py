import os

cd = os.path.dirname(os.path.realpath(__file__))

DEMO_CONFIG_YAML = cd + "/scripts/demo_config.yaml"
DATA_PATH = cd + "/data"
DATA_TXT_PATH = cd + "/data/txt"
DATA_VIDEOS_PATH = cd + "/data/videos"

GPT_QUERY_CONFIG_YAML = cd + "/analysis/gpt_query_config.yaml"
GPT_OUTPUT_PATH = cd + "/data/gpt_outputs.json"
CUSTOM_GPT_OUTPUT_PATH = lambda x: cd + f"/data/gpt_outputs/{x}_gpt_output.json"
