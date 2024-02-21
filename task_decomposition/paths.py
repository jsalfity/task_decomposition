import os

cd = os.path.dirname(os.path.realpath(__file__))

DEMO_CONFIG_YAML = cd + "/scripts/demo_config.yaml"
DATA_PATH = cd + "/data"
DATA_GT_TXT_PATH = cd + "/data/txt/groundtruths"
DATA_RAW_TXT_PATH = cd + "/data/txt/raw"
DATA_VIDEOS_PATH = cd + "/data/videos"

GPT_QUERY_CONFIG_YAML = cd + "/analysis/gpt_query_config.yaml"
GPT_OUTPUT_PATH = cd + "/data/gpt_outputs"
CUSTOM_GPT_OUTPUT_PATH = lambda env, runid: cd + f"/data/gpt_outputs/{env}/{runid}.json"
