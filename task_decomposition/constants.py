# For querying
GPT_MAX_TOKENS_PER_MINUE = 40000
MAX_RESPONSE_TOKENS = 2500
WAITTIME = 60

START_STEP_IDX = 0
END_STEP_IDX = 1
DESCRIPTION_IDX = 2

TEMPORAL_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.5

ENVIRONMENT_NAMES = [
    "Door",
    "Lift",
    "NutAssemblySquare",
    "PickPlace",
    "Stack",
]  # ToolHang Not implementeed yet.

POSSIBLE_SUBTASKS = [
    "Pick up object from a table",
    "Move object from one location to another",
    "Place object on a shelf",
    "Sort objects by color",
    "Remove object from a container",
    "Open a door",
    "Close a door",
    "Turn on a light",
    "Turn off a light",
    "Set a table",
    "Clear a table",
    "Mount object on wall",
    "Remove object from wall",
    "Unload objects from a truck",
    "Unload objects from a conveyor belt",
    "Load objects onto conveyor belt",
    "Pack objects into a box",
    "Unpack objects from a box",
    "Prepare a meal",
    "Serve a meal",
]

USE_MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
