# For querying
GPT_MAX_TOKENS_PER_MINUE = 40000
GPT_MAX_RESPONSE_TOKENS = 1200
WAITTIME = 60

START_STEP_IDX = 0
END_STEP_IDX = 1
DESCRIPTION_IDX = 2

TEMPORAL_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.5

ENVIRONMENT_NAMES = [
    "Door",
    "Lift",
    "PickPlace",
    "Stack",
    # "NutAssemblySquare",
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

SHAKESPEARE_SUBTASKS = [
    "To be, or not to be: that is the question.",
    "This above all: to thine own self be true.",
    "There are more things in heaven and earth, Horatio, than are dreamt of in your philosophy.",
    "Something is rotten in the state of Denmark.",
    "The lady doth protest too much, methinks.",
    "Frailty, thy name is woman!",
    "Neither a borrower nor a lender be; for loan oft loses both itself and friend."
    "Give every man thy ear, but few thy voice.",
    "To thine own self be true, and it must follow, as the night the day, thou canst not then be false to any man.",
    "Though this be madness, yet there is method in't.",
    "What a piece of work is a man, how noble in reason, how infinite in faculties, in form and moving how express and admirable, in action how like an angel, in apprehension how like a god!",
    "The play's the thing wherein I'll catch the conscience of the king.",
    "Brevity is the soul of wit.",
    "Doubt thou the stars are fire; Doubt that the sun doth move; Doubt truth to be a liar But never doubt I love.",
    "O, woe is me, to have seen what I have seen, see what I see!",
    "Rich gifts wax poor when givers prove unkind.",
    "That it should come to this!",
    "There is nothing either good or bad but thinking makes it so.",
    "When sorrows come, they come not single spies, but in battalions.",
    "My words fly up, my thoughts remain below: Words without thoughts never to heaven go",
]

USE_MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
BERT_MODEL = "bert-base-uncased"
