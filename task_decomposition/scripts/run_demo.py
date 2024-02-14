import numpy as np
import pandas as pd
from datetime import datetime
import yaml

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

from task_decomposition.utils.logging import (
    save_video_fn,
    save_df_to_txt,
    gpt_annotate_video_fn,
)
from task_decomposition.paths import DEMO_CONFIG_YAML

controller_config = load_controller_config(default_controller="OSC_POSE")

reset_sampler = UniformRandomSampler(
    name="ObjectSampler",
    mujoco_objects=None,
    x_range=[-0.2, 0.2],  # HARDCODED VALUES
    y_range=[-0.2, 0.2],  # HARDCODED VALUES
    rotation=None,
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=True,
    reference_pos=np.array((0, 0, 0.8)),
    z_offset=0.00,
)


def get_data_to_record(env_name: str):
    """
    This function returns the data to record for a given environment for the text columns.
    It's up to the user which data to record, but the following are examples:
    By default, the `step` is always recorded.
    """
    if env_name == "Stack":
        actions_to_record = ["action"]
        meta_data_to_record = []
        obs_to_record = [
            "robot0_eef_pos",
            "cubeA_pos",
            "cubeB_pos",
            "gripper_to_cubeA",
            "gripper_to_cubeB",
            "cubeA_to_cubeB",
        ]
    elif env_name == "Lift":
        actions_to_record = ["action"]
        meta_data_to_record = []
        obs_to_record = [
            "robot0_eef_pos",
            "cube_pos",
            "gripper_to_cube_pos",
        ]
    elif env_name == "Door":
        actions_to_record = ["action"]
        meta_data_to_record = []
        obs_to_record = [
            "robot0_eef_pos",
            "door_pos",
            "handle_pos",
            "door_to_eef_pos",
            "handle_to_eef_pos",
        ]
    elif env_name == "PickPlace":
        actions_to_record = ["action"]
        meta_data_to_record = []
        obs_to_record = [
            "robot0_eef_pos",
            "Can_pos",
            "Can_to_robot0_eef_pos",
        ]
    else:
        raise ValueError(f"Environment {env_name} has no defined data to record.")

    return ["step"] + actions_to_record + obs_to_record + meta_data_to_record


def get_state_machine(env: str):
    """
    Each environment has a different state machine to control the robot.
    """
    if env == "Stack":
        from task_decomposition.scripts.state_machines import (
            stack_state_machine as state_machine,
        )
    elif env == "Lift":
        from task_decomposition.scripts.state_machines import (
            lift_state_machine as state_machine,
        )
    elif env == "Door":
        from task_decomposition.scripts.state_machines import (
            open_door_state_machine as state_machine,
        )
    elif env == "PickPlace":
        from task_decomposition.scripts.state_machines import (
            pickplace_state_machine as state_machine,
        )
    else:
        raise ValueError(f"Environment {env} has no defined state machine.")

    return state_machine


def run_demo(demo_config: dict, filename: str):
    """
    Main script to run a demo to generate data
    """
    # Need to parse the demo_config to get the suite environment details
    env_config = {}
    env_config["env_name"] = demo_config["env_name"]
    env_config["robots"] = "Panda"
    env_config["controller_configs"] = controller_config
    env_config["camera_names"] = ["frontview"]
    env_config["camera_heights"] = 480
    env_config["camera_widths"] = 480
    env_config["control_freq"] = 10
    env_config["horizon"] = demo_config["horizon"]
    if (
        "custom_placement_initializer" in demo_config
        and demo_config["custom_placement_initializer"]
    ):
        env_config["placement_initializer"] = reset_sampler
    if "object_type" in demo_config:
        env_config["object_type"] = demo_config["object_type"]
    env_config["has_renderer"] = True
    env_config["use_camera_obs"] = True

    # Save variables
    save_txt = demo_config["save_txt"]
    save_video = demo_config["save_video"]
    save_gt = demo_config["save_gt"]
    gpt_annotate = demo_config["gpt_annotate"]

    # Make the environment
    env = suite.make(**env_config)

    # Get the state machine and data to record
    state_machine = get_state_machine(demo_config["env_name"])
    data_to_record = get_data_to_record(demo_config["env_name"])

    # Episode varialbes
    render = demo_config["render"]
    obs = env.reset()
    stage = 0
    stage_counter = 0
    done = False

    k = 0
    df = pd.DataFrame(columns=data_to_record)
    gt_df = pd.DataFrame(columns=["step", "subtask", "stage"])

    frames = []
    while not done:
        action, stage, stage_counter, subtask = state_machine(env, stage, stage_counter)
        obs, reward, done, info = env.step(action)
        frame = np.flip(obs["frontview_image"], axis=0)
        frames.append(frame)
        env.render() if render else None

        row_data = {}
        for o in data_to_record:
            if o == "step":
                row_data[o] = k
            elif o == "action":
                row_data[o] = np.around(action, 2).tolist()
            else:
                row_data[o] = np.around(obs[o], 2).tolist()

        df.loc[k] = row_data

        gt_row_data = {"step": k, "subtask": subtask, "stage": stage}
        gt_df.loc[k] = gt_row_data

        k += 1

    print(" Done Running Simulation.")
    save_video_fn(frames=frames, filename=filename) if save_video else None
    gpt_annotate_video_fn(frames=frames, filename=filename) if gpt_annotate else None
    save_df_to_txt(df=df, filename=filename) if save_txt else None
    save_df_to_txt(df=gt_df, filename=filename + "_gt") if save_gt else None


def main():
    with open(DEMO_CONFIG_YAML, "r") as f:
        demo_config = yaml.safe_load(f)

    for n in range(demo_config["n_demos"]):
        print(f"Running demo {n+1} of {demo_config['n_demos']}")
        timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        idx = timenow + f"_{n}"
        run_demo(
            demo_config=demo_config,
            filename=demo_config["env_name"] + f"_{idx}",
        )


if __name__ == "__main__":
    main()
