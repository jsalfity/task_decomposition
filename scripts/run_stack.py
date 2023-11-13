import argparse
import numpy as np
import pandas as pd

import robosuite as suite
from robosuite import load_controller_config

from task_decomposition.utils.logging import (
    # save_df_to_csv,
    save_df_to_txt,
    save_groundtruth_to_txt,
)

# config = load_controller_config(default_controller='IK_POSE')
config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name="Stack",  # try with other tasks like "Stack" and "Door"
    robots="Panda",
    controller_configs=config,
    camera_names=["frontview"],
    camera_heights=128,
    camera_widths=128,
    control_freq=10,
    horizon=80,
    # placement_initializer=reset_sampler
    has_renderer=True,
    # has_offscreen_renderer=False,
    # use_camera_obs=False,
)

actions_to_record = ["action"]
meta_data_to_record = []
obs_to_record = [
    # "robot0_joint_pos_cos",
    # "robot0_joint_pos_sin",
    # "robot0_joint_vel",
    "robot0_eef_pos",
    # "robot0_eef_quat",
    # "robot0_gripper_qpos",
    # "robot0_gripper_qvel",
    # "frontview_image",
    "cubeA_pos",
    # "cubeA_quat",
    "cubeB_pos",
    # "cubeB_quat",
    "gripper_to_cubeA",
    "gripper_to_cubeB",
    "cubeA_to_cubeB",
    # "robot0_proprio-state",
    # "object-state",
]

data_to_record = ["step"] + actions_to_record + obs_to_record + meta_data_to_record


def _setup_parser():
    """Set up Python's ArgumentParser with params"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--filename", type=str, default="stack.txt")

    return parser


def run_demo(render: bool = True, save: bool = True, filename: str = "stack.txt"):
    obs = env.reset()
    stage = 0
    stage_counter = 0
    done = False

    k = 0
    df = pd.DataFrame(columns=data_to_record)
    gt_df = pd.DataFrame(columns=["step", "subtask", "stage"])

    while not done:
        cube_a_pos = env.sim.data.body_xpos[env.cubeA_body_id]
        cube_b_pos = env.sim.data.body_xpos[env.cubeB_body_id]
        gripper_pos = np.array(
            env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        )

        action = np.zeros(7)
        if stage == 0:
            subtask = "Move to above Cube A"
            action[:2] = cube_a_pos[:2] - gripper_pos[:2]
            action[-1] = -1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 1
            action[:3] *= 10

        if stage == 1:
            subtask = "Move directly down to Cube A"
            action[:3] = cube_a_pos - gripper_pos
            action[-1] = -1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 2
            action[:3] *= 10

        if stage == 2:
            subtask = "Grasp Cube A"
            action[:] = 0
            action[-1] = 1
            stage_counter += 1
            if stage_counter == 3:
                stage = 3
                stage_counter = 0

        if stage == 3:
            subtask = "Vertically pick up Cube A"
            action[:] = 0
            action[2] = 0.25
            action[-1] = 1
            stage_counter += 1
            if stage_counter == 15:
                stage = 4
                stage_counter = 0

        if stage == 4:
            subtask = "Align Cube A with Cube B"
            action[:2] = cube_b_pos[:2] - cube_a_pos[:2]
            action[-1] = 1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 5
            action[:3] *= 10

        if stage == 5:
            subtask = "Move Cube A vertically down to Cube B"
            action[:3] = cube_b_pos + np.array([0, 0, 0.05]) - cube_a_pos
            action[-1] = 1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 6
                stage_counter = 0
            action[:3] *= 10

        if stage == 6:
            subtask = "Release Cube A onto Cube B"
            action[:] = 0
            action[-1] = -1
            stage_counter += 1
            if stage_counter == 5:
                stage = 7
                stage_counter = 0

        if stage == 7:
            subtask = "Return Home"
            action[:] = 0
            action[2] = 1
            action[-1] = -1
            stage_counter += 1
            if stage_counter >= 5:
                action[2] = 0

        obs, reward, done, info = env.step(action)
        env.render() if render else None

        row_data = {
            "step": k,
            "action": np.around(action, 2).tolist(),
            "robot0_eef_pos": np.around(obs["robot0_eef_pos"], 2).tolist(),
            "cubeA_pos": np.around(obs["cubeA_pos"], 2).tolist(),
            "cubeB_pos": np.around(obs["cubeB_pos"], 2).tolist(),
            "gripper_to_cubeA": np.around(obs["gripper_to_cubeA"], 2).tolist(),
            "gripper_to_cubeB": np.around(obs["gripper_to_cubeB"], 2).tolist(),
            "cubeA_to_cubeB": np.around(obs["cubeA_to_cubeB"], 2).tolist(),
        }
        df.loc[k] = row_data

        gt_row_data = {"step": k, "subtask": subtask, "stage": stage}
        gt_df.loc[k] = gt_row_data

        k += 1

    # save_df_to_csv(df=df, filename="open_door.csv") if save else None
    save_df_to_txt(df=df, filename=filename) if save else None
    save_groundtruth_to_txt(
        df=gt_df, filename=filename.split(".")[0] + "_gt." + filename.split(".")[-1]
    ) if save else None


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    run_demo(render=args.render, save=args.save, filename=args.filename)


if __name__ == "__main__":
    main()
