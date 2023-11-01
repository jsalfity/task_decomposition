import argparse
import numpy as np
import pandas as pd

import robosuite as suite
from robosuite import load_controller_config

from task_decomposition.utils.utils import save_df_to_csv, save_df_to_txt

# config = load_controller_config(default_controller='IK_POSE')
config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name="Door",  # try with other tasks like "Stack" and "Door"
    robots="Panda",
    controller_configs=config,
    camera_names=["frontview", "robot0_eye_in_hand"],
    camera_heights=128,
    camera_widths=128,
    control_freq=10,
    horizon=75,
    # placement_initializer=reset_sampler
    has_renderer=True,
    # has_offscreen_renderer=False,
    # use_camera_obs=False,
)

actions_to_record = ["action"]
meta_data_to_record = []
obs_to_record = [
    #     "robot0_joint_pos_cos",
    #     "robot0_joint_pos_sin",
    #     "robot0_joint_vel",
    "robot0_eef_pos",
    #     "robot0_eef_quat",
    #     "robot0_gripper_qpos",
    #     "robot0_gripper_qvel",
    #     "frontview_image",
    #     "robot0_eye_in_hand_image",
    "door_pos",
    "handle_pos",
    "door_to_eef_pos",
    "handle_to_eef_pos",
    #     "hinge_qpos",
    #     "handle_qpos",
    #     "robot0_proprio-state",
    #     "object-state",
]


def _setup_parser():
    """Set up Python's ArgumentParser with params"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--save_episode", type=int, default=1)
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--filename", type=str, default="open_door.txt")

    return parser


def run_demo(
    render: bool = True, save_episode: bool = True, filename: str = "open_door.txt"
):
    obs = env.reset()
    stage = 0
    stage_counter = 0
    done = False

    k = 0
    df = pd.DataFrame(columns=obs_to_record + actions_to_record + meta_data_to_record)
    while not done:
        handle_pos = env._handle_xpos
        gripper_pos = np.array(
            env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        )

        action = np.zeros(7)
        if stage == 0:
            action[:] = 0
            action[2] = 1
            action[-1] = -1
            stage_counter += 1
            if stage_counter == 8:
                stage = 1
                stage_counter = 0

        if stage == 1:
            action[:2] = handle_pos[:2] - gripper_pos[:2] - np.array([0, -0.025])
            action[-1] = -1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 2
            action[:3] *= 10

        if stage == 2:
            action[:3] = handle_pos - gripper_pos - np.array([0, -0.02, 0.05])
            action[-1] = -1
            if gripper_pos[2] < 0.1:
                action[4] = -1
            if gripper_pos[2] < 0.915:
                stage = 3
            action[:3] *= 10

        if stage == 3:
            action[:3] = handle_pos - gripper_pos - np.array([0, -0.1, 0])
            action[-1] = -1
            action[:3] *= 10

        obs, reward, done, info = env.step(action)
        k += 1
        env.render() if render else None

        row_data = {
            "action": np.around(action, 2).tolist(),
            "robot0_eef_pos": np.around(obs["robot0_eef_pos"], 2).tolist(),
            "door_pos": np.around(obs["door_pos"], 2).tolist(),
            "handle_pos": np.around(obs["handle_pos"], 2).tolist(),
            "door_to_eef_pos": np.around(obs["door_to_eef_pos"], 2).tolist(),
            "handle_to_eef_pos": np.around(obs["handle_to_eef_pos"], 2).tolist(),
        }
        df.loc[k] = row_data

    # save_df_to_csv(df=df, filename="open_door.csv") if save_episode else None
    save_df_to_txt(df=df, filename="open_door.txt") if save_episode else None


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    run_demo(render=args.render, save_episode=args.save_episode, filename=args.filename)


if __name__ == "__main__":
    main()
