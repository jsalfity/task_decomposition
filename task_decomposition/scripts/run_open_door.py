import argparse
import numpy as np
import pandas as pd

import robosuite as suite
from robosuite import load_controller_config

from task_decomposition.utils.logging import (
    save_video_fn,
    save_df_to_txt,
    gpt_annotate_video_fn,
)

# config = load_controller_config(default_controller='IK_POSE')
config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name="Door",  # try with other tasks like "Stack" and "Door"
    robots="Panda",
    controller_configs=config,
    camera_names=["frontview", "robot0_eye_in_hand"],
    camera_heights=480,
    camera_widths=480,
    control_freq=10,
    horizon=75,
    # placement_initializer=reset_sampler
    has_renderer=True,
    # has_offscreen_renderer=False,
    use_camera_obs=True,
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

data_to_record = ["step"] + actions_to_record + obs_to_record + meta_data_to_record


def _setup_parser():
    """Set up Python's ArgumentParser with params"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--save_gt", type=int, default=0)
    parser.add_argument("--save_video", type=int, default=0)
    parser.add_argument("--save_txt", type=int, default=0)
    parser.add_argument("--gpt_annotate", type=int, default=0)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--filename", type=str, default="open_door")

    return parser


def run_demo(
    save_gt: bool = True,
    save_txt: bool = True,
    save_video: bool = True,
    gpt_annotate: bool = False,
    render: bool = True,
    filename: str = "open_door",
):
    obs = env.reset()
    stage = 0
    stage_counter = 0
    done = False

    k = 0
    df = pd.DataFrame(columns=data_to_record)
    gt_df = pd.DataFrame(columns=["step", "subtask", "stage"])

    frames = []
    while not done:
        handle_pos = env._handle_xpos
        gripper_pos = np.array(
            env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        )

        action = np.zeros(7)
        if stage == 0:
            subtask = "Align manipulator height with Door"
            action[:] = 0
            action[2] = 1
            action[-1] = -1
            stage_counter += 1
            if stage_counter == 8:
                stage = 1
                stage_counter = 0

        if stage == 1:
            subtask = "Get closer to Door"
            action[:2] = handle_pos[:2] - gripper_pos[:2] - np.array([0, -0.025])
            action[-1] = -1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 2
            action[:3] *= 10

        if stage == 2:
            subtask = "Turn Door handle"
            action[:3] = handle_pos - gripper_pos - np.array([0, -0.02, 0.05])
            action[-1] = -1
            if gripper_pos[2] < 0.1:
                action[4] = -1
            if gripper_pos[2] < 0.915:
                stage = 3
            action[:3] *= 10

        if stage == 3:
            subtask = "Open Door"
            action[:3] = handle_pos - gripper_pos - np.array([0, -0.1, 0])
            action[-1] = -1
            action[:3] *= 10

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

    print("Done Running Simulation.")
    save_video_fn(frames=frames, filename=filename) if save_video else None
    gpt_annotate_video_fn(frames=frames, filename=filename) if gpt_annotate else None
    save_df_to_txt(df=df, filename=filename) if save_txt else None
    save_df_to_txt(df=gt_df, filename=filename + "_gt") if save_gt else None


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    run_demo(
        save_gt=args.save_gt,
        save_txt=args.save_txt,
        save_video=args.save_video,
        gpt_annotate=args.gpt_annotate,
        render=args.render,
        filename=args.filename,
    )


if __name__ == "__main__":
    main()
