import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

from task_decomposition.utils.logging import (
    save_video_fn,
    save_df_to_txt,
    gpt_annotate_video_fn,
)

config = load_controller_config(default_controller="OSC_POSE")

reset_sampler = UniformRandomSampler(
    name="ObjectSampler",
    mujoco_objects=None,
    x_range=[-0.2, 0.2],
    y_range=[-0.2, 0.2],
    rotation=None,
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=True,
    reference_pos=np.array((0, 0, 0.8)),
    z_offset=0.00,
)

# create environment instance
env = suite.make(
    env_name="Lift",  # try with other tasks like "Stack" and "Door"
    robots="Panda",
    controller_configs=config,
    camera_names=["frontview"],
    camera_heights=480,
    camera_widths=480,
    control_freq=10,
    horizon=40,
    placement_initializer=reset_sampler,
    has_renderer=True,
    # has_offscreen_renderer=False,
    use_camera_obs=True,
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
    # "robot0_eye_in_hand_image",
    "cube_pos",
    # "cube_quat",
    "gripper_to_cube_pos",
    # "robot0_proprio-state",
    # "object-state",
]

data_to_record = ["step"] + actions_to_record + obs_to_record + meta_data_to_record


def _setup_parser():
    """Set up Python's ArgumentParser with params"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--save_gt", type=int, default=0)
    parser.add_argument("--save_video", type=int, default=0)
    parser.add_argument("--save_txt", type=int, default=0)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--gpt_annotate", type=int, default=0)
    parser.add_argument("--filename", type=str, default="lift")
    parser.add_argument("--n_demos", type=int, default=1)

    return parser


def run_demo(
    save_gt: bool = True,
    save_txt: bool = True,
    save_video: bool = True,
    gpt_annotate: bool = True,
    render: bool = True,
    filename: str = "lift.txt",
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
        cube_pos = env.sim.data.body_xpos[env.cube_body_id]
        gripper_pos = np.array(
            env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        )

        action = np.zeros(7)
        if stage == 0:
            subtask = "Move to cube"
            action[:3] = cube_pos - gripper_pos
            action[-1] = -1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 1
            action[:3] *= 10

        if stage == 1:
            subtask = "Grasp Cube"
            action[:] = 0
            action[-1] = 1
            stage_counter += 1
            if stage_counter == 3:
                stage = 2
                stage_counter = 0

        if stage == 2:
            subtask = "Lift Cube"
            action[:] = 0
            action[2] = 0.25
            action[-1] = 1
            stage_counter += 1
            if stage_counter >= 10:
                action[2] = 0

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
    parser = _setup_parser()
    args = parser.parse_args()

    for n in range(args.n_demos):
        print(f"Running demo {n} of {args.n_demos}")
        timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        idx = timenow + f"_{n}"
        run_demo(
            save_gt=args.save_gt,
            save_txt=args.save_txt,
            save_video=args.save_video,
            gpt_annotate=args.gpt_annotate,
            render=args.render,
            filename=args.filename + f"_{idx}",
        )


if __name__ == "__main__":
    main()
