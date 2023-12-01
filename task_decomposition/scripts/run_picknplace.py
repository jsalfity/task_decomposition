import argparse
import numpy as np
import pandas as pd

import robosuite as suite
from robosuite import load_controller_config

# from robosuite.utils.placement_samplers import UniformRandomSampler

from task_decomposition.utils.logging import save_video_fn, save_df_to_txt

config = load_controller_config(default_controller="OSC_POSE")

# reset_sampler = UniformRandomSampler(
#     name="ObjectSampler",
#     mujoco_objects=None,
#     x_range=[-0.2, 0.2],
#     y_range=[-0.2, 0.2],
#     rotation=None,
#     ensure_object_boundary_in_range=False,
#     ensure_valid_placement=True,
#     reference_pos=np.array((0, 0, 0.8)),
#     z_offset=0.01,
# )

# create environment instance
env = suite.make(
    env_name="PickPlace",
    robots="Panda",
    controller_configs=config,
    camera_names=["frontview", "robot0_eye_in_hand"],
    camera_heights=480,
    camera_widths=480,
    control_freq=10,
    horizon=120,
    single_object_mode=2,
    object_type="can",
    bin1_pos=(0.1, -0.27, 0.8),
    bin2_pos=(0.1, 0.27, 0.8),
    # placement_initializer=reset_sampler,
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
    "Can_pos",
    # "Can_quat",
    "Can_to_robot0_eef_pos",
    # "Can_to_robot0_eef_quat",
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
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--filename", type=str, default="picknplace")

    return parser


def run_demo(
    save_gt: bool = True,
    save_txt: bool = True,
    save_video: bool = True,
    render: bool = True,
    filename: str = "picknplace",
):
    print("Running Simulation...")
    obs = env.reset()
    stage = 0
    stage_counter = 0
    done = False

    k = 0
    df = pd.DataFrame(columns=data_to_record)
    gt_df = pd.DataFrame(columns=["step", "subtask", "stage"])

    frames = []
    while not done:
        obj_pos = env.sim.data.body_xpos[env.obj_body_id["Can"]]
        goal_pos = env.target_bin_placements[env.object_to_id["can"]]
        gripper_pos = np.array(
            env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        )

        action = np.zeros(7)
        if stage == 0:
            subtask = "Move to above can"
            action[:3] = obj_pos - gripper_pos
            action[2] = 0
            action[-1] = -1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 1
            action[:3] *= 10

        if stage == 1:
            subtask = "Move down to can"
            action[:3] = obj_pos + np.array([0, 0, 0.015]) - gripper_pos
            action[-1] = -1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 2
            action[:3] *= 10

        if stage == 2:
            subtask = "Grasp can"
            action[:] = 0
            action[-1] = 1
            stage_counter += 1
            if stage_counter == 3:
                stage = 3
                stage_counter = 0

        if stage == 3:
            subtask = "Pick up can"
            action[:] = 0
            action[2] = 1
            action[-1] = 1
            stage_counter += 1
            if stage_counter == 8:
                stage = 4
                stage_counter = 0

        if stage == 4:
            subtask = "Move to goal position"
            action[:2] = goal_pos[:2] - obj_pos[:2]
            action[-1] = 1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 5
            action[:3] *= 10

        if stage == 5:
            action[:3] = goal_pos + np.array([0, 0, 0.05]) - obj_pos
            action[-1] = 1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 6
                stage_counter = 0
                input("stage 5 -- unlabeled")
            action[:3] *= 10

        if stage == 6:
            action[:] = 0
            action[-1] = -1
            stage_counter += 1
            if stage_counter == 8:
                stage = 7
                stage_counter = 0
                input("stage 6 -- unlabeled")

        if stage == 7:
            action[:] = 0
            action[2] = 1
            action[-1] = -1
            stage_counter += 1
            if stage_counter >= 5:
                action[2] = 0
                input("stage 7 -- unlabeled")

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

    env.close()

    print("Done Running Simulation.")
    save_video_fn(frames=frames, filename=filename) if save_video else None
    save_df_to_txt(df=df, filename=filename) if save_txt else None
    save_df_to_txt(df=gt_df, filename=filename + "_gt") if save_gt else None


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    run_demo(
        save_gt=args.save_gt,
        save_txt=args.save_txt,
        save_video=args.save_video,
        render=args.render,
        filename=args.filename,
    )


if __name__ == "__main__":
    main()
