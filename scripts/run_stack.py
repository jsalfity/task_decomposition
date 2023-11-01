import argparse
import numpy as np

import robosuite as suite
from robosuite import load_controller_config

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


def _setup_parser():
    """Set up Python's ArgumentParser with params"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--render", type=int, default=1)

    return parser


def run_demo(n_trajectories: int = 1, render: bool = True):
    for _ in range(n_trajectories):
        obs = env.reset()
        stage = 0
        stage_counter = 0
        done = False

        while not done:
            cube_a_pos = env.sim.data.body_xpos[env.cubeA_body_id]
            cube_b_pos = env.sim.data.body_xpos[env.cubeB_body_id]
            gripper_pos = np.array(
                env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
            )

            action = np.zeros(7)
            if stage == 0:
                action[:2] = cube_a_pos[:2] - gripper_pos[:2]
                action[-1] = -1
                if (action[:3] ** 2).sum() < 0.0001:
                    stage = 1
                action[:3] *= 10

            if stage == 1:
                action[:3] = cube_a_pos - gripper_pos
                action[-1] = -1
                if (action[:3] ** 2).sum() < 0.0001:
                    stage = 2
                action[:3] *= 10

            if stage == 2:
                action[:] = 0
                action[-1] = 1
                stage_counter += 1
                if stage_counter == 3:
                    stage = 3
                    stage_counter = 0

            if stage == 3:
                action[:] = 0
                action[2] = 0.25
                action[-1] = 1
                stage_counter += 1
                if stage_counter == 15:
                    stage = 4
                    stage_counter = 0

            if stage == 4:
                action[:2] = cube_b_pos[:2] - cube_a_pos[:2]
                action[-1] = 1
                if (action[:3] ** 2).sum() < 0.0001:
                    stage = 5
                action[:3] *= 10

            if stage == 5:
                action[:3] = cube_b_pos + np.array([0, 0, 0.05]) - cube_a_pos
                action[-1] = 1
                if (action[:3] ** 2).sum() < 0.0001:
                    stage = 6
                    stage_counter = 0
                action[:3] *= 10

            if stage == 6:
                action[:] = 0
                action[-1] = -1
                stage_counter += 1
                if stage_counter == 5:
                    stage = 7
                    stage_counter = 0

            if stage == 7:
                action[:] = 0
                action[2] = 1
                action[-1] = -1
                stage_counter += 1
                if stage_counter >= 5:
                    action[2] = 0

            obs, reward, done, info = env.step(action)
            env.render() if render else None


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    run_demo(n_trajectories=args.n_trajectories, render=args.render)


if __name__ == "__main__":
    run_demo()
