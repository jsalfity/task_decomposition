import argparse
import numpy as np

import robosuite as suite
from robosuite import load_controller_config

config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name="Lift",  # try with other tasks like "Stack" and "Door"
    robots="Panda",
    controller_configs=config,
    camera_names=["frontview", "robot0_eye_in_hand"],
    camera_heights=128,
    camera_widths=128,
    control_freq=10,
    horizon=40,
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
            cube_pos = env.sim.data.body_xpos[env.cube_body_id]
            gripper_pos = np.array(
                env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
            )

            action = np.zeros(7)
            if stage == 0:
                action[:3] = cube_pos - gripper_pos
                action[-1] = -1
                if (action[:3] ** 2).sum() < 0.0001:
                    stage = 1
                action[:3] *= 10

            if stage == 1:
                action[:] = 0
                action[-1] = 1
                stage_counter += 1
                if stage_counter == 3:
                    stage = 2
                    stage_counter = 0

            if stage == 2:
                action[:] = 0
                action[2] = 0.25
                action[-1] = 1
                stage_counter += 1
                if stage_counter >= 10:
                    action[2] = 0

            obs, reward, done, info = env.step(action)
            env.render() if render else None


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    run_demo(n_trajectories=args.n_trajectories, render=args.render)


if __name__ == "__main__":
    main()
