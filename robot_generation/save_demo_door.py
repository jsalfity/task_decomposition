import numpy as np

import robosuite as suite
from robosuite import load_controller_config

# config = load_controller_config(default_controller='IK_POSE')
config = load_controller_config(default_controller="OSC_POSE")

NUM_DEMOS = 10
RENDER = True
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


def run_demo(n_demos: int = NUM_DEMOS):
    for _ in range(n_demos):
        obs = env.reset()
        stage = 0
        stage_counter = 0
        done = False

        while not done:
            handle_pos = env._handle_xpos
            gripper_pos = np.array(
                env.sim.data.site_xpos[
                    env.sim.model.site_name2id("gripper0_grip_site")
                ]
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
            action[:2] = (
                handle_pos[:2] - gripper_pos[:2] - np.array([0, -0.025])
            )
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
        env.render() if RENDER else None


if __name__ == "__main__":
    run_demo(NUM_DEMOS)
