import numpy as np

import robosuite as suite
from robosuite import load_controller_config

# from robosuite.utils.placement_samplers import UniformRandomSampler

# config = load_controller_config(default_controller='IK_POSE')
config = load_controller_config(default_controller="OSC_POSE")

NUM_DEMOS = 5
RENDER = True
# reset_sampler = UniformRandomSampler(
#     name="ObjectSampler",
#     mujoco_objects=None,
#     x_range=[-0.2, 0.2],
#     y_range=[-0.2, 0.2],
#     rotation=None,
#     ensure_object_boundary_in_range=False,
#     ensure_valid_placement=True,
#     reference_pos=np.array((0, 0, 0.8)),
#     z_offset=0.01)

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
    # has_renderer=True,
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
            cube_pos = env.sim.data.body_xpos[env.cube_body_id]
            gripper_pos = np.array(
                env.sim.data.site_xpos[
                    env.sim.model.site_name2id("gripper0_grip_site")
                ]
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
            env.render() if RENDER else None


if __name__ == "__main__":
    run_demo()
