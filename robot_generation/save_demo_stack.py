import numpy as np

import robosuite as suite
from robosuite import load_controller_config

# config = load_controller_config(default_controller='IK_POSE')
config = load_controller_config(default_controller='OSC_POSE')

NUM_DEMOS = 10

# create environment instance
env = suite.make(
    env_name="Stack",  # try with other tasks like "Stack" and "Door"
    robots="Panda",
    controller_configs=config,
    camera_names=['frontview'],
    camera_heights=128,
    camera_widths=128,
    control_freq=10,
    horizon=80,
    # placement_initializer=reset_sampler
    has_renderer=True,
    # has_offscreen_renderer=False,
    # use_camera_obs=False,
)

obs_list = []
next_obs_list = []
action_list = []
reward_list = []
not_done_list = []

stage = 0
stage_counter = 0

demo_starts = []
demo_ends = []

i = 0
while i < NUM_DEMOS:
    obs = env.reset()
    demo_starts.append(len(obs_list))
    img_obs = np.concatenate([obs['frontview_image'][::-1],
                              obs['robot0_eye_in_hand_image'][::-1]], axis=2).transpose((2, 0, 1))
    while True:
        cube_a_pos = env.sim.data.body_xpos[env.cubeA_body_id]
        cube_b_pos = env.sim.data.body_xpos[env.cubeB_body_id]
        gripper_pos = np.array(env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_grip_site')])

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

        next_obs, r, d, info = env.step(action)
        env.render()
        next_img_obs = np.concatenate([next_obs['frontview_image'][::-1],
                                       next_obs['robot0_eye_in_hand_image'][::-1]], axis=2).transpose((2, 0, 1))
        obs_list.append(img_obs)
        next_obs_list.append(next_img_obs)
        action_list.append(action)
        r = -1 if r <= 0 else 100
        if r == 100:
            d = True
        reward_list.append([r])
        not_done_list.append([not d])
        img_obs = next_img_obs

        if d:
            if r <= 0:
                demo_starts = demo_starts[:-1]
                obs_list = obs_list[:demo_ends[-1]]
                next_obs_list = next_obs_list[:demo_ends[-1]]
                action_list = action_list[:demo_ends[-1]]
                reward_list = reward_list[:demo_ends[-1]]
                not_done_list = not_done_list[:demo_ends[-1]]
                print('redo!')
            else:
                demo_ends.append(len(obs_list))
                print(i)
                i = i + 1
            stage = 0
            stage_counter = 0
            break

payload = [np.array(obs_list), np.array(next_obs_list), np.array(action_list),
           np.array(reward_list), np.array(not_done_list)]


if __name__ == "__main__":
    run_demo()