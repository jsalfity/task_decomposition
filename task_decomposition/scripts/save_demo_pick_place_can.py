import numpy as np
import os
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

# config = load_controller_config(default_controller='IK_POSE')
config = load_controller_config(default_controller='OSC_POSE')

NUM_DEMOS = 20
ROOT_FOLDER = './demo/robosuite_pick_place_can/'
target_folder = ROOT_FOLDER + str(NUM_DEMOS)

reset_sampler = UniformRandomSampler(
    name="ObjectSampler",
    mujoco_objects=None,
    x_range=[-0.2, 0.2],
    y_range=[-0.2, 0.2],
    rotation=None,
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=True,
    reference_pos=np.array((0, 0, 0.8)),
    z_offset=0.01)

# create environment instance
env = suite.make(
    env_name="PickPlace",  # try with other tasks like "Stack" and "Door"
    robots="Panda",
    controller_configs=config,
    camera_names=['frontview', 'robot0_eye_in_hand'],
    camera_heights=128,
    camera_widths=128,
    control_freq=10,
    horizon=120,
    single_object_mode=2,
    object_type='can',
    bin1_pos=(0.1, -0.27, 0.8),
    bin2_pos=(0.1, 0.27, 0.8),
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

for i in range(NUM_DEMOS):
    obs = env.reset()
    demo_starts.append(len(obs_list))
    img_obs = np.concatenate([obs['frontview_image'][::-1],
                              obs['robot0_eye_in_hand_image'][::-1]], axis=2).transpose((2, 0, 1))
    while True:
        obj_pos = env.sim.data.body_xpos[env.obj_body_id['Can']]
        goal_pos = env.target_bin_placements[env.object_to_id['can']]
        gripper_pos = np.array(env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_grip_site')])

        action = np.zeros(7)

        if stage == 0:
            action[:3] = obj_pos - gripper_pos
            action[2] = 0
            action[-1] = -1
            if (action[:3] ** 2).sum() < 0.0001:
                stage = 1
            action[:3] *= 10

        if stage == 1:
            action[:3] = obj_pos + np.array([0, 0, 0.015]) - gripper_pos
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
            action[2] = 1
            action[-1] = 1
            stage_counter += 1
            if stage_counter == 8:
                stage = 4
                stage_counter = 0

        if stage == 4:
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
            action[:3] *= 10

        if stage == 6:
            action[:] = 0
            action[-1] = -1
            stage_counter += 1
            if stage_counter == 8:
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
            demo_ends.append(len(obs_list))
            stage = 0
            stage_counter = 0
            break

# payload = [np.array(obs_list), np.array(next_obs_list), np.array(action_list),
#            np.array(reward_list), np.array(not_done_list)]
# if not os.path.isdir(target_folder):
#     os.makedirs(target_folder)
# torch.save(payload, target_folder + '/0_' + str(len(obs_list)) + '.pt')
# np.save(target_folder + '/demo_starts.npy', np.array(demo_starts))
# np.save(target_folder + '/demo_ends.npy', np.array(demo_ends))
