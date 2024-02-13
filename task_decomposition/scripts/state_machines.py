import numpy as np
from typing import Union

from robosuite import environments as robosuite_envs


def stack_state_machine(
    env: robosuite_envs, stage: int, stage_counter: int
) -> Union[np.array, int, int]:
    """
    State machine to control the robot that stacks cube A on top of cube B.

    :param env: The simulation environment from robosuite.
    :param stage: The current stage of the stacking task.
    :param stage_counter: The number of times the current stage has been attempted.

    :returns: A tuple containing the action to be taken by the robot (as a numpy array),
              the updated stage of the task, and the updated stage counter.
    """
    cube_a_pos = env.sim.data.body_xpos[env.cubeA_body_id]
    cube_b_pos = env.sim.data.body_xpos[env.cubeB_body_id]
    gripper_pos = np.array(
        env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
    )

    action = np.zeros(7)
    if stage == 0:
        subtask = "Move to above Cube A"
        action[:2] = cube_a_pos[:2] - gripper_pos[:2]
        action[-1] = -1
        if (action[:3] ** 2).sum() < 0.0001:
            stage = 1
        action[:3] *= 10

    if stage == 1:
        subtask = "Move directly down to Cube A"
        action[:3] = cube_a_pos - gripper_pos
        action[-1] = -1
        if (action[:3] ** 2).sum() < 0.0001:
            stage = 2
        action[:3] *= 10

    if stage == 2:
        subtask = "Grasp Cube A"
        action[:] = 0
        action[-1] = 1
        stage_counter += 1
        if stage_counter == 3:
            stage = 3
            stage_counter = 0

    if stage == 3:
        subtask = "Vertically pick up Cube A"
        action[:] = 0
        action[2] = 0.25
        action[-1] = 1
        stage_counter += 1
        if stage_counter == 15:
            stage = 4
            stage_counter = 0

    if stage == 4:
        subtask = "Align Cube A with Cube B"
        action[:2] = cube_b_pos[:2] - cube_a_pos[:2]
        action[-1] = 1
        if (action[:3] ** 2).sum() < 0.0001:
            stage = 5
        action[:3] *= 10

    if stage == 5:
        subtask = "Move Cube A vertically down to Cube B"
        action[:3] = cube_b_pos + np.array([0, 0, 0.05]) - cube_a_pos
        action[-1] = 1
        if (action[:3] ** 2).sum() < 0.0001:
            stage = 6
            stage_counter = 0
        action[:3] *= 10

    if stage == 6:
        subtask = "Release Cube A onto Cube B"
        action[:] = 0
        action[-1] = -1
        stage_counter += 1
        if stage_counter == 5:
            stage = 7
            stage_counter = 0

    if stage == 7:
        subtask = "Return Home"
        action[:] = 0
        action[2] = 1
        action[-1] = -1
        stage_counter += 1
        if stage_counter >= 5:
            action[2] = 0

    return action, stage, stage_counter, subtask


def lift_state_machine(
    env: robosuite_envs, stage: int, stage_counter: int
) -> Union[np.array, int, int]:
    """
    State machine to control the robot to lift a cube.

    :param env: The simulation environment from robosuite.
    :param stage: The current stage of the stacking task.
    :param stage_counter: The number of times the current stage has been attempted.

    :returns: A tuple containing the action to be taken by the robot (as a numpy array),
              the updated stage of the task, and the updated stage counter.
    """
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

    return action, stage, stage_counter, subtask


def open_door_state_machine(
    env: robosuite_envs, stage: int, stage_counter: int
) -> Union[np.array, int, int]:
    """
    State machine to control the robot to open a door

    :param env: The simulation environment from robosuite.
    :param stage: The current stage of the stacking task.
    :param stage_counter: The number of times the current stage has been attempted.

    :returns: A tuple containing the action to be taken by the robot (as a numpy array),
              the updated stage of the task, and the updated stage counter.
    """
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

    return action, stage, stage_counter, subtask


def pickplace_state_machine(
    env: robosuite_envs, stage: int, stage_counter: int
) -> Union[np.array, int, int]:
    """
    State machine to control the robot to pick and place an object.

    :param env: The simulation environment from robosuite.
    :param stage: The current stage of the stacking task.
    :param stage_counter: The number of times the current stage has been attempted.

    :returns: A tuple containing the action to be taken by the robot (as a numpy array),
              the updated stage of the task, and the updated stage counter.
    """
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

    return action, stage, stage_counter, subtask
