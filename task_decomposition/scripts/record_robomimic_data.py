"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset for task_decomposition evaluation.

"""

import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase

from task_decomposition.utils.logging import save_video_fn, save_df_to_txt

IMAGE_VIEW_TO_RECORD = "sideview_image"
# IMAGE_VIEW_TO_RECORD = "frontview_image"


def get_data_to_record(env_name: str):
    """
    This function returns the data to record for a given environment for the text columns.
    It's up to the user which data to record, but the following are examples:
    By default, the `step` is always recorded.
    """
    # Common to all environments
    actions_to_record = ["action"]
    obs_to_record = ["robot0_eef_pos"]

    # Unique to each environment
    if env_name == "NutAssemblySquare":
        meta_data_to_record = ["nut0_pos", "nut1_pos"]
    elif env_name == "ToolHang":
        meta_data_to_record = ["tool_pos", "frame_pos", "stand_pos"]
    else:
        raise ValueError(f"Environment {env_name} has no defined data to record.")

    return ["step"] + actions_to_record + obs_to_record + meta_data_to_record


def query_sim_for_data(env, desired_obs):
    """
    Helper function to query the simulator for data to record.
    """
    if env.name == "NutAssemblySquare":
        if desired_obs == "nut0_pos":
            return env.env.sim.data.body_xpos[env.env.obj_body_id[env.env.nuts[0].name]]
        elif desired_obs == "nut1_pos":
            return env.env.sim.data.body_xpos[env.env.obj_body_id[env.env.nuts[1].name]]
        else:
            raise ValueError(f"Environment {env.name} has no defined data to record.")
    elif env.name == "ToolHang":
        if desired_obs == "tool_pos":
            return env.env.sim.data.body_xpos[env.env.obj_body_id[env.env.tool.name]]
        elif desired_obs == "frame_pos":
            return env.env.sim.data.body_xpos[env.env.obj_body_id[env.env.frame.name]]
        elif desired_obs == "stand_pos":
            return env.env.sim.data.body_xpos[env.env.obj_body_id[env.env.stand.name]]
        else:
            raise ValueError(f"Environment {env.name} has no defined data to record.")
    else:
        raise ValueError(f"Environment {env.name} has no defined data to record.")


def extract_trajectory(env, initial_state, states, actions, done_mode):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    data_to_record = get_data_to_record(env_name=env.name)
    df = pd.DataFrame(columns=get_data_to_record(env_name=env.name))

    traj_len = states.shape[0]
    frames = []
    for k in range(traj_len):

        obs = env.reset_to({"states": states[k]})
        frame = obs[IMAGE_VIEW_TO_RECORD]
        frames.append(frame)

        # infer reward signal
        # r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (k == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        row_data = {}
        for o in data_to_record:
            if o == "step":
                row_data[o] = k
            elif o == "action":
                row_data[o] = np.around(actions[k], 2).tolist()
            elif o == "robot0_eef_pos":
                row_data[o] = np.around(obs[o], 2).tolist()
            else:
                row_data[o] = np.around(
                    query_sim_for_data(env, desired_obs=o), 2
                ).tolist()
        df.loc[k] = row_data

    print(" Done Running Simulation.")
    return df, frames


def record_dataset(args):
    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
        use_depth_obs=args.depth,
        render_offscreen=True,
    )

    save_txt = True if args.save_txt == 1 else False
    save_video = True if args.save_video == 1 else False
    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    # output file in same directory as input file
    print("input file: {}".format(args.dataset))

    for idx in tqdm(range(len(demos))):
        ep = demos[idx]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        idx = timenow + f"_{idx}"
        df, frames = extract_trajectory(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            done_mode=args.done_mode,
        )

        # save data
        filename = env.name + f"_{idx}"
        save_video_fn(frames=frames, filename=filename) if save_video else None
        save_df_to_txt(df=df, filename=filename) if save_txt else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="path to input hdf5 dataset",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(Required) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--save_txt",
        type=int,
        default=1,
        help="(Required) but default to True, save txt files",
    )

    parser.add_argument(
        "--save_video",
        type=int,
        default=1,
        help="(Required) but default to True, save video files",
    )

    ####################################################################################
    # THE BELOW ARGS ARE REQUIRED BY THE ROBOMIMIC UTILITIES TO CREATE THE ENVIRONMENT
    # see original script `robomimic/robomimic/scripts/dataset_states_to_obs.py`
    parser.add_argument(
        "--shaped",
        action="store_true",
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["frontview", "sideview"],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=480,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=480,
        help="(optional) width of image observations",
    )

    # flag for including depth observations per camera
    parser.add_argument(
        "--depth",
        action="store_true",
        help="(optional) use depth observations for each camera",
    )

    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )
    ####################################################################################

    args = parser.parse_args()
    record_dataset(args)
