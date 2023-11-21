import robosuite
from robosuite import load_controller_config
import numpy as np

import robosuite as suite
from robosuite import load_controller_config

from task_decomposition.utils.logging import get_device, save_frames_fn, save_df_to_txt


controller_config = load_controller_config(default_controller="OSC_POSE")

env = suite.make(
    env_name="ToolHang",
    robots="Panda",
    controller_config=controller_config,
    camera_names=["frontview"],
    camera_heights=128,
    camera_widths=128,
    control_freq=20,
    horizon=80,
    has_renderer=True,
)

actions_to_record = ["action"]
meta_data_to_record = []
obs_to_record = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

data_to_record = ["step"] + actions_to_record + obs_to_record + meta_data_to_record


def _setup_parser():
    """Set up Python's ArgumentParser with params"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--filename", type=str, default="lift.txt")

    return parser


def run_demo(
    device: str = "spacemouse",
    save_gt: bool = True,
    save_txt: bool = True,
    save_frames: bool = True,
    render: bool = True,
    filename: str = "toolhang",
):
    # initialize device
    device = get_device(device)
    device.start_control()

    print("Running Simulation...")
    obs = env.reset()
    while True:
        active_robot = env.robots[0]

        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=args.arm,
            env_configuration=args.config,
        )

        # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
        # toggle arm control and / or camera viewing angle if requested
        if last_grasp < 0 < grasp:
            if args.switch_on_grasp:
                args.arm = "left" if args.arm == "right" else "right"
            if args.toggle_camera_on_grasp:
                cam_id = (cam_id + 1) % num_cam
                env.viewer.set_camera(camera_id=cam_id)
        # Update last grasp
        last_grasp = grasp

        # Fill out the rest of the action space if necessary
        rem_action_dim = env.action_dim - action.size
        if rem_action_dim > 0:
            # Initialize remaining action space
            rem_action = np.zeros(rem_action_dim)
            # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
            if args.arm == "right":
                action = np.concatenate([action, rem_action])
            elif args.arm == "left":
                action = np.concatenate([rem_action, action])
            else:
                # Only right and left arms supported
                print(
                    "Error: Unsupported arm specified -- "
                    "must be either 'right' or 'left'! Got: {}".format(args.arm)
                )
        elif rem_action_dim < 0:
            # We're in an environment with no gripper action space, so trim the action space to be the action dim
            action = action[: env.action_dim]

        # Step through the simulation and render
        obs, reward, done, info = env.step(action)
        env.render()


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    run_demo(
        device=args.device,
        save_gt=args.save_gt,
        save_txt=args.save_txt,
        save_frames=args.save_frames,
        render=args.render,
        filename=args.filename,
    )


if __name__ == "__main__":
    main()
