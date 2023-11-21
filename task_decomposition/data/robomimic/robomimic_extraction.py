import robosuite
from robosuite import load_controller_config
import numpy as np

# controller_config = load_controller_config(default_controller="OSC_POSE")
env_kwargs = {
    "has_renderer": False,
    "has_offscreen_renderer": True,
    "ignore_done": True,
    "use_object_obs": True,
    "use_camera_obs": False,
    "control_freq": 20,
    "controller_configs": load_controller_config(default_controller="OSC_POSE"),
    "robots": ["Panda"],
    "camera_depths": False,
    "camera_heights": 84,
    "camera_widths": 84,
    "reward_shaping": False,
}

env = robosuite.make(env_name="ToolHang", **env_kwargs)

env.reset()

action = np.zeros(7)
obs, reward, done, info = env.step(action)
print(obs)
