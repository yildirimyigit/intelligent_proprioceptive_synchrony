import time

import gymnasium as gym

import mujoco
import rcs
import duobench  # noqa: F401
from bimanual_rope.tasks import sling_hook  # noqa: F401  (import registers the env)
from bimanual_rope.utils.sling_state import SlingState


ENV_ID = "bimanual_rope/sling_hook"
REQUIRED_INFO_KEYS = ("instruction", "stage", "max_stage")

if __name__ == "__main__":
    assert ENV_ID in gym.registry, f"{ENV_ID!r} was not registered with Gymnasium"
    creator = gym.spec(ENV_ID).entry_point
    cfg = creator.config()
    cfg.headless = False
    cfg.camera_cfgs = None
    cfg.camera_adds = None

    env = gym.make(ENV_ID, cfg=cfg, disable_env_checker=True)

    # cfg = sling_hook.SlingHookEnvConfig().config()
    # cfg.headless = False


    try: 
        obs, info = env.reset()

        print("Keys:", info.keys())
        # print("Instruction:", info["instruction"])
        # print(info["stage"], info["max_stage"], info["current_subinstruction"])

        tot_reward = 0.0
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            tot_reward += reward
            time.sleep(1.0 / 30.0)  # pace so the viewer motion is watchable
            if terminated or truncated:
                break
    finally:
        print(tot_reward)
        env.close()

    