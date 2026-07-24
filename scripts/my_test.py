import time

import gymnasium as gym
from duobench.tasks import pour_marbles

if __name__ == "__main__":
    cfg = pour_marbles.PourMarblesEnvConfig().config()
    cfg.headless = False

    env = gym.make("duobench/pour_marbles", cfg=cfg)

    try: 
        obs, info = env.reset()

        print("Keys:", info.keys())
        print("Instruction:", info["instruction"])
        print(info["stage"], info["max_stage"], info["current_subinstruction"])

        tot_reward = 0.0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            tot_reward += reward
            time.sleep(1.0 / 30.0)  # pace so the viewer motion is watchable
            if terminated or truncated:
                break
    finally:
        print(tot_reward)
        env.close()

    