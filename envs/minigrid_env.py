import gymnasium as gym
import minigrid  # noqa: F401

def make_env(config: dict) -> gym.Env:
    env_name  = config.get("env_name", "MiniGrid-DoorKey-5x5-v0")
    seed      = config.get("seed", 42)
    max_steps = config.get("max_episode_steps", 300)
    env = gym.make(env_name, max_episode_steps=max_steps)
    env.reset(seed=seed)
    print(f"[make_env] Created '{env_name}' | seed={seed} | max_steps={max_steps}")
    print(f"[make_env] Observation space : {env.observation_space}")
    print(f"[make_env] Action space      : {env.action_space}")
    return env
