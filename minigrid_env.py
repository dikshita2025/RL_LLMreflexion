"""
envs/minigrid_env.py

MiniGrid environment wrapper for RL_LLMreflexion.
Reads env_name and seed from config dict and returns a ready-to-use gym env.

Used by all three experiment branches:
  - Person 1: PPO only
  - Person 2: PPO + static LLM4Teach teacher
  - Person 3: PPO + Reflexion teacher
"""

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium


def make_env(config: dict) -> gym.Env:
    """
    Create and return a MiniGrid gymnasium environment.

    Args:
        config (dict): Must contain:
            - env_name (str): e.g. "MiniGrid-DoorKey-5x5-v0"
            - seed     (int): random seed for reproducibility
            - max_episode_steps (int, optional): step limit per episode

    Returns:
        gym.Env: A fully initialised, seeded environment.
    """
    env_name        = config.get("env_name", "MiniGrid-DoorKey-5x5-v0")
    seed            = config.get("seed", 42)
    max_steps       = config.get("max_episode_steps", 300)

    # MiniGrid envs accept max_episode_steps at construction time
    env = gym.make(env_name, max_episode_steps=max_steps)

    # Seed the environment for reproducibility
    env.reset(seed=seed)

    print(f"[make_env] Created '{env_name}' | seed={seed} | max_steps={max_steps}")
    print(f"[make_env] Observation space : {env.observation_space}")
    print(f"[make_env] Action space      : {env.action_space}")

    return env
