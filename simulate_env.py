"""
scripts/simulate_env.py

Visual simulator for the MiniGrid environment.
Runs a random-action episode, captures every frame, and saves:
  - An animated GIF  → logs/simulation.gif
  - A summary image  → logs/first_frame.png

Usage:
    python scripts/simulate_env.py

No display required — works on headless servers.
To watch it interactively with a display:  render_mode="human" (see comment below).
"""

import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[simulate_env] Pillow not installed. Run:  pip install Pillow")
    print("[simulate_env] Falling back to frame-count only mode.\n")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_simulation(config: dict, num_steps: int = 64, gif_fps: int = 6):
    env_name  = config.get("env_name", "MiniGrid-DoorKey-5x5-v0")
    seed      = config.get("seed", 42)
    max_steps = config.get("max_episode_steps", 300)

    # ── render_mode="rgb_array" works without a display ──────────────────────
    # To open a live pygame window instead, change to render_mode="human"
    env = gym.make(env_name, max_episode_steps=max_steps, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    frames = [env.render()]   # capture first frame
    total_reward = 0.0
    step_log = []

    print(f"[simulate_env] Running {num_steps} steps in '{env_name}' ...")
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        total_reward += reward

        step_log.append({
            "step": step + 1,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

        if terminated or truncated:
            print(f"  Episode ended at step {step+1}. Resetting ...")
            obs, _ = env.reset()

    env.close()

    # ── Save GIF ──────────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)

    if PIL_AVAILABLE:
        pil_frames = [Image.fromarray(f) for f in frames]
        gif_path = os.path.join("logs", "simulation.gif")
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / gif_fps),
            loop=0,
        )
        print(f"[simulate_env] ✓ GIF saved  →  {gif_path}  ({len(frames)} frames @ {gif_fps} fps)")

        png_path = os.path.join("logs", "first_frame.png")
        pil_frames[0].save(png_path)
        print(f"[simulate_env] ✓ PNG saved  →  {png_path}")
    else:
        print(f"[simulate_env] Captured {len(frames)} frames (install Pillow to save GIF).")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[simulate_env] Total reward over {num_steps} steps: {total_reward:.3f}")
    print("\n  Step  Action  Reward  Done")
    print("  " + "-" * 32)
    for entry in step_log[:15]:   # show first 15 rows
        done = entry["terminated"] or entry["truncated"]
        print(f"  {entry['step']:>4}  {entry['action']:>6}  {entry['reward']:>6.3f}  {done}")
    if len(step_log) > 15:
        print(f"  ... ({len(step_log) - 15} more steps)")

    return frames


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_config.yaml")
    config = load_config(config_path)
    run_simulation(config, num_steps=64, gif_fps=6)
