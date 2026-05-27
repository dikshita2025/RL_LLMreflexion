import os, sys, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gymnasium as gym
import minigrid  # noqa: F401

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[simulate_env] Install Pillow:  pip install Pillow")

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def run_simulation(config, num_steps=64, gif_fps=6):
    env_name  = config.get("env_name", "MiniGrid-DoorKey-5x5-v0")
    seed      = config.get("seed", 42)
    max_steps = config.get("max_episode_steps", 300)
    # Change render_mode to "human" for a live pygame window (needs a display)
    env = gym.make(env_name, max_episode_steps=max_steps, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    os.makedirs("logs", exist_ok=True)
    if PIL_AVAILABLE:
        pil_frames = [Image.fromarray(f) for f in frames]
        gif_path = os.path.join("logs", "simulation.gif")
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           duration=int(1000/gif_fps), loop=0)
        print(f"[simulate_env] ✓ GIF saved → {gif_path}  ({len(frames)} frames @ {gif_fps} fps)")
    else:
        print(f"[simulate_env] Captured {len(frames)} frames (install Pillow to save GIF).")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_config.yaml")
    config = load_config(config_path)
    run_simulation(config, num_steps=64, gif_fps=6)
