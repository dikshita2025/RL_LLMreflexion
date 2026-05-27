import os, sys, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.minigrid_env import make_env

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_config.yaml")
    config = load_config(config_path)
    print("=" * 60)
    print(f"  Experiment : {config['experiment_name']}")
    print(f"  Env        : {config['env_name']}")
    print(f"  Seed       : {config['seed']}")
    print("=" * 60)
    env = make_env(config)
    obs, info = env.reset(seed=config["seed"])
    print("\n[test_env] Initial observation:")
    if isinstance(obs, dict):
        for key, val in obs.items():
            shape = getattr(val, "shape", type(val))
            print(f"  obs['{key}'] — type: {type(val).__name__}, shape: {shape}")
    NUM_STEPS = 10
    print(f"\n[test_env] Running {NUM_STEPS} random steps ...\n")
    print(f"  {'Step':>4}  {'Action':>6}  {'Reward':>8}  {'Terminated':>10}  {'Truncated':>9}")
    print("  " + "-" * 46)
    for step in range(NUM_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  {step+1:>4}  {action:>6}  {reward:>8.3f}  {str(terminated):>10}  {str(truncated):>9}")
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print("\n[test_env] ✓ Environment test passed. Setup is ready.\n")

if __name__ == "__main__":
    main()
