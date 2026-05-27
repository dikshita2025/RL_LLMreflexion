# RL_LLMreflexion

Research project: PPO + LLM4Teach + Reflexion on MiniGrid.

## 3 Experiments
| Branch | What |
|---|---|
| exp1/ppo-only | Baseline PPO, no teacher |
| exp2/ppo-llm4teach | PPO + static LLM teacher |
| exp3/ppo-reflexion | PPO + LLM teacher with Reflexion |

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python scripts/test_env.py       # verify setup
python scripts/simulate_env.py   # generate simulation GIF → logs/simulation.gif
python -m minigrid.manual_control --env MiniGrid-DoorKey-5x5-v0  # keyboard control
```
