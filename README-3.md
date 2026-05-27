# RL_LLMreflexion

A research project combining **PPO**, **LLM4Teach-style teacher guidance**, and **Reflexion** for training RL agents in MiniGrid environments.

---

## Project Goal

We investigate whether an LLM-based teacher that improves through self-reflection (Reflexion) can help a PPO agent learn faster and more robustly than:
- A baseline PPO agent with no teacher, and
- A PPO agent guided by a static (non-reflecting) LLM teacher.

---

## Three Experiment Runs

| Branch | What it does |
|---|---|
| **Exp 1 — PPO only** | Standard PPO on MiniGrid. Baseline. No teacher. |
| **Exp 2 — PPO + static LLM4Teach** | PPO agent receives action hints from an LLM teacher. Teacher prompt does not change across episodes. |
| **Exp 3 — PPO + LLM4Teach + Reflexion** | Same as Exp 2, but after each episode the LLM reflects on what went wrong and updates its own guidance for the next episode. |

---

## About This Repository

This is **our own clean implementation**. We studied the structure and ideas from [LLM4Teach](https://github.com/ZJLAB-AMMI/LLM4Teach) as a reference only. No code was copied from that repository.

---

## Project Structure

```
RL_LLMreflexion/
├── envs/               # Shared MiniGrid environment wrapper
│   ├── __init__.py
│   └── minigrid_env.py
├── configs/            # YAML config files
│   └── default_config.yaml
├── scripts/            # Runnable scripts
│   └── test_env.py
├── algos/              # RL algorithm implementations (PPO etc.)
│   └── __init__.py
├── teacher/            # LLM teacher modules
│   └── __init__.py
├── reflexion/          # Reflexion feedback loop modules
│   └── __init__.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/dikshanta2025/RL_LLMreflexion.git
cd RL_LLMreflexion

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Run the Environment Test

```bash
python scripts/test_env.py
```

This will:
1. Load `configs/default_config.yaml`
2. Create the MiniGrid environment (`MiniGrid-DoorKey-5x5-v0`)
3. Reset the environment with the configured seed
4. Take 10 random steps
5. Print observation type/shape, action, reward, and episode status for each step

A successful run confirms the shared base is working and you are ready to start your experiment branch.

---

## Team Branches

| Person | Branch | Task |
|---|---|---|
| Person 1 | `exp1/ppo-only` | Implement PPO in `algos/ppo.py` |
| Person 2 | `exp2/ppo-llm4teach` | Add static LLM teacher in `teacher/llm_teacher.py` |
| Person 3 | `exp3/ppo-reflexion` | Add Reflexion loop in `reflexion/` |

All branches share the `envs/` and `configs/` packages from `main`. Do not modify those unless all three agree.
