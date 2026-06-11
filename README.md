# LLM4Teach: Large Language Models as Policy Teachers for RL Agents

**A Comprehensive Guide for Getting Started with the Codebase**

> This is an implementation of "[Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents](https://arxiv.org/abs/2311.13373)" from IJCAI 2024.

---

## Table of Contents

1. [Quick Overview](#quick-overview)
2. [Directory Structure & File Descriptions](#directory-structure--file-descriptions)
3. [Training Pipeline Explained](#training-pipeline-explained)
4. [Architecture Components](#architecture-components)
5. [Setup Instructions by Platform](#setup-instructions-by-platform)
6. [Running Your First Training](#running-your-first-training)
7. [Understanding the Output](#understanding-the-output)
8. [Troubleshooting](#troubleshooting)

---

## Quick Overview

**What is LLM4Teach?**

LLM4Teach trains a small, specialized RL agent (student) using guidance from a large language model (teacher). The system works like this:

```
┌─────────────────────────────────────────────────────────────┐
│ Teacher (LLM: Qwen)                                         │
│ ├─ Generates high-level plans ("go to key, pick it up")   │
│ ├─ Learns from past failures (reflection memory)          │
│ └─ Intervenes when student gets stuck                     │
└──────────────┬──────────────────────────────────────────────┘
               │ Guidance (high-level plans)
               │
┌──────────────▼──────────────────────────────────────────────┐
│ Student (PPO Neural Network)                               │
│ ├─ Executes low-level actions (move, turn, pick up)       │
│ ├─ Learns from environment rewards                        │
│ └─ Eventually surpasses teacher performance               │
└──────────────┬──────────────────────────────────────────────┘
               │ Actions
               │
┌──────────────▼──────────────────────────────────────────────┐
│ Environment (MiniGrid: SimpleDoorKey, etc.)               │
│ ├─ Agent navigates maze, finds key, opens door           │
│ ├─ Provides step-based rewards for progress              │
│ └─ Tracks success (episode termination when goal reached) │
└─────────────────────────────────────────────────────────────┘
```

**Key Innovation:** The teacher LLM provides *semantic guidance* (what to do) while the student PPO policy provides *robust execution* (how to do it). This hybrid approach is more efficient than pure RL or pure LLM approaches.

---

## Directory Structure & File Descriptions

```
LLM4Teach/
├── README_DETAILED.md                  # This file
├── Research_Ablation_Notebook.ipynb    # ★ Main entry: 3-phase ablation study (local + Kaggle)
│
├── main.py                             # CLI entry — train/eval one phase (used by run_train.bat)
├── Game.py                             # Training loop orchestrator (collect + PPO update + logging)
├── experiment_config.py                # ExperimentConfig + phase definitions (ppo_only/planner/reflection/…)
├── experiment_runner.py                # build_game(), run_phase(), run_research_pipeline(), metrics
├── planner.py                          # LLM-based symbolic planner (cache + confidence invalidation)
├── mediator.py                         # RL ↔ LLM translator (RL2LLM / RL2LLM_rich / LLM2RL)
├── teacher_policy.py                   # Teacher: skill dispatch + FailureDetector interventions
│
├── diagnostics.py                      # Behavior-free teacher→PPO + pickup-pipeline diagnostics
├── verify_reflection_influence.py      # Audits whether reflection ever reached the planner
├── sanity_test.py                      # Quick environment/sanity check
├── test_requirements.py                # Smoke + unit test suite (python test_requirements.py)
├── requirements.txt                    # Python dependencies (pip install -r)
│
├── algos/                              # RL algorithm
│   ├── base.py                         # PPO base / network plumbing
│   ├── buffer.py                       # Rollout buffer (obs, actions, teacher_probs, returns)
│   ├── model.py                        # Actor-Critic (MLP) network
│   └── ppo.py                          # PPO update + kickstarting (KS) loss + KS-coef schedule
│
├── skill/                              # Grounded skill controllers (emit primitive actions)
│   ├── base_skill.py                   # BaseSkill (obs decode) + Pickup / Toggle / Drop / Wait
│   ├── explore.py                      # Frontier-based exploration
│   └── goto_goal.py                    # A* navigation to a target cell
│
├── env/                                # Custom MiniGrid environments
│   ├── historicalobs.py                # Parent: persistent fog-of-war observation
│   ├── doorkey.py                      # SimpleDoorKey (main benchmark)
│   ├── coloreddoorkey.py               # Colored keys/doors variant
│   ├── lavadoorkey.py                  # Lava-obstacle variant
│   └── twodoor.py                      # Two-door variant (room 17–20)
│
├── memory/                             # Reflection & experience storage
│   ├── reflection.py                   # QwenReflector + cluster/coordinate guards
│   └── memory_buffer.py                # ReflectionMemory (hash dedup, clusters, top-k retrieval)
│
├── simulator/                          # Logging backbone
│   ├── trajectory_logger.py            # ★ Per-step + per-episode logger (used by Game)
│   ├── live_dashboard.py               # LEGACY streamlit dashboard (CLI only; see viz/)
│   ├── visualize_training.py           # LEGACY matplotlib plots (CLI only)
│   └── render_episode.py               # LEGACY episode renderer (CLI only)
│
├── viz/                                # ★ Live web dashboard (run_viz.bat)
│   ├── server.py                       # FastAPI app: /start /stop /ws (WebSocket) + dashboard
│   ├── run_viz.py                      # Launcher (python viz/run_viz.py --port 7860)
│   ├── event_bus.py                    # Thread→async event bus consumed by Game.collect()
│   ├── dashboard.html                  # Single-file UI (grid, charts, event log)
│   └── requirements_viz.txt            # fastapi + uvicorn
│
├── utils/                              # Utilities
│   ├── qwen_llm.py                     # Qwen LLM client (Ollama; DashScope optional)
│   ├── symbolic_parser.py              # Strict symbolic-plan grammar validator
│   ├── env.py                          # make_env_fn / WrapEnv / obs preprocessor
│   ├── log.py                          # create_logger (log dir + config dump)
│   ├── format.py, global_param.py, logo.py   # misc helpers (CLI)
│   └── chatglm-api.py                  # LEGACY ChatGLM client (unused)
│
├── prompt/
│   └── task_info.json                  # Per-task env configs + prompt metadata
│
├── notebooks/                          # Legacy/auxiliary notebooks (not the main flow)
│   ├── kaggle_train.ipynb
│   ├── reflection_debugger.ipynb
│   └── reflection_llmplanner_experiments.ipynb
│
├── setup.bat                           # Windows one-time setup (venv + deps)
├── run_viz.bat                         # Launch the live web dashboard (viz/)
├── run_train.bat                       # CLI training launcher (main.py)
│
├── log/                                # Per-run logs (created at runtime)
│   └── ppo/<task>/.../trajectory/      # training.log, trajectories.jsonl, prompts.log, metrics.csv
├── results/                            # Per-phase outputs (created at runtime; download these)
│   └── <phase>/                        #   phase ∈ ppo_only | planner | reflection
│       ├── metrics.csv                 #   per-ITERATION metrics (deltas, totals, agreement, …)
│       ├── episodes.csv                #   per-EPISODE metrics (episode_id, iteration_id, …)
│       ├── history.json                #   full history arrays
│       ├── eval_metrics.json           #   final evaluation
│       ├── llm_stats.json              #   planner/cache/reflector/memory counters
│       └── plots/                      #   training + ablation PNGs
└── checkpoints/                        # Saved PPO weights (created at runtime)
```

### **File Descriptions (Key Files)**

#### **Core Training Files**

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `Research_Ablation_Notebook.ipynb` | ★ Main flow — runs the 3-phase ablation (local + Kaggle) | edit Cell 3 only |
| `experiment_config.py` | Phase definitions + per-phase config | `ExperimentConfig`, `build_experiment_config()` |
| `experiment_runner.py` | Builds games, runs phases, computes/saves metrics | `build_game()`, `run_phase()`, `run_research_pipeline()` |
| `main.py` | CLI entry — parses args, runs one phase (used by `run_train.bat`) | `train()`, `eval()` |
| `Game.py` | Training loop - runs episodes, stores data, trains PPO | `Game.collect()`, `Game.evaluate()` |
| `planner.py` | LLM-based symbolic planner - generates high-level plans | `Planner.plan()`, cache management |
| `teacher_policy.py` | Failure detection & intervention - decides when LLM helps | `TeacherPolicy()`, `FailureDetector` |
| `mediator.py` | Translates between RL observations and LLM language | `RL2LLM()`, `LLM2RL()` |

#### **RL Components**

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `algos/base.py` | Base PPO / network plumbing | forward/backward passes |
| `algos/buffer.py` | Rollout buffer | `store()`, `sample()`, `ep_returns`, `ep_lens` |
| `algos/model.py` | Actor-Critic network | `ActorCritic` (MLP) |
| `algos/ppo.py` | PPO + kickstarting | `update_policy()`, KS loss, `ks_coef` schedule |

#### **Skills (Grounded Behaviors)**

| File | Purpose | How It Works |
|------|---------|--------------|
| `skill/base_skill.py` | Base skill class, observation parsing | Unpacks obs into (agent_pos, agent_dir, carrying, map) |
| `skill/explore.py` | Frontier-based exploration | Explores unseen areas, tracks visited cells, detects oscillation |
| `skill/goto_goal.py` | Navigation to target | Pathfinding to coordinate, handles obstacles |

#### **Environments**

| File | Task | Difficulty |
|------|------|-----------|
| `env/doorkey.py` | Find key → pick up → navigate to door → open | Easy |
| `env/coloreddoorkey.py` | Match colored key to colored door | Medium |
| `env/lavadoorkey.py` | Navigate around lava obstacles | Hard |
| `env/twodoor.py` | Two sequential doors to open | Hard |

#### **Memory & Reflection**

| File | Purpose | Output |
|------|---------|--------|
| `memory/reflection.py` | Generates semantic reflections from failed episodes | Text: "Agent failed to pick up key before opening door..." |
| `memory/memory_buffer.py` | Stores up to 20 recent reflections, retrieves top-5 relevant | Injected into planner prompts |

#### **Logging, Visualization & Diagnostics**

| File | Purpose | Output |
|------|---------|--------|
| `simulator/trajectory_logger.py` | Per-step + per-episode logger (used by `Game`) | `trajectories.jsonl`, `training.log`, per-episode `metrics.csv` |
| `experiment_runner.py` | Per-iteration metrics + saving | `results/<phase>/metrics.csv`, `episodes.csv`, `history.json`, `llm_stats.json` |
| `viz/server.py` + `viz/run_viz.py` | **Live web dashboard** (FastAPI + WebSocket) — `run_viz.bat` | Real-time grid, LLM calls, success/KS/entropy charts |
| `viz/event_bus.py` | Thread→async bridge consumed by `Game.collect()` | live `step` / `episode_end` events |
| `diagnostics.py` | Behavior-free teacher→PPO + pickup-pipeline audit | `results/diagnostics.txt` |
| `verify_reflection_influence.py` | Checks whether reflection reached the planner | console verdict |
| `simulator/live_dashboard.py`, `visualize_training.py`, `render_episode.py` | LEGACY matplotlib/streamlit viz (CLI `main.py` only) | superseded by `viz/` |

---

## Training Pipeline Explained

### **High-Level Overview**

The training loop is an iteration of **data collection → policy training → evaluation**:

```
┌─────────────────────────────────────────────────────────────────┐
│ TRAINING LOOP (main.py → Game.train())                         │
└─────────────────────────────────────────────────────────────────┘

FOR each iteration (e.g., 50 iterations, each with 500 episodes):

   STEP 1: DATA COLLECTION
   ───────────────────────────────────────────────────────────────
   │
   ├─ FOR each episode:
   │  │
   │  ├─ Reset environment & agents
   │  ├─ FOR each timestep (max 150 steps):
   │  │  │
   │  │  ├─ [PPO NORMAL PATH] (90% of time)
   │  │  │  ├─ Student PPO policy generates action
   │  │  │  └─ Execute in environment
   │  │  │
   │  │  └─ [TEACHER INTERVENTION] (10% of time, on failures)
   │  │     ├─ Failure detector triggers (stuck, oscillation, timeout)
   │  │     ├─ Teacher generates plan: "go to key, pick up, go to door, open"
   │  │     ├─ Skill controllers execute high-level plan
   │  │     └─ Student learns from this trajectory
   │  │
   │  ├─ Episode ends (success or timeout)
   │  ├─ Generate reflection: "Agent failed to navigate..."
   │  └─ Store in replay buffer
   │
   └─ Collect ~50k steps per iteration

   STEP 2: POLICY TRAINING
   ───────────────────────────────────────────────────────────────
   │
   ├─ Sample minibatches from replay buffer
   ├─ Compute PPO loss: Actor loss + Critic loss + Entropy bonus
   ├─ Backprop & update neural network weights
   └─ Repeat for 3 epochs

   STEP 3: LOGGING & MONITORING
   ───────────────────────────────────────────────────────────────
   │
   ├─ Write metrics to CSV: success_rate, avg_reward, llm_calls, etc.
   ├─ Log to TensorBoard
   └─ Save checkpoint: acmodel.pt (neural network weights)

   STEP 4: EVALUATION (optional, every N iterations)
   ───────────────────────────────────────────────────────────────
   │
   └─ Run test episodes without teacher intervention
      (measure if student has learned to solve task independently)

```

### **Detailed Step-by-Step: One Episode**

Let's trace a single episode from start to finish:

```
EPISODE 591 (from actual training logs)
═══════════════════════════════════════════════════════════════════

STEP 0: RESET
────────────
• Environment spawns: agent at (7,2), key at (4,3), door at (1,1)
• Teacher policy resets (clears failure counters)
• Observation: "Agent sees <nothing>, holds <nothing>"

STEPS 1-50: EXPLORATION (PPO learning)
───────────────────────────────────────
• Step 1-10: PPO explores randomly
  Action sequence: turn_left, turn_right, move_forward, move_forward, ...
  Observation: still "sees <nothing>"

• Step 51: Observation changes!
  Observation: "Agent sees <key>, holds <nothing>"
  → Agent has discovered the key location
  → Teacher is watching for failures now

• Steps 52-114: Agent approaches key
  Action sequence: move_forward, turn_left, move_forward, ...
  Failure detector tracks: position, oscillation, progress
  Observations: Still "sees <key>"

• Step 115: KEY PICKUP SUCCESS
  Observation: "Agent sees <nothing>, holds <key>"
  → Trajectory logger detects: hold changed from "nothing" to "key"
  → This is progress! Reset intervention cooldown

• Steps 116-124: Navigate toward door
  Action sequence: move_forward, move_forward, ...
  Observations: "sees <nothing>, holds <key>"
  → No interventions needed yet, PPO is making progress

• Step 125-131: Find door
  Observation changes: "Agent sees <door>, holds <key>"
  → System detects final goal is visible
  → Prepare for completion

STEP 132: GOAL ACHIEVED (door_open)
────────────────────────────────────
• Action: toggle_open (the door)
• Environment returns: reward = 0.202, terminated = True
• Trajectory logger records: success=1, reward=0.202, steps=132

EPISODE END: POST-EPISODE PROCESSING
─────────────────────────────────────
• Generate reflection:
  "Agent successfully found and picked up key, then navigated to
   door and opened it. Key pattern: explore → find key → go to door."

• Store in reflection memory (top 20 most recent)

• Add trajectory to replay buffer for PPO training

• Metrics recorded:
  episode, reward=0.202, success=1, steps=132, llm_calls=8,
  interventions=0, reflections=1, ...

═══════════════════════════════════════════════════════════════════
```

### **When Does Teacher Intervene?**

Teacher (LLM) intervenes when failure detectors trigger:

```
FAILURE DETECTOR THRESHOLDS (from teacher_policy.py)
═══════════════════════════════════════════════════════════════════

STUCK DETECTION (threshold: 20 steps)
│
├─ If agent position hasn't changed for 20 consecutive steps
├─ Teacher generates plan to unstick: "explore new areas"
└─ LLM call made, new plan cached

OSCILLATION DETECTION (threshold: 16 step window)
│
├─ If agent repeats same action 8+ times in 16 steps
│  (e.g., left-right-left-right-...)
├─ Teacher intervenes: "explore a different direction"
└─ LLM call made, cache invalidated

FAILED INTERACTION (threshold: 5 consecutive failures)
│
├─ If toggle or pickup action fails 5 times in a row
├─ Teacher rplans: "you're missing something, re-explore"
└─ LLM call made

NO PROGRESS (20 steps without seeing anything new)
│
├─ If agent's observation hasn't changed meaningfully
├─ Teacher suggests: "you need to move in a different direction"
└─ LLM call made

INTERVENTION COOLDOWN: 30 steps
│
└─ After teacher intervenes, wait 30 steps before next intervention
   (let student practice the new plan)
```

### **Reflection Memory Integration**

```
HOW REFLECTION HELPS THE TEACHER
═══════════════════════════════════════════════════════════════════

COLLECT PHASE:
  Episode 4: Failed to find key
  → Reflection: "Agent explored but never reached key area"
  → Stored in memory

COLLECT PHASE:
  Episode 7: Failed to pick up key
  → Reflection: "Agent saw key but didn't approach it directly"
  → Stored in memory
  
  ...many episodes...

PLANNER RECEIVES NEW OBSERVATION:
  Obs: "Agent sees nothing, in middle of maze"
  
  → Planner queries: "What did we learn about exploring?"
  → Retrieves top-3 relevant reflections:
     1. "Explore systematically, don't oscillate"
     2. "Once you see a target, go directly to it"
     3. "Mark visited areas to avoid loops"
  
  → Adds to LLM prompt:
     "Past experience: [3 reflections]
      Current state: [observation]
      Generate new plan..."
  
  → Qwen generates better plan using lessons learned

RESULT:
  Teacher makes smarter suggestions, student learns faster
```

---

## Architecture Components

### **1. PPO Algorithm (Student Policy)**

**What it does:** Neural network that learns to generate good actions from observations.

**How it works:**
```
Observation (from environment)
        ↓
   [Neural Network]
        ↓
   Policy Head → Action probabilities (softmax)
   Value Head → Estimated future reward (scalar)

Loss = Actor loss + Critic loss + Entropy bonus
    Actor: maximize expected reward
    Critic: minimize TD error (value prediction error)
    Entropy: encourage exploration (don't get stuck)

Backprop → Update network weights → Better actions next time
```

**Configuration:**
- Network type: MLP (2 hidden layers, 512 neurons each) or LSTM (recurrent)
- Optimizer: Adam
- Learning rate: varies with training
- Entropy coefficient: 0.01 (exploration bonus)

### **2. Failure Detector**

**What it does:** Monitors agent behavior and detects when it's stuck/failing.

**Triggers:**
- Same position for 20+ steps → STUCK
- Alternating actions for 16 steps → OSCILLATING  
- Same failed action 5+ times → FAILED_INTERACTION
- No new observations → NO_PROGRESS

### **3. Planner (LLM Teacher)**

**What it does:** Uses Qwen LLM (running locally via Ollama) to generate symbolic plans.

**Input:** Observation text + reflection memory
**Output:** Plan list like `["go to <key>", "pick up <key>", "go to <door>", "open <door>"]`

**Caching:** Plans are cached by observation text. If same observation seen again, reuse plan (avoid redundant LLM calls).

**Cache invalidation:** When a plan fails 3+ times, discard it and ask LLM for new plan.

### **4. Skill Controllers**

Skills are **state machines** that verify pre/post conditions:

```
PICKUP SKILL:
─────────────
State: CHECK
  ├─ Precondition: Is there a pickable object in front?
  ├─ YES → Action: pickup (action_id=3), go to VERIFY
  └─ NO → Fail: "no pickable object"

State: VERIFY (after pickup is executed)
  ├─ Postcondition: Are we now carrying something?
  ├─ YES → Success: done
  ├─ NO → Retry (max 2 retries)
  └─ Out of retries → Fail: "pickup failed after 2 attempts"

Similar state machines for:
  • Explore (visit unseen frontier)
  • GoTo (navigate to coordinate)
  • Toggle (open/close door)
  • Wait (idle)
  • Drop (release carried object)
```

### **5. Mediator (RL ↔ LLM Bridge)**

Converts between two domains:

```
RL DOMAIN (numbers)                 LLM DOMAIN (language)
───────────────────────────────────────────────────────

obs = [7, 2, 0, 1, ...]        ←→  "Agent at (7,2), sees
                                    <nothing>, holds <nothing>,
                                    door ahead to north"

action = 2 (move_forward)      ←→  "move_forward"

reward = 0.202                 ←→  "good progress! +0.2 reward"

Agent position tracked         ←→  "going from A to B"
```

**RL2LLM:** Converts observation arrays → natural language description
**LLM2RL:** Converts plan strings → skill controller calls → primitive actions

---

## Setup Instructions by Platform

### **Platform 1: Linux / Mac (Local Machine)**

#### Prerequisites
- Python 3.11+
- `curl` (pre-installed on most systems)
- ~30GB disk space (for Ollama models)

#### Step 1: Run Setup Script

```bash
# Download and extract the project
cd ~/Downloads
# (Make sure you have LLM4Teach-main folder)

cd LLM4Teach-main

# Run setup (takes 10-15 minutes first time)
bash setup.sh
```

**What this does:**
1. Creates Python virtual environment (`venv/`)
2. Installs PyTorch (CPU version, ~500MB)
3. Installs RL libraries: gymnasium, minigrid, numpy, tensorboard, etc.
4. Downloads & installs Ollama (~100MB CLI)
5. Pulls Qwen models (~6.7GB total):
   - `qwen2.5:3b` (2GB, for planner)
   - `qwen2.5:7b` (4.7GB, for reflection)

#### Step 2: Start Training

```bash
bash run_train.sh
```

**What happens:**
1. Activates virtual environment
2. Checks if Ollama is running (starts if needed)
3. Launches training with default settings:
   - Task: SimpleDoorKey
   - Planner: qwen2.5:3b via Ollama
   - Reflection: qwen2.5:7b via Ollama
   - Duration: 50 iterations × 500 episodes = 25,000 episodes

#### Monitoring Training

```bash
# In another terminal, watch training progress with TensorBoard
tensorboard --logdir=log
# Then open: http://localhost:6006
```

---

### **Platform 2: Windows (Local Machine)**

#### Prerequisites
- Python 3.11+ from [python.org](https://www.python.org/downloads/)
- Git Bash or PowerShell
- ~30GB disk space

#### Step 1: Install Ollama

Download & install from: **https://ollama.ai/download/windows**

After installation, open a **NEW terminal** and pull models:
```powershell
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
```

(Takes ~10 minutes, shows download progress)

#### Step 2: Run Setup Script

```batch
# In Command Prompt or PowerShell
cd \Users\YourUsername\Downloads\LLM4Teach-main

setup.bat
```

Press `Enter` when it asks you to (takes 5-10 minutes).

#### Step 3: Start Training

Open a **NEW terminal** and run:

```batch
run_train.bat
```

The training will start! You should see:
```
Starting training with Qwen planner + reflection...
(Ollama must already be running: ollama serve)

Iteration 1...
Episode 1 | Reward: 0.0 | Success: No | Steps: 150
Episode 2 | Reward: 0.1 | Success: Yes | Steps: 95
...
```

#### Monitoring Training

```batch
# In another terminal
tensorboard --logdir=log
```

Then open: http://localhost:6006

---

### **Platform 3: Kaggle (Cloud GPU, Free)**

Kaggle provides free GPU and 30GB storage. Perfect for training!

#### Step 1: Upload Project

1. Go to [kaggle.com](https://kaggle.com) (create account if needed)
2. Create **Dataset**:
   - Click "Create" → "Dataset"
   - Upload `LLM4Teach-main.zip`
   - Name it: `llm4teach`
   - Set to "Public"
   - Click "Create"

3. Create **Notebook**:
   - Click "Create" → "Notebook"
   - Select "Python"
   - In right sidebar: "Add Data" → Select your `llm4teach` dataset

#### Step 2: Run Setup Cells

Copy-paste **each cell** from `kaggle_setup.sh` into separate Kaggle cells:

**CELL 1 - Install Libraries:**
```bash
cd /kaggle/working

# Unzip your uploaded project
unzip -q /kaggle/input/llm4teach/LLM4Teach-main.zip -d .

# Install Python libraries
pip install -q torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q \
    gymnasium==1.3.0 \
    minigrid==3.1.0 \
    numpy==1.26.4 \
    tensorboard==2.20.0 \
    opencv-python-headless==4.9.0.80 \
    matplotlib==3.10.0 \
    streamlit==1.35.0 \
    requests==2.31.0

echo "✅ Python libraries installed."
```

**CELL 2 - Install Ollama:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
echo "✅ Ollama installed."
```

**CELL 3 - Start Ollama Server:**
```bash
ollama serve &> /tmp/ollama.log &
sleep 5

# Confirm it's alive
curl -sf http://localhost:11434/api/tags && echo "✅ Ollama server is running." || echo "❌ Ollama failed"
```

**CELL 4 - Pull Models (takes ~15 minutes):**
```bash
echo "Pulling qwen2.5:3b (planner, ~2 GB)..."
ollama pull qwen2.5:3b

echo "Pulling qwen2.5:7b (reflector, ~4.7 GB)..."
ollama pull qwen2.5:7b

echo "✅ Both models ready."
ollama list
```

**CELL 5 - Run Training:**
```bash
cd /kaggle/working/LLM4Teach-main

python main.py train \
    --task SimpleDoorKey \
    --savedir run1 \
    --llm_backend ollama \
    --llm_model qwen2.5:3b \
    --reflection \
    --reflection_backend ollama \
    --reflection_model qwen2.5:7b
```

Training will run for ~2-4 hours (depending on GPU).

**CELL 6 (Optional) - Save Outputs:**
```bash
cp -r /kaggle/working/LLM4Teach-main/log /kaggle/working/output_logs
cp -r /kaggle/working/LLM4Teach-main/saved_model /kaggle/working/output_models 2>/dev/null || true

echo "✅ Logs and models saved"
```

#### Important Kaggle Notes:
- Kaggle notebooks timeout after 9 hours, but training saves checkpoints automatically
- Output is saved to `/kaggle/working/output_*` for permanent storage
- You can resume training by loading the checkpoint
- Free GPU (T4 or P100) is sufficient for this task

---

### **Platform 4: Docker (Containerized)**

Docker packages everything into an isolated environment. Great for reproducibility!

#### Prerequisites
- Docker & Docker Compose installed ([get here](https://docs.docker.com/get-docker/))
- ~20GB disk space (final image)

#### Step 1: Build Docker Image

```bash
cd LLM4Teach-main

docker build -t llm4teach .
```

(Takes 10-20 minutes - downloads all dependencies, pulls Qwen models, builds image)

#### Step 2: Run Training

```bash
docker compose up train
```

Output logs to console:
```
==> Starting Ollama server (CPU mode)...
==> Ollama is ready.
==> Running: python main.py train ...

Iteration 1...
Episode 1 | Reward: 0.0
Episode 2 | Reward: 0.202
...
```

#### Monitoring (in separate terminals)

**TensorBoard:**
```bash
docker compose up tensorboard
# Visit: http://localhost:6006
```

**Streamlit Dashboard (live plots):**
```bash
docker compose up dashboard
# Visit: http://localhost:8501
```

**Jupyter Notebook (interactive):**
```bash
docker compose up jupyter
# Visit: http://localhost:8888
```

#### Stop Training
```bash
docker compose down
```

---

## Running Your First Training

### **Quick Start Command (All Platforms)**

After setup is complete, just run:

**Linux/Mac:**
```bash
bash run_train.sh
```

**Windows:**
```batch
run_train.bat
```

**Docker:**
```bash
docker compose up train
```

### **Training Configuration Options**

You can customize training by modifying the command:

```bash
python main.py train \
    --task SimpleDoorKey                    # Environment (see env/ folder)
    --savedir run1                          # Output directory name
    --llm_backend ollama                    # "ollama" or "dashscope" (cloud)
    --llm_model qwen2.5:3b                  # LLM model to use
    --reflection                            # Enable reflection learning
    --reflection_backend ollama             # Reflection LLM backend
    --reflection_model qwen2.5:7b           # Reflection model
    --n_itr 50                              # Number of training iterations
    --traj_per_itr 500                      # Episodes per iteration
    --batch_size 128                        # PPO batch size
    --learning_rate 0.001                   # Learning rate
```

### **Testing Without Training (Sanity Check)**

Quick test to verify everything works:

```bash
python sanity_test.py
```

Should show:
```
Testing SimpleDoorKey environment...
✅ Environment reset successfully
✅ Step 1: action=2, reward=0.0, done=False
✅ Step 2: action=0, reward=0.0, done=False
✅ All systems operational!
```

---

## Understanding the Output

### **Directory Structure After Training**

After first training run, you'll see:

```
log/
├── ppo/
│   └── simpledoorkey/
│       └── run1/                          # Your --savedir
│           ├── acmodel.pt                 # Trained neural network (weights)
│           ├── config.json                # Hyperparameters used
│           ├── events.out.tfevents.*      # TensorBoard data
│           └── logs/
│               ├── metrics.csv            # Key metrics per episode
│               ├── training.log           # Detailed event log
│               ├── trajectories.jsonl     # Every step of every episode
│               ├── reflection_memory.json # Reflections learned
│               └── debug.log              # Errors/warnings
```

### **Metrics CSV Columns**

`log/.../metrics.csv` contains:

| Column | Meaning | Example |
|--------|---------|---------|
| episode | Episode number | 1, 2, 3, ... |
| reward | Total episode reward | 0.0, 0.202, 0.5 |
| success | Did agent reach goal? | 0 or 1 |
| steps | Number of steps taken | 150, 95, 73 |
| llm_calls | How many times teacher planned | 5, 10, 0 |
| interventions | How many failures detected | 2, 0, 1 |
| reflections | How many reflections generated | 1, 0, 0 |
| cache_invalidations | Plans that failed & were replaced | 0, 1, 0 |
| avg_ppo_entropy | PPO policy randomness | 1.8, 1.6, 1.4 |

### **TensorBoard Dashboard**

Run `tensorboard --logdir=log` then open http://localhost:6006

You'll see charts:
- **Episode Reward**: Goes from ~0 to ~0.5 as agent learns
- **Success Rate**: Percentage of episodes that reach goal (improves over time)
- **LLM Calls**: How often teacher intervenes (should decrease as student learns)
- **PPO Entropy**: Policy randomness (lower = more confident)
- **Failure Attribution**: What causes failures (exploration_failure, timeout, etc.)

### **Example Good Training Run**

```
Iteration 1 (Episodes 1-500):
  Avg Reward: 0.01
  Success Rate: 0%
  LLM Calls: 50
  Status: Learning phase - agent mostly exploring

Iteration 5 (Episodes 2001-2500):
  Avg Reward: 0.12
  Success Rate: 1.2%
  LLM Calls: 36
  Status: Student making progress

Iteration 25 (Episodes 12001-12500):
  Avg Reward: 0.28
  Success Rate: 3.2%
  LLM Calls: 20
  Status: Getting good! Teacher intervenes less

Iteration 50 (Episodes 24501-25000):
  Avg Reward: 0.35
  Success Rate: 4.1%
  LLM Calls: 12
  Status: Student has learned, teacher barely needed
```

---

## Troubleshooting

### **Issue: "Ollama command not found"**

**Solution:**
- Linux/Mac: Run `curl -fsSL https://ollama.ai/install.sh | sh` manually
- Windows: Download from https://ollama.ai/download/windows and run installer
- Verify: Type `ollama --version` in terminal

### **Issue: "CUDA out of memory"**

**Solution:**
- This shouldn't happen - code uses CPU by default
- If GPU error: reinstall torch CPU version
  ```bash
  pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
  ```

### **Issue: Ollama models won't download**

**Solution:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Manually pull models
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b

# Verify they're installed
ollama list
```

### **Issue: "Module not found: gymnasium"**

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate.bat  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### **Issue: Training is very slow**

**Possible causes:**
1. **Ollama models not cached**: First episode is slow while Ollama loads models
2. **LLM calls too frequent**: Reflection overhead, try `--no-reflection`
3. **Dataset too large**: trajectories.jsonl growing large (746MB possible)

**Solutions:**
- Increase `--batch_size` to 256
- Disable reflection: remove `--reflection` flag
- Use smaller model: `--llm_model qwen2.5:1.5b` (if available)

### **Issue: No progress after 10 iterations**

**Likely causes:**
1. Environment too hard, teacher not helping
2. Hyperparameters suboptimal
3. Reflection memory hurting more than helping

**Try:**
```bash
# Disable reflection to see if it's the culprit
python main.py train --task SimpleDoorKey --savedir test_no_reflection

# Use more aggressive teacher
# (Modify teacher_policy.py: stuck_threshold=10 instead of 20)

# Try different environment
python main.py train --task ColoredDoorKey --savedir test_colored
```

### **Issue: Docker build fails**

**Solution:**
```bash
# Clear Docker cache and rebuild
docker system prune -a
docker build -t llm4teach .
```

---

## Common Questions

### **Q: What's the difference between student and teacher?**

**A:** 
- **Teacher (LLM Qwen)**: Slow but smart. Generates high-level plans like "go to key, pick it up, go to door". Helps when student is stuck.
- **Student (PPO)**: Fast but initially naive. Learns to execute primitive actions (move, turn, pick up). Gets faster and more independent over time.

### **Q: Why do we need both?**

**A:** 
- Pure RL: Takes millions of steps to learn, very sample-inefficient
- Pure LLM: Too slow for real-time, can't handle unseen variants of task
- Hybrid: RL is fast, learns quickly, LLM provides guidance when stuck. Best of both worlds!

### **Q: How long does training take?**

**A:**
- Local CPU: 2-4 hours per 5000 episodes
- Local GPU: 30-60 min per 5000 episodes
- Kaggle free GPU: 2-3 hours per 5000 episodes

### **Q: Can I train on different tasks?**

**A:** Yes! Available tasks in `env/`:
```
SimpleDoorKey   (Easy, 1 key-door pair)
ColoredDoorKey  (Medium, colored keys/doors must match)
LavaDoorKey     (Hard, obstacles to avoid)
TwoDoor         (Hard, two sequential doors)
```

Use: `python main.py train --task ColoredDoorKey --savedir run_colored`

### **Q: How do I use a trained model for evaluation?**

**A:**
```bash
python main.py eval \
    --task SimpleDoorKey \
    --loaddir run1 \
    --savedir eval_run1 \
    --n_episodes 100
```

### **Q: Can I modify the LLM prompts?**

**A:** Yes! See `prompt/` folder. Modify task descriptions to change how LLM thinks about the problem.

### **Q: Is there a GUI to visualize training?**

**A:** Yes! Three options:
1. **TensorBoard**: `tensorboard --logdir=log` (best for metrics)
2. **Streamlit**: `streamlit run simulator/live_dashboard.py` (live dashboard)
3. **Jupyter**: `jupyter notebook` (analysis notebooks)

---

## Next Steps 

1. **Run sanity test first**: `python sanity_test.py`
2. **Start with SimpleDoorKey**: Easiest task to debug with
3. **Monitor with TensorBoard**: See learning progress in real-time
4. **Try modifying hyperparameters**: See impact on learning
5. **Experiment with reflection**: Compare with/without reflection learning

---

## Reference: Key Hyperparameters

In `main.py`, you can adjust:

```python
# PPO Training
--n_itr 50                      # Iterations (default: 50)
--traj_per_itr 500              # Episodes per iteration
--batch_size 128                # PPO batch size
--learning_rate 0.001           # Learning rate
--gamma 0.99                    # Discount factor
--gae_lambda 0.95               # GAE parameter

# LLM Teacher
--llm_backend ollama            # "ollama" or "dashscope"
--llm_model qwen2.5:3b          # Model name
--llm_temperature 0.7           # Randomness (0=deterministic, 1=random)

# Reflection
--reflection                    # Enable/disable
--reflection_backend ollama     # Reflection LLM backend
--reflection_model qwen2.5:7b   # Reflection model
--reflection_maxlen 20          # Max reflections to keep in memory
--reflection_top_k 5            # Top-k reflections to inject in prompt

# Teacher Interventions
--offline false                 # Use LLM (false) vs precomputed plans (true)
```



## Support & Resources

- **Paper**: https://arxiv.org/abs/2311.13373
- **Original Repo**: https://github.com/ZJLAB-AMMI/LLM4Teach
- **Ollama**: https://ollama.ai
- **MiniGrid**: https://github.com/Farama-Foundation/Minigrid
- **Gymnasium**: https://gymnasium.farama.org/
- **PyTorch**: https://pytorch.org/

---
