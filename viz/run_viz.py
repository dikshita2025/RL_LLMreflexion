"""
viz/run_viz.py — Launch the MiniGrid Training Visualizer.

Usage:
    cd <project_root>
    python viz/run_viz.py                    # default: 0.0.0.0:7860
    python viz/run_viz.py --port 8080
    python viz/run_viz.py --host 127.0.0.1

Then open http://localhost:7860 in your browser.

The dashboard lets you:
  - Pick a phase (ppo_only / planner / reflection)
    or run all 3 phases sequentially
  - Configure iterations, episodes/itr, PPO epochs, batch, LR
  - Toggle Online LLM (requires Ollama + qwen2.5:3b running)
  - See the MiniGrid grid rendered in real time, cell by cell
  - Watch LLM calls (query, plan, latency, cache hit/miss)
  - Track success rate, reward, KS coefficient, entropy with live charts
  - See the full event log with interventions, episode outcomes, iterations
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="MiniGrid Training Visualizer")
    parser.add_argument("--host",  default="0.0.0.0",  help="Bind host (default 0.0.0.0)")
    parser.add_argument("--port",  default=7860, type=int, help="Port (default 7860)")
    parser.add_argument("--reload", action="store_true",   help="Hot-reload (dev mode)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MiniGrid Training Visualizer")
    print(f"  http://localhost:{args.port}")
    print("=" * 60)
    print("  Open the URL above in your browser, then press ▶ Start.")
    print()

    import uvicorn
    uvicorn.run(
        "viz.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",   # suppress per-request noise
    )


if __name__ == "__main__":
    main()
