#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
test_viz.py — End-to-end smoke test for the live training visualizer.

Drives viz.server._run_training_thread() directly (no browser / uvicorn needed)
with the event bus initialised, then asserts the full event stream flows:
phase_start → step (+ frame) → episode_end → iteration_end → phase_end.
Also checks Stop and reset-then-run-again.

Run:  python test_viz.py
"""

import os, sys, asyncio, threading, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viz.event_bus import VIZ


def _init_bus():
    # No subscribers in this test, so the loop is never actually used by emit();
    # initialise with a real loop object anyway to mirror the server.
    loop = asyncio.new_event_loop()
    VIZ.initialize(loop)


def _run(phase, iters=2):
    from viz import server
    cfg = {
        "phase": phase, "run_all_phases": False,
        "total_iterations": iters, "episodes_per_itr": 2,
        "ppo_epochs": 1, "batch_size": 16, "viz_speed_hz": 1000,
        "use_online_llm": False,
    }
    server._run_training_thread(cfg, threading.Event())


def main():
    _init_bus()

    # ── single phase: full event stream ───────────────────────────────────────
    _run("planner", iters=2)
    hist = VIZ.get_history(5000)
    types = set(e["type"] for e in hist)
    print("event types seen:", sorted(types))
    for needed in ["phase_start", "step", "episode_end", "iteration_end", "phase_end"]:
        assert needed in types, f"missing '{needed}' event"

    # a live frame must have been produced for the dashboard
    state = VIZ.get_state()
    assert state.get("last_frame", "").startswith("data:image/png;base64,"), \
        "no rendered frame emitted"
    # step events carry the symbolic context the dashboard shows
    step_ev = next(e for e in hist if e["type"] == "step")
    assert "action" in step_ev and "obs_text" in step_ev and "plan" in step_ev
    print(f"step sample: action={step_ev['action']!r} obs={step_ev['obs_text']!r}")

    # ── reset + run-again ─────────────────────────────────────────────────────
    VIZ.reset()
    assert len(VIZ.get_history()) == 0, "reset did not clear history"
    assert VIZ.get_state()["phase"] is None
    _run("ppo_only", iters=1)
    types2 = set(e["type"] for e in VIZ.get_history(5000))
    assert "phase_start" in types2 and "iteration_end" in types2, "run-again did not emit"
    print("reset + run-again OK")

    # ── Stop responsiveness: a pre-set stop event ends the run immediately ─────
    from viz import server
    VIZ.reset()
    stop = threading.Event(); stop.set()
    server._run_training_thread(
        {"phase": "planner", "total_iterations": 5, "episodes_per_itr": 2,
         "ppo_epochs": 1, "batch_size": 16, "viz_speed_hz": 1000}, stop)
    # phase_start fires, but run_phase breaks on the set stop event before iterating
    h = VIZ.get_history(5000)
    iters_done = [e for e in h if e["type"] == "iteration_end"]
    assert len(iters_done) == 0, f"stop ignored: {len(iters_done)} iterations ran"
    print("stop honored (0 iterations ran with pre-set stop)")

    print("\n[PASS] viz end-to-end smoke test")


if __name__ == "__main__":
    try:
        main()
    finally:
        # tidy scratch training artifacts created by build_game/run_phase
        for d in ["checkpoints", "log"]:
            try:
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
