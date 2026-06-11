#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Sanity tests for LLM4Teach on Windows CPU.
Run from repo root:  python sanity_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import traceback

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = []

def run_test(name, fn):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print('='*55)
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"{FAIL} {name}")
        print(tb)
        results.append((name, False, tb))

# ──────────────────────────────────────────────────────────
# Test 1 – torch / device check
# ──────────────────────────────────────────────────────────
def test_torch():
    import torch
    print(f"{INFO} torch version : {torch.__version__}")
    print(f"{INFO} cuda available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{INFO} device        : {device}")
    # Defensive CUDA seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    assert device == "cpu", "Expected CPU-only environment"

# ──────────────────────────────────────────────────────────
# Test 2 – import Game
# ──────────────────────────────────────────────────────────
def test_import_game():
    import Game
    print(f"{INFO} Game.__file__ = {Game.__file__}")

# ──────────────────────────────────────────────────────────
# Test 3 – minimal Game initialisation (CPU, offline planner)
# ──────────────────────────────────────────────────────────
def make_args():
    from types import SimpleNamespace
    return SimpleNamespace(
        task           = "simpledoorkey",
        seed           = 42,
        frame_stack    = 1,
        offline_planner= True,
        soft_planner   = False,
        device         = "cpu",
        batch_size     = 32,
        recurrent      = False,
        gamma          = 0.99,
        lam            = 0.95,
        n_itr          = 1,
        traj_per_itr   = 1,
        num_eval       = 1,
        eval_interval  = 10,
        save_interval  = 100,
        logdir         = "log_sanity",
        policy         = "ppo",
        savedir        = "test_run_0",
        loaddir        = None,
        loadmodel      = "acmodel",
    )

game = None   # shared across tests

def test_game_init():
    global game
    from Game import Game
    args = make_args()
    game = Game(args)
    print(f"{INFO} obs_space   : {game.obs_space}")
    print(f"{INFO} action_space: {game.action_space}")
    print(f"{INFO} max_ep_len  : {game.max_ep_len}")
    print(f"{INFO} device      : {game.device}")
    assert game.device == "cpu"
    assert game.max_ep_len == 150

# ──────────────────────────────────────────────────────────
# Test 4 – teacher policy evaluation
# ──────────────────────────────────────────────────────────
def test_teacher_evaluate():
    assert game is not None, "Game must be initialised first"
    ep_return, ep_len, ep_success = game.evaluate(
        teacher_policy=True,
        record_frames=False,
        deterministic=False,
    )
    print(f"{INFO} teacher eval → return={ep_return:.3f}  len={ep_len}  success={ep_success}")
    assert isinstance(ep_return, float)
    assert isinstance(ep_len, int) and ep_len > 0

# ──────────────────────────────────────────────────────────
# Test 5 – student policy evaluation (untrained, random-ish)
# ──────────────────────────────────────────────────────────
def test_student_evaluate():
    assert game is not None
    ep_return, ep_len, ep_success = game.evaluate(
        teacher_policy=False,
        record_frames=False,
        deterministic=False,
    )
    print(f"{INFO} student eval → return={ep_return:.3f}  len={ep_len}  success={ep_success}")
    assert isinstance(ep_return, float)
    assert isinstance(ep_len, int) and ep_len > 0

# ──────────────────────────────────────────────────────────
# Test 6 – rollout collection
# ──────────────────────────────────────────────────────────
def test_collect():
    assert game is not None
    game.buffer.clear()
    game.collect()
    n = len(game.buffer)
    print(f"{INFO} buffer size after collect: {n} steps")
    assert n > 0, "Buffer should have at least 1 step after collect()"

# ──────────────────────────────────────────────────────────
# Test 7 – one training iteration (CPU)
# ──────────────────────────────────────────────────────────
def test_train():
    assert game is not None
    # Re-init a fresh game with n_itr=1 to run one clean iteration
    from Game import Game
    args = make_args()
    args.n_itr = 1
    args.traj_per_itr = 1
    args.batch_size = 32
    args.savedir = "test_train_0"
    g = Game(args)
    g.train()
    print(f"{INFO} Training completed 1 iteration on CPU. total_steps={g.total_steps}")
    assert g.total_steps > 0

# ──────────────────────────────────────────────────────────
# Run all tests in order
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_test("Test 1 – torch/device check",          test_torch)
    run_test("Test 2 – import Game",                  test_import_game)
    run_test("Test 3 – Game initialisation",          test_game_init)
    run_test("Test 4 – teacher policy evaluation",    test_teacher_evaluate)
    run_test("Test 5 – student policy evaluation",    test_student_evaluate)
    run_test("Test 6 – rollout collection",           test_collect)
    run_test("Test 7 – one training iteration",       test_train)

    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    for name, ok, _ in results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
    print(f"\n  {passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)
