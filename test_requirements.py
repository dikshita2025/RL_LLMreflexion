#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
test_requirements.py — Smoke + unit tests for the 15-requirement spec.

Run from repo root:  python test_requirements.py

Fast by design (no training to convergence): each phase only builds and runs a
couple of rollouts; skills and memory are unit-tested with scripted observations.
"""

import os
import sys
import json
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
INFO = "[INFO]"
results = []


def run_test(name, fn):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    try:
        status = fn()
        if status == "skip":
            print(f"{SKIP} {name}")
            results.append((name, "skip"))
        else:
            print(f"{PASS} {name}")
            results.append((name, True))
    except Exception:
        print(f"{FAIL} {name}")
        print(traceback.format_exc())
        results.append((name, False))


# ── scripted-observation helper ───────────────────────────────────────────────

def make_obs(w, h, agent_pos, agent_dir, objects=None, carrying=None):
    """
    Build a (w, h, 4) observation array.

    channel 0 = object type (default empty=1)
    channel 1 = color
    channel 2 = state
    channel 3 = agent dir at the agent cell, 4 everywhere else

    objects  : list of (pos, type, color, state)
    carrying : (type, color) placed at the agent cell (encodes what is held)
    """
    obs = np.zeros((w, h, 4), dtype=np.uint8)
    obs[:, :, 0] = 1     # empty
    obs[:, :, 3] = 4     # "no agent" marker
    for (pos, otype, color, state) in (objects or []):
        obs[pos[0], pos[1], 0] = otype
        obs[pos[0], pos[1], 1] = color
        obs[pos[0], pos[1], 2] = state
    ar, ac = agent_pos
    obs[ar, ac, 3] = agent_dir
    if carrying is not None:
        obs[ar, ac, 0] = carrying[0]
        obs[ar, ac, 1] = carrying[1]
    else:
        obs[ar, ac, 0] = 1
    return obs


# ── 1. pipeline builds for the 3 phases (req 2, 4) ────────────────────────────

def test_pipeline_builds():
    from experiment_config import build_experiment_config
    from experiment_runner import build_game
    from Game import _NoTeacher

    overrides = dict(total_iterations=2, batch_size=16, episodes_per_iteration=2,
                     ppo_epochs=1, num_eval=1, eval_interval=100, save_interval=100)
    for phase in ["ppo_only", "planner", "reflection"]:
        cfg = build_experiment_config(phase, overrides=overrides)
        game = build_game(cfg, task="SimpleDoorKey", device="cpu", seed=0)

        if phase == "ppo_only":
            assert isinstance(game.teacher_policy, _NoTeacher), \
                "ppo_only must use the uniform _NoTeacher"
        else:
            assert not isinstance(game.teacher_policy, _NoTeacher), \
                f"{phase} must use the real TeacherPolicy"

        game.buffer.clear()
        for _ in range(2):
            game.collect()
        assert len(game.buffer) > 0, f"{phase}: empty buffer after collect"
        game.student_policy.update_policy(game.buffer, 0.0)
        print(f"{INFO} {phase:<10} built + 2 rollouts + 1 update OK "
              f"(buffer={len(game.buffer)}, teacher={type(game.teacher_policy).__name__})")


# ── 2. backward-compatible Game(args) positional call (req 2) ─────────────────

def test_game_positional():
    from types import SimpleNamespace
    from Game import Game
    args = SimpleNamespace(
        task="simpledoorkey", seed=1, frame_stack=1, offline_planner=True,
        soft_planner=False, device="cpu", batch_size=16, recurrent=False,
        gamma=0.99, lam=0.95, n_itr=1, traj_per_itr=1, num_eval=1,
        eval_interval=10, save_interval=100, logdir="log_test", policy="ppo",
        savedir="test_req_pos", loaddir=None, loadmodel="acmodel",
    )
    game = Game(args)   # no exp_config → legacy behavior
    assert game.exp_config is None
    assert game.max_ep_len == 150
    print(f"{INFO} Game(args) positional OK, max_ep_len={game.max_ep_len}")


# ── 3. TwoDoor room size 17-20 (req 1) ────────────────────────────────────────

def test_twodoor_size():
    import gymnasium as gym
    import env  # noqa: F401  (registers the env ids)

    e = gym.make("MiniGrid-TwoDoor-Min17-Max20")
    u = e.unwrapped
    assert u.minRoomSize == 17 and u.maxRoomSize == 20, \
        f"expected Min17-Max20, got Min{u.minRoomSize}-Max{u.maxRoomSize}"

    # Room size = bounding box of non-empty cells. The room interior is None in
    # the MiniGrid grid; only the boundary walls / doors / key are set, so the
    # max populated column/row equals sizeX-1 / sizeY-1 regardless of where doors
    # sit on the walls.
    sizes = set()
    for s in range(40):
        obs, _ = e.reset(seed=s)
        maxx = maxy = 0
        for x in range(u.grid.width):
            for y in range(u.grid.height):
                if u.grid.get(x, y) is not None:
                    maxx = max(maxx, x)
                    maxy = max(maxy, y)
        sizes.add(maxx + 1)
        sizes.add(maxy + 1)
        assert obs["image"].shape == (20, 20, 4), \
            f"obs canvas must stay 20x20x4, got {obs['image'].shape}"
    assert sizes, "no room sizes observed"
    assert min(sizes) >= 17 and max(sizes) <= 20, f"room sizes out of range: {sorted(sizes)}"
    print(f"{INFO} observed TwoDoor room sizes: {sorted(sizes)} (obs canvas fixed 20x20x4)")


# ── 4. reflection memory: dedup / cluster / retrieval / guards (req 9-12,15) ──

def test_memory():
    from memory.memory_buffer import ReflectionMemory
    from memory.reflection import CLUSTER_LABELS

    mem = ReflectionMemory(maxlen=20, top_k=5)

    # hash dedup → frequency (req 9)
    txt = "Pick up the key before opening the door."
    assert mem.add_memory(txt, success=False) is True
    assert mem.add_memory(txt, success=True) is False      # merged, not new
    assert len(mem) == 1
    entry = mem.get_recent(1)[0]
    assert entry["frequency"] == 2, f"frequency should be 2, got {entry['frequency']}"
    assert entry["cluster"] == "KEY_FIRST", f"cluster={entry['cluster']}"
    assert entry["success_count"] == 1 and entry["failure_count"] == 1

    # coordinate / location guards (req 5,15)
    assert mem.add_memory("Move to (7,3) then open the door.") is False
    assert mem.add_memory("The key was located near the upper-right corner.") is False
    assert mem.add_memory("Go to row 3 then column 7.") is False
    assert mem.stats["total_rejected"] >= 3
    assert len(mem) == 1, "coordinate reflections must not be stored"

    # cluster-prioritized retrieval (req 11/12)
    mem.add_memory("Search unexplored frontier regions when no target is visible.",
                   success=True)
    ctx = mem.get_context(current_cluster="KEY_FIRST")
    first_line = [l for l in ctx.splitlines() if l.strip().startswith("[")][0]
    assert "KEY_FIRST" in first_line, f"KEY_FIRST should rank first:\n{ctx}"

    # success-rate then frequency ordering within equal cluster relevance
    m2 = ReflectionMemory(top_k=5)
    m2.add_memory("Lesson one stands on its own merit.", success=True)    # rate 1.0
    m2.add_memory("Lesson two stands on its own merit.", success=False)   # rate 0.0
    ctx2 = m2.get_context(current_cluster=None)
    assert ctx2.index("Lesson one") < ctx2.index("Lesson two"), \
        f"higher success-rate must rank first:\n{ctx2}"

    # save / load round-trip + legacy entry (req 10)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.json")
        mem.save(path)
        reloaded = ReflectionMemory()
        reloaded.load(path)
        assert len(reloaded) == len(mem)
        e0 = reloaded.get_recent(99)
        assert all("cluster" in e and "frequency" in e for e in e0)

        # legacy schema (no cluster/frequency) must still load
        legacy = os.path.join(d, "legacy.json")
        with open(legacy, "w") as f:
            json.dump({"entries": [
                {"reflection": "Old lesson about getting the key first.",
                 "success": True}]}, f)
        lm = ReflectionMemory()
        lm.load(legacy)
        assert len(lm) == 1
        le = lm.get_recent(1)[0]
        assert le["frequency"] == 1 and le["cluster"] in CLUSTER_LABELS
    print(f"{INFO} memory dedup/cluster/retrieval/guards/save-load OK | {mem.summary()}")


# ── 5. Toggle key-check (req 8) ───────────────────────────────────────────────

def test_toggle():
    from skill.base_skill import Toggle
    # agent at (5,5) facing east(0); LOCKED blue(2) door directly in front (6,5)
    door = [((6, 5), 4, 2, 2)]   # type door, color blue, state locked

    # A) no key held → fail (no toggle)
    obs = make_obs(10, 10, (5, 5), 0, objects=door, carrying=None)
    a, term = Toggle()(obs)
    assert a is None and term is True
    print(f"{INFO} no key  → terminated, reason='{Toggle().__class__.__name__}' no-toggle OK")

    # B) holding matching blue key → issue toggle (action 5)
    obs = make_obs(10, 10, (5, 5), 0, objects=door, carrying=(5, 2))  # key, blue
    t = Toggle(); a, term = t(obs)
    assert a == 5 and term is False, f"expected toggle(5,False), got ({a},{term})"

    # C) holding mismatched red(0) key → fail with color reason
    obs = make_obs(10, 10, (5, 5), 0, objects=door, carrying=(5, 0))  # key, red
    t = Toggle(); a, term = t(obs)
    assert a is None and term is True and "color" in t.failure_reason.lower(), \
        f"expected color failure, got ({a},{term}) reason='{t.failure_reason}'"
    print(f"{INFO} matching key → toggle; mismatched key → '{t.failure_reason}'")


# ── 6. Pickup recovery (req 14) ───────────────────────────────────────────────

def test_pickup_recovery():
    from skill.base_skill import Pickup

    # A) key directly in front (6,5) → pickup (action 3)
    obs = make_obs(10, 10, (5, 5), 0, objects=[((6, 5), 5, 0, 0)], carrying=None)
    a, term = Pickup()(obs)
    assert a == 3 and term is False, f"key in front should pickup, got ({a},{term})"

    # B) key visible but not adjacent (5,8) → recovery emits a move/turn (not None)
    obs = make_obs(10, 10, (5, 5), 0, objects=[((5, 8), 5, 0, 0)], carrying=None)
    a, term = Pickup()(obs)
    assert a is not None and a in (0, 1, 2) and term is False, \
        f"distant key should trigger navigation, got ({a},{term})"

    # C) no key visible → soft fail (None, False)
    obs = make_obs(10, 10, (5, 5), 0, objects=[], carrying=None)
    p = Pickup(); a, term = p(obs)
    assert a is None and term is False and p.failure_reason, \
        f"no key should soft-fail, got ({a},{term})"

    # D) already carrying the key → success (None, True)
    obs = make_obs(10, 10, (5, 5), 0, objects=[], carrying=(5, 0))
    p = Pickup(); a, term = p(obs)
    assert a is None and term is True and p.success
    print(f"{INFO} pickup: in-front→3, distant→nav, none→soft-fail, holding→success OK")


# ── 7. offline + online teacher (req 3, 12) ───────────────────────────────────

def test_offline_online():
    from planner import Planner

    # offline: cached symbolic plan, no LLM
    p = Planner("simpledoorkey", offline=True, soft=False, prefix="")
    plans, probs = p.plan("Agent sees <key>, holds <nothing>.")
    assert plans and "pick up" in str(plans).lower(), f"offline plan unexpected: {plans}"

    # online: stub LLM + reflection memory → context injected into the prompt
    from memory.memory_buffer import ReflectionMemory

    class _StubLLM:
        def call_with_retry(self, user_prompt, system_prompt=None, correction_prompt=None):
            self.last_prompt = user_prompt
            return "explore", "raw"

    p2 = Planner("simpledoorkey", offline=False, soft=False, prefix="")
    stub = _StubLLM()
    p2.set_llm(stub)
    mem = ReflectionMemory(top_k=5)
    mem.add_memory("Pick up the key before opening the door.", success=True)
    p2.set_reflection_memory(mem)

    prompt = p2._build_prompt_with_reflection("Agent sees <door>, holds <nothing>.")
    assert "Pick up the key before opening the door." in prompt, \
        f"reflection context not injected:\n{prompt}"
    # and query path returns the stub plan without crashing
    out = p2.query_codex("Agent sees <door>, holds <nothing>.")
    assert isinstance(out, str)
    print(f"{INFO} offline cached plan OK; online prompt injects reflection context OK")


# ── 8. viz import (req 4) ─────────────────────────────────────────────────────

def test_reflector_backend():
    """Reflector: Ollama is primary, offline is the automatic backup.
    Default resolves to 'ollama' when reachable, else 'offline'; an explicit
    'offline' never touches the network."""
    from experiment_config import build_experiment_config
    from experiment_runner import build_game, _ollama_available
    from memory.reflection import EpisodeTrajectory

    cfg = build_experiment_config(
        "reflection",
        overrides=dict(total_iterations=1, batch_size=16, episodes_per_iteration=1),
    )

    # Default: Ollama primary / offline backup — must match the reachability probe.
    expected = "ollama" if _ollama_available() else "offline"
    g = build_game(cfg, task="SimpleDoorKey", device="cpu", seed=1)
    refl = getattr(g, "_reflector", None)
    assert refl is not None, "reflector not wired for reflection phase"
    assert refl.backend == expected, \
        f"default reflector backend '{refl.backend}' != probe-expected '{expected}'"
    print(f"{INFO} default reflector backend='{refl.backend}' (Ollama probe={expected=='ollama'})")

    # Explicit offline → no network call; reflect() returns None.
    g2 = build_game(cfg, task="SimpleDoorKey", device="cpu", seed=1,
                    reflector_backend="offline")
    assert g2._reflector.backend == "offline"
    traj = EpisodeTrajectory()
    traj.add_step("Agent sees <key>, holds <nothing>.", "explore", 0.0)
    traj.finish(False)
    assert g2._reflector.reflect(traj) is None, "offline reflector must return None"
    print(f"{INFO} explicit offline → reflect()=None (no network)")


def test_viz_import():
    try:
        import viz.server  # noqa: F401
    except ModuleNotFoundError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else str(e)
        if missing.split(".")[0] in ("fastapi", "uvicorn", "starlette", "pydantic",
                                      "websockets", "anyio"):
            print(f"{INFO} viz deps not installed ({missing}) — import skipped")
            return "skip"
        raise
    print(f"{INFO} viz.server imported OK")


if __name__ == "__main__":
    run_test("1. pipeline builds (ppo_only / planner / reflection)", test_pipeline_builds)
    run_test("2. Game(args) positional backward-compat",            test_game_positional)
    run_test("3. TwoDoor room size 17-20",                          test_twodoor_size)
    run_test("4. reflection memory dedup/cluster/retrieval/guards", test_memory)
    run_test("5. Toggle key-check",                                 test_toggle)
    run_test("6. Pickup recovery",                                  test_pickup_recovery)
    run_test("7. offline + online teacher",                         test_offline_online)
    run_test("8. reflector backend (ollama primary / offline backup)", test_reflector_backend)
    run_test("9. viz import",                                       test_viz_import)

    print("\n" + "=" * 60 + "\n  SUMMARY\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok is True)
    skipped = sum(1 for _, ok in results if ok == "skip")
    failed = sum(1 for _, ok in results if ok is False)
    for name, ok in results:
        tag = PASS if ok is True else (SKIP if ok == "skip" else FAIL)
        print(f"  {tag}  {name}")
    print(f"\n  {passed} passed, {skipped} skipped, {failed} failed")
    sys.exit(1 if failed else 0)
