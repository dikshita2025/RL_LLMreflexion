#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
diagnostics.py — behavior-free verification of the teacher → PPO supervision signal
and the Pickup execution pipeline.

Drives the TEACHER policy through full episodes (it does NOT change training,
PPO, planner, reflection, memory, exploration, env, or rewards — it only reads
state and steps the env with the teacher's own action), and records:

  * Pickup pipeline classification (A/B/C/D) — action mapping, alignment,
    env interaction, observation decode, planner-state.
  * Teacher supervision signal — at "sees key / holds key+door / sees nothing":
    symbolic obs, plan, teacher_probs, teacher argmax, PPO sampled action,
    plus match rates (teacher↔intended skill, PPO↔teacher).

Output is printed AND written to a file (default results/diagnostics.txt) so it is
captured in Kaggle cell output and in the downloadable results zip.

Usage (notebook):
    from diagnostics import run_all_diagnostics
    run_all_diagnostics(game=game, out_path=os.path.join(OUTPUT_DIR,'results','diagnostics.txt'))
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

try:
    import torch
except Exception:
    torch = None

from skill.base_skill import (BaseSkill, OBJ_KEY, OBJ_DOOR, PICKABLE_OBJECTS)

PICKUP, TOGGLE = 3, 5
ANAME = {0: "left", 1: "right", 2: "forward", 3: "PICKUP", 4: "drop", 5: "TOGGLE", 6: "done"}
_CARRY = {1: "nothing", 5: "key", 6: "ball", 7: "box"}


class _Tee:
    """Write lines to stdout and (optionally) a file."""
    def __init__(self, path=None):
        self._f = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._f = open(path, "w", encoding="utf-8")
        self.path = path

    def __call__(self, line=""):
        print(line)
        if self._f:
            self._f.write(str(line) + "\n")

    def close(self):
        if self._f:
            self._f.close()


def _build_game_if_needed(game, task, device, seed, qwen_llm):
    if game is not None:
        return game, False
    from experiment_config import build_experiment_config
    from experiment_runner import build_game
    cfg = build_experiment_config("planner")
    g = build_game(cfg, task=task, device=device, seed=seed, qwen_llm=qwen_llm)
    return g, True


def run_pickup_diagnostic(game, log, n_ep=6, seed0=100):
    env, med, bs = game.env, game.teacher_policy.planner.mediator, BaseSkill()
    try:
        env_pick = int(env.unwrapped.actions.pickup)
    except Exception as e:
        env_pick = -1
    log("=" * 64)
    log("PICKUP PIPELINE DIAGNOSTIC (teacher-driven)")
    log("=" * 64)
    log(f"ACTION_ID_PICKUP = {PICKUP}    ENV_PICKUP_ID = {env_pick}    MATCH = {env_pick == PICKUP}")
    cls = {"ok": 0, "A": 0, "B": 0, "C": 0}
    solved = 0
    for ep in range(n_ep):
        obs = env.reset(seed=seed0 + ep)
        game.teacher_policy.reset()
        done, steps, picked, lastr = False, 0, False, 0.0
        while not done and steps < game.max_ep_len:
            a = int(np.argmax(np.asarray(game.teacher_policy(obs[0]), dtype=float)))
            if a == PICKUP:
                bs.unpack_obs(obs); fwd = bs._fwd_cell()
                aligned = fwd[0] in PICKABLE_OBJECTS
                obs, r, d, _ = env.step(np.array([a]))
                bs.unpack_obs(obs); post = _CARRY.get(int(bs.carrying), "?")
                env_has = getattr(env.unwrapped, "carrying", None) is not None
                holds = med.RL2LLM(obs[0]).split("holds")[-1].strip()
                log(f"  ep{ep} s{steps} PICKUP aligned(front={fwd[0]})={aligned} "
                    f"-> env.carrying={'key' if env_has else None} obs={post} mediator_holds={holds}")
                if not aligned and not env_has: cls["A"] += 1
                elif aligned and not env_has:   cls["B"] += 1
                elif env_has and post == "nothing": cls["C"] += 1
                elif env_has: cls["ok"] += 1
                if env_has: picked = True
            else:
                obs, r, d, _ = env.step(np.array([a]))
            lastr = float(r.squeeze()) if hasattr(r, "squeeze") else float(r)
            done = bool(d.squeeze()) if hasattr(d, "squeeze") else bool(d)
            steps += 1
        solved += int(lastr > 0)
        log(f"  ep{ep}: steps={steps} key_picked={picked} reward>0={lastr > 0}")
    log(f"  RESULT: ok={cls['ok']} A(misaligned)={cls['A']} B(env-miss)={cls['B']} "
        f"C(decode)={cls['C']}  teacher_solved={solved}/{n_ep}")
    verdict = "D (pipeline healthy)" if (cls["A"] + cls["B"] + cls["C"] == 0) else \
              ("A" if cls["A"] else "B" if cls["B"] else "C")
    log(f"  PICKUP VERDICT: {verdict}")
    log("")
    return {"ok": cls["ok"], "A": cls["A"], "B": cls["B"], "C": cls["C"],
            "teacher_solved": solved, "n_ep": n_ep, "action_map_ok": env_pick == PICKUP}


def run_teacher_signal_diagnostic(game, log, n_ep=8, seed0=200):
    env, med, bs = game.env, game.teacher_policy.planner.mediator, BaseSkill()
    dev = getattr(game, "device", "cpu")
    mask = torch.FloatTensor([1]).to(dev) if torch is not None else None
    pick = {"n": 0, "m": 0}; tog = {"n": 0, "m": 0}; agree = {"n": 0, "m": 0}
    unif = []; ex = {"key": None, "door": None, "nothing": None}
    for ep in range(n_ep):
        obs = env.reset(seed=seed0 + ep)
        game.teacher_policy.reset()
        done, steps = False, 0
        while not done and steps < game.max_ep_len:
            otext = med.RL2LLM(obs[0])
            probs = np.asarray(game.teacher_policy(obs[0]), dtype=float)
            targ = int(np.argmax(probs))
            ppo_a = targ
            if torch is not None:
                try:
                    dist, _, _ = game.student_policy(torch.Tensor(obs).to(dev), mask, None)
                    ppo_a = int(dist.sample().cpu().numpy().reshape(-1)[0])
                except Exception:
                    pass
            bs.unpack_obs(obs); fwd_t = bs._fwd_cell()[0]; carry = bs._is_carrying()
            plan = game._get_plan_text() if hasattr(game, "_get_plan_text") else ""
            agree["n"] += 1; agree["m"] += int(ppo_a == targ)
            if fwd_t == OBJ_KEY and not carry:
                pick["n"] += 1; pick["m"] += int(targ == PICKUP)
                ex["key"] = ex["key"] or (otext, plan, probs.round(3).tolist(), targ, ppo_a)
            if fwd_t == OBJ_DOOR and carry:
                tog["n"] += 1; tog["m"] += int(targ == TOGGLE)
                ex["door"] = ex["door"] or (otext, plan, probs.round(3).tolist(), targ, ppo_a)
            if otext.startswith("Agent sees <nothing>"):
                unif.append(float(np.max(probs)))
                ex["nothing"] = ex["nothing"] or (otext, plan, probs.round(3).tolist(), targ, ppo_a)
            obs, r, d, _ = env.step(np.array([targ]))
            done = bool(d.squeeze()) if hasattr(d, "squeeze") else bool(d)
            steps += 1
    nact = game.action_space
    pc = lambda d: 100.0 * d["m"] / max(d["n"], 1)
    log("=" * 64)
    log("TEACHER SUPERVISION SIGNAL DIAGNOSTIC")
    log("=" * 64)
    log("Representative examples (obs | plan | teacher_probs | t_argmax | ppo_sampled):")
    for tag in ("key", "door", "nothing"):
        e = ex[tag]
        if e:
            o, p, pr, ta, pa = e
            log(f"  [{tag:7}] obs='{o}'")
            log(f"            plan='{p}'")
            log(f"            probs={pr}  t_argmax={ta}({ANAME.get(ta)})  ppo={pa}({ANAME.get(pa)})")
    log("")
    log(f"(1) adjacent-key & not-carrying → PICKUP : {pick['m']}/{pick['n']} ({pc(pick):.0f}%)")
    log(f"(2) carrying & adjacent-door   → TOGGLE  : {tog['m']}/{tog['n']} ({pc(tog):.0f}%)")
    if unif:
        mu = float(np.mean(unif))
        log(f"(3) exploration mean max(probs)={mu:.3f} (uniform={1/nact:.3f}) "
            f"→ {'NOT collapsed' if mu > 1.5/nact else 'COLLAPSED'}")
    inter_m, inter_n = pick["m"] + tog["m"], pick["n"] + tog["n"]
    log(f"(4a) teacher argmax matches intended skill action: {inter_m}/{inter_n} "
        f"({100.0*inter_m/max(inter_n,1):.0f}%)")
    log(f"(4b) PPO sampled matches teacher argmax           : {agree['m']}/{agree['n']} ({pc(agree):.0f}%)")
    log("")
    # Classification
    sig_ok = (inter_n > 0 and 100.0 * inter_m / inter_n >= 80)
    if not sig_ok:
        verdict = "A or B (teacher targets wrong — check planner/plan→action)"
    elif pc(agree) < 60:
        verdict = "C (teacher signal correct & useful; PPO under-imitates it)"
    else:
        verdict = "OK (signal correct AND PPO imitating well)"
    log(f"SIGNAL VERDICT: {verdict}")
    log("")
    return {"pickup_state_match_pct": pc(pick), "toggle_state_match_pct": pc(tog),
            "explore_mean_maxprob": (float(np.mean(unif)) if unif else None),
            "teacher_intended_match_pct": 100.0*inter_m/max(inter_n,1),
            "ppo_teacher_match_pct": pc(agree), "verdict": verdict}


def run_all_diagnostics(game=None, out_path=None, task="SimpleDoorKey",
                        device="cpu", seed=0, qwen_llm=None, n_pickup=6, n_signal=8):
    """Run both diagnostics; print and (if out_path) save to a text file.

    Pass the trained `game` from your phase to make (4b) reflect the learned
    policy; if game is None a fresh game is built (student untrained)."""
    game, built = _build_game_if_needed(game, task, device, seed, qwen_llm)
    tee = _Tee(out_path)
    try:
        tee(f"LLM4Teach diagnostics — task={getattr(game,'task',task)} "
            f"student={'fresh/untrained' if built else 'from training run'}")
        tee("")
        p = run_pickup_diagnostic(game, tee, n_ep=n_pickup)
        s = run_teacher_signal_diagnostic(game, tee, n_ep=n_signal)
        if out_path:
            tee(f"[diagnostics] saved → {out_path}")
        return {"pickup": p, "signal": s}
    finally:
        tee.close()


if __name__ == "__main__":
    run_all_diagnostics(out_path="results/diagnostics.txt")
