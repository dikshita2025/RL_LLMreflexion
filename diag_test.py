#!/usr/bin/env python
"""
diag_test.py — Architecture diagnostic for SimpleDoorKey
Run: python diag_test.py
"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

results = []

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def log(tag, msg):
    print(f"  {tag} {msg}")

def make_game(**overrides):
    from Game import Game
    from types import SimpleNamespace
    base = dict(
        task='simpledoorkey', seed=42, frame_stack=1,
        offline_planner=True, soft_planner=False,
        device='cpu', batch_size=32, recurrent=False,
        gamma=0.99, lam=0.95, n_itr=5, traj_per_itr=2,
        num_eval=5, eval_interval=10, save_interval=100,
        logdir='log_diag', policy='ppo',
        savedir='diag_run', loaddir=None, loadmodel='acmodel',
    )
    base.update(overrides)
    return Game(SimpleNamespace(**base))


# ── TEST 1: Environment + obs shape ──────────────────────────────────────────
section("TEST 1: MiniGrid Environment")
try:
    g1 = make_game(savedir='diag_t1')
    obs = g1.env.reset()
    log(INFO, f"obs shape       : {obs.shape}")
    log(INFO, f"action_space    : {g1.action_space}")
    log(INFO, f"max_ep_len      : {g1.max_ep_len}")

    total_r = 0
    for _ in range(20):
        a = np.array([g1.env.action_space.sample()])
        obs, r, done, info = g1.env.step(a)
        total_r += float(r.squeeze())
        if done.squeeze():
            obs = g1.env.reset()

    log(PASS, f"20 random steps OK. cumulative_reward={total_r:.3f}")
    results.append(("Env basic step", True))
except Exception as e:
    log(FAIL, str(e)); traceback.print_exc()
    results.append(("Env basic step", False))


# ── TEST 2: Mediator — RL2LLM + obj_coordinate ───────────────────────────────
section("TEST 2: Mediator — RL2LLM + obj_coordinate")
try:
    g2 = make_game(savedir='diag_t2')
    mediator = g2.teacher_policy.planner.mediator

    # Walk until key is visible
    obs2 = g2.env.reset()
    mediator.reset()
    key_found_text = None
    for step in range(150):
        obs_text = mediator.RL2LLM(obs2)
        if 'key' in obs_text and 'holds <nothing>' in obs_text:
            key_found_text = obs_text
            break
        obs2, _, done, _ = g2.env.step(np.array([2]))
        if done.squeeze():
            obs2 = g2.env.reset()
            mediator.reset()

    if key_found_text:
        log(INFO, f"Key-visible obs      : '{key_found_text}'")
        log(INFO, f"obj_coordinate       : {mediator.obj_coordinate}")
        log(PASS,  "RL2LLM populates obj_coordinate when key visible")
    else:
        log(WARN, "Key never appeared in 150 steps — exploration broken")

    # CRITICAL: simulate cache-hit scenario (episode boundary)
    mediator.reset()
    log(INFO, f"obj_coordinate after reset(): {mediator.obj_coordinate}")

    # Parser called without RL2LLM — simulates what happens on every cache hit
    skill_list = mediator.parser("go to <key>, pick up <key>")
    actions = [s['action'] for s in skill_list]
    coords  = [s['coordinate'] for s in skill_list]
    log(INFO, f"Parser result (no coordinate): actions={actions}, coords={coords}")

    if 1 in actions:  # action 1 = GoTo
        log(FAIL, "BUG: GoTo generated with None/stale coordinate — will navigate to wrong cell")
        results.append(("Mediator obj_coordinate", False))
    else:
        log(PASS, "Parser correctly falls back to explore when coordinate missing")
        results.append(("Mediator obj_coordinate", True))

except Exception as e:
    log(FAIL, str(e)); traceback.print_exc()
    results.append(("Mediator obj_coordinate", False))


# ── TEST 3: Planner — cache hit does NOT refresh coordinate ──────────────────
section("TEST 3: Planner — coordinate stale after cache hit")
try:
    g3 = make_game(savedir='diag_t3')
    planner = g3.teacher_policy.planner

    log(INFO, f"Pre-seeded plans_dict keys ({len(planner.plans_dict)}):")
    for k in planner.plans_dict:
        plans, probs = planner.plans_dict[k]
        log(INFO, f"  '{k}' → {plans}")

    # Ep1: see key, populate coordinate, cache plan
    obs3 = g3.env.reset()
    for _ in range(150):
        obs_t = planner.mediator.RL2LLM(obs3)
        if 'key' in obs_t and 'holds <nothing>' in obs_t:
            plans, probs = planner.plan(obs_t)
            coord_ep1 = dict(planner.mediator.obj_coordinate)
            log(INFO, f"Ep1 key coord: {coord_ep1}")
            break
        obs3, _, done, _ = g3.env.step(np.array([2]))
        if done.squeeze():
            obs3 = g3.env.reset()

    # Ep2: new episode, mediator reset, same obs text → cache hit
    planner.reset()  # calls mediator.reset() → obj_coordinate = {}
    log(INFO, f"After planner.reset(), obj_coordinate: {planner.mediator.obj_coordinate}")

    obs3 = g3.env.reset()
    for _ in range(150):
        obs_t2 = planner.mediator.RL2LLM(obs3)
        if 'key' in obs_t2 and 'holds <nothing>' in obs_t2:
            # This is a cache hit — RL2LLM already ran so coord IS populated
            # But in actual training flow, planner.plan() is called with text
            # from teacher_policy which calls RL2LLM separately
            # Simulate the bug: call plan() directly with cached text
            cached_text = 'Agent sees <key>, holds <nothing>.'
            planner.mediator.reset()  # simulate episode start, coord cleared
            log(INFO, f"Coord before plan() call: {planner.mediator.obj_coordinate}")
            plans2, _ = planner.plan(cached_text)  # cache hit, no RL2LLM
            log(INFO, f"Coord after plan() cache hit: {planner.mediator.obj_coordinate}")
            skill_list2, _ = planner.mediator.LLM2RL(plans2, [1.0])
            coords2 = [s[0]['coordinate'] if s else None for s in skill_list2]
            log(INFO, f"GoTo coordinates used: {coords2}")
            if any(c is None for c in coords2) or not planner.mediator.obj_coordinate:
                log(FAIL, "BUG CONFIRMED: plan says GoTo but coordinate is None — silent explore fallback")
                results.append(("Planner stale coordinate", False))
            else:
                log(PASS, "Coordinate present after cache hit")
                results.append(("Planner stale coordinate", True))
            break
        obs3, _, done, _ = g3.env.step(np.array([2]))
        if done.squeeze():
            obs3 = g3.env.reset()
    else:
        log(WARN, "Could not reach key-visible state")
        results.append(("Planner stale coordinate", False))

except Exception as e:
    log(FAIL, str(e)); traceback.print_exc()
    results.append(("Planner stale coordinate", False))


# ── TEST 4: Teacher-only — can teacher solve it? ─────────────────────────────
section("TEST 4: Teacher-only episodes (10 runs, pure teacher actions)")
try:
    g4 = make_game(savedir='diag_t4')
    tp = g4.teacher_policy

    successes = 0
    key_pickups = 0
    door_opens  = 0
    toggle_no_key = 0
    drop_with_key = 0

    for ep in range(10):
        obs4 = g4.env.reset()
        tp.reset()
        ep_reward = 0.0
        holding_key = False

        for step in range(150):
            teacher_probs = tp(obs4)
            action = int(np.argmax(teacher_probs))

            obs_text = tp.planner.mediator.RL2LLM(obs4)
            holding_key = 'holds <key>' in obs_text

            if action == 5 and not holding_key:
                toggle_no_key += 1
            if action == 4 and holding_key:
                drop_with_key += 1

            obs4, r, done, _ = g4.env.step(np.array([action]))
            ep_reward += float(r.squeeze())

            new_text = tp.planner.mediator.RL2LLM(obs4)
            if 'holds <key>' in new_text and not holding_key:
                key_pickups += 1
            if ep_reward > 0 and action == 5:
                door_opens += 1

            if done.squeeze():
                break

        if ep_reward > 0:
            successes += 1
        log(INFO, f"  ep {ep+1:2d}: reward={ep_reward:.3f}  steps={step+1:3d}  {'SUCCESS' if ep_reward>0 else 'fail'}")

    log(INFO, f"Results: {successes}/10 success")
    log(INFO, f"Key pickups: {key_pickups}, Door opens: {door_opens}")
    log(INFO, f"toggle_without_key: {toggle_no_key}, drop_with_key: {drop_with_key}")
    log(INFO, f"Interventions: {tp.intervention_stats}")

    if toggle_no_key > 0:
        log(WARN, f"Teacher fires toggle without key {toggle_no_key} times — plan execution bug")
    if drop_with_key > 0:
        log(WARN, f"Teacher drops key {drop_with_key} times — plan execution bug")
    if successes == 0:
        log(FAIL, "Teacher CANNOT solve task alone — teacher pipeline broken")
        results.append(("Teacher-only solve", False))
    elif successes < 5:
        log(WARN, f"Teacher solves only {successes}/10 — partially working")
        results.append(("Teacher-only solve", True))
    else:
        log(PASS, f"Teacher solves {successes}/10")
        results.append(("Teacher-only solve", True))

except Exception as e:
    log(FAIL, str(e)); traceback.print_exc()
    results.append(("Teacher-only solve", False))


# ── TEST 5: PPO kickstarting decay ───────────────────────────────────────────
section("TEST 5: PPO kickstarting coefficient decay")
try:
    g5 = make_game(savedir='diag_t5')
    ppo = g5.student_policy

    log(INFO, f"Initial ks_coef     : {ppo.ks_coef}")
    log(INFO, f"ks_coef_minimum     : {ppo.ks_coef_minimum}")
    log(INFO, f"ks_coef_descent     : {ppo.ks_coef_descent}")
    log(INFO, f"iter_with_ks        : {ppo.iter_with_ks}")
    log(INFO, f"entropy_coef        : {ppo.entropy_coef}")

    # Simulate decay at 0% success (actual run condition)
    import copy
    ppo_sim = copy.copy(ppo)
    ppo_sim.iter = 0
    ppo_sim.ks_coef = 1.0

    history = []
    for i in range(100):
        ppo_sim.update_kickstarting_coef(recent_success_rate=0.0)
        history.append((i+1, ppo_sim.ks_coef))

    floor_iter = next((i for i, c in history if c <= ppo.ks_coef_minimum), None)
    log(INFO, f"ks_coef hits minimum ({ppo.ks_coef_minimum}) at iteration: {floor_iter}")
    log(INFO, f"ks_coef at iter 10 : {history[9][1]:.4f}")
    log(INFO, f"ks_coef at iter 47 : {history[46][1]:.4f}")
    log(INFO, f"ks_coef at iter 100: {history[99][1]:.4f}")

    if floor_iter and floor_iter < 50:
        log(FAIL, f"ks_coef hits floor after {floor_iter} iters with 0% success — teacher signal lost too early")
        log(WARN, "With ~1% success rate, PPO loses teacher guidance before learning anything")
        results.append(("PPO ks_coef decay", False))
    else:
        log(PASS, f"ks_coef decay OK (floor at iter {floor_iter})")
        results.append(("PPO ks_coef decay", True))

    # Check entropy dynamics
    log(INFO, "Entropy coefficient dynamics:")
    log(INFO, f"  Normal entropy: coef={ppo.entropy_coef}")
    log(INFO, f"  If entropy<0.5: coef={ppo.entropy_coef*2.0} (2x boost)")
    log(INFO, f"  If entropy<0.2: coef={ppo.entropy_coef*3.0} (3x boost)")
    log(INFO, "  Actual run entropy ~1.74 → normal coef — pushes toward uniform random")

except Exception as e:
    log(FAIL, str(e)); traceback.print_exc()
    results.append(("PPO ks_coef decay", False))


# ── TEST 6: PPO loss components ───────────────────────────────────────────────
section("TEST 6: PPO loss — kickstarting vs policy vs entropy")
try:
    g6 = make_game(savedir='diag_t6')

    loss_history = []
    ks_coef_history = []

    for itr in range(5):
        g6.buffer.clear()
        g6.collect()
        loss = g6.student_policy.update_policy(g6.buffer, recent_success_rate=0.0)
        loss_history.append(loss)
        ks_coef_history.append(g6.student_policy.ks_coef)

    log(INFO, "Loss components [total, entropy, kickstart, policy, value]:")
    for i, (l, k) in enumerate(zip(loss_history, ks_coef_history)):
        log(INFO, f"  iter {i+1}: ks_coef={k:.3f}  losses={[f'{x:.4f}' for x in l]}")

    avg_ks   = np.mean([l[2] for l in loss_history])
    avg_pol  = np.mean([l[3] for l in loss_history])
    avg_ent  = np.mean([l[1] for l in loss_history])

    log(INFO, f"Avg kickstarting loss : {avg_ks:.4f}")
    log(INFO, f"Avg policy loss       : {avg_pol:.4f}")
    log(INFO, f"Avg entropy           : {avg_ent:.4f}")

    ks_fraction = (g6.student_policy.ks_coef * avg_ks) / max(abs(avg_pol) + abs(avg_ent) + g6.student_policy.ks_coef * avg_ks, 1e-6)
    log(INFO, f"Kickstarting fraction of total gradient: {ks_fraction*100:.1f}%")

    if avg_ks < 0.1:
        log(WARN, "Kickstarting loss very small — teacher barely influencing PPO")
    else:
        log(PASS, f"Kickstarting loss active: {avg_ks:.4f}")

    results.append(("PPO loss components", True))

except Exception as e:
    log(FAIL, str(e)); traceback.print_exc()
    results.append(("PPO loss components", False))


# ── TEST 7: Action distribution — PPO vs teacher ─────────────────────────────
section("TEST 7: Action quality — PPO vs Teacher distribution")
try:
    g7 = make_game(savedir='diag_t7')
    tp7 = g7.teacher_policy
    ppo7 = g7.student_policy

    obs7 = g7.env.reset()
    tp7.reset()

    ppo_actions = []
    teacher_actions = []
    agreements = 0

    for step in range(150):
        obs_t = torch.Tensor(obs7).to('cpu')
        mask  = torch.FloatTensor([1.0])
        dist, _, _ = ppo7(obs_t, mask, None)
        ppo_action = int(dist.sample().item())

        teacher_probs = tp7(obs7)
        teacher_action = int(np.argmax(teacher_probs))

        ppo_actions.append(ppo_action)
        teacher_actions.append(teacher_action)
        if ppo_action == teacher_action:
            agreements += 1

        obs7, _, done, _ = g7.env.step(np.array([ppo_action]))
        if done.squeeze():
            break

    anames = {0:'left',1:'right',2:'fwd',3:'pickup',4:'drop',5:'toggle',6:'wait'}

    ppo_dist = {}
    for a in ppo_actions:
        ppo_dist[anames[a]] = ppo_dist.get(anames[a], 0) + 1

    teach_dist = {}
    for a in teacher_actions:
        teach_dist[anames[a]] = teach_dist.get(anames[a], 0) + 1

    log(INFO, f"PPO action distribution     : {ppo_dist}")
    log(INFO, f"Teacher action distribution : {teach_dist}")
    log(INFO, f"PPO/Teacher agreement       : {agreements}/{len(ppo_actions)} ({100*agreements/len(ppo_actions):.1f}%)")

    bad_ppo = (ppo_dist.get('drop',0) + ppo_dist.get('wait',0) + ppo_dist.get('toggle',0))
    bad_pct = bad_ppo / len(ppo_actions) * 100
    log(INFO, f"PPO wasted actions (drop+wait+toggle-no-key): {bad_pct:.1f}%")

    if agreements / len(ppo_actions) < 0.2:
        log(FAIL, f"PPO agrees with teacher only {100*agreements/len(ppo_actions):.1f}% — kickstarting not working")
        results.append(("PPO/Teacher agreement", False))
    else:
        log(PASS, f"PPO/Teacher agreement {100*agreements/len(ppo_actions):.1f}%")
        results.append(("PPO/Teacher agreement", True))

except Exception as e:
    log(FAIL, str(e)); traceback.print_exc()
    results.append(("PPO/Teacher agreement", False))


# ── TEST 8: Skill execution — does GoTo actually navigate? ───────────────────
section("TEST 8: Skill execution — GoTo navigates, Pickup picks up")
try:
    g8 = make_game(savedir='diag_t8')
    tp8 = g8.teacher_policy
    mediator8 = tp8.planner.mediator

    from skill import GoTo_Goal, Explore
    from skill.base_skill import Pickup

    obs8 = g8.env.reset()
    mediator8.reset()

    # Walk until key visible
    key_coord = None
    for _ in range(150):
        obs_text = mediator8.RL2LLM(obs8)
        if 'key' in mediator8.obj_coordinate:
            key_coord = mediator8.obj_coordinate['key']
            break
        obs8, _, done, _ = g8.env.step(np.array([2]))
        if done.squeeze():
            obs8 = g8.env.reset()
            mediator8.reset()

    if key_coord:
        log(INFO, f"Key found at coordinate: {key_coord}")
        goto = GoTo_Goal(key_coord)
        steps_taken = 0
        reached = False
        for _ in range(50):
            action, terminated = goto(obs8)
            if action is None or terminated:
                log(INFO, f"  GoTo terminated: failure='{goto.failure_reason}' msg='{goto.message}'")
                reached = (goto.message != 'no path found' and not goto.failure_reason)
                break
            obs8, _, done, _ = g8.env.step(np.array([action]))
            steps_taken += 1
            if done.squeeze():
                break

        log(INFO, f"GoTo steps taken: {steps_taken}")

        if goto.failure_reason:
            log(FAIL, f"GoTo failed: {goto.failure_reason}")
        else:
            log(PASS, "GoTo completed — agent positioned adjacent to key")

        # Now try pickup
        pickup = Pickup()
        action, _ = pickup(obs8)
        log(INFO, f"Pickup action returned: {action}")
        if action == 3:
            obs8, r, done, _ = g8.env.step(np.array([3]))
            new_text = mediator8.RL2LLM(obs8)
            if 'holds <key>' in new_text:
                log(PASS, "Pickup SUCCESS — agent now holding key")
            else:
                log(WARN, f"Pickup action fired but agent still not holding key: '{new_text}'")
        elif action is None:
            log(WARN, f"Pickup returned None — not facing key. Failure: '{pickup.failure_reason}'")

        results.append(("Skill GoTo+Pickup", not bool(goto.failure_reason)))
    else:
        log(WARN, "Key never found in 150 steps — skipping GoTo test")
        results.append(("Skill GoTo+Pickup", False))

except Exception as e:
    log(FAIL, str(e)); traceback.print_exc()
    results.append(("Skill GoTo+Pickup", False))


# ── SUMMARY ──────────────────────────────────────────────────────────────────
section("DIAGNOSTIC SUMMARY")
passed = sum(1 for _, ok in results if ok)
total  = len(results)
for name, ok in results:
    print(f"  {'[PASS]' if ok else '[FAIL]'}  {name}")
print(f"\n  {passed}/{total} passed\n")

fails = [n for n, ok in results if not ok]
if fails:
    print("  Components needing fixes:")
    for f in fails:
        print(f"    - {f}")
