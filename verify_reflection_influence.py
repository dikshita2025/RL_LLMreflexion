#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
verify_reflection_influence.py — Did reflection memory ever influence planning?

Reads the reflection phase's saved counters (results/reflection/llm_stats.json,
written by run_phase → save_llm_stats) and prints final totals for:

    reflections_generated   reflector produced a validated reflection
    reflections_stored      unique reflections added to memory
    memory_retrievals        get_context() calls that returned >=1 entry
    prompt_injections        times reflection text was injected into an LLM prompt
    planner_cache_misses     planner_calls - cache_hits (the only path that can inject)

Verdict: if memory_retrievals == 0 AND prompt_injections == 0, reflection
never influenced planning (offline / fully-cached planner short-circuits it).

Usage:
    python verify_reflection_influence.py [path/to/llm_stats.json]
    # default: results/reflection/llm_stats.json under cwd or $OUTPUT_DIR
"""
import os, sys, json


def _find_stats(argv):
    if len(argv) > 1:
        return argv[1]
    roots = [os.environ.get("OUTPUT_DIR", ""), os.getcwd(), "/kaggle/working"]
    for r in roots:
        if not r:
            continue
        p = os.path.join(r, "results", "reflection", "llm_stats.json")
        if os.path.exists(p):
            return p
    return os.path.join("results", "reflection", "llm_stats.json")


def verify(path=None):
    path = path or _find_stats(sys.argv)
    if not os.path.exists(path):
        print(f"[verify] llm_stats.json not found at: {path}")
        print("[verify] Run the reflection phase first (it writes this file).")
        return None

    d = json.load(open(path, encoding="utf-8"))
    planner   = d.get("planner", {})
    reflector = d.get("reflector", {})
    memory    = d.get("memory", {})

    totals = {
        "reflections_generated": int(reflector.get("generated", 0)),
        "reflections_stored":    int(memory.get("total_added", 0)),
        "memory_retrievals":     int(memory.get("total_retrievals", 0)),
        "prompt_injections":     int(planner.get("prompt_injections", 0)),
        "planner_cache_misses":  int(planner.get("cache_misses",
                                       max(0, planner.get("planner_calls", 0) - planner.get("cache_hits", 0)))),
    }

    print("=" * 56)
    print(f"  Reflection-influence audit  ({path})")
    print(f"  reflector backend: {reflector.get('backend','?')} | "
          f"planner_calls={planner.get('planner_calls',0)} cache_hits={planner.get('cache_hits',0)}")
    print("=" * 56)
    for k, v in totals.items():
        print(f"  {k:<24} = {v}")
    print("-" * 56)

    if totals["memory_retrievals"] == 0 and totals["prompt_injections"] == 0:
        print("  VERDICT: ❌ Reflection NEVER influenced planning.")
        print("           (0 retrievals, 0 prompt injections — the planner")
        print("            never hit a cache-miss, so reflection was never read.)")
        if totals["planner_cache_misses"] == 0:
            print("           Cause: planner_cache_misses=0 → fully cached / OFFLINE planner.")
    else:
        print(f"  VERDICT: ✅ Reflection influenced planning "
              f"({totals['prompt_injections']} prompt injections from "
              f"{totals['memory_retrievals']} retrievals).")
    print("=" * 56)
    return totals


if __name__ == "__main__":
    verify()
