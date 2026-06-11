"""
viz/server.py — FastAPI training visualisation server.

Endpoints
---------
GET  /                  → dashboard HTML
GET  /state             → current training state snapshot
GET  /history           → last 200 events as JSON array
WS   /ws                → live event WebSocket (receives JSON strings)
POST /start             → start training (body: TrainConfig JSON)
POST /stop              → signal training to stop
GET  /health            → server health check
"""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Make sure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from viz.event_bus import VIZ

# ── app setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="MiniGrid Visualizer", version="1.0")

_training_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


@app.on_event("startup")
async def _startup():
    loop = asyncio.get_event_loop()
    VIZ.initialize(loop)
    print("[VizServer] Event bus initialised.")


# ── static pages ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── REST helpers ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "training": VIZ.get_state().get("running", False)}


@app.get("/state")
async def get_state():
    return JSONResponse(VIZ.get_state())


@app.get("/history")
async def get_history():
    return JSONResponse(VIZ.get_history(200))


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    q = VIZ.subscribe()

    # Immediately send current state + last 100 events so the UI bootstraps
    state = VIZ.get_state()
    # Re-attach latest frame to the init state so the grid shows on reconnect
    init_msg = {
        "type":    "init",
        "state":   state,
        "history": VIZ.get_history(100),
    }
    # Inject last rendered frame as a synthetic step event at the front
    if state.get("last_frame"):
        init_msg["last_frame"] = state["last_frame"]
    await websocket.send_text(json.dumps(init_msg))

    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=20.0)
                await websocket.send_text(msg)
            except asyncio.TimeoutError:
                # Keep-alive ping
                await websocket.send_text(json.dumps({"type": "ping"}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        VIZ.unsubscribe(q)


# ── Training control ──────────────────────────────────────────────────────────

@app.post("/start")
async def start_training(body: dict):
    global _training_thread, _stop_event

    if _training_thread is not None and _training_thread.is_alive():
        return JSONResponse({"error": "Training already running."}, status_code=409)

    _stop_event.clear()
    VIZ.reset()

    _training_thread = threading.Thread(
        target=_run_training_thread,
        args=(body, _stop_event),
        daemon=True,
        name="training",
    )
    _training_thread.start()
    return JSONResponse({"status": "started", "config": body})


@app.post("/stop")
async def stop_training():
    _stop_event.set()
    return JSONResponse({"status": "stop_requested"})


# ── Training thread ───────────────────────────────────────────────────────────

def _run_training_thread(config: dict, stop_event: threading.Event):
    """
    Full training run executed in a background thread.

    Config keys (all optional — sensible defaults provided):
        phase            : "ppo_only" | "planner" | "reflection"   (default: "reflection")
        total_iterations : int   (default: 200)
        episodes_per_itr : int   (default: 5)
        ppo_epochs       : int   (default: 3)
        batch_size       : int   (default: 64)
        learning_rate    : float (default: 3e-4)
        use_online_llm   : bool  (default: False)
        viz_speed_hz     : float (default: 30.0 — max step-event rate)
        run_all_phases   : bool  (default: False — run the 3-phase pipeline)
    """
    try:
        from experiment_config import build_experiment_config
        from experiment_runner import build_game, run_phase

        phase_name   = config.get("phase", "reflection")
        run_all      = config.get("run_all_phases", False)
        speed_hz     = float(config.get("viz_speed_hz", 30.0))
        use_online   = config.get("use_online_llm", False)

        # Current research pipeline = 3 phases.
        phases = (
            ["ppo_only", "planner", "reflection"]
            if run_all else [phase_name]
        )

        for pname in phases:
            if stop_event.is_set():
                break

            overrides = {
                "total_iterations":    config.get("total_iterations",  200),
                "episodes_per_iteration": config.get("episodes_per_itr", 5),
                "ppo_epochs":          config.get("ppo_epochs",         3),
                "batch_size":          config.get("batch_size",         64),
                "learning_rate":       config.get("learning_rate",      3e-4),
            }
            cfg = build_experiment_config(pname, overrides=overrides)

            # Store speed limit so Game.collect() can read it
            cfg._viz_speed_hz = speed_hz
            cfg._viz_stop     = stop_event

            VIZ.emit("phase_start", {
                "phase_name":       pname,
                "total_iterations": cfg.total_iterations,
                "config": {
                    "episodes_per_itr": cfg.episodes_per_iteration,
                    "ppo_epochs":       cfg.ppo_epochs,
                    "batch_size":       cfg.batch_size,
                    "lr":               cfg.learning_rate,
                    "use_teacher":      cfg.use_teacher_policy,
                    "use_planner":      cfg.use_planner,
                    "use_ks":           cfg.use_kickstarting,
                    "use_reflect":      cfg.use_reflection,
                    "use_replan":       cfg.use_mid_episode_replanning,
                },
            })

            qwen_llm = None
            if use_online and cfg.use_planner:
                try:
                    from utils.qwen_llm import QwenLLM
                    qwen_llm = QwenLLM(backend="ollama")
                    VIZ.emit("log", {"msg": f"[{pname}] Online Qwen connected."})
                except Exception as e:
                    VIZ.emit("log", {"msg": f"[{pname}] Qwen unavailable: {e} — offline mode."})

            try:
                game = build_game(cfg, task="SimpleDoorKey", device="cpu",
                                  seed=42, qwen_llm=qwen_llm)
                # Attach stop event so collect() can break early
                game._viz_stop = stop_event

                history = run_phase(cfg, game, verbose=True)

                final_sr = (
                    float(sum(history["success_rate"][-20:]) / max(len(history["success_rate"][-20:]), 1))
                    if history.get("success_rate") else 0.0
                )
                VIZ.emit("phase_end", {
                    "phase_name":        pname,
                    "final_success_rate": final_sr,
                    "total_steps":       game.total_steps,
                })

            except Exception as exc:
                import traceback
                VIZ.emit("error", {
                    "phase": pname,
                    "msg":   str(exc),
                    "trace": traceback.format_exc()[-800:],
                })
                raise

    except Exception as exc:
        import traceback
        VIZ.emit("error", {"msg": str(exc), "trace": traceback.format_exc()[-800:]})


# ── Entry point ───────────────────────────────────────────────────────────────

def run(host: str = "0.0.0.0", port: int = 7860, reload: bool = False):
    uvicorn.run(
        "viz.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run()
