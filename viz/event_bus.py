"""
viz/event_bus.py — Thread-safe singleton event bus.

Training runs in a background thread and calls VIZ.emit() / VIZ.emit_throttled().
FastAPI WebSocket handler subscribes asyncio Queues and receives broadcasts via
asyncio.run_coroutine_threadsafe, safely bridging the training thread → the
uvicorn event loop.

Usage (training side):
    from viz.event_bus import VIZ
    VIZ.emit("step", {...})
    VIZ.emit_throttled("step", {...}, max_rate_hz=30)

Usage (server side):
    VIZ.initialize(loop)      # called once at startup
    q = VIZ.subscribe()       # asyncio.Queue; receive events
    VIZ.unsubscribe(q)        # cleanup on disconnect
"""

import asyncio
import json
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Set


# ── MiniGrid object-type → human label ────────────────────────────────────────
OBJ_NAMES = {
    0: "unseen", 1: "empty", 2: "wall", 3: "floor",
    4: "door", 5: "key", 6: "ball", 7: "box",
    8: "goal", 9: "lava", 10: "agent",
}

ACTION_NAMES = {
    0: "turn_left", 1: "turn_right", 2: "move_fwd",
    3: "pick_up", 4: "drop", 5: "toggle", 6: "done",
}


def encode_obs_grid(obs_4d) -> Optional[Dict]:
    """
    Encode a (1, H, W, 4) observation as a compact JSON-serialisable dict.

    Returns None on failure so callers can skip gracefully.

    Grid channels:
        0 – object type  (int 0-10)
        1 – color        (int 0-5)
        2 – state        (int: door open/closed/locked)
        3 – agent dir at agent cell (0-3), else 4
    """
    try:
        import numpy as np
        g = obs_4d[0]               # (H, W, 4)
        H, W = g.shape[:2]

        obj   = g[:, :, 0].tolist()
        color = g[:, :, 1].tolist()
        state = g[:, :, 2].tolist()
        amap  = g[:, :, 3]

        # Agent position: cell where channel-3 != 4
        ap = np.argwhere(amap != 4)
        if len(ap):
            r, c = int(ap[0][0]), int(ap[0][1])
            agent = [r, c]
            adir  = int(amap[r, c])
        else:
            agent = [0, 0]
            adir  = 0

        return {"obj": obj, "color": color, "state": state,
                "agent": agent, "dir": adir, "h": H, "w": W}
    except Exception:
        return None


class EventBus:
    """Thread-safe global event bus (singleton)."""

    _instance: Optional["EventBus"] = None
    _class_lock = threading.Lock()

    def __new__(cls):
        with cls._class_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._ready      = False
                inst._loop       = None
                inst._subs: Set[asyncio.Queue] = set()
                inst._sub_lock   = threading.Lock()
                inst._history    = deque(maxlen=1000)
                inst._state: Dict[str, Any] = {
                    "running":      False,
                    "phase":        None,
                    "iteration":    0,
                    "total_itr":    0,
                    "episode":      0,
                    "step":         0,
                    "total_steps":  0,
                    "success_rate": 0.0,
                    "avg_reward":   0.0,
                    "ep_len":       0.0,
                    "ks_coef":      1.0,
                    "entropy":      0.0,
                    "plan":         "—",
                    "obs_text":     "—",
                    "action":       "—",
                    "reward":       0.0,
                    "ep_reward":    0.0,
                    "grid":         None,
                    "last_llm":     None,
                    "phase_config": {},
                }
                # rate-limit timestamps per event type
                inst._last_emit: Dict[str, float] = {}
                cls._instance = inst
        return cls._instance

    # ── initialisation ─────────────────────────────────────────────────────────

    def initialize(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once from FastAPI startup with the running asyncio loop."""
        self._loop  = loop
        self._ready = True

    # ── subscription management ────────────────────────────────────────────────

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._sub_lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._sub_lock:
            self._subs.discard(q)

    # ── emit ───────────────────────────────────────────────────────────────────

    def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit immediately, broadcasting to all WebSocket subscribers."""
        if not self._ready:
            return
        event = {"type": event_type, "ts": time.time(), **data}
        # Store a stripped copy in history (drop large base64 frames to keep
        # history lean — replayed on reconnect shouldn't flood with images)
        history_event = {k: v for k, v in event.items() if k != "frame"}
        self._history.append(history_event)
        self._update_state(event_type, data)
        msg = json.dumps(event, default=_json_default)
        with self._sub_lock:
            subs = list(self._subs)
        for q in subs:
            try:
                asyncio.run_coroutine_threadsafe(
                    _queue_put_nowait(q, msg), self._loop
                )
            except Exception:
                pass

    def emit_throttled(self, event_type: str, data: Dict[str, Any],
                       max_rate_hz: float = 20.0) -> None:
        """Emit only if the minimum interval has elapsed (for high-frequency events)."""
        now = time.time()
        min_gap = 1.0 / max(max_rate_hz, 0.1)
        if now - self._last_emit.get(event_type, 0.0) < min_gap:
            return
        self._last_emit[event_type] = now
        self.emit(event_type, data)

    # ── state queries ──────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def get_history(self, n: int = 200) -> list:
        return list(self._history)[-n:]

    def reset(self) -> None:
        """Clear history and reset state (between pipeline runs)."""
        self._history.clear()
        self._last_emit.clear()
        self._state.update({
            "running": False, "phase": None, "iteration": 0,
            "episode": 0, "step": 0, "grid": None,
        })

    # ── internal state update ──────────────────────────────────────────────────

    def _update_state(self, event_type: str, data: Dict[str, Any]) -> None:
        s = self._state
        if event_type == "phase_start":
            s["running"]      = True
            s["phase"]        = data.get("phase_name")
            s["total_itr"]    = data.get("total_iterations", 0)
            s["phase_config"] = data.get("config", {})
            s["iteration"]    = 0
            s["episode"]      = 0
        elif event_type == "phase_end":
            s["running"] = False
        elif event_type == "step":
            s["step"]      = data.get("step", s["step"])
            s["episode"]   = data.get("episode", s["episode"])
            s["iteration"] = data.get("iteration", s["iteration"])
            s["plan"]      = data.get("plan", s["plan"])
            s["obs_text"]  = data.get("obs_text", s["obs_text"])
            s["action"]    = data.get("action", s["action"])
            s["reward"]    = data.get("reward", 0.0)
            s["ep_reward"] = data.get("ep_reward", 0.0)
            # Keep latest rendered frame for reconnecting clients
            if data.get("frame"):
                s["last_frame"] = data["frame"]
            if data.get("grid") is not None:
                s["grid"] = data["grid"]
        elif event_type == "episode_end":
            s["episode"] = data.get("episode", s["episode"])
        elif event_type == "iteration_end":
            s["iteration"]    = data.get("iteration", s["iteration"])
            s["success_rate"] = data.get("success_rate", s["success_rate"])
            s["avg_reward"]   = data.get("avg_reward", s["avg_reward"])
            s["ep_len"]       = data.get("ep_len", s["ep_len"])
            s["ks_coef"]      = data.get("ks_coef", s["ks_coef"])
            s["entropy"]      = data.get("entropy", s["entropy"])
            s["total_steps"]  = data.get("total_steps", s["total_steps"])
        elif event_type == "llm_call":
            s["last_llm"] = data


# ── helpers ────────────────────────────────────────────────────────────────────

async def _queue_put_nowait(q: asyncio.Queue, msg: str) -> None:
    try:
        q.put_nowait(msg)
    except asyncio.QueueFull:
        try:
            q.get_nowait()   # drop oldest
            q.put_nowait(msg)
        except Exception:
            pass


def _json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


# ── module-level singleton ────────────────────────────────────────────────────
VIZ = EventBus()
