#!/usr/bin/env python3
"""CS013 Self-Driving Car Sim — gated pipeline (pitch CS032)."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

WORK_DIR = Path("/home/box/case-studies/_nospace/cs013_work")
REAL_DIR = Path("/home/box/case-studies/013 - Self-Driving Car Sim")
VENV_PYTHON = "/home/box/case-studies/venv/bin/python"
STATUS_PATH = WORK_DIR / "step_status.json"
LOG_PATH = WORK_DIR / "steps.log"
PROC = WORK_DIR / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("cs013")


def now_ist() -> str:
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()


def load_status() -> List[Dict[str, Any]]:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text())
    return []


def upsert_status(step: int, name: str, state: str, detail: str = "") -> None:
    rows = load_status()
    row = {"step": step, "name": name, "state": state, "detail": detail, "updated_at_ist": now_ist()}
    found = False
    for i, r in enumerate(rows):
        if r.get("step") == step:
            rows[i] = row
            found = True
            break
    if not found:
        rows.append(row)
    rows.sort(key=lambda r: r.get("step", 0))
    STATUS_PATH.write_text(json.dumps(rows, indent=2))


def step_01_env_check() -> Dict[str, Any]:
    """Verify venv + gymnasium + box2d / CarRacing import path."""
    PROC.mkdir(parents=True, exist_ok=True)
    py_ver = subprocess.run([VENV_PYTHON, "--version"], capture_output=True, text=True)
    detail: Dict[str, Any] = {
        "python": (py_ver.stdout or py_ver.stderr or "").strip(),
        "venv_python": VENV_PYTHON,
        "packages": {},
        "carracing": None,
        "ok": False,
    }

    # Try imports; install gymnasium[box2d] once if missing
    check = subprocess.run(
        [VENV_PYTHON, "-c", "import gymnasium; print(gymnasium.__version__)"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        log.info("Installing gymnasium[box2d] into case-studies venv…")
        inst = subprocess.run(
            [VENV_PYTHON, "-m", "pip", "install", "-q", "gymnasium[box2d]"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        detail["pip_install"] = {
            "returncode": inst.returncode,
            "stderr_tail": (inst.stderr or "")[-800:],
        }
        if inst.returncode != 0:
            detail["error"] = "pip install gymnasium[box2d] failed"
            (PROC / "env_check.json").write_text(json.dumps(detail, indent=2))
            raise RuntimeError(detail["error"] + "\n" + detail["pip_install"]["stderr_tail"])

    code = r"""
import json, sys
out = {"packages": {}, "carracing": None}
try:
    import gymnasium as gym
    out["packages"]["gymnasium"] = gym.__version__
except Exception as e:
    out["packages"]["gymnasium"] = f"ERR: {e}"
    print(json.dumps(out)); sys.exit(1)
for name in ("numpy", "torch", "sklearn", "PIL"):
    try:
        m = __import__(name if name != "PIL" else "PIL")
        ver = getattr(m, "__version__", "ok")
        out["packages"][name] = ver
    except Exception as e:
        out["packages"][name] = f"missing: {e}"
# Box2D / CarRacing
try:
    import Box2D
    out["packages"]["Box2D"] = getattr(Box2D, "__version__", "ok")
except Exception as e:
    out["packages"]["Box2D"] = f"missing: {e}"
try:
    env = gym.make("CarRacing-v3", render_mode=None)
    obs, info = env.reset(seed=0)
    out["carracing"] = {
        "id": "CarRacing-v3",
        "obs_shape": list(getattr(obs, "shape", [])),
        "action_space": str(env.action_space),
    }
    env.close()
    out["ok"] = True
except Exception as e:
    out["carracing"] = {"error": str(e)}
    out["ok"] = False
print(json.dumps(out))
sys.exit(0 if out.get("ok") else 2)
"""
    r = subprocess.run([VENV_PYTHON, "-c", code], capture_output=True, text=True, timeout=180)
    if r.stdout.strip():
        try:
            detail.update(json.loads(r.stdout.strip().splitlines()[-1]))
        except json.JSONDecodeError:
            detail["raw_stdout"] = r.stdout[-1000:]
    if r.returncode != 0 and not detail.get("ok"):
        detail["stderr_tail"] = (r.stderr or "")[-800:]
        (PROC / "env_check.json").write_text(json.dumps(detail, indent=2))
        raise RuntimeError(f"env_check failed: {detail.get('carracing') or detail.get('stderr_tail')}")
    detail["ok"] = True
    (PROC / "env_check.json").write_text(json.dumps(detail, indent=2))
    return {"detail": json.dumps(detail)[:500]}



def step_02_sim_smoke() -> Dict[str, Any]:
    """Run a short CarRacing episode headless; save sample frames."""
    script = WORK_DIR / "step02_sim_smoke.py"
    r = subprocess.run(
        [VENV_PYTHON, str(script)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(WORK_DIR),
        env={**__import__("os").environ, "SDL_VIDEODRIVER": "dummy", "PYGLET_HEADLESS": "1"},
    )
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "sim_smoke failed")[-1200:])
    line = (r.stdout or "").strip().splitlines()[-1]
    summary = json.loads(line)
    return {
        "detail": f"steps={summary.get('steps')} reward={float(summary.get('total_reward', 0)):.2f} frames={len(summary.get('frames', []))}"
    }


def _stub(n: int, name: str) -> Callable[[], Dict[str, Any]]:
    def _f() -> Dict[str, Any]:
        raise RuntimeError(f"Step {n} ({name}) not implemented yet — scaffold only")

    return _f


STEPS: List[Dict[str, Any]] = [
    {"num": 1, "name": "env_check", "fn": step_01_env_check, "desc": "Verify Python, venv, gymnasium, CarRacing-v3", "deps": []},
    {"num": 2, "name": "sim_smoke", "fn": step_02_sim_smoke, "desc": "Smoke episode + frames", "deps": [1]},
    {"num": 3, "name": "extract_stub_notebook", "fn": _stub(3, "extract_stub_notebook"), "desc": "Notebook outline", "deps": []},
    {"num": 4, "name": "pid_baseline", "fn": _stub(4, "pid_baseline"), "desc": "PID baseline", "deps": [2]},
    {"num": 5, "name": "collect_expert", "fn": _stub(5, "collect_expert"), "desc": "Expert rollouts", "deps": [4]},
    {"num": 6, "name": "bc_train", "fn": _stub(6, "bc_train"), "desc": "Behavioral cloning", "deps": [5]},
    {"num": 7, "name": "bc_eval", "fn": _stub(7, "bc_eval"), "desc": "BC vs PID", "deps": [6]},
    {"num": 8, "name": "rl_ppo", "fn": _stub(8, "rl_ppo"), "desc": "Time-boxed RL", "deps": [2]},
    {"num": 9, "name": "rl_eval", "fn": _stub(9, "rl_eval"), "desc": "Metrics ladder", "deps": [8]},
    {"num": 10, "name": "xai_policy", "fn": _stub(10, "xai_policy"), "desc": "XAI", "deps": [6]},
    {"num": 11, "name": "feature_clusters", "fn": _stub(11, "feature_clusters"), "desc": "Clusters", "deps": [5]},
    {"num": 12, "name": "dl_light", "fn": _stub(12, "dl_light"), "desc": "Light DL", "deps": [5]},
    {"num": 13, "name": "foundation_skip", "fn": _stub(13, "foundation_skip"), "desc": "FM skip", "deps": []},
    {"num": 14, "name": "genai_tracks_stub", "fn": _stub(14, "genai_tracks_stub"), "desc": "Synth tracks", "deps": [2]},
    {"num": 15, "name": "genai_features", "fn": _stub(15, "genai_features"), "desc": "GenAI features", "deps": [14, 6]},
    {"num": 16, "name": "agentic_stub", "fn": _stub(16, "agentic_stub"), "desc": "Agentic stub", "deps": [9]},
    {"num": 17, "name": "hybrid_endgame_note", "fn": _stub(17, "hybrid_endgame_note"), "desc": "Hybrid note", "deps": [16]},
    {"num": 18, "name": "executed_notebook", "fn": _stub(18, "executed_notebook"), "desc": "Execute notebook", "deps": [1]},
    {"num": 19, "name": "fill_conclusion", "fn": _stub(19, "fill_conclusion"), "desc": "Conclusion", "deps": [18]},
    {"num": 20, "name": "sync_to_real", "fn": _stub(20, "sync_to_real"), "desc": "Sync", "deps": [18, 19]},
    {"num": 21, "name": "readme_update", "fn": _stub(21, "readme_update"), "desc": "README", "deps": [20]},
    {"num": 22, "name": "git_commit", "fn": _stub(22, "git_commit"), "desc": "PR", "deps": [21]},
]


def run_step(num: int) -> None:
    meta = next(s for s in STEPS if s["num"] == num)
    name = meta["name"]
    upsert_status(num, name, "running", "started")
    log.info("Step %s [running]: %s — started", num, name)
    try:
        result = meta["fn"]()
        detail = result.get("detail", "ok") if isinstance(result, dict) else str(result)
        upsert_status(num, name, "done", detail)
        log.info("Step %s [done]: %s — %s", num, name, detail[:200])
    except Exception as e:
        upsert_status(num, name, "failed", str(e))
        log.exception("Step %s failed", num)
        raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--step", type=int)
    args = p.parse_args()
    if args.list:
        for s in STEPS:
            print(f"{s['num']:2d}  {s['name']:24s}  {s['desc']}")
        return
    if args.status:
        print(json.dumps(load_status(), indent=2))
        return
    if args.step:
        run_step(args.step)
        return
    p.print_help()


if __name__ == "__main__":
    main()
