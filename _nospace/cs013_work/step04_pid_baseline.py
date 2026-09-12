#!/usr/bin/env python3
"""CS013 step 4: vision PID / lane-centering baseline on CarRacing-v3."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGLET_HEADLESS", "1")

import numpy as np
from PIL import Image
import gymnasium as gym

WORK = Path("/home/box/case-studies/_nospace/cs013_work")
REAL = Path("/home/box/case-studies/013 - Self-Driving Car Sim")
PLOTS = WORK / "plots"
BASE = WORK / "baselines"
PROC = WORK / "data" / "processed"
for d in (PLOTS, BASE, PROC, REAL / "plots"):
    d.mkdir(parents=True, exist_ok=True)


def road_mask(obs: np.ndarray) -> np.ndarray:
    """Heuristic: track is grayish; grass is green-dominant."""
    img = obs.astype(np.float32)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # grass: g high vs r/b; road: channels close and mid brightness
    greenish = (g > r + 15) & (g > b + 15)
    grayish = (np.abs(r - g) < 20) & (np.abs(r - b) < 20) & (np.abs(g - b) < 20)
    bright = (r + g + b) / 3.0
    road = grayish & (bright > 40) & (bright < 220) & (~greenish)
    return road


def cross_track_error(obs: np.ndarray) -> float:
    """Offset in [-1, 1]: negative = road center left of image center."""
    h, w = obs.shape[:2]
    # look a few rows ahead of the car (lower-middle band)
    y0, y1 = int(h * 0.55), int(h * 0.78)
    band = road_mask(obs)[y0:y1, :]
    ys, xs = np.where(band)
    if len(xs) < 30:
        # fallback: wider band
        band = road_mask(obs)[int(h * 0.45) : int(h * 0.85), :]
        ys, xs = np.where(band)
    if len(xs) < 10:
        return 0.0
    cx = float(xs.mean())
    return (cx - (w / 2.0)) / (w / 2.0)


def pid_action(err: float, prev_err: float, integ: float, dt: float = 1.0):
    # tuned lightly for smoke-quality baseline (not SOTA)
    Kp, Ki, Kd = 0.85, 0.02, 0.25
    integ = float(np.clip(integ + err * dt, -2.0, 2.0))
    deriv = (err - prev_err) / dt
    steer = float(np.clip(-(Kp * err + Ki * integ + Kd * deriv), -1.0, 1.0))
    # throttle up when centered; brake a touch when error large
    gas = float(np.clip(0.35 + 0.25 * (1.0 - min(abs(err), 1.0)), 0.0, 1.0))
    brake = float(np.clip(0.15 * abs(err) - 0.05, 0.0, 0.4))
    return np.array([steer, gas, brake], dtype=np.float32), integ


def run_episode(env, seed: int, save_frames: bool = False, tag: str = "pid"):
    obs, info = env.reset(seed=seed)
    prev_err = 0.0
    integ = 0.0
    total_r = 0.0
    steps = 0
    abs_err = []
    frames = []
    tiles = 0
    for t in range(800):
        err = cross_track_error(obs)
        abs_err.append(abs(err))
        action, integ = pid_action(err, prev_err, integ)
        prev_err = err
        obs, r, terminated, truncated, info = env.step(action)
        total_r += float(r)
        steps += 1
        if "tile_visited_count" in (info or {}):
            tiles = int(info["tile_visited_count"])
        if save_frames and t in (0, 100, 250, 500, 799):
            frames.append((t, np.asarray(obs).copy()))
        if terminated or truncated:
            break
    mean_abs_err = float(np.mean(abs_err)) if abs_err else None
    return {
        "seed": seed,
        "steps": steps,
        "total_reward": total_r,
        "mean_abs_cte": mean_abs_err,
        "tile_visited_count": tiles,
        "frames": frames,
    }


def main():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    seeds = [0, 1, 2]
    episodes = []
    showcase = None
    for i, seed in enumerate(seeds):
        ep = run_episode(env, seed=seed, save_frames=(i == 0), tag="pid")
        frames = ep.pop("frames")
        episodes.append(ep)
        if i == 0:
            showcase = frames
    env.close()

    saved = []
    if showcase:
        for t, fr in showcase:
            path = PLOTS / f"pid_t{t:03d}.png"
            Image.fromarray(fr.astype(np.uint8)).save(path)
            dest = REAL / "plots" / path.name
            dest.write_bytes(path.read_bytes())
            saved.append(str(path))

    rewards = [e["total_reward"] for e in episodes]
    ctes = [e["mean_abs_cte"] for e in episodes if e["mean_abs_cte"] is not None]
    summary = {
        "controller": "vision_PID_lane_center",
        "env": "CarRacing-v3",
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_abs_cte": float(np.mean(ctes)) if ctes else None,
        "frames": saved,
        "gains": {"Kp": 0.85, "Ki": 0.02, "Kd": 0.25},
        "notes": "Heuristic gray-road mask + PD/PID on cross-track error; exploratory baseline, not SOTA.",
    }
    (BASE / "pid_scores.json").write_text(json.dumps(summary, indent=2))
    (PROC / "pid_baseline.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "mean_reward": summary["mean_reward"],
        "std_reward": summary["std_reward"],
        "mean_abs_cte": summary["mean_abs_cte"],
        "n_episodes": len(episodes),
        "frames": len(saved),
        "scores_path": str(BASE / "pid_scores.json"),
    }))


if __name__ == "__main__":
    main()
