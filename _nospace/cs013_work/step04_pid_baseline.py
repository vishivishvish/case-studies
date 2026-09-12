#!/usr/bin/env python3
"""CS013 step 4: retuned vision PID on CarRacing-v3 (non-grass mask, correct steer sign)."""
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
    """Track ≈ not grass. Grass is strongly green-dominant in CarRacing."""
    img = obs.astype(np.float32)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:,:, 2]
    grass = (g > r + 20) & (g > b + 20) & (g > 100)
    # also drop near-black UI / borders
    bright = (r + g + b) / 3.0
    return (~grass) & (bright > 25)


def cross_track_and_coverage(obs: np.ndarray):
    """
    CTE in [-1, 1]: positive => road center is right of image center => steer right.
    coverage: fraction of road pixels in the sensing band.
    """
    h, w = obs.shape[:2]
    mask = road_mask(obs)
    # multi-row lookahead: weight nearer rows less than farther? use mid-lower band
    y0, y1 = int(h * 0.45), int(h * 0.82)
    band = mask[y0:y1, :]
    # downweight edges a bit by requiring a min count
    ys, xs = np.where(band)
    coverage = float(band.mean())
    if len(xs) < 40:
        # wider fallback
        band = mask[int(h * 0.35) : int(h * 0.90), :]
        ys, xs = np.where(band)
        coverage = float(band.mean())
    if len(xs) < 15:
        return 0.0, coverage, False
    # emphasize farther rows (smaller y in image = farther ahead in this view? in CarRacing
    # lower y is toward top of image = farther ahead). Weight by (y1 - y).
    weights = (y1 - (ys + y0)).astype(np.float32) + 1.0
    cx = float(np.average(xs, weights=weights))
    err = (cx - (w / 2.0)) / (w / 2.0)
    return float(np.clip(err, -1.5, 1.5)), coverage, True


def pid_action(err: float, prev_err: float, integ: float, saw_road: bool, coverage: float, dt: float = 1.0):
    # Retuned: correct sign (steer toward road), stronger lateral, sensible speed
    Kp, Ki, Kd = 1.1, 0.01, 0.35
    if not saw_road:
        # lost track: brake and hold last steer direction softly
        steer = float(np.clip(prev_err * 0.5, -1.0, 1.0))
        return np.array([steer, 0.05, 0.4], dtype=np.float32), integ

    integ = float(np.clip(integ + err * dt, -1.5, 1.5))
    deriv = (err - prev_err) / dt
    # POSITIVE err (road to the right) => POSITIVE steer (right)
    steer = float(np.clip(Kp * err + Ki * integ + Kd * deriv, -1.0, 1.0))
    # speed: more gas when centered and road visible
    gas = float(np.clip(0.2 + 0.55 * (1.0 - min(abs(err), 1.0)) * min(coverage * 4.0, 1.0), 0.0, 0.85))
    brake = float(np.clip(0.35 * abs(err) - 0.08, 0.0, 0.5))
    return np.array([steer, gas, brake], dtype=np.float32), integ


def run_episode(env, seed: int, save_frames: bool = False):
    obs, info = env.reset(seed=seed)
    prev_err = 0.0
    integ = 0.0
    total_r = 0.0
    steps = 0
    abs_err = []
    frames = []
    tiles = 0
    for t in range(1000):
        err, coverage, saw = cross_track_and_coverage(obs)
        if saw:
            abs_err.append(abs(err))
        action, integ = pid_action(err, prev_err, integ, saw, coverage)
        if saw:
            prev_err = err
        obs, r, terminated, truncated, info = env.step(action)
        total_r += float(r)
        steps += 1
        if info and "tile_visited_count" in info:
            tiles = int(info["tile_visited_count"])
        if save_frames and t in (0, 50, 150, 400, 800):
            frames.append((t, np.asarray(obs).copy()))
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "steps": steps,
        "total_reward": total_r,
        "mean_abs_cte": float(np.mean(abs_err)) if abs_err else None,
        "tile_visited_count": tiles,
        "frames": frames,
    }


def main():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    seeds = [0, 1, 2]
    episodes = []
    showcase = None
    for i, seed in enumerate(seeds):
        ep = run_episode(env, seed=seed, save_frames=(i == 0))
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
    tiles = [e["tile_visited_count"] for e in episodes]
    summary = {
        "controller": "vision_PID_lane_center_v2",
        "env": "CarRacing-v3",
        "retune": "non-grass mask; steer sign flipped toward road; coverage-aware throttle; lost-track brake",
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_abs_cte": float(np.mean(ctes)) if ctes else None,
        "mean_tiles": float(np.mean(tiles)),
        "frames": saved,
        "gains": {"Kp": 1.1, "Ki": 0.01, "Kd": 0.35},
        "notes": "v2 retune after v1 mean reward ~-55 / 0 tiles (inverted steer + weak mask).",
    }
    (BASE / "pid_scores.json").write_text(json.dumps(summary, indent=2))
    (PROC / "pid_baseline.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "mean_reward": summary["mean_reward"],
        "std_reward": summary["std_reward"],
        "mean_abs_cte": summary["mean_abs_cte"],
        "mean_tiles": summary["mean_tiles"],
        "n_episodes": len(episodes),
        "frames": len(saved),
        "episodes": episodes,
    }))


if __name__ == "__main__":
    main()
