#!/usr/bin/env python3
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
out_dir = WORK / "plots"
proc = WORK / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)
proc.mkdir(parents=True, exist_ok=True)
(REAL / "plots").mkdir(parents=True, exist_ok=True)

env = gym.make("CarRacing-v3", render_mode="rgb_array")
obs, info = env.reset(seed=0)
frames = []
total_r = 0.0
steps = 0
for t in range(60):
    action = np.array([0.0, 0.3, 0.0], dtype=np.float32)
    obs, r, terminated, truncated, info = env.step(action)
    total_r += float(r)
    steps += 1
    if t in (0, 20, 40, 59):
        frames.append((t, np.asarray(obs).copy()))
    if terminated or truncated:
        break
env.close()

saved = []
for t, fr in frames:
    path = out_dir / f"smoke_t{t:03d}.png"
    Image.fromarray(fr.astype(np.uint8)).save(path)
    dest = REAL / "plots" / path.name
    dest.write_bytes(path.read_bytes())
    saved.append(str(path))

summary = {
    "env": "CarRacing-v3",
    "steps": steps,
    "total_reward": total_r,
    "frames": saved,
    "obs_shape": list(frames[0][1].shape) if frames else None,
}
(proc / "sim_smoke.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
