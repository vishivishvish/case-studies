"""
Shared cross-process rate limiter for the NVIDIA NIM API.
ALL agents/notebooks calling the NVIDIA API in this batch of retrofits MUST use this
(or an identical mkdir-lock scheme against the SAME lock path) so the combined request
rate across all parallel agents stays under the free-tier cap of 40 requests/minute.

Usage:
    from nvidia_rate_limited_call import nvidia_chat
    content = nvidia_chat("your prompt here")
"""
import os
import time
import json
import random
import tempfile

# A fixed, session-independent temp path (NOT a Claude-session scratchpad, which gets wiped
# once that session ends) so this lock survives across separate notebook runs and processes.
_COORD_DIR = os.path.join(tempfile.gettempdir(), "case_studies_nvidia_coord")
os.makedirs(_COORD_DIR, exist_ok=True)
LOCK_DIR = os.path.join(_COORD_DIR, "lock")
TS_FILE = os.path.join(_COORD_DIR, "last_call.txt")
MIN_INTERVAL = 1.8  # seconds between ANY two calls across ALL agents => ~33 req/min, safely under the 40 RPM cap

STALE_LOCK_SECONDS = 30  # a lock held this long was almost certainly abandoned by a crashed process

def _acquire_slot():
    wait_started = time.time()
    while True:
        try:
            os.mkdir(LOCK_DIR)
            break
        except FileExistsError:
            # Stale-lock recovery: a crashed holder (e.g. a segfaulted kernel) can leave LOCK_DIR
            # behind forever, since its `finally: os.rmdir(LOCK_DIR)` never gets to run. Without this,
            # every future call across every process sharing this lock would spin here indefinitely.
            try:
                lock_age = time.time() - os.stat(LOCK_DIR).st_mtime
            except FileNotFoundError:
                lock_age = 0  # lock disappeared between the failed mkdir and this stat; just retry
            if lock_age > STALE_LOCK_SECONDS:
                try:
                    os.rmdir(LOCK_DIR)
                except OSError:
                    pass  # lost the race to another process also clearing it; fine, loop and retry
                continue
            if time.time() - wait_started > 120:
                raise TimeoutError(
                    "Timed out after 120s waiting for the shared NVIDIA rate-limit lock "
                    f"({LOCK_DIR}). Heavy contention from other concurrent callers, or a lock "
                    "that is stale but younger than the staleness threshold. Failing loudly "
                    "instead of hanging indefinitely."
                )
            time.sleep(0.05 + random.uniform(0, 0.05))
    try:
        last = 0.0
        if os.path.exists(TS_FILE):
            with open(TS_FILE) as f:
                content = f.read().strip()
                last = float(content) if content else 0.0
        now = time.time()
        wait = MIN_INTERVAL - (now - last)
        if wait > 0:
            time.sleep(wait)
        with open(TS_FILE, "w") as f:
            f.write(str(time.time()))
    finally:
        os.rmdir(LOCK_DIR)

def nvidia_chat(prompt, api_key=None, max_tokens=900, temperature=0.2, model="nvidia/nemotron-3-ultra-550b-a55b", max_retries=5):
    import requests
    if api_key is None:
        env_path = os.path.expanduser("~/.config/secrets/.env")
        with open(env_path) as f:
            for line in f:
                if line.startswith("NVIDIA_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                    break
    last_exc = None
    for attempt in range(max_retries):
        _acquire_slot()
        try:
            resp = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "detailed thinking off"},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=(30, 150)
            )
        except requests.exceptions.RequestException as exc:
            # Covers read/connect timeouts, dropped connections, DNS hiccups, etc. A single slow
            # or stalled response should not kill the whole notebook run: back off and retry
            # rather than propagating immediately, mirroring the 429 handling below.
            last_exc = exc
            time.sleep(2 ** attempt + random.uniform(0, 1))
            continue
        if resp.status_code == 429:
            time.sleep(2 ** attempt + random.uniform(0, 1))
            continue
        if resp.status_code >= 500:
            last_exc = RuntimeError(f"NVIDIA API returned {resp.status_code}: {resp.text[:300]}")
            time.sleep(2 ** attempt + random.uniform(0, 1))
            continue
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or ""
    if last_exc is not None:
        raise RuntimeError(f"NVIDIA API failed after {max_retries} retries: {last_exc}") from last_exc
    raise RuntimeError("NVIDIA API rate-limited after max retries")
