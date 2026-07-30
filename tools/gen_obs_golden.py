"""Generate observation golden hashes BEFORE the ObservationBuilder rewrite.

Usage::

    python3 -m tools.gen_obs_golden   # writes tests/golden/obs_hashes.json

The file is the pin-down artifact for Phase 0.4 M2. Floating-point belief
normalisation may differ in the last few bits between Apple Silicon and
Linux x86, so the portable golden digest quantises values to 1e-6 before
hashing. The direct builder-vs-wrapper test remains byte-identical within
each runtime.

Scenarios covered (12 total):

  * 3 random seeds:   0, 42, 2026
  * 2 steps-taken:    0 (opening), 100 (mid-game)
  * 2 show modes:     BRIGHT, HALF_DARK
  * observer:         SOUTH  (all four seats would be redundant — rotation
                              is exercised separately in rotation tests)

We hash the ``(spatial.tobytes(), global_.tobytes())`` pair with SHA-256,
along with a compact metadata header so drift reports are legible.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from junqi_core.info_model import BeliefTensor
from junqi_core.observation import build_observation
from junqi_core.rules import Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState


GOLDEN_HASH_KIND = "quantized_i32_1e6"
_GOLDEN_SCALE = 1_000_000.0


def hash_observation(obs) -> str:
    """Return an endian-stable digest at micro-unit numeric precision."""
    h = hashlib.sha256()
    for arr in (obs.spatial, obs.global_):
        if not np.isfinite(arr).all():
            raise ValueError("observation contains non-finite values")
        quantized = np.rint(
            arr.astype(np.float64, copy=False) * _GOLDEN_SCALE
        ).astype("<i4")
        h.update(np.asarray(arr.shape, dtype="<i4").tobytes())
        h.update(quantized.tobytes())
    return h.hexdigest()


def _advance(state: GameState, belief: BeliefTensor, steps: int, rng: random.Random):
    for _ in range(steps):
        if state.terminated:
            break
        legal = state.legal_actions()
        if not legal:
            break
        action = rng.choice(legal)
        new_state, result = state.step(action)
        if not new_state.info[belief.observer].dead:
            belief.update(state, new_state, result)
        state = new_state
    return state, belief


def _scenarios():
    for seed in (0, 42, 2026):
        for steps in (0, 100):
            for show_mode in (ShowMode.BRIGHT, ShowMode.HALF_DARK):
                yield (seed, steps, show_mode)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "tests" / "golden" / "obs_hashes.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)

    records = []
    for seed, steps, show_mode in _scenarios():
        rng = random.Random(seed)
        setups = generate_random_setup(rng)
        state = GameState.new_game(setups, show_mode=show_mode)
        belief = BeliefTensor.initial(state, Seat.SOUTH)
        state, belief = _advance(state, belief, steps, rng)

        if state.terminated:
            # Skip terminated states; builder behavior on terminal states is
            # a distinct concern.
            continue

        obs = build_observation(state, belief, Seat.SOUTH)

        digest = hash_observation(obs)

        # Also stash a tiny fingerprint (sum + max per channel group) for
        # human-readable drift diagnosis.
        spatial_sums = {
            name: float(obs.channel(name).sum())
            for name in obs.channel.__self__.__class__.__mro__[0].__dict__  # lint-safe sentinel
            if False  # placeholder; we use CHANNEL_LAYOUT explicitly below
        }
        # Explicit per-name sums (safer).
        from junqi_core.observation import CHANNEL_LAYOUT, GLOBAL_LAYOUT
        spatial_sums = {
            name: float(obs.channel(name).sum()) for name in CHANNEL_LAYOUT
        }
        global_sums = {
            name: float(obs.global_slice(name).sum()) for name in GLOBAL_LAYOUT
        }

        records.append({
            "seed": seed,
            "steps": steps,
            "show_mode": show_mode.name,
            "observer": "SOUTH",
            "hash_kind": GOLDEN_HASH_KIND,
            "sha256": digest,
            "spatial_sums": spatial_sums,
            "global_sums": global_sums,
        })

    out_path.write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} golden obs records to {out_path}")


if __name__ == "__main__":
    main()
