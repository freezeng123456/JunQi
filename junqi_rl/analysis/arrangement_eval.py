"""Full agents choose their own setups; held-out scenarios swap team seats."""
from __future__ import annotations

import random
import numpy as np
import torch

from junqi_core.rules import PieceType
from junqi_core.setup import generate_random_setup
from junqi_rl.arrangement.sampling import generate_arrangements
from junqi_rl.networks.arrangement_net import VOCAB_IDX_TO_PIECE_TYPE_VALUE
from junqi_rl.analysis.random_eval import evaluate_paired_head_to_head
from junqi_rl.analysis.protocol import paired_score_interval


def evaluate_arrangement_agent(policy, baseline, arrangement_net, *, num_games,
        num_envs, device, seed, max_moves, autocast_dtype=None):
    if num_games <= 0 or num_games % 2:
        raise ValueError("arrangement-agent evaluation requires an even game budget")
    n = num_games // 2
    with torch.random.fork_rng(devices=[arrangement_net.device.index or 0]):
        torch.manual_seed(seed)
        generated = generate_arrangements(4 * n, arrangement_net, max_resample=0)
    if generated.fallback_mask.any():
        raise RuntimeError("setup evaluation requires real arrangement-policy samples")
    lut = np.array(VOCAB_IDX_TO_PIECE_TYPE_VALUE, dtype=np.int8)
    learned = lut[generated.samples.argmax(-1).cpu().numpy()].reshape(n, 4, 30)
    random_setups = [generate_random_setup(random.Random(seed + i)) for i in range(n)]

    def factory(game_id, team):
        return tuple(tuple(PieceType(int(v)) for v in learned[game_id, seat])
            if seat % 2 == team else random_setups[game_id][seat] for seat in range(4))

    records = []
    metrics = evaluate_paired_head_to_head(policy, baseline, num_games=num_games,
        num_envs=num_envs, device=device, seed=seed, max_moves=max_moves,
        autocast_dtype=autocast_dtype, greedy=True, game_records=records,
        setup_factory=factory)
    for row in records:
        row["seat_setup_sources"] = ["arrangement_policy" if seat % 2 == row["first_team"]
                                    else "uniform_random" for seat in range(4)]
    paired = paired_score_interval(records, seed=seed, pair_key="scenario_pair_id")
    paired.update(wins=metrics["h2h/wins"], losses=metrics["h2h/losses"], draws=metrics["h2h/draws"],
        comparison="learned_setup_and_move_vs_frozen_move_with_random_setup",
        commander_hq_rate=float((learned[:, :, [26, 28]] == PieceType.SILING.value).any(-1).mean()))
    return paired, records
