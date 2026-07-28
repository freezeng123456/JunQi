"""Ataraxos vs JunQi compute + parameter comparison.

Quick-reference table of where our config sits vs the Ataraxos paper's
published values, with an honest assessment of which gaps matter on a
single T4 GPU.
"""
from __future__ import annotations


def main():
    # ----- Compute scale -----
    print("=" * 72)
    print("  COMPUTE SCALE")
    print("=" * 72)
    atar_total = 208_000_000_000
    junqi_500R = 128 * 512 * 500
    junqi_fps = 4965  # measured from v25 logs
    print(f"Ataraxos total RL run (16×H100×1 week): {atar_total:>15,} env-steps")
    print(f"JunQi v25 500-rollout run (1×T4, 2h10m): {junqi_500R:>15,} env-steps")
    print(f"Ratio: {atar_total / junqi_500R:>44,.0f}×")
    print()
    print(f"JunQi measured fps: {junqi_fps:,} env-steps/s")
    print(f"To match Ataraxos scale on single T4: "
          f"{atar_total / junqi_fps / 3600 / 24 / 365:.1f} years "
          f"(obviously not happening)")
    print()

    # Their single-H100 ablation in Fig 13 — ~45h reached 1900 Elo
    # H100 fp16 matmul is ~4× faster than T4 ⇒ T4 needs ~180h
    single_h100_fps = 625_000       # 10M/s over 16 GPUs
    atar_single = single_h100_fps * 45 * 3600
    print(f"Ataraxos SINGLE-H100 ablation (Fig 13, 45h): {atar_single:>10,} env-steps")
    print(f"Ratio to v25 500R: {atar_single / junqi_500R:.0f}×")
    print(f"Equivalent T4 wall-clock (T4 ≈ H100/4 for fp16): "
          f"{atar_single / (junqi_fps) / 3600:.0f}h ≈ "
          f"{atar_single / junqi_fps / 3600 / 24:.1f} days")
    print()

    # Realistic targets for us
    print("-" * 72)
    print("  REALISTIC T4 TARGETS")
    print("-" * 72)
    for steps, label in [
        (1e8,  "100M env-steps (3× our 500R)"),
        (5e8,  "500M env-steps (15× our 500R)"),
        (1e9,  "1B  env-steps (30× our 500R)"),
    ]:
        hours = steps / junqi_fps / 3600
        rollouts = steps / (128 * 512)
        print(f"  {label:38s} = {int(rollouts):>5,} rollouts = {hours:>5.1f}h "
              f"({hours/24:.1f} days)")
    print()

    # ----- Parameter gap -----
    print("=" * 72)
    print("  MOVE-NET ARCHITECTURE GAP")
    print("=" * 72)
    rows = [
        ("hyperparam",  "Ataraxos",   "JunQi v17-v25",  "If we scaled"),
        ("depth",       "8",          "4",              "6 (half gap)"),
        ("embed_dim",   "384",        "128",            "256 (half gap)"),
        ("n_head",      "8",          "4",              "8"),
        ("ff_factor",   "4 → 1536",   "4 → 512",        "4 → 1024"),
        ("total params","14.7 M",     "~1.03 M",        "~4.5 M"),
        ("input chans", "455",        "256",            "256 (unchanged)"),
        ("value head",  "3-cat",      "3-cat ✓",        "3-cat"),
    ]
    for r in rows:
        print(f"  {r[0]:>14}  {r[1]:>10}  {r[2]:>16}  {r[3]:>18}")
    print()
    print("Note: Ataraxos's 14.7M move net runs at batch 1536 on H100. We")
    print("physically can't match that on T4, but the shape ratio matters:")
    print("  Ours = 1.03M / our 4N=512 batch  →  per-sample compute density")
    print("  Theirs = 14.7M / 1536 batch       →  ~2.8× our density")
    print()

    # ----- Belief net gap -----
    print("=" * 72)
    print("  BELIEF-NET ARCHITECTURE GAP")
    print("=" * 72)
    rows = [
        ("hyperparam",   "Ataraxos prod",    "JunQi v25",     "Cost on T4"),
        ("encoder depth","6",                "4",             "+25% forward"),
        ("decoder",      "4 AR blocks",      "0 (linear head)","+4 blocks of ~50%"),
        ("embed_dim",    "512",              "256",           "4× memory"),
        ("dropout",      "0.2",              "0.0",           "free"),
        ("temporal attn","Yes (RNN-ish)",    "No (stateless)","+embed_dim"),
        ("total params", "57.1 M",           "~3.89 M",       "~15× larger"),
    ]
    for r in rows:
        print(f"  {r[0]:>14}  {r[1]:>14}  {r[2]:>16}  {r[3]:>18}")
    print()

    # ----- Hyperparameter deltas to close TODAY -----
    print("=" * 72)
    print("  CONFIG DELTAS TO CLOSE TODAY (no code change needed)")
    print("=" * 72)
    deltas = [
        # (field, current, ataraxos, impact)
        ("ppo.num_epochs_per_rollout",     "1",     "1",     "✓ match"),
        ("ppo.gae_lambda (advantage λ)",    "0.5",   "0.5",   "✓ match"),
        ("ppo.td_lambda  (outcome λ)",      "0.8",   "0.8",   "✓ match"),
        ("ppo.vf_coef",                     "1.0",   "1.0",   "✓ match"),
        ("ppo.kl_coef (main policy KL)",    "0.1",   "0.1",   "✓ match"),
        ("ppo.clip_range",                  "0.2",   "0.2",   "✓ match"),
        ("ppo.ema_decay",                   "0.999", "0.999", "✓ match"),
        ("arr.ppo.kl_coef",                 "0.1",   "0.1",   "✓ was 0.01 in v16, fixed in v17+"),
        ("arr.ppo.ent_pred_coef",           "0.5",   "1.0",   "🟡 ours half — minor"),
        ("ppo.lr_ceil",                     "1e-4",  "1e-4",  "✓ match"),
        ("ppo.lr_floor",                    "5e-6",  "5e-6",  "✓ match"),
        ("ppo.lr_decay (t^exp)",            "1.1",   "1.1",   "🟡 was 0.6 in v16; check v17+"),
        ("ppo.adv_filt_rate",               "0.75",  "0.75",  "✓ match"),
        ("ppo.adv_filt_thresh",             "0.01",  "0.01",  "✓ match"),
        ("arr.reg_temp schedule (t^0.3)",   "has it","0.1/t^0.3","✓ has temp_init/decay"),
        ("ppo.temperature_decay (magnet)",  "0.3",   "0.3",   "✓ match"),
        ("env.num_envs",                    "128",   "1536",  "❌ T4 hardware-bound"),
        ("env.steps_per_env",               "512",   "202",   "🟡 different cadence"),
    ]
    for r in deltas:
        print(f"  {r[0]:>30s}  cur={r[1]:>7s}  atar={r[2]:>7s}  {r[3]}")
    print()

    # ----- The actual recommendation -----
    print("=" * 72)
    print("  RECOMMENDED v26 CONFIG CHANGES (ranked by expected gain / cost)")
    print("=" * 72)
    recs = [
        # (rank, change, effort, expected, risk)
        ("1", "net.depth 4→6, embed_dim 128→192",  "1×edit",  "+0.05 win_rate ceiling",  "LOW — fits on T4"),
        ("2", "total_rollouts 500→1000 (4h, ~100M env-steps)", "-",  "closer to paper signal regime",   "LOW — just time"),
        ("3", "scale net + double rollouts together", "above", "cumulative: peak 0.85+?", "MEDIUM — 4h run"),
        ("4", "check+fix lr_decay 0.6→1.1",         "1×edit",  "small, possibly positive late", "LOW"),
        ("5", "confirm arr.kl_coef = 0.1",          "verify",  "marginal",                "LOW"),
        ("6", "belief net cat_vf?",                 "verify",  "?",                       "--"),
    ]
    for r in recs:
        print(f"  [{r[0]}] {r[1]:55s} {r[2]:>10s}  → {r[3]}")
        print(f"         risk: {r[4]}")
    print()


if __name__ == "__main__":
    main()
