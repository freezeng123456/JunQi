"""tools/measure_engineer_endgame.py — Engineer mobility in 2-seat endgames."""
import random
import numpy as np
from junqi_rl.env import JunqiEnv
from junqi_core.rules import PieceType


def measure():
    stats = {
        'max_moves': 0,
        'max_at': None,
        'two_dead_states': 0,
        'hist': [],
    }
    N_GAMES = 300
    MAX_STEPS = 600

    for g in range(N_GAMES):
        rng = random.Random(g)
        env = JunqiEnv()
        env.reset(seed=g)
        for step in range(MAX_STEPS):
            if env.state.terminated:
                break
            st = env.state
            n_dead = sum(1 for info in st.info.values() if info.dead)

            if n_dead >= 2:
                stats['two_dead_states'] += 1
                acting = st.turn
                if not st.info[acting].dead:
                    aids = env.legal_action_ids()
                    for pid in range(120):
                        if not st.alive[pid]:
                            continue
                        if st.piece_seat_arr[pid] != acting.value:
                            continue
                        if st.piece_type_arr[pid] != PieceType.GONGB.value:
                            continue
                        src_flat = int(st.pos_y[pid]) * 17 + int(st.pos_x[pid])
                        n_moves = int(((aids // 289) == src_flat).sum())
                        stats['hist'].append(n_moves)
                        if n_moves > stats['max_moves']:
                            stats['max_moves'] = n_moves
                            stats['max_at'] = (g, step, pid, src_flat, n_dead)

            aids = env.legal_action_ids()
            if aids.size == 0:
                break
            env._step_game_only(int(rng.choice(aids)))

    print(f"Games: {N_GAMES}, states with >=2 dead: {stats['two_dead_states']}")
    print(f"Engineer obs: {len(stats['hist'])}")
    print(f"Max engineer moves: {stats['max_moves']}")
    print(f"  context: {stats['max_at']}")

    if stats['hist']:
        h = np.array(stats['hist'])
        print(f"min={h.min()} median={int(np.median(h))} p90={int(np.percentile(h, 90))} p99={int(np.percentile(h, 99))} max={h.max()}")
        print("Histogram:")
        for bucket in range(0, int(h.max()) + 2, 2):
            cnt = int(((h >= bucket) & (h < bucket + 2)).sum())
            bar = '#' * (cnt * 60 // max(1, len(h)))
            print(f"  {bucket:2d}-{bucket+1:2d}: {cnt:5d} {bar}")


if __name__ == "__main__":
    measure()
