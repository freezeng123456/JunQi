"""Tests for junqi_rl.networks.junqi_net.

Focused on GraphStem, which replaced the convolutional stem. The board is a
cross of 129 playable cells embedded in a 17x17 array, so a padded 3x3
convolution worked on 160 positions that do not exist and blurred them into
the ones that do.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


# ===========================================================================
# GraphStem — the board's graphs, not a grid
# ===========================================================================


class TestGraphStem:
    """The stem replaced a padded 3x3 convolution over the 17x17 encoding.

    That encoding holds 160 positions that are not on the board. A padded
    convolution spent 55% of its work on them and, for the 81 on-board cells
    whose window overlaps the void, made "off the board" look the same as
    "empty square". These pin the properties that made the swap worth doing.
    """

    def test_neighbour_tables_match_the_engine_topology(self) -> None:
        from junqi_core.board import (
            COMPACT_RAIL_DEGREE,
            COMPACT_RAIL_NEIGHBORS,
            COMPACT_ROAD_DEGREE,
            COMPACT_ROAD_NEIGHBORS,
            COMPACT_TO_FLAT,
            FLAT_TO_COMPACT,
            NUM_ON_BOARD_CELLS,
            is_camp,
        )
        from junqi_core.rail_topology import RAIL_ADJ

        on_board = {int(f) for f in COMPACT_TO_FLAT}
        for compact in range(NUM_ON_BOARD_CELLS):
            flat = int(COMPACT_TO_FLAT[compact])
            x, y = flat % 17, flat // 17

            want_road = set()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < 17 and 0 <= ny < 17):
                    continue
                nf = ny * 17 + nx
                if nf not in on_board:
                    continue
                diagonal = dx != 0 and dy != 0
                if diagonal and not (is_camp(x, y) or is_camp(nx, ny)):
                    continue
                want_road.add(int(FLAT_TO_COMPACT[nf]))
            got_road = set(
                COMPACT_ROAD_NEIGHBORS[compact][: COMPACT_ROAD_DEGREE[compact]].tolist()
            )
            assert got_road == want_road, f"road mismatch at ({x},{y})"

            want_rail = {
                int(FLAT_TO_COMPACT[int(v)]) for v in RAIL_ADJ[flat] if int(v) in on_board
            }
            got_rail = set(
                COMPACT_RAIL_NEIGHBORS[compact][: COMPACT_RAIL_DEGREE[compact]].tolist()
            )
            assert got_rail == want_rail, f"rail mismatch at ({x},{y})"

    def test_adjacency_is_symmetric(self) -> None:
        from junqi_core.board import (
            COMPACT_RAIL_DEGREE,
            COMPACT_RAIL_NEIGHBORS,
            COMPACT_ROAD_DEGREE,
            COMPACT_ROAD_NEIGHBORS,
            NUM_ON_BOARD_CELLS,
        )

        for nb, deg, name in (
            (COMPACT_ROAD_NEIGHBORS, COMPACT_ROAD_DEGREE, "road"),
            (COMPACT_RAIL_NEIGHBORS, COMPACT_RAIL_DEGREE, "rail"),
        ):
            for i in range(NUM_ON_BOARD_CELLS):
                for j in nb[i][: deg[i]].tolist():
                    assert i in nb[j][: deg[j]].tolist(), (
                        f"{name}: {i}->{j} but not {j}->{i}"
                    )

    def test_padded_slots_contribute_nothing(self) -> None:
        """A cell's output must depend only on itself and its real neighbours.

        The pad index is one past the last cell and gathers an appended zero
        row, so perturbing any non-neighbour must leave a cell untouched.
        """
        from junqi_core.board import (
            COMPACT_RAIL_DEGREE,
            COMPACT_RAIL_NEIGHBORS,
            COMPACT_ROAD_DEGREE,
            COMPACT_ROAD_NEIGHBORS,
        )
        from junqi_rl.networks.junqi_net import GraphStem

        torch.manual_seed(0)
        stem = GraphStem(8, 6, num_layers=1, ffn_factor=2).eval()
        x = torch.zeros(1, 129, 8)
        with torch.no_grad():
            base = stem(x)

        cell = 0
        neighbours = (
            set(COMPACT_ROAD_NEIGHBORS[cell][: COMPACT_ROAD_DEGREE[cell]].tolist())
            | set(COMPACT_RAIL_NEIGHBORS[cell][: COMPACT_RAIL_DEGREE[cell]].tolist())
            | {cell}
        )
        far = next(i for i in range(129) if i not in neighbours)
        x_far = x.clone()
        x_far[0, far] = 5.0
        with torch.no_grad():
            moved = stem(x_far)
        assert torch.allclose(base[0, cell], moved[0, cell], atol=1e-6), (
            f"cell {cell} changed when non-neighbour {far} did; "
            "one round of aggregation must stay local"
        )
        assert not torch.allclose(base[0, far], moved[0, far]), (
            "perturbation had no effect at all; test is vacuous"
        )

    def test_every_parameter_receives_gradient(self) -> None:
        from junqi_rl.networks.junqi_net import GraphStem

        stem = GraphStem(16, 12, num_layers=2, ffn_factor=2)
        out = stem(torch.randn(3, 129, 16))
        out.square().mean().backward()
        missing = [n for n, p in stem.named_parameters() if p.grad is None]
        assert not missing, f"no gradient reached: {missing}"
