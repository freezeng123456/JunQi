"""Replay files are data, including when their contents are hostile."""

import random
from pathlib import Path

import numpy as np
import pytest

from junqi_core.replay import Trajectory
from junqi_core.replay_viewer import load_replay
from junqi_core.replay_with_policy import TrajectoryWithPolicy
from junqi_core.setup import generate_random_setup


class _Payload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return Path.touch, (self.marker,)


@pytest.mark.parametrize(
    "loader,field",
    [(Trajectory.load, "setups"), (TrajectoryWithPolicy.load, "setups"), (load_replay, "kind")],
)
def test_replay_rejects_pickle_without_executing(tmp_path, loader, field):
    marker = tmp_path / "executed"
    path = tmp_path / "hostile.npz"
    np.savez(path, **{field: np.array(_Payload(marker), dtype=object)})
    try:
        with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
            loader(path)
    finally:
        assert not marker.exists(), "replay loader executed pickle payload"


@pytest.mark.parametrize("cls", [Trajectory, TrajectoryWithPolicy])
def test_saved_replay_contains_only_non_object_arrays(tmp_path, cls):
    replay = cls(
        setups=generate_random_setup(random.Random(17)),
        actions=np.empty((0, 5), dtype=np.int16),
        rng_seed=17,
        final_state_hash=0,
    )
    path = tmp_path / "safe.npz"
    replay.save(path)
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            assert not data[key].dtype.hasobject
    assert cls.load(path).rng_seed == 17
