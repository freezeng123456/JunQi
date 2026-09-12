"""Exercise the real HTTP serialization boundary, including metadata."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from junqi_core.replay import record_trajectory
from junqi_core.rules import Seat, ShowMode
from junqi_viz.app import create_app


@pytest.mark.parametrize("mode", list(ShowMode))
def test_http_player_perspectives_and_fail_closed_query(tmp_path, mode):
    path = tmp_path / "game.npz"
    record_trajectory(rng_seed=42, max_steps=10, show_mode=mode).save(path)
    with TestClient(create_app(str(path))) as client:
        assert client.get("/api/frame/0").json()["observer"] == "SOUTH"
        for observer in Seat:
            query = {"observer": observer.name}
            frame = client.get("/api/frame/0", params=query).json()
            for piece in frame["pieces"]:
                owner = Seat[piece["seat"]]
                expected_known = (
                    mode is ShowMode.BRIGHT or owner is observer
                    or (mode is ShowMode.HALF_DARK and owner.team == observer.team)
                )
                assert (piece["type"] != "DARK") == expected_known
                if not expected_known:
                    assert piece["label"] == "暗棋"
                    assert piece["deduced_label"] is None
            assert frame["policy"] is None
            meta = client.get("/api/meta", params=query).json()
            assert meta["observer"] == observer.name
            assert meta["key_events"] == []
            assert meta["meta"] == {}
        full = client.get("/api/frame/0", params={"observer": "OMNISCIENT"}).json()
        assert all(piece["type"] != "DARK" for piece in full["pieces"])
        for endpoint in ("/api/meta", "/api/frame/0"):
            assert client.get(endpoint, params={"observer": "unknown"}).status_code == 422
        assert client.get("/api/frame/-1").status_code == 404
        assert client.get("/api/frame/999999").status_code == 404
