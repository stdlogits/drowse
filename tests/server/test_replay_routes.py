"""``GET /instruments/token-anatomy`` and ``GET /instruments/units/activity``.

The node replay's two routes, beside the token readout: real behavior
through a tiny session, argument forwarding and refusals through the
autospec session.
"""

from __future__ import annotations

from typing import Any

import pytest

from drowse.generation.sampling import SamplingConfig
from drowse.loom.node_replay import decode_population
from tests.support.server import mock_session

pytestmark = pytest.mark.usefixtures("isolated_home")

_BASE = "/drowse/v1/sessions/default/instruments"


def _node(session: Any) -> Any:
    session.generate("hello", sampling=SamplingConfig(max_tokens=4, seed=0))
    return session.tree.nodes[session.tree.active_node_id]


def test_token_anatomy_route_returns_the_replay_envelope(
    tiny_session: Any, app_client: Any,
) -> None:
    node = _node(tiny_session)
    client = app_client(tiny_session)
    resp = client.get(
        f"{_BASE}/token-anatomy",
        params={"node_id": node.id, "raw_index": 1, "k": 2, "units": 3, "patterns": 0},
    )
    assert resp.status_code == 200, resp.text
    measurements = resp.json()["measurements"]
    assert measurements["scope"] == "replay"
    anatomy = measurements["anatomy"]
    assert anatomy["token"]["raw_index"] == 1
    assert len(anatomy["layers"]) == len(tiny_session.layers)
    layer = anatomy["layers"][0]
    assert set(layer) == {
        "layer", "depth", "reads_as", "attention", "mlp", "steering", "neurons", "heads",
    }
    assert layer["reads_as"]["of"] == "residual" and layer["mlp"]["of"] == "direction"
    assert anatomy["patterns"] is None
    bad = client.get(f"{_BASE}/token-anatomy", params={"node_id": node.id, "raw_index": 1, "k": 0})
    assert bad.status_code == 400 and "k must be" in bad.json()["detail"]
    missing = client.get(f"{_BASE}/token-anatomy", params={"node_id": "nope", "raw_index": 0})
    assert missing.status_code == 404


def test_unit_activity_route_parses_units_and_layers(
    tiny_session: Any, app_client: Any,
) -> None:
    node = _node(tiny_session)
    client = app_client(tiny_session)
    resp = client.get(
        f"{_BASE}/units/activity",
        params={
            "node_id": node.id, "top": 4, "units": "neuron/L1n3, head/L2h1",
            "raw_index": 1, "layers": "1-2",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["node_id"] == node.id and body["unit"] == "activation_over_scale"
    assert set(body["traces"]) == {"neuron/L1n3", "head/L2h1"}
    assert len(body["traces"]["neuron/L1n3"]) == len(body["tokens"]) == len(node.raw_token_ids)
    assert len(body["top"]) == 4
    population = body["population"]
    assert [layer["layer"] for layer in population["layers"]] == [1, 2]
    first = population["layers"][0]
    assert decode_population(first["neurons"]).shape == (first["n_neurons"],)
    # A population layer needs a token; an unknown unit is a 400.
    assert client.get(
        f"{_BASE}/units/activity", params={"node_id": node.id, "layers": "1"},
    ).status_code == 400
    assert client.get(
        f"{_BASE}/units/activity", params={"node_id": node.id, "units": "neuron/L99n0"},
    ).status_code == 400


def test_replay_routes_forward_their_knobs_and_the_units_422(app_client: Any) -> None:
    from drowse.units.runtime import UnitsUnavailableError

    session = mock_session()
    session.token_anatomy.return_value = {"measurements": None}
    session.unit_activity.side_effect = UnitsUnavailableError(
        "units are unavailable for stub/model: no unit statistics",
    )
    client = app_client(session)
    resp = client.get(
        f"{_BASE}/token-anatomy",
        params={"node_id": "n", "raw_index": 3, "patterns": 0, "steered": "false", "raw": "true"},
    )
    assert resp.status_code == 200 and resp.json() == {"measurements": None}
    session.token_anatomy.assert_called_once_with(
        "n", 3, k=8, units=8, patterns=0, apply_steering=False, raw=True,
    )
    resp = client.get(f"{_BASE}/units/activity", params={"node_id": "n", "top": 5})
    assert resp.status_code == 422
    assert "no unit statistics" in resp.json()["detail"]
    session.unit_activity.assert_called_once_with(
        "n", top=5, units=[], raw_index=None, layers=None, apply_steering=True, raw=False,
    )
