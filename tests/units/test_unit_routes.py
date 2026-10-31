"""The units routes over a real tiny session: the unit view, top, find, the
catalog status and the catalog fetch preparation."""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch

from drowse.model.blocks import model_spec
from drowse.model.units import neuron_write
from drowse.naming import NeuronRef

pytestmark = pytest.mark.usefixtures("isolated_home")

_BASE = "/drowse/v1/sessions/default/instruments/units"


def test_read_answers_the_unit_view(tiny_session: Any, app_client: Any) -> None:
    client = app_client(tiny_session)
    resp = client.get(f"{_BASE}/read", params={"unit": "neuron/L1n3", "k": 3})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == tiny_session.units.read("neuron/L1n3", k=3).to_dict()
    assert body["kind"] == "neuron" and len(body["poles"]) == 2
    head = client.get(f"{_BASE}/read", params={"unit": "head/L2h1", "hide_hubs": "true"}).json()
    assert head["kind"] == "head" and set(head["inputs"]) == {"query", "key", "value"}
    bad = client.get(f"{_BASE}/read", params={"unit": "neuron/L1n9999"})
    assert bad.status_code == 400 and "outside block 1" in bad.json()["detail"]
    assert client.get(f"{_BASE}/read", params={"unit": "honest"}).status_code == 400
    assert client.get(f"{_BASE}/read", params={"unit": "neuron/L1n1", "k": 0}).status_code == 400


def test_top_and_find(tiny_session: Any, app_client: Any) -> None:
    client = app_client(tiny_session)
    top = client.get(f"{_BASE}/top", params={"token": "a", "by": "fires", "layers": "1,2", "k": 3})
    assert top.status_code == 200, top.text
    body = top.json()
    assert body["by"] == "fires" and set(body["layers"]) == {"1", "2"}
    # Without lenses no layer has a read: no hits, and each layer says why.
    assert body["hits"] == [] and body["layers"]["1"]["absent"]["R"]
    assert client.get(f"{_BASE}/top", params={"token": "a", "by": "says"}).status_code == 400
    w = neuron_write(model_spec(tiny_session.model).blocks[2], 5)
    found = client.post(f"{_BASE}/find", json={"vector": w.tolist(), "layer": 2, "k": 2})
    assert found.status_code == 200, found.text
    row = found.json()["layers"]["2"]
    assert row["neurons"][0]["unit"] == "neuron/L2n5" and row["neurons"][0]["pole"] == "+"
    by_selector = client.post(f"{_BASE}/find", json={"selector": "neuron/L1n7", "k": 1}).json()
    assert by_selector["layers"]["1"]["neurons"][0]["unit"] == "neuron/L1n7"
    for bad in ({}, {"selector": "x", "vector": [1.0]}, {"vector": [1.0, 2.0]},
                {"vector": [1.0, 2.0], "layer": 1}, {"selector": "x", "extra": 1}):
        assert client.post(f"{_BASE}/find", json=bad).status_code == 400, bad


def test_catalog_status_and_fetch_preparation(
    tiny_session: Any, app_client: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.units import catalog as cat

    client = app_client(tiny_session)
    status = client.get(f"{_BASE}/catalog").json()
    assert status["available"] is False and status["bound"] is False
    assert "no neuron-lens catalog" in status["reason"]
    refused = client.post(f"{_BASE}/preparations", json={"operation": "fetch"})
    assert refused.status_code == 404 and "no neuron-lens catalog" in refused.json()["detail"]
    assert client.post(f"{_BASE}/preparations", json={"operation": "fit"}).status_code == 400

    monkeypatch.setattr(cat, "CATALOG_MODELS", {
        **cat.CATALOG_MODELS,
        tiny_session.model_id.lower(): cat.CatalogModel(tiny_session.model_id, "tiny-test"),
    })
    calls: list[bool] = []

    def fetch(model_id: str, *, force: bool, on_progress: Any) -> dict[str, Any]:
        calls.append(force)
        on_progress("1/1 tiny-test/heads/labels.parquet")
        return {"slug": "tiny-test"}

    monkeypatch.setattr(cat, "fetch_catalog", fetch)
    started = client.post(f"{_BASE}/preparations", json={"operation": "fetch", "force": True})
    assert started.status_code == 202, started.text
    assert started.json()["operation"] == "fetch"
    deadline = time.monotonic() + 10
    while client.get(f"{_BASE}/preparations").json()["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    final = client.get(f"{_BASE}/preparations").json()
    assert final["state"] == "done" and final["slug"] == "tiny-test" and calls == [True]
    assert client.delete(f"{_BASE}/preparations").status_code == 409


def test_find_on_a_whitened_selector_matches_lens_decompose(tiny_session: Any) -> None:
    out = tiny_session.lens.decompose("neuron/L1n4", dictionary="neurons", k=2)
    assert out[1][1][0][0] == NeuronRef(layer=1, index=4).qualified
    assert out[1][0] == pytest.approx(1.0, abs=1e-4)
    find = tiny_session.units.find("neuron/L1n4", k=2).to_dict()
    assert find["layers"]["1"]["neurons"][0]["coeff"] == pytest.approx(out[1][1][0][1], rel=1e-5)
    with pytest.raises(Exception, match="residual-width"):
        tiny_session.units.find(torch.ones(3), layer=1)
