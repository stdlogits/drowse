"""The probe routes never copy off the model's device outside the GPU section.

MPS serializes device work through one global command buffer that is not
thread-safe: a copy off the device on a request thread while another thread's
model op holds the buffer aborts the process (``failed assertion _status <
MTLCommandBufferStatusCommitted``). That is how the Units room's first load
killed the server: the dashboard's lock-free ``GET /probes`` copied each
geometry probe's resident ``node_coords`` while the room's node replay uploaded
its ids. The listing now serves the host copy taken at attach, and the
inspector's geometry copies inside the exclusive-GPU section. On CPU the race
cannot abort, so these pin where the reads happen.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import torch

from tests.support.sessions import register_vectors

_PROBES = "/drowse/v1/sessions/default/probes"


def _session_with_probe(make_tiny_session: Any) -> Any:
    session = make_tiny_session()
    register_vectors(session, "mood")
    session.add_probe("mood")
    return session


def test_probe_listing_serves_the_attach_time_host_copy(
    make_tiny_session: Any, app_client: Callable[..., Any],
) -> None:
    session = _session_with_probe(make_tiny_session)
    probe = session.monitor.attached_probes()["mood"]
    expected = [[float(x) for x in row] for row in probe.manifold.node_coords.tolist()]
    assert probe.node_coords_host == expected
    # Any read of the resident tensor now fails: a meta tensor has no data to copy.
    object.__setattr__(
        probe.manifold, "node_coords",
        torch.empty(probe.manifold.node_coords.shape, device="meta"),
    )

    resp = app_client(session).get(_PROBES)

    assert resp.status_code == 200
    row = next(r for r in resp.json()["probes"] if r["name"] == "mood")
    assert row["node_coords"] == expected


def test_probe_geometry_copies_inside_the_gpu_section(
    make_tiny_session: Any, app_client: Callable[..., Any],
    monkeypatch: Any,
) -> None:
    session = _session_with_probe(make_tiny_session)
    client = app_client(session)
    real = session.monitor.probe_geometry
    held: list[bool] = []

    def recording(name: str, **kwargs: Any) -> Any:
        held.append(session.gen_lock._is_owned())
        return real(name, **kwargs)

    monkeypatch.setattr(session.monitor, "probe_geometry", recording)

    resp = client.get(f"{_PROBES}/mood/geometry")
    assert resp.status_code == 200
    assert resp.json()["name"] == "mood"
    assert held == [True]

    # Another thread's model op holds the GPU: a 409, never a racing copy.
    acquired, release = threading.Event(), threading.Event()

    def hold() -> None:
        with session.gen_lock:
            acquired.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert acquired.wait(10)
        resp = client.get(f"{_PROBES}/mood/geometry")
        assert resp.status_code == 409
        assert "in flight" in resp.json()["detail"]
    finally:
        release.set()
        holder.join()
    assert held == [True], "no geometry was read while another thread held the GPU"

    assert client.get(f"{_PROBES}/nope/geometry").status_code == 404
