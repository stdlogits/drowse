"""The probe routes never copy off the model's device outside the GPU section.

MPS serializes device work through one global command buffer that is not
thread-safe: a copy off the device on a request thread while another thread's
model op holds the buffer aborts the process (``failed assertion _status <
MTLCommandBufferStatusCommitted``). That is how the Units room's first load
killed the server: the dashboard's lock-free ``GET /probes`` copied each
geometry probe's resident ``node_coords`` while the room's node replay uploaded
its ids. The listing and the inspector's geometry now serve host copies taken
at attach. On CPU the race cannot abort, so these pin where the reads happen.
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


def test_probe_geometry_answers_from_the_host_payload_beside_a_reply(
    make_tiny_session: Any, app_client: Callable[..., Any],
) -> None:
    session = _session_with_probe(make_tiny_session)
    client = app_client(session)
    expected = client.get(f"{_PROBES}/mood/geometry").json()
    assert expected["name"] == "mood"
    # Any read of the resident tensors now fails: a meta tensor has no data.
    probe = session.monitor.attached_probes()["mood"]
    for sub in probe.manifold.layers.values():
        for attr in ("mean", "basis"):
            object.__setattr__(sub, attr, torch.empty(getattr(sub, attr).shape, device="meta"))
    for wh in probe.whitened.values():
        for attr in ("node_white", "neutral_white", "chol"):
            setattr(wh, attr, torch.empty(getattr(wh, attr).shape, device="meta"))

    # A streamed reply holds the server's session lock and the GPU for its
    # whole length. A geometry read that waited on them stalled the inspector
    # until the reply ended (or the lock's timeout); it answers beside the
    # reply instead.
    session_lock = client.app.state.session_lock
    client.portal.call(session_lock.acquire)
    acquired, release = threading.Event(), threading.Event()

    def hold() -> None:
        with session.gen_lock:
            acquired.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    answers: list[Any] = []
    try:
        assert acquired.wait(10)
        reader = threading.Thread(
            target=lambda: answers.append(client.get(f"{_PROBES}/mood/geometry")),
        )
        reader.start()
        reader.join(10)
        assert not reader.is_alive(), "geometry queued behind the reply's locks"
    finally:
        release.set()
        holder.join()
        client.portal.call(session_lock.release)
    assert answers[0].status_code == 200
    assert answers[0].json() == expected

    assert client.get(f"{_PROBES}/nope/geometry").status_code == 404
