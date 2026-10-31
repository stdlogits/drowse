"""The lock-free probe readers never copy off the model's device.

MPS serializes device work through one global command buffer that is not
thread-safe: a copy off the device on a request thread while another thread's
model op holds the buffer aborts the process (``failed assertion _status <
MTLCommandBufferStatusCommitted``). Under a stress loop on SmolLM2, the
dashboard's lock-free ``GET /probes`` (each geometry probe's resident
``node_coords``), ``GET /probes/{name}/geometry`` (the whitened factors) and
``POST /tree/transcript`` (the probe digest, folded and copied from the
resident basis) all copied off MPS while a reply held the GPU. All three now
come from host copies taken at attach. On CPU the race cannot abort, so these
pin where the reads happen: a resident tensor swapped for a meta tensor (which
has no data to copy) fails any reader that still walks it.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import torch

from drowse.readout.monitor_attach import probe_digest
from tests.support.sessions import register_vectors

_S = "/drowse/v1/sessions/default"


def _session_with_probe(make_tiny_session: Any) -> Any:
    session = make_tiny_session()
    register_vectors(session, "mood")
    session.add_probe("mood")
    return session


def _strip_resident(manifold: Any) -> None:
    """Swap the resident tensors for meta tensors: reading one now raises."""
    object.__setattr__(
        manifold, "node_coords",
        torch.empty(manifold.node_coords.shape, device="meta"),
    )
    for sub in manifold.layers.values():
        object.__setattr__(sub, "mean", torch.empty(sub.mean.shape, device="meta"))
        object.__setattr__(sub, "basis", torch.empty(sub.basis.shape, device="meta"))


def test_probe_listing_serves_the_attach_time_host_copy(
    make_tiny_session: Any, app_client: Callable[..., Any],
) -> None:
    session = _session_with_probe(make_tiny_session)
    probe = session.monitor.attached_probes()["mood"]
    expected = probe.manifold.node_coords.tolist()
    assert probe.node_coords_host == expected
    _strip_resident(probe.manifold)

    resp = app_client(session).get(f"{_S}/probes")

    assert resp.status_code == 200
    row = next(r for r in resp.json()["probes"] if r["name"] == "mood")
    assert row["node_coords"] == expected


def test_probe_digest_is_taken_at_attach(make_tiny_session: Any) -> None:
    session = _session_with_probe(make_tiny_session)
    probe = session.monitor.attached_probes()["mood"]
    expected = probe_digest(probe.manifold)
    assert probe.digest == expected
    _strip_resident(probe.manifold)

    assert session.geometry.probe_hash("mood") == expected
    assert session.geometry.probe_hash("nope") is None


def test_transcript_export_never_walks_the_resident_tensors(
    make_tiny_session: Any, app_client: Callable[..., Any],
) -> None:
    session = _session_with_probe(make_tiny_session)
    probe = session.monitor.attached_probes()["mood"]
    expected = probe_digest(probe.manifold)
    _strip_resident(probe.manifold)

    # Nothing has asked for a digest yet, so the roster's cache is cold and
    # the export reaches the instrument.
    resp = app_client(session).post(f"{_S}/tree/transcript", json={})

    assert resp.status_code == 200
    assert f"sha256: {expected}" in resp.json()["yaml"]


def test_probe_geometry_answers_from_the_host_payload_beside_a_reply(
    make_tiny_session: Any, app_client: Callable[..., Any],
) -> None:
    session = _session_with_probe(make_tiny_session)
    client = app_client(session)
    expected = client.get(f"{_S}/probes/mood/geometry").json()
    assert expected["name"] == "mood"
    probe = session.monitor.attached_probes()["mood"]
    _strip_resident(probe.manifold)
    for wh in probe.whitened.values():
        for attr in ("node_white", "neutral_white", "chol"):
            setattr(wh, attr, torch.empty(getattr(wh, attr).shape, device="meta"))

    # A streamed reply holds the server's session lock and the GPU for its
    # whole length. Under a stress loop, a geometry read that waited on the
    # session lock outlived a 30 s client timeout; the inspector answers
    # beside the reply instead.
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
            target=lambda: answers.append(client.get(f"{_S}/probes/mood/geometry")),
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

    assert client.get(f"{_S}/probes/nope/geometry").status_code == 404
