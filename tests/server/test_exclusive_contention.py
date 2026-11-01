"""Routes that touch the device while another thread holds the model.

MPS serializes device work through one global command buffer that is not
thread-safe, so every upload, load and model read has to run inside the
exclusive-GPU section (``session.gen_lock``).  A background lens fit or SAE
train holds that lock without the server's session lock, so these routes can
run beside it.  Each test holds the lock on another thread and checks that the
route refuses (409) or only looks things up, and that the device work it would
have done never ran.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch

from tests.support.sessions import register_vectors

pytestmark = pytest.mark.usefixtures("isolated_home")

_BASE = "/drowse/v1/sessions/default"


@contextmanager
def gpu_held_elsewhere(session: Any, seconds: float = 30.0) -> Iterator[None]:
    """Hold ``session.gen_lock`` on another thread (a background fit)."""
    acquired, release = threading.Event(), threading.Event()

    def hold() -> None:
        with session.gen_lock:
            acquired.set()
            release.wait(seconds)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert acquired.wait(10)
    try:
        yield
    finally:
        release.set()
        holder.join()


def _spy(target: Any, name: str, session: Any, calls: list[bool], monkeypatch: Any) -> None:
    """Record, for each call of ``target.name``, whether another thread held
    the model lock at that moment."""
    real = getattr(target, name)

    def spy(*args: Any, **kwargs: Any) -> Any:
        lock = session.gen_lock
        free = lock.acquire(blocking=False)
        if free:
            lock.release()
        calls.append(not free)
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, spy)


def test_validate_refuses_without_resolving_while_the_model_is_held(
    tiny_session: Any, app_client: Any, monkeypatch: Any,
) -> None:
    """Regression: validate resolved ``neuron/`` atoms (a device upload) before
    taking the model lock, beside a fit holding it."""
    client = app_client(tiny_session)
    calls: list[bool] = []
    _spy(tiny_session.units.runtime, "write_profile", tiny_session, calls, monkeypatch)
    with gpu_held_elsewhere(tiny_session):
        resp = client.post(f"{_BASE}/steering/validate", json={"expression": "0.2 neuron/L2n6"})
    assert resp.status_code == 409
    assert calls == []
    # Free again, the same expression validates and resolves once.
    resp = client.post(f"{_BASE}/steering/validate", json={"expression": "0.2 neuron/L2n6"})
    assert resp.status_code == 200 and resp.json()["valid"] is True
    assert calls == [False]


def test_validate_does_not_block_the_event_loop(tiny_session: Any, app_client: Any) -> None:
    """Regression: validate opened the steering scope on the event loop and
    waited there for the model lock, so every other request stalled until a
    fit finished."""
    register_vectors(tiny_session, "mood")
    client = app_client(tiny_session)
    assert client.post(f"{_BASE}/steering/validate", json={"expression": "0.3 mood"}).json()["valid"]
    with gpu_held_elsewhere(tiny_session):
        start = time.monotonic()
        resp = client.post(f"{_BASE}/steering/validate", json={"expression": "0.3 mood"})
        waited = time.monotonic() - start
        assert resp.status_code == 409
        start = time.monotonic()
        assert client.get("/drowse/v1/sessions").status_code == 200
        other = time.monotonic() - start
    assert waited < 2.0 and other < 2.0


def test_units_find_does_not_resolve_its_selector_while_the_model_is_held(
    tiny_session: Any, app_client: Any, monkeypatch: Any,
) -> None:
    """Regression: units/find resolved its selector (a device upload) before
    the decomposition's exclusive section."""
    client = app_client(tiny_session)
    calls: list[bool] = []
    _spy(tiny_session.units.runtime, "write_profile", tiny_session, calls, monkeypatch)
    with gpu_held_elsewhere(tiny_session):
        resp = client.post(f"{_BASE}/instruments/units/find", json={"selector": "neuron/L2n6"})
    assert resp.status_code == 409
    assert calls == []


def test_a_lazy_profile_load_refuses_while_the_model_is_held(tiny_session: Any) -> None:
    """The registry's miss path runs in the exclusive section, so a resolver
    called off the composer (any caller) cannot share the device."""
    from drowse.errors import ConcurrentExtractionError

    with gpu_held_elsewhere(tiny_session), pytest.raises(ConcurrentExtractionError):
        tiny_session.registry.ensure_profile_registered("neuron/L2n6")
    assert "neuron/L2n6" not in tiny_session.registry.profiles
    assert tiny_session.registry.ensure_profile_registered("neuron/L2n6")
    assert "neuron/L2n6" in tiny_session.registry.profiles


def _save_identity_lens(session: Any, scale: float) -> None:
    from drowse.lens.jlens import JacobianLens
    from drowse.lens.key import LensKey
    from drowse.lens.store import LensProvenance, save_lens
    from drowse.model.fingerprint import loaded_model_fingerprint

    d = session.model.get_input_embeddings().weight.shape[1]
    n = len(session.layers)
    provenance = LensProvenance(
        corpus_spec="test", corpus_sha256="a" * 64, seq_len=8, dim_batch=1,
        model_fingerprint=loaded_model_fingerprint(session.model, session.model_id),
    )
    save_lens(
        JacobianLens(
            {layer: scale * torch.eye(d) for layer in range(n - 1)},
            n_prompts=3, d_model=d, key=LensKey("R"),
        ),
        session.model_id, provenance,
    )


def test_live_lens_toggle_refuses_while_the_model_is_held(
    tiny_session: Any, app_client: Any, monkeypatch: Any,
) -> None:
    """Regression: switching the live readout on uploaded the J stack holding
    only the lens state lock."""
    _save_identity_lens(tiny_session, 1.0)
    client = app_client(tiny_session)
    calls: list[bool] = []
    _spy(tiny_session.lens.runtime, "transport_stack", tiny_session, calls, monkeypatch)
    with gpu_held_elsewhere(tiny_session):
        resp = client.post(f"{_BASE}/instruments/lens/live", json={"enabled": True})
    assert resp.status_code == 409
    assert calls == []
    assert client.post(f"{_BASE}/instruments/lens/live", json={"enabled": True}).status_code == 200
    assert calls and not any(calls)


def test_lens_compatibility_checks_never_reload_while_the_model_is_held(
    tiny_session: Any, app_client: Any, monkeypatch: Any,
) -> None:
    """Regression: with a resident lens whose artifact was replaced on disk,
    the lock-free compatibility check reloaded, adopted and rebuilt the live
    stack on the device."""
    _save_identity_lens(tiny_session, 1.0)
    client = app_client(tiny_session)
    assert client.post(f"{_BASE}/instruments/lens/live", json={"enabled": True}).status_code == 200
    resident = tiny_session.lens.resident_lens
    _save_identity_lens(tiny_session, 2.0)
    calls: list[bool] = []
    _spy(tiny_session.lens.runtime, "transport_stack", tiny_session, calls, monkeypatch)
    with gpu_held_elsewhere(tiny_session):
        assert client.get(f"{_BASE}/instruments/lens/sources").status_code == 200
        assert client.get(f"{_BASE}").status_code == 200
    assert calls == []
    assert tiny_session.lens.resident_lens is resident


def test_transcript_export_never_hashes_on_the_device(
    tiny_session: Any, app_client: Any, monkeypatch: Any,
) -> None:
    """Regression: the lock-free transcript export hashed a freshly attached
    geometry probe's device tensors on a digest-cache miss."""
    from drowse.readout.geometry import GeometryInstrument

    register_vectors(tiny_session, "review.axis")
    tiny_session.add_probe("review.axis")
    recorded = tiny_session.geometry.probe_hash("review.axis")
    assert recorded is not None
    client = app_client(tiny_session)
    calls: list[bool] = []
    _spy(GeometryInstrument, "_record_digest", tiny_session, calls, monkeypatch)
    with gpu_held_elsewhere(tiny_session):
        resp = client.post(f"{_BASE}/tree/transcript", json={})
    assert resp.status_code == 200
    assert calls == []
    assert recorded in resp.text
