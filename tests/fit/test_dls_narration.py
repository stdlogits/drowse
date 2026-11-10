"""DLS narrates the layers it drops, at fit time and on serve's startup roster.

Regression: on Qwen3-8B DLS kept 2 of 36 layers of the former bundled
``default/honest.deceptive`` and nothing said so; the probe looked healthy
until a scoped term was refused.  Serve's startup then printed a degenerate
axis twice, the DLS line and an older ``UserWarning`` from
``compute_dls_axes``; the DLS line is the one message.
"""
from __future__ import annotations

import types
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.events import EventBus
from drowse.fit import capture
from drowse.fit.pipeline import (
    ManifoldExtractionPipeline,
    dls_drop_note,
    dls_fit_note,
    dls_load_note,
)
from drowse.geometry.subspace import LayerSubspace
from drowse.store.manifold_authoring import create_discover_manifold_folder
from tests.support.capture import DIM, N_LAYERS, StubHandle, install_capture
from tests.support.sessions import baseline_corpus


def _sub(*coords: float) -> LayerSubspace:
    return LayerSubspace.affine(
        mean=torch.zeros(2), basis=torch.tensor([[1.0, 0.0]]),
        read=torch.tensor([[1.0, 0.0]]),
        node_coords=torch.tensor([[c] for c in coords]),
    )


def test_the_drop_note_names_kept_and_dropped_layers() -> None:
    note = dls_drop_note("honest.deceptive", [1, 2], list(range(36)))
    assert note == (
        "DLS kept 2 of 36 layers of 'honest.deceptive' (1-2); at 0,3-35 its "
        "nodes sit on one side of the neutral mean on every axis"
    )
    assert dls_drop_note("x", [0, 1, 2], [0, 1, 2]) is None


def test_the_fit_note_names_the_all_fail_fallback() -> None:
    same_side = {0: _sub(1.2, 0.1), 1: _sub(1.1, 0.05)}
    note = dls_fit_note("honest.deceptive", [0, 1], same_side)
    assert note is not None and note.startswith(
        "DLS: at no layer of 'honest.deceptive' do its nodes sit on both sides"
    )
    assert "all 2 layers kept" in note
    straddling = {1: _sub(0.4, -0.6)}
    assert dls_fit_note("x", [0, 1], straddling) == dls_drop_note("x", [1], [0, 1])
    assert dls_fit_note("x", [1], straddling) is None


def test_the_load_note_reads_only_sidecar_metadata() -> None:
    manifold: Any = types.SimpleNamespace(
        layers={1: object(), 2: object()},
        metadata={"node_spread_per_layer": {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}},
    )
    assert dls_load_note("default/x", manifold) == dls_drop_note("default/x", [1, 2], [0, 1, 2, 3])
    manifold.metadata = {}
    assert dls_load_note("default/x", manifold) is None


def _keep_first_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    real = capture.compute_dls_axes

    def keep_first(centroids: Any, bases: Any, means: Any) -> dict[int, set[int]]:
        out = real(centroids, bases, means)
        first = min(out)
        return {first: out[first]}

    monkeypatch.setattr(capture, "compute_dls_axes", keep_first)


def _author(namespace: str, name: str, tags: list[str]) -> None:
    create_discover_manifold_folder(
        namespace, name, "test axis", fit_mode="pca",
        node_corpora={
            "calm": baseline_corpus("I feel calm and settled today."),
            "angry": baseline_corpus("I am furious about all of this!"),
        },
        hyperparams={"max_dim": 1}, tags=tags,
    )


@pytest.mark.usefixtures("isolated_home")
def test_a_fit_narrates_dropped_layers(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.base.paths import manifold_dir

    _keep_first_layer(monkeypatch)
    _author("local", "mood", [])
    lines: list[str] = []
    manifold = tiny_session.fit(manifold_dir("local", "mood"), on_progress=lines.append)
    notes = [line for line in lines if line.startswith("DLS")]
    n_layers = len(tiny_session.layers)
    assert len(manifold.layers) == 1 and n_layers > 1
    assert len(notes) == 1
    assert notes == [dls_drop_note("mood", sorted(manifold.layers), list(range(n_layers)))]


@pytest.mark.usefixtures("isolated_home")
def test_serve_startup_narrates_a_probe_once_whether_fitted_or_cached(
    make_tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _keep_first_layer(monkeypatch)
    _author("default", "w4axis", ["w4test"])

    first: list[str] = []
    make_tiny_session(probes=["w4test"], on_progress=first.append)
    fit_notes = [line for line in first if line.startswith("DLS") and "w4axis" in line]
    assert len(fit_notes) == 1, first

    again: list[str] = []
    session = make_tiny_session(probes=["w4test"], on_progress=again.append)
    load_notes = [line for line in again if line.startswith("DLS") and "w4axis" in line]
    n_layers = len(session.layers)
    kept = sorted(session.manifolds["default/w4axis"].layers)
    assert load_notes == [dls_drop_note("default/w4axis", kept, list(range(n_layers)))]


def test_a_degenerate_axis_is_reported_once_by_the_dls_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neutral far past both nodes: every layer fails the straddle, the fit
    keeps them all and says so in one DLS line, and no ``UserWarning``
    repeats it (serve printed both)."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "home"))
    install_capture(monkeypatch)
    folder = create_discover_manifold_folder(
        "local", "lopsided", "both poles on one side of neutral", fit_mode="pca",
        node_corpora={
            "pos": [f"pos statement {i}" for i in range(4)],
            "neg": [f"neg statement {i}" for i in range(4)],
        },
        hyperparams={"max_dim": 1},
    )
    far = {layer: torch.full((DIM,), 1000.0) for layer in range(N_LAYERS)}
    pipe = ManifoldExtractionPipeline(StubHandle(layer_means=far), EventBus())
    lines: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manifold = pipe.fit(folder, on_progress=lines.append)
    assert sorted(manifold.layers) == list(range(N_LAYERS))
    notes = [line for line in lines if line.startswith("DLS")]
    assert len(notes) == 1 and notes[0].startswith(
        "DLS: at no layer of 'lopsided' do its nodes sit on both sides"
    ), lines
    assert not [w for w in caught if "DLS" in str(w.message)]
