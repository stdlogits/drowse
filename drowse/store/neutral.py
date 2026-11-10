"""Per-model neutral-activation cache.

The model's activations on the bundled neutral statements, one ``[N, D]`` fp32
row block per layer, stored as immutable per-layer shard generations behind an
atomic JSON pointer (:mod:`drowse.base.shards`).  It is the single per-model
neutral artifact: the probe-centering means are its per-layer ``X.mean(0)``,
the :class:`~drowse.geometry.whitener.LayerWhitener` covariance is built from
the stack, and cross-model transfer aligns two models' caches.

Each layer's shard also carries that block's **unit statistics** when the
model's blocks are described (:mod:`drowse.units.stats`: a neuron's mean
activation and per-pole scale, a head's mean ``z_h`` and write-norm scale,
measured on the same forward), as ``<name>_<layer>`` tensors beside
``layer_<layer>``.  The sidecar's ``units`` block says which: ``{"available":
true, "n_documents": N, "schema": {layer: {name: shape}}}``, or ``{"available":
false, "reason": …}`` for a model whose blocks the description refuses (its
cache is otherwise complete).  This module stores and validates the tensors
generically; their meaning is the units package's.

Each shard also carries the layer's **mid mean** (``mid_mean_<layer>``,
``[D]`` fp32): the residual after the block's attention (the MLP's input, the
lens ``mid`` site) averaged over the same pooled positions as the activations,
on the same forward.  It is the centring mean of an S-family read of a ``mid``
residual, as ``X.mean(0)`` of the activations is at ``out``.  The sidecar's
``mid`` block is ``{"available": true}``, or ``{"available": false, "reason":
…}`` for a model whose blocks the description refuses (the ``mid`` site needs
it).  A cache without the block is stale and recomputes.

Cache identity binds the exact rendered token rows, the loaded model weights
(``model_fingerprint``), the off-model checkpoint identity
(``model_source_fingerprint``), and the layer set.  The sidecar reader needs
those keys and ignores any other (6.1's per-shard ``tensor_sha256`` included).
Readers never mutate the store; superseded generations are collected by the
next writer.

This module is pure IO plus the offline whitener loader
(:func:`load_neutral_whitener`); rendering the neutral corpus and capturing it
on a miss is :mod:`drowse.fit.neutral`.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, NamedTuple

import torch
from safetensors.torch import (
    load as load_safetensors,
    save as save_safetensors,
)

from drowse.errors import WhitenerError
from drowse.base.atomic import fsync_directory, write_bytes_atomic, write_json_atomic
from drowse.base.paths import model_dir
from drowse.base.shards import (
    cleanup_generations,
    fit_lock,
    generation_path,
    json_pointer_matches as _json_pointer_matches,
    shard_paths,
)
from drowse.geometry.whitener import DEFAULT_RIDGE_SCALE, LayerWhitener

log = logging.getLogger(__name__)

_NEUTRAL_LABEL = "neutral activation cache"
_NEUTRAL_ACTS_NAME = "neutral_activations"
_NEUTRAL_CACHE_FORMAT_VERSION = 5
#: Bumped when the neutral forward measures something new (2: the mid means),
#: so a cache captured without it is stale and recomputes.
NEUTRAL_CAPTURE_VERSION = 2
#: The per-layer mid-mean tensor name in a shard (``mid_mean_<layer>``).
MID_MEAN = "mid_mean"

#: The sidecar keys that identify one cache generation; alignment pointers
#: embed them (:func:`neutral_cache_identity`).
_NEUTRAL_IDENTITY_KEYS = (
    "format_version", "capture_version", "model_fingerprint",
    "model_source_fingerprint", "capture_sha256", "layers", "n_prompts",
    "tensor_files",
)


class NeutralUnits(NamedTuple):
    """A cache's unit statistics as stored: ``{layer: {name: tensor}}`` fp32
    CPU, with the document count, or ``tensors=None`` and the ``reason`` units
    are unavailable."""

    tensors: dict[int, dict[str, torch.Tensor]] | None
    n_documents: int
    reason: str | None


class NeutralMid(NamedTuple):
    """A cache's mid means as stored: ``{layer: [D]}`` fp32 CPU, or
    ``means=None`` and the ``reason`` they are unavailable."""

    means: dict[int, torch.Tensor] | None
    reason: str | None


class NeutralCache(NamedTuple):
    """One neutral cache: per-layer activations, unit statistics, mid means
    and the sidecar, read in the same transaction."""

    activations: dict[int, torch.Tensor]
    units: NeutralUnits
    mid: NeutralMid
    sidecar: dict[str, Any]


def _neutral_acts_paths(model_id: str) -> tuple[Path, Path]:
    """Return the stable lock anchor and atomic pointer sidecar."""
    md = model_dir(model_id)
    return (
        md / f"{_NEUTRAL_ACTS_NAME}.safetensors",
        md / f"{_NEUTRAL_ACTS_NAME}.json",
    )


def _neutral_shard_paths(
    anchor: Path, sidecar: Mapping[str, Any], layers: list[int],
) -> dict[int, Path]:
    return shard_paths(anchor, sidecar, layers, label=_NEUTRAL_LABEL)


def _cleanup_neutral_generations(
    anchor: Path, sidecar: Mapping[str, Any],
) -> None:
    cleanup_generations(anchor, sidecar, label=_NEUTRAL_LABEL)


def _units_block(units: Any) -> dict[str, list[str]]:
    """The sidecar's ``units`` block as ``{layer: [tensor name, ...]}`` (empty
    when units are unavailable)."""
    if not isinstance(units, dict) or not isinstance(units.get("available"), bool):
        raise ValueError("neutral activation cache has no units block")
    if not units["available"]:
        return {}
    schema = units.get("schema")
    n_documents = units.get("n_documents")
    if (
        not isinstance(schema, dict)
        or isinstance(n_documents, bool)
        or not isinstance(n_documents, int)
        or n_documents <= 0
    ):
        raise ValueError("neutral activation cache units block is malformed")
    return {
        str(layer): [str(name) for name in row]
        for layer, row in schema.items() if isinstance(row, dict)
    }


def _mid_block(mid: Any) -> bool:
    """Whether the sidecar's ``mid`` block says mid means are stored."""
    if not isinstance(mid, dict) or not isinstance(mid.get("available"), bool):
        raise ValueError("neutral activation cache has no mid block")
    return mid["available"]


@contextmanager
def neutral_fit_lock(model_id: str) -> Iterator[None]:
    """Single-flight lock for one model's expensive neutral forward pass."""
    ts_path, _ = _neutral_acts_paths(model_id)
    with fit_lock(ts_path):
        yield


def validate_neutral_cache_metadata(model_id: str) -> dict[str, Any]:
    """Read a neutral cache's sidecar without touching its shards.

    The cheap identity preflight for exact transfer repeats and
    source-to-loaded fingerprint bridging; the shards are read (and their
    tensors checked) by :func:`load_validated_neutral_cache`.
    """
    from drowse.base.atomic import artifact_lock

    ts_path, _ = _neutral_acts_paths(model_id)
    with artifact_lock(ts_path):
        return _read_neutral_sidecar_locked(model_id)


def _read_neutral_sidecar_locked(model_id: str) -> dict[str, Any]:
    """The current-format sidecar, with the neutral pair lock already held."""
    ts_path, sc_path = _neutral_acts_paths(model_id)
    if not sc_path.exists():
        raise FileNotFoundError(f"neutral activation cache missing for {model_id}")
    with open(sc_path) as handle:
        sidecar = json.load(handle)
    if not isinstance(sidecar, dict):
        raise ValueError("neutral activation cache sidecar must be a JSON object")
    for key, current in (
        ("format_version", _NEUTRAL_CACHE_FORMAT_VERSION),
        ("capture_version", NEUTRAL_CAPTURE_VERSION),
    ):
        value = sidecar.get(key)
        if isinstance(value, bool) or value != current:
            raise ValueError(
                f"neutral activation cache {key} is {value!r}, not {current}"
            )
    layers = sidecar.get("layers")
    if (
        not isinstance(layers, list)
        or not layers
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in layers
        )
    ):
        raise ValueError("neutral activation cache sidecar has no layer list")
    _units_block(sidecar.get("units"))
    _mid_block(sidecar.get("mid"))
    _neutral_shard_paths(ts_path, sidecar, layers)
    return sidecar


def load_validated_neutral_cache(
    model_id: str,
    *,
    requested_layers: Iterable[int] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Materialize and fully validate a neutral cache's activations (all
    layers, or ``requested_layers``) without loading the model."""
    cache = _load_neutral_cache(model_id, requested_layers=requested_layers)
    return cache.activations, cache.sidecar


def load_neutral_cache(model_id: str) -> NeutralCache:
    """Materialize and fully validate every layer of a neutral cache: the
    activations, the unit statistics (when the cache has them) and the
    sidecar, from one read of each shard."""
    return _load_neutral_cache(model_id, requested_layers=None)


def _load_neutral_cache(
    model_id: str, *, requested_layers: Iterable[int] | None,
) -> NeutralCache:
    from drowse.base.atomic import artifact_lock

    ts_path, _ = _neutral_acts_paths(model_id)
    with artifact_lock(ts_path):
        sidecar = _read_neutral_sidecar_locked(model_id)
        all_layers = sorted(int(layer) for layer in sidecar["layers"])
        selected = (
            all_layers if requested_layers is None else sorted(
                set(all_layers) & {int(layer) for layer in requested_layers}
            )
        )
        unit_names = _units_block(sidecar["units"])
        has_mid = _mid_block(sidecar["mid"])
        paths = _neutral_shard_paths(ts_path, sidecar, all_layers)
        shards = {
            layer: load_safetensors(paths[layer].read_bytes()) for layer in selected
        }

    def finite(tensor: torch.Tensor | None, *, ndim: int | None = None) -> torch.Tensor:
        # fp32 because the whitener inverts a covariance built from these (a
        # bf16 source reopens the precision seam); non-finite rows would
        # poison it silently.
        if (
            tensor is None
            or tensor.dtype != torch.float32
            or ndim is not None and tensor.ndim != ndim
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("neutral activation cache tensor failed validation")
        return tensor

    out: dict[int, torch.Tensor] = {}
    for layer, shard in shards.items():
        out[layer] = finite(shard.get(f"layer_{layer}"), ndim=2)
    units_meta = sidecar["units"]
    if not unit_names:
        units = NeutralUnits(None, 0, str(units_meta.get("reason") or "unavailable"))
    else:
        unit_tensors: dict[int, dict[str, torch.Tensor]] = {}
        for layer, shard in shards.items():
            names = unit_names.get(str(layer))
            if names is None:
                raise ValueError(f"neutral unit statistics do not cover layer {layer}")
            unit_tensors[layer] = {
                name: finite(shard.get(f"{name}_{layer}")) for name in names
            }
        units = NeutralUnits(unit_tensors, int(units_meta["n_documents"]), None)
    if has_mid:
        mid_means: dict[int, torch.Tensor] = {}
        for layer, shard in shards.items():
            tensor = finite(shard.get(f"{MID_MEAN}_{layer}"), ndim=1)
            if int(tensor.shape[0]) != int(out[layer].shape[1]):
                raise ValueError(f"neutral mid mean at layer {layer} failed validation")
            mid_means[layer] = tensor
        mid = NeutralMid(mid_means, None)
    else:
        mid = NeutralMid(None, str(sidecar["mid"].get("reason") or "unavailable"))
    return NeutralCache(out, units, mid, sidecar)


def neutral_cache_identity(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Stable identity subset embedded into dependent alignment artifacts.

    ``tensor_files`` names the cache's immutable shard generation (a fresh
    random name per publish), so a re-captured cache never shares an identity.
    """
    if (
        any(key not in sidecar for key in _NEUTRAL_IDENTITY_KEYS)
        or sidecar["format_version"] != _NEUTRAL_CACHE_FORMAT_VERSION
    ):
        raise ValueError("neutral cache identity requires current metadata")
    return {key: sidecar[key] for key in _NEUTRAL_IDENTITY_KEYS}


def neutral_cache_matches(
    sidecar: Mapping[str, Any],
    *,
    capture_sha256: str,
    model_fingerprint: str,
    model_source_fingerprint: str | None,
    layers: list[int],
) -> bool:
    """True iff a validated sidecar was captured under exactly this identity."""
    return (
        sidecar.get("format_version") == _NEUTRAL_CACHE_FORMAT_VERSION
        and sidecar.get("capture_version") == NEUTRAL_CAPTURE_VERSION
        and sidecar.get("capture_sha256") == capture_sha256
        and sidecar.get("model_fingerprint") == model_fingerprint
        and sidecar.get("model_source_fingerprint") == model_source_fingerprint
        and sidecar.get("layers") == layers
    )


def publish_neutral_cache(
    model_id: str,
    activations: Mapping[int, torch.Tensor],
    *,
    capture_sha256: str,
    model_fingerprint: str,
    model_source_fingerprint: str | None,
    units: NeutralUnits,
    mid: NeutralMid,
) -> dict[str, Any]:
    """Write a fresh neutral-cache generation and publish its pointer.

    Stores fp32 — the whitener covariance is built and inverted from these,
    so the compute path (this fresh fp32 store) and the cache-hit path stay
    bit-identical (no precision seam) and the inversion is full-precision.
    fp32 carries the exponent range fp16 lacked (gemma-3's late layers exceed
    the fp16 max of 65504 and overflowed to ±inf) without bf16's input loss,
    at ~2× disk on this one cache.  ``units`` rides in the same shards (its
    tensors, one set per layer with the same names everywhere) or records the
    reason it is unavailable, and so does ``mid`` (one ``[D]`` mean per
    layer, or the reason).  Returns the published sidecar.  The caller holds
    :func:`neutral_fit_lock`.
    """
    from drowse.base.atomic import artifact_lock

    ts_path, sc_path = _neutral_acts_paths(model_id)
    sc_path.parent.mkdir(parents=True, exist_ok=True)
    fp32 = {
        int(idx): t.contiguous().to(torch.float32).cpu()
        for idx, t in activations.items()
    }
    unit_fp32: dict[int, dict[str, torch.Tensor]] = {}
    units_block: dict[str, Any]
    if units.tensors is None:
        units_block = {"available": False, "reason": str(units.reason or "unavailable")}
    else:
        if set(units.tensors) != set(fp32):
            raise ValueError("neutral unit statistics must cover every cached layer")
        unit_fp32 = {
            int(idx): {
                str(name): t.contiguous().to(torch.float32).cpu()
                for name, t in row.items()
            }
            for idx, row in units.tensors.items()
        }
        units_block = {
            "available": True,
            "n_documents": int(units.n_documents),
            "schema": {
                str(idx): {name: list(t.shape) for name, t in sorted(row.items())}
                for idx, row in sorted(unit_fp32.items())
            },
        }
    mid_fp32: dict[int, torch.Tensor] = {}
    mid_block: dict[str, Any]
    if mid.means is None:
        mid_block = {"available": False, "reason": str(mid.reason or "unavailable")}
    else:
        if set(mid.means) != set(fp32):
            raise ValueError("neutral mid means must cover every cached layer")
        mid_fp32 = {
            int(idx): t.reshape(-1).contiguous().to(torch.float32).cpu()
            for idx, t in mid.means.items()
        }
        if any(
            int(t.shape[0]) != int(fp32[idx].shape[1]) for idx, t in mid_fp32.items()
        ):
            raise ValueError("a neutral mid mean must match its layer's width")
        mid_block = {"available": True}
    with artifact_lock(ts_path):
        tensor_files: dict[str, str] = {}
        created: list[Path] = []
        try:
            for idx, tensor in sorted(fp32.items()):
                path = generation_path(ts_path, idx)
                payload = save_safetensors({
                    f"layer_{idx}": tensor,
                    **{
                        f"{name}_{idx}": unit
                        for name, unit in unit_fp32.get(idx, {}).items()
                    },
                    **(
                        {f"{MID_MEAN}_{idx}": mid_fp32[idx]} if idx in mid_fp32 else {}
                    ),
                })
                write_bytes_atomic(path, payload)
                created.append(path)
                tensor_files[str(idx)] = path.name
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        sidecar: dict[str, Any] = {
            "method": "neutral_activations",
            "format_version": _NEUTRAL_CACHE_FORMAT_VERSION,
            "capture_version": NEUTRAL_CAPTURE_VERSION,
            "capture_sha256": capture_sha256,
            "model_fingerprint": model_fingerprint,
            "model_source_fingerprint": model_source_fingerprint,
            "tensor_files": tensor_files,
            "layers": sorted(activations),
            "tensor_schema": {
                str(idx): {"shape": list(t.shape), "dtype": str(t.dtype)}
                for idx, t in sorted(activations.items())
            },
            "n_prompts": (
                next(iter(activations.values())).shape[0] if activations else 0
            ),
            "n_layers": len(activations),
            "units": units_block,
            "mid": mid_block,
        }
        try:
            # Persist every new directory entry before publishing the pointer.
            # A crash can therefore expose either the complete old generation
            # or the complete new one, never a pointer to absent shards.
            fsync_directory(sc_path.parent)
            write_json_atomic(sc_path, sidecar)
        except BaseException:
            if not _json_pointer_matches(sc_path, sidecar):
                for path in created:
                    path.unlink(missing_ok=True)
            raise
        # The pointer rename is now authoritative. Preserve its generations if
        # this durability barrier itself fails; deleting them would break the
        # successfully replaced pointer immediately.
        fsync_directory(sc_path.parent)
        _cleanup_neutral_generations(ts_path, sidecar)
        fsync_directory(sc_path.parent)
    return sidecar


def load_neutral_whitener(
    model_id: str, *, ridge_scale: float = DEFAULT_RIDGE_SCALE,
) -> LayerWhitener:
    """Load a model's whitener from its neutral-activation cache (no model load).

    The single offline whitener loader — used by ``manifold compare`` and the
    offline bake.  Derives the per-layer centering mean from the activations
    themselves (``X.mean(0)``): the neutral mean *is* the probe-centering
    baseline (same corpus, same pooling), so there is no separate
    ``layer_means`` cache.  Raises :class:`WhitenerError` with a
    populating-command hint when the cache is missing, and when it is stale,
    corrupt, or legacy — offline tools never silently load a model.

    A live session builds its whitener from the activations it already holds
    (:meth:`LayerWhitener.from_neutral_activations`).
    """
    _, pointer_path = _neutral_acts_paths(model_id)
    if not pointer_path.is_file():
        raise WhitenerError(
            f"whitener cache missing for {model_id} "
            f"(expected {pointer_path.name} under {pointer_path.parent}); populate "
            f"via any flow that loads the model + neutral activations (e.g. run "
            f"any session-level extract on this model, or prepare a transfer "
            f"that targets {model_id})"
        )
    try:
        acts_raw, _sidecar = load_validated_neutral_cache(model_id)
    except Exception as exc:
        raise WhitenerError(
            f"whitener cache for {model_id} is stale, corrupt, or legacy; "
            "load the model once to regenerate neutral activations"
        ) from exc
    # ``load_validated_neutral_cache`` already rejects a non-fp32 or
    # non-finite store (re-raised above as ``WhitenerError``).
    means = {layer: tensor.mean(dim=0) for layer, tensor in acts_raw.items()}
    return LayerWhitener.from_neutral_activations(
        acts_raw, means, ridge_scale=ridge_scale,
    )
