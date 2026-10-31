"""Load or capture a model's neutral activations and unit statistics.

The compute half of the per-model neutral cache: render the neutral corpus,
derive the cache identity (rendered token rows, loaded weights, checkpoint
identity, layer set), reuse a validated cache whose identity matches, and run
the capture forward on a miss.  The same forward feeds the unit statistics
(:class:`~drowse.units.stats.UnitStatsAccumulator`: one hook pair per block)
when the model's blocks are described; a model they cannot describe, or a
unit hook that fails, records units as unavailable and never fails the pass.
The cache file format, validation and publication are pure IO in
:mod:`drowse.store.neutral`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable

import torch

from drowse.fit import capture as _capture
from drowse.model.fingerprint import loaded_model_fingerprint
from drowse.store.neutral import (
    NEUTRAL_CAPTURE_VERSION,
    NeutralCache,
    NeutralUnits,
    load_neutral_cache,
    neutral_cache_matches,
    neutral_fit_lock,
    publish_neutral_cache,
)
from drowse.units.stats import UNIT_TENSOR_NAMES, UnitStatsAccumulator

log = logging.getLogger(__name__)


def load_or_compute_neutral_activations(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> dict[int, torch.Tensor]:
    """The activations of :func:`load_or_compute_neutral_cache`."""
    return load_or_compute_neutral_cache(
        model, tokenizer, layers, model_id=model_id, force=force,
        on_progress=on_progress,
    ).activations


def load_or_compute_neutral_activations_with_metadata(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """The activations plus the sidecar proven in the same transaction.

    Alignment fitting needs both the materialized neutral rows and their stable
    cache identity; returning the validated sidecar avoids a second full-file
    digest pass.
    """
    cache = load_or_compute_neutral_cache(
        model, tokenizer, layers, model_id=model_id, force=force,
        on_progress=on_progress,
    )
    return cache.activations, cache.sidecar


def load_or_compute_neutral_cache(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> NeutralCache:
    """Single-flight neutral-cache load/capture: activations, unit statistics
    and the validated sidecar.

    ``on_progress`` narrates the cache-miss capture loop (per forward chunk);
    a cache hit reports nothing because it runs no model.
    """
    with neutral_fit_lock(model_id):
        return _load_or_compute_neutral_cache_locked(
            model, tokenizer, layers, model_id=model_id, force=force,
            on_progress=on_progress,
        )


def _load_or_compute_neutral_activations_with_metadata_locked(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """:func:`load_or_compute_neutral_activations_with_metadata` with
    :func:`~drowse.store.neutral.neutral_fit_lock` already held (transfer
    alignment holds it across a model load)."""
    cache = _load_or_compute_neutral_cache_locked(
        model, tokenizer, layers, model_id=model_id, force=force,
        on_progress=on_progress,
    )
    return cache.activations, cache.sidecar


def _units_well_formed(units: NeutralUnits) -> bool:
    """A cached units record the units package can read: unavailable, or
    exactly the current tensor names at every layer."""
    if units.tensors is None:
        return True
    names = set(UNIT_TENSOR_NAMES)
    return all(set(row) == names for row in units.tensors.values())


def _load_or_compute_neutral_cache_locked(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> NeutralCache:
    """Disk-cached per-statement activations and unit statistics for one model.

    Cache identity binds the exact rendered token rows and pooling positions,
    the loaded model weights, and the fitted layer set. The tensor payload is
    digest-verified before reuse. A stale cache is recomputed and replaced.

    Activations are ``{layer_idx: [N, D] fp32 CPU tensor}``, stored fp32 so the
    whitener covariance is built and inverted at full precision and the
    compute path (cache miss) and cache-hit path return bit-identical tensors
    (no precision seam — bf16's ~0.4% input error landed on exactly the
    inverted quantity).  fp16 was abandoned because its 65504 ceiling
    overflows gemma-3's extreme late-layer channels to ±inf (poisoning ``λ`` /
    ``K``); fp32 has the range and, unlike the former bf16 store, no
    input-precision loss, at the cost of ~2× disk on this one cache.
    """
    pairs = _capture._neutral_pairs()
    prompts = [prompt for prompt, _ in pairs]
    responses = [response for _, response in pairs]
    prepared = _capture._prepare_capture_batch(
        tokenizer, prompts, responses,
    )
    rendered_payload = [
        {"input_ids": ids[0].tolist(), "content_end": int(content_end)}
        for ids, content_end in prepared
    ]
    capture_sha = hashlib.sha256(json.dumps(
        {"capture_version": NEUTRAL_CAPTURE_VERSION, "rows": rendered_payload},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    model_fingerprint = loaded_model_fingerprint(model, model_id)
    base_model = getattr(model, "_orig_mod", model)
    model_source_fingerprint = getattr(
        base_model, "_drowse_source_fingerprint", None,
    )
    identity = {
        "capture_sha256": capture_sha,
        "model_fingerprint": model_fingerprint,
        "model_source_fingerprint": model_source_fingerprint,
    }

    if not force:
        try:
            cache = load_neutral_cache(model_id)
        except FileNotFoundError:
            pass
        except Exception as e:
            log.warning("Corrupt neutral activations cache for %s, recomputing: %s", model_id, e)
        else:
            if not _units_well_formed(cache.units):
                log.info(
                    "Neutral cache unit statistics are not the current set; "
                    "recomputing for %s", model_id,
                )
            elif neutral_cache_matches(
                cache.sidecar, layers=list(range(len(layers))), **identity,
            ):
                log.debug("Loaded cached neutral activations for %s", model_id)
                return cache
            else:
                log.info(
                    "Neutral activations stale (model/token/corpus/layer identity "
                    "changed); recomputing for %s",
                    model_id,
                )

    log.info("Computing neutral activations (one-time per model)...")
    if on_progress is not None:
        on_progress(
            f"Computing neutral activations for {model_id} "
            f"(one-time per model, {len(pairs)} prompts)..."
        )
    accumulator, reason = _unit_accumulator(base_model)
    if accumulator is None:
        activations = _capture.compute_neutral_activations(
            model, tokenizer, layers, rendered=prepared, on_progress=on_progress,
        )
        units = NeutralUnits(None, 0, reason)
    else:
        with accumulator:
            activations = _capture.compute_neutral_activations(
                model, tokenizer, layers, rendered=prepared,
                on_progress=on_progress, observer=accumulator,
            )
        units = _finish_units(accumulator)
    sidecar = publish_neutral_cache(model_id, activations, units=units, **identity)
    return NeutralCache(activations, units, sidecar)


def _unit_accumulator(
    model: Any,
) -> tuple[UnitStatsAccumulator | None, str | None]:
    """The unit-statistics accumulator for ``model``, or why there is none.

    Block discovery runs small forwards (the caller holds the exclusive
    section).  Any failure here is a reason, never an error.
    """
    from drowse.model.blocks import model_spec

    try:
        spec = model_spec(model)
        device = next(model.parameters()).device
        return UnitStatsAccumulator(spec, device=device), None
    except Exception as exc:  # noqa: BLE001 — units never fail the neutral pass
        reason = f"{type(exc).__name__}: {exc}"
        log.info("Unit statistics unavailable for this model: %s", reason)
        return None, reason


def _finish_units(accumulator: UnitStatsAccumulator) -> NeutralUnits:
    try:
        stats = accumulator.finish()
    except Exception as exc:  # noqa: BLE001 — units never fail the neutral pass
        reason = f"{type(exc).__name__}: {exc}"
        log.warning("Unit statistics failed on the neutral pass: %s", reason)
        return NeutralUnits(None, 0, reason)
    return NeutralUnits(stats.tensors(), stats.n_documents, None)


__all__ = [
    "load_or_compute_neutral_activations",
    "load_or_compute_neutral_activations_with_metadata",
    "load_or_compute_neutral_cache",
]
