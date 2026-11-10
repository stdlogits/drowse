"""Load or capture a model's neutral activations and unit statistics.

The compute half of the per-model neutral cache: render the neutral corpus,
derive the cache identity (rendered token rows, loaded weights, checkpoint
identity, layer set), reuse a validated cache whose identity matches, and run
the capture forward on a miss.  The same forward feeds the unit statistics
(:class:`~drowse.units.stats.UnitStatsAccumulator`: one hook pair per block)
when the model's blocks are described; a model they cannot describe, or a
unit hook that fails, records units as unavailable and never fails the pass.
The same forward also measures each block's **mid mean** (:class:`MidMeans`:
the residual after attention, the MLP's input, pooled at the activations'
own positions), the centring mean of an S-family read at the lens ``mid``
site; a model the description refuses records it as unavailable.  Only
that refusal lasts: a cache recording units or mid means unavailable for a
model the description accepts (a failure on that pass, such as an
out-of-memory error at capture batch size 1) is stale, and the next load
recomputes it.  The cache file format, validation and publication are pure IO in
:mod:`drowse.store.neutral`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Any, Callable

import torch

from drowse.fit import capture as _capture
from drowse.model.fingerprint import loaded_model_fingerprint
from drowse.store.neutral import (
    NEUTRAL_CAPTURE_VERSION,
    NeutralCache,
    NeutralMid,
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
    cache identity; returning the sidecar read in the same transaction avoids a
    second read.
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
    the current tensor names at every layer (extra names are ignored)."""
    if units.tensors is None:
        return True
    names = set(UNIT_TENSOR_NAMES)
    return all(names <= set(row) for row in units.tensors.values())


def _failed_on_a_described_model(cache: NeutralCache, model: Any) -> bool:
    """Whether ``cache`` records its unit statistics or mid means as
    unavailable for a model the block description accepts.

    Only a refusal (``model_spec`` raises: the blocks are outside the
    description) is a lasting property of the model.  Anything else that
    left them unavailable (an out-of-memory error at capture batch size 1, a
    hook that raised) was a failure of that pass, so the record is stale and
    the next load recomputes it rather than refusing every unit or ``mid``
    read for the cache's lifetime.  The check runs block discovery only for
    such a cache (memoized per live model; the caller holds the section a
    cache miss would need).
    """
    if cache.units.tensors is not None and cache.mid.means is not None:
        return False
    from drowse.model.blocks import model_spec

    try:
        model_spec(model)
    except Exception:  # noqa: BLE001 — refused (or not describable now): the record stands
        return False
    return True


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
    the loaded model weights, and the fitted layer set. A stale or unreadable
    cache is recomputed and replaced.

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
            log.warning(
                "Unreadable or stale neutral activations cache for %s, recomputing: %s",
                model_id, e,
            )
        else:
            if not _units_well_formed(cache.units):
                log.info(
                    "Neutral cache unit statistics are not the current set; "
                    "recomputing for %s", model_id,
                )
            elif not neutral_cache_matches(
                cache.sidecar, layers=list(range(len(layers))), **identity,
            ):
                log.info(
                    "Neutral activations stale (model/token/corpus/layer identity "
                    "changed); recomputing for %s",
                    model_id,
                )
            elif _failed_on_a_described_model(cache, base_model):
                log.info(
                    "Neutral cache records unit statistics or mid means as "
                    "unavailable (%s) for a model the block description "
                    "accepts; recomputing for %s",
                    cache.mid.reason or cache.units.reason, model_id,
                )
            else:
                log.debug("Loaded cached neutral activations for %s", model_id)
                return cache

    log.info("Computing neutral activations (one-time per model)...")
    if on_progress is not None:
        on_progress(
            f"Computing neutral activations for {model_id} "
            f"(one-time per model, {len(pairs)} prompts)..."
        )
    accumulator, mids, reason = _observers(base_model, prepared)
    if accumulator is None or mids is None:
        activations = _capture.compute_neutral_activations(
            model, tokenizer, layers, rendered=prepared, on_progress=on_progress,
        )
        units = NeutralUnits(None, 0, reason)
        mid = NeutralMid(None, reason)
    else:
        with accumulator, mids:
            activations = _capture.compute_neutral_activations(
                model, tokenizer, layers, rendered=prepared,
                on_progress=on_progress, observer=_Observers(accumulator, mids),
            )
        units = _finish_units(accumulator)
        mid = _finish_mid(mids)
    sidecar = publish_neutral_cache(
        model_id, activations, units=units, mid=mid, **identity,
    )
    return NeutralCache(activations, units, mid, sidecar)


class MidMeans:
    """Streams each block's mid mean off the neutral capture forward.

    One forward hook per block on the MLP input norm (its input is the
    block's ``mid`` residual, as :class:`~drowse.model.blocks.BlockSpec`
    defines it), gathering each row at its pooled position (the rendered
    row's ``content_end``, the activations' own pooling) and summing in
    fp32 on the device.  It frames forwards like the unit accumulator (a
    :class:`~drowse.fit.capture.ChunkObserver`): a chunk's rows are the next
    ``len(lengths)`` prepared rows, committed or discarded with the forward;
    ``abandon`` records why the means are unavailable.
    """

    def __init__(
        self, spec: Any, rendered: Sequence[tuple[torch.Tensor, int]],
        *, device: torch.device,
    ) -> None:
        self._blocks = list(spec.blocks)
        self._rendered = list(rendered)
        self._device = device
        d = int(spec.d_model)
        self._sums = {
            block.layer: torch.zeros(d, dtype=torch.float32, device=device)
            for block in self._blocks
        }
        self._pending: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._cursor = 0
        self._chunk = 0
        self._rows: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self.count = 0
        self.failure: str | None = None

    def install(self) -> None:
        if self._handles:
            return
        try:
            for block in self._blocks:
                self._handles.append(
                    block.mlp_in_norm.module.register_forward_hook(self._hook(block.layer)),
                )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "MidMeans":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _hook(self, layer: int) -> Any:
        def hook(_module: torch.nn.Module, args: tuple[Any, ...], _output: Any) -> None:
            if self.failure is not None or self._positions is None or self._rows is None:
                return
            x = args[0]
            positions = self._positions.clamp(max=int(x.shape[1]) - 1)
            self._pending[layer] = x[self._rows, positions, :].float().sum(dim=0)
        return hook

    def begin_chunk(self, lengths: Sequence[int]) -> None:
        rows = self._rendered[self._cursor:self._cursor + len(lengths)]
        self._pending = {}
        if [int(ids.shape[1]) for ids, _ in rows] != [int(n) for n in lengths]:
            if self.failure is None:
                self.failure = "the neutral capture framed its rows out of order"
            return
        self._chunk = len(rows)
        self._rows = torch.arange(len(rows), device=self._device)
        self._positions = torch.tensor(
            [int(end) for _, end in rows], dtype=torch.long, device=self._device,
        )

    def commit_chunk(self) -> None:
        self._rows = self._positions = None
        if self.failure is not None:
            return
        if set(self._pending) != set(self._sums):
            self.failure = "a block's mid hook did not fire on a neutral forward"
            return
        for layer, total in self._pending.items():
            self._sums[layer] += total
        self._pending = {}
        self._cursor += self._chunk
        self.count += self._chunk

    def discard_chunk(self) -> None:
        self._rows = self._positions = None
        self._pending = {}

    def abandon(self, reason: str) -> None:
        self.close()
        self.discard_chunk()
        if self.failure is None:
            self.failure = reason

    def finish(self) -> dict[int, torch.Tensor]:
        """``{layer: [D]}`` fp32 CPU; raises when a row was missed."""
        if self.failure is not None:
            raise RuntimeError(self.failure)
        if self.count != len(self._rendered):
            raise RuntimeError(
                f"the mid means saw {self.count} of {len(self._rendered)} neutral rows"
            )
        return {layer: (total / self.count).cpu() for layer, total in self._sums.items()}


class _Observers:
    """The unit accumulator and the mid means, framed as one observer."""

    def __init__(self, *observers: Any) -> None:
        self._observers = observers

    def begin_chunk(self, lengths: Sequence[int]) -> None:
        for observer in self._observers:
            observer.begin_chunk(lengths)

    def commit_chunk(self) -> None:
        for observer in self._observers:
            observer.commit_chunk()

    def discard_chunk(self) -> None:
        for observer in self._observers:
            observer.discard_chunk()

    def abandon(self, reason: str) -> None:
        for observer in self._observers:
            observer.abandon(reason)


def _observers(
    model: Any, rendered: Sequence[tuple[torch.Tensor, int]],
) -> tuple[UnitStatsAccumulator | None, MidMeans | None, str | None]:
    """The unit-statistics accumulator and the mid means for ``model``, or
    why there are none (both read the block description).

    Block discovery runs small forwards (the caller holds the exclusive
    section).  Any failure here is a reason, never an error.
    """
    accumulator, reason = _unit_accumulator(model)
    if accumulator is None:
        return None, None, reason
    from drowse.model.blocks import model_spec

    device = next(model.parameters()).device
    return accumulator, MidMeans(model_spec(model), rendered, device=device), None


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


def _finish_mid(mids: MidMeans) -> NeutralMid:
    try:
        means = mids.finish()
    except Exception as exc:  # noqa: BLE001 — mid means never fail the neutral pass
        reason = f"{type(exc).__name__}: {exc}"
        log.warning("Mid means failed on the neutral pass: %s", reason)
        return NeutralMid(None, reason)
    return NeutralMid(means, None)


__all__ = [
    "MidMeans",
    "load_or_compute_neutral_activations",
    "load_or_compute_neutral_activations_with_metadata",
    "load_or_compute_neutral_cache",
]
