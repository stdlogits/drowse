"""The lens estimator: :func:`fit_lens`, transport over a corpus.

One estimator fits every lens kind (J, R, K, S) at either or both sites
(``out``, ``mid``).  Each document is linearized once
(:func:`drowse.lens.transport.linearize`); then ``dim_batch`` basis seeds at
a time sweep through every block's map: down from the last block's output
for the output kinds (J/R, cotangents, :func:`~drowse.lens.transport.vjp`),
up from the embedding for the input kinds (K/S, tangents,
:func:`~drowse.lens.transport.jvp`).  One sweep covers every layer and both
sites, so a document costs one forward plus ``ceil(d_model/dim_batch)``
sweeps, whatever the layer request; restricting layers only trims the sweep
(an output fit stops at its lowest layer, an input fit at its highest).

**Weighting.** Seeds go in at every valid position ``[skip_first, T − 1)``
and each site's read is the mean over the same positions, so each document
contributes one per-document estimate and documents weigh equally at
ragged lengths.  Documents of ``skip_first + 1`` tokens or fewer are
dropped up front; ``n_prompts`` counts consumed documents.

**Split-half.** Documents alternate between two halves by their index in
the corpus (even, odd).  The estimator keeps the total sums and the odd
half's, so every fit reports, per site and layer, ``‖mean_even −
mean_odd‖ / ‖mean‖``: how differently two halves of the corpus read, which
also says whether more documents would help.  It costs one more fp32 lens
in RAM; ``split_half=False`` drops it.

**Memory and devices.** Accumulators are fp32 on the CPU.  A sweep stages
its reads on the device and moves them to the host in one transfer per pass
(on MPS that transfer also drains Metal's command queue, whose exhaustion
silently zeroes work); an all-zero row from an accelerator is that
asynchronous failure and retries the pass at a smaller width, as an OOM
does.  ``dim_batch`` halves on OOM and stays down for the rest of the fit.
Cancellation is checked between passes and documents: a stop between
documents checkpoints the completed prefix, one inside a document abandons
that document's partial sums (the caller reloads from the last checkpoint).
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from drowse.errors import RequestError, is_out_of_memory_error
from drowse.lens.jlens import (
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_DIM_BATCH,
    DEFAULT_SEQ_LEN,
    SKIP_FIRST_POSITIONS,
    JacobianLens,
    JacobianLensCancelled,
    JacobianLensError,
)
from drowse.lens.key import FITTABLE_KINDS, LENS_SITES, LensKey, site_layers
from drowse.lens.transport import (
    blocks_needed,
    forward_sweep,
    jvp,
    linearize,
    reverse_sweep,
    rules_for,
    seed_block,
    step_for,
    vjp,
)
from drowse.model.blocks import ModelSpec, verify_model_spec

log = logging.getLogger(__name__)

#: Both sites, the default of every fit.
DEFAULT_SITES: tuple[str, ...] = LENS_SITES

SiteLayers = Mapping[str, Sequence[int]]


def fit_site_layers(
    kind: str, n_layers: int, *, sites: Sequence[str] = DEFAULT_SITES,
    layers: SiteLayers | None = None,
) -> dict[str, list[int]]:
    """The validated ``{site: sorted layers}`` a fit covers.

    ``layers`` (per site) wins over ``sites``; without it every site in
    ``sites`` takes its whole pool (:func:`~drowse.lens.key.site_layers`).
    A layer outside its site's pool raises :class:`RequestError`.
    """
    if kind not in FITTABLE_KINDS:
        raise JacobianLensError(
            f"no estimator fits {kind}-lenses (fittable: {', '.join(FITTABLE_KINDS)})"
        )
    if layers is None:
        chosen = [site for site in LENS_SITES if site in set(sites)]
        if len(chosen) != len(tuple(sites)) or not chosen:
            raise RequestError(
                f"sites must be a non-empty subset of {', '.join(LENS_SITES)}, "
                f"got {list(sites)}"
            )
        return {site: site_layers(kind, site, n_layers) for site in chosen}
    out: dict[str, list[int]] = {}
    for site in LENS_SITES:
        if site not in layers:
            continue
        pool = set(site_layers(kind, site, n_layers))
        wanted = sorted({int(layer) for layer in layers[site]})
        bad = [layer for layer in wanted if layer not in pool]
        if bad:
            where = (
                "the last block is the output kinds' target, not an out-site "
                "source; "
                if _output_out(kind, site) and n_layers - 1 in bad else ""
            )
            raise RequestError(
                f"{kind}:{site} layers {bad} are outside [0, {max(pool, default=-1) + 1}) "
                f"({where}a {n_layers}-block model)"
            )
        if not wanted:
            raise RequestError(f"{kind}:{site} names no layers")
        out[site] = wanted
    unknown = set(layers) - set(LENS_SITES)
    if unknown or not out:
        raise RequestError(
            f"layers must map sites ({', '.join(LENS_SITES)}) to layers, got {sorted(layers)}"
        )
    return out


def _output_out(kind: str, site: str) -> bool:
    return LensKey(kind, site).side == "output" and site == "out"


@dataclass
class LensSums:
    """The estimator's raw state: fp32 CPU sums per site and layer.

    ``total`` sums every document's estimate, ``odd`` the odd-indexed
    documents' (``None`` when the split-half check is off or a resume base
    lacks it).  Checkpoints persist both; :meth:`finish` turns ``total`` into
    the lens in place.
    """

    kind: str
    d_model: int
    n_prompts: int
    total: dict[str, dict[int, torch.Tensor]]
    odd: dict[str, dict[int, torch.Tensor]] | None = None

    @property
    def key(self) -> LensKey:
        return LensKey(self.kind)

    @property
    def n_odd(self) -> int:
        return self.n_prompts // 2

    @property
    def layers(self) -> dict[str, list[int]]:
        return {site: sorted(matrices) for site, matrices in self.total.items()}

    @classmethod
    def zeros(
        cls, kind: str, d_model: int, layers: SiteLayers, *, split_half: bool,
    ) -> "LensSums":
        def make() -> dict[str, dict[int, torch.Tensor]]:
            return {
                site: {int(layer): torch.zeros(d_model, d_model) for layer in wanted}
                for site, wanted in layers.items()
            }

        return cls(kind, int(d_model), 0, make(), make() if split_half else None)

    @classmethod
    def from_means(
        cls,
        lens: JacobianLens,
        odd: Mapping[str, Mapping[int, torch.Tensor]] | None = None,
    ) -> "LensSums":
        """Take ownership of an averaged lens (and its odd-half means): the
        matrices are scaled to sums in place, so the caller drops ``lens``."""
        n = int(lens.n_prompts)
        if n <= 0:
            raise ValueError("a resume base must hold at least one prompt")
        total = {site: lens.matrices(site) for site in lens.sites}
        for matrices in total.values():
            for matrix in matrices.values():
                matrix.mul_(n)
        lens._atom_norm_cache.clear()
        n_odd = n // 2
        odd_sums: dict[str, dict[int, torch.Tensor]] | None = None
        if odd is not None:
            if {site: sorted(m) for site, m in odd.items()} != {
                site: sorted(m) for site, m in total.items()
            }:
                raise ValueError("odd-half matrices must cover the base's sites and layers")
            odd_sums = {
                site: {int(layer): m.to(torch.float32).mul_(n_odd) for layer, m in matrices.items()}
                for site, matrices in odd.items()
            }
        elif n_odd == 0:
            # One document: the odd half is empty, so its sum is exactly zero.
            odd_sums = {
                site: {layer: torch.zeros_like(m) for layer, m in matrices.items()}
                for site, matrices in total.items()
            }
        return cls(lens.kind, lens.d_model, n, total, odd_sums)

    def split_half(self) -> dict[str, dict[int, float]]:
        """Per site and layer ``‖mean_even − mean_odd‖ / ‖mean‖`` (empty
        when the odd half is not tracked or either half is empty)."""
        n, n_odd = self.n_prompts, self.n_odd
        n_even = n - n_odd
        if self.odd is None or n_odd == 0 or n_even == 0:
            return {}
        out: dict[str, dict[int, float]] = {}
        for site, matrices in self.total.items():
            odd = self.odd[site]
            values: dict[int, float] = {}
            for layer, total in matrices.items():
                odd_sum = odd[layer]
                diff = (total - odd_sum).div_(n_even).sub_(odd_sum, alpha=1.0 / n_odd)
                scale = float(total.norm()) / n
                values[layer] = float(diff.norm()) / scale if scale > 0 else float("inf")
            out[site] = values
        return out

    def finish(self) -> JacobianLens:
        """The averaged lens, normalizing ``total`` in place (the sums are
        consumed)."""
        if self.n_prompts <= 0:
            raise JacobianLensError("no usable prompts were fitted")
        inv = 1.0 / self.n_prompts
        for matrices in self.total.values():
            for matrix in matrices.values():
                matrix.mul_(inv)
        return JacobianLens(
            self.total.get("out"), mid=self.total.get("mid"),
            n_prompts=self.n_prompts, d_model=self.d_model, key=self.key,
        )


@dataclass(frozen=True)
class LensFit:
    """A fitted lens and its per-site, per-layer split-half differences
    (``{}`` for a site or fit that did not measure them)."""

    lens: JacobianLens
    split_half: dict[str, dict[int, float]] = field(default_factory=dict)


class _AsyncZeroRows(JacobianLensError):
    """An accelerator returned an all-zero row: work silently lost (an
    asynchronous out of memory on the command queue)."""


class _Steps:
    """The per-block maps as they run on the fit's device (compiled on CUDA,
    falling back to eager when compilation fails)."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.forward: Callable[..., tuple[torch.Tensor, torch.Tensor]] = step_for(jvp, device)
        self.reverse: Callable[..., tuple[torch.Tensor, torch.Tensor]] = step_for(vjp, device)

    @property
    def compiled(self) -> bool:
        return self.forward is not jvp or self.reverse is not vjp

    def eager(self) -> None:
        self.forward, self.reverse = jvp, vjp


def _prepare_rows(
    tokenizer: Any,
    prompts: Sequence[str],
    input_id_rows: Sequence[Sequence[int]] | None,
    *,
    max_seq_len: int,
    skip_first: int,
) -> list[torch.Tensor]:
    if input_id_rows is not None and len(input_id_rows) != len(prompts):
        raise ValueError(
            "input_id_rows must be aligned with prompts "
            f"({len(input_id_rows)} != {len(prompts)})"
        )
    rows: list[torch.Tensor] = []
    for index, prompt in enumerate(prompts):
        if input_id_rows is None:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0, :max_seq_len]
            ids = ids.to(dtype=torch.long, device="cpu")
        else:
            ids = torch.tensor(
                [int(tok) for tok in input_id_rows[index]][:max_seq_len], dtype=torch.long,
            )
        if int(ids.numel()) < skip_first + 2:
            log.warning(
                "lens fit: skipping prompt %d — %d tokens, need > %d",
                index, int(ids.numel()), skip_first + 1,
            )
            continue
        rows.append(ids)
    return rows


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _drain(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def fit_lens(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    kind: str,
    sites: Sequence[str] = DEFAULT_SITES,
    layers: SiteLayers | None = None,
    dim_batch: int = DEFAULT_DIM_BATCH,
    max_seq_len: int = DEFAULT_SEQ_LEN,
    skip_first: int = SKIP_FIRST_POSITIONS,
    split_half: bool = True,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    on_checkpoint: Callable[[LensSums], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
    progress_base: int = 0,
    input_id_rows: Sequence[Sequence[int]] | None = None,
    cancel_event: Any | None = None,
    initial: LensSums | None = None,
    spec: ModelSpec | None = None,
) -> LensFit:
    """Fit a ``kind`` lens (J, R, K or S) at ``sites`` over ``prompts``.

    ``layers`` (``{site: layers}``) restricts the fit and names its sites;
    without it each of ``sites`` (default both) covers its whole pool.
    ``initial`` is a resume base (:meth:`LensSums.from_means`): its sums
    become the accumulator and must name exactly the requested sites and
    layers; documents continue its count, so the split-half parity follows
    the corpus index.  ``on_checkpoint`` receives the live sums after every
    ``checkpoint_every`` documents (and on a stop between documents); it
    must not keep them.  ``input_id_rows`` is the pre-tokenized sibling of
    ``prompts`` (still cut at ``max_seq_len``).  ``spec`` is a verified
    :class:`~drowse.model.blocks.ModelSpec`; without one the model is
    verified first (:func:`~drowse.model.blocks.verify_model_spec`), so the
    caller holds it exclusively and unsteered.  An unsupported block raises
    :class:`~drowse.errors.UnsupportedBlockError` before any work.

    Raises :class:`JacobianLensError` when no prompt is long enough (each
    needs more than ``skip_first + 1`` tokens).
    """
    if dim_batch <= 0:
        raise ValueError("dim_batch must be > 0")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be > 0")
    if progress_base < 0:
        raise ValueError("progress_base must be >= 0")
    if skip_first < 0:
        raise ValueError("skip_first must be >= 0")
    key = LensKey(kind) if kind in FITTABLE_KINDS else None
    if key is None:
        raise JacobianLensError(
            f"no estimator fits {kind}-lenses (fittable: {', '.join(FITTABLE_KINDS)})"
        )
    eager = getattr(model, "_orig_mod", model)
    if spec is None:
        spec = verify_model_spec(eager)
    d = spec.d_model
    plan = fit_site_layers(kind, spec.n_layers, sites=sites, layers=layers)
    if initial is not None:
        if initial.kind != kind:
            raise ValueError(f"the resume base is a {initial.kind} lens, not {kind}")
        if initial.layers != plan:
            raise ValueError("the resume base's sites and layers must match the fit request")
        if initial.d_model != d:
            raise ValueError("the resume base's d_model does not match the loaded model")
    rows = _prepare_rows(
        tokenizer, prompts, input_id_rows, max_seq_len=max_seq_len, skip_first=skip_first,
    )
    if not rows:
        raise JacobianLensError(
            f"no usable prompts: every prompt had <= {skip_first + 1} tokens"
        )
    if cancel_event is not None and cancel_event.is_set():
        raise JacobianLensCancelled("lens fit cancelled before start")
    if initial is None:
        sums = LensSums.zeros(kind, d, plan, split_half=split_half)
    else:
        sums = initial
        if not split_half:
            sums.odd = None
        elif sums.odd is None:
            log.warning(
                "lens fit: the resume base carries no odd-half sums; this fit "
                "records no split-half difference (refit with force to measure it)"
            )
    return _run(
        spec, rows, sums, key=key, plan=plan, dim_batch=int(dim_batch),
        skip_first=int(skip_first), checkpoint_every=int(checkpoint_every),
        on_checkpoint=on_checkpoint, on_progress=on_progress,
        progress_base=int(progress_base), cancel_event=cancel_event,
    )


def _run(
    spec: ModelSpec,
    rows: list[torch.Tensor],
    sums: LensSums,
    *,
    key: LensKey,
    plan: dict[str, list[int]],
    dim_batch: int,
    skip_first: int,
    checkpoint_every: int,
    on_checkpoint: Callable[[LensSums], None] | None,
    on_progress: Callable[[str], None] | None,
    progress_base: int,
    cancel_event: Any | None,
) -> LensFit:
    device = spec.embed.weight.device
    work = torch.promote_types(spec.embed.weight.dtype, torch.float32)
    d = spec.d_model
    direction = key.side
    rules = rules_for(key.kind)
    blocks = blocks_needed(direction, spec.n_layers, plan)
    slots = [(site, layer) for site in plan for layer in plan[site]]
    slot_of = {entry: index for index, entry in enumerate(slots)}
    out_layers, mid_layers = plan.get("out", []), plan.get("mid", [])
    steps = _Steps(device)
    batch = min(dim_batch, d)
    n_done = 0
    next_checkpoint = checkpoint_every
    started = time.perf_counter()

    def checkpoint() -> None:
        if on_checkpoint is not None:
            on_checkpoint(sums)

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    maps: Any = None
    try:
        with torch.no_grad():
            for ids in rows:
                if cancelled():
                    if n_done:
                        checkpoint()
                    raise JacobianLensCancelled(f"lens fit cancelled after {n_done} prompts")
                odd = sums.n_prompts % 2 == 1
                maps = linearize(spec, ids, rules=rules, skip_first=skip_first, blocks=blocks)
                n_rows = next(iter(maps.values())).n_rows
                start = 0
                while start < d:
                    b = min(batch, d - start)
                    try:
                        host = _pass(
                            maps, steps, direction, start, b, n_rows,
                            n_layers=spec.n_layers, d=d, device=device, work=work,
                            out_layers=out_layers, mid_layers=mid_layers,
                            slot_of=slot_of,
                        )
                    except RuntimeError as exc:
                        if steps.compiled and not is_out_of_memory_error(exc):
                            log.warning(
                                "lens fit: the compiled block maps failed (%s); "
                                "running them eagerly", exc,
                            )
                            steps.eager()
                            continue
                        if not is_out_of_memory_error(exc) or batch <= 1:
                            raise
                        batch = max(1, batch // 2)
                        _empty_device_cache(device)
                        log.warning(
                            "lens fit: out of memory — retrying the pass at dim_batch=%d", batch,
                        )
                        continue
                    _commit(sums, host, slots, start, b, odd=odd, direction=direction)
                    start += b
                    if start < d and cancelled():
                        # A partial document is not a consistent sum: abandon
                        # it; the caller resumes from the last checkpoint.
                        _drain(device)
                        raise JacobianLensCancelled(
                            f"lens fit cancelled inside a document after {n_done} prompts"
                        )
                maps = None
                sums.n_prompts += 1
                n_done += 1
                if on_progress is not None:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    on_progress(
                        f"prompt {progress_base + n_done}/{progress_base + len(rows)} "
                        f"(dim_batch={batch}, elapsed={elapsed:.1f}s, "
                        f"rate={n_done / elapsed:.3f}/s)"
                    )
                if cancelled():
                    checkpoint()
                    raise JacobianLensCancelled(f"lens fit cancelled after {n_done} prompts")
                if n_done >= next_checkpoint:
                    checkpoint()
                    # Allocator hygiene at checkpoint cadence only.
                    if device.type == "mps":
                        _empty_device_cache(device)
                    next_checkpoint = (n_done // checkpoint_every + 1) * checkpoint_every
    finally:
        maps = None
        if device.type in {"mps", "cuda"}:
            _drain(device)
            gc.collect()
            _empty_device_cache(device)
    split = sums.split_half()
    sums.odd = None
    return LensFit(lens=sums.finish(), split_half=split)


def _pass(
    maps: Mapping[int, Any],
    steps: _Steps,
    direction: str,
    start: int,
    b: int,
    n_rows: int,
    *,
    n_layers: int,
    d: int,
    device: torch.device,
    work: torch.dtype,
    out_layers: Sequence[int],
    mid_layers: Sequence[int],
    slot_of: Mapping[tuple[str, int], int],
) -> torch.Tensor:
    """One sweep of ``b`` seeds; the staged reads on the host, ``[slots, b, d]``."""
    stage = torch.empty(len(slot_of), b, d, device=device, dtype=torch.float32)
    filled: list[int] = []

    def emit(site: str, layer: int, est: torch.Tensor) -> None:
        slot = slot_of.get((site, layer))
        if slot is not None:
            stage[slot].copy_(est)
            filled.append(slot)

    seeds = seed_block(b, n_rows, d, start, device=device, dtype=work)
    if direction == "output":
        reverse_sweep(
            maps, seeds, n_layers=n_layers, out_layers=out_layers,
            mid_layers=mid_layers, emit=emit, step=steps.reverse,
        )
    else:
        forward_sweep(
            maps, seeds, out_layers=out_layers, mid_layers=mid_layers,
            emit=emit, step=steps.forward,
        )
    if len(filled) != len(slot_of):
        raise JacobianLensError("a transport sweep skipped a requested site or layer")
    if device.type == "cpu":
        return stage
    host = stage.to("cpu")
    # Metal reports command-queue exhaustion asynchronously: nothing raises
    # and the work comes back as zeros.  No lens row is ever exactly zero
    # (the residual path alone contributes), so a zero row is lost work.
    if bool((host.abs().sum(dim=-1) == 0).any()):
        raise _AsyncZeroRows(
            "a transport pass came back with zero rows from the device — an "
            "asynchronous out of memory on the command queue"
        )
    return host


def _commit(
    sums: LensSums,
    host: torch.Tensor,
    slots: Sequence[tuple[str, int]],
    start: int,
    b: int,
    *,
    odd: bool,
    direction: str,
) -> None:
    targets = [sums.total]
    if odd and sums.odd is not None:
        targets.append(sums.odd)
    for index, (site, layer) in enumerate(slots):
        rows = host[index]
        for target in targets:
            matrix = target[site][layer]
            if direction == "output":
                matrix[start:start + b].add_(rows)
            else:
                matrix[:, start:start + b].add_(rows.T)


__all__ = [
    "DEFAULT_SITES",
    "LensFit",
    "LensSums",
    "fit_lens",
    "fit_site_layers",
]
