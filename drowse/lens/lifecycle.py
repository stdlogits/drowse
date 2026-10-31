"""The lens fit lifecycle: a pure planner and the executor that runs it.

A fit names a kind and, per site (``out``, ``mid``), its layers.  It has
four outcomes, decided from metadata alone:

- ``noop``: the durable artifact already covers this corpus and every
  requested site and layer.  Nothing is fitted; a crash-left checkpoint is
  reaped.
- ``topup``: the durable artifact covers the corpus but misses layers at
  some site.  Only those are fitted (resuming their own checkpoint) and
  unioned in.
- ``resume``: a matching durable artifact or checkpoint over a prefix of the
  corpus is the starting point; the remainder is fitted and merged.
- ``fresh``: nothing reusable (or ``force``); fit from prompt zero.

:func:`plan_lens_fit` is the decision, a pure function of the corpus
identity, the durable and checkpoint sidecars and the resident lens.
:class:`LensFitter` executes a plan.  When a payload the plan relies on fails
to load (a corrupt shard, a concurrent removal) the executor drops that
sidecar and plans again, which yields exactly the fallback the failed payload
calls for.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol


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
from drowse.model.layers import LayerSelector
from drowse.lens.store import (
    LensProvenance,
    lens_estimator_policy,
    lens_identity,
    lens_layers,
    lens_split_half,
)
from drowse.errors import RequestError

if TYPE_CHECKING:
    from drowse.lens.fit import LensFit, LensSums

#: The corpus identity a fit records: sha256 of the consumed token-id rows.
CORPUS_HASH_KIND = "token_ids_v1"

FitAction = Literal["noop", "topup", "resume", "fresh"]
DurableUse = Literal["none", "resident", "full", "selective"]
#: ``{site: sorted layers}``: what a fit, an artifact or a resident lens covers.
SiteLayers = Mapping[str, tuple[int, ...]]


def raw_corpus_sha256(prompts: Sequence[str]) -> str:
    """The raw-document identity a fit records (``raw_corpus_sha256``)."""
    return hashlib.sha256(repr(list(prompts)).encode("utf-8")).hexdigest()


def _rows_sha256(rows: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(repr([list(row) for row in rows]).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LensFitCorpus:
    """The usable documents of a fit and their consumed token-id rows.

    Documents too short for the estimator (at most ``SKIP_FIRST_POSITIONS +
    1`` tokens) are dropped up front, so ``n_prompts`` counts consumed
    documents exactly; that is what makes prefix resume sound.
    """

    prompts: tuple[str, ...]
    token_rows: tuple[tuple[int, ...], ...]
    raw_sha256: str
    raw_count: int
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sha256", _rows_sha256(self.token_rows))

    @property
    def n_usable(self) -> int:
        return len(self.prompts)

    def prefix_sha256(self, n: int) -> str:
        """The token-row identity of the first ``n`` usable documents."""
        return _rows_sha256(self.token_rows[:n])

    def id_rows(self, start: int = 0) -> list[list[int]]:
        return [list(row) for row in self.token_rows[start:]]


def tokenize_lens_corpus(
    tokenizer: Any, prompts: Sequence[str], *, seq_len: int,
) -> LensFitCorpus:
    """Tokenize ``prompts`` at ``seq_len`` and keep the usable ones."""
    prompt_list = list(prompts)
    usable: list[str] = []
    rows: list[tuple[int, ...]] = []
    for prompt in prompt_list:
        try:
            ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=seq_len,
            )["input_ids"]
        except TypeError:
            # Minimal tokenizers may not take HF truncation kwargs.
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"][:, :seq_len]
        if ids.shape[1] > SKIP_FIRST_POSITIONS + 1:
            usable.append(prompt)
            rows.append(tuple(int(tok) for tok in ids[0].tolist()))
    return LensFitCorpus(
        prompts=tuple(usable),
        token_rows=tuple(rows),
        raw_sha256=raw_corpus_sha256(prompt_list),
        raw_count=len(prompt_list),
    )


def resolve_fit_layers(
    selector: LayerSelector,
    *,
    kind: str,
    n_layers: int,
    sites: Sequence[str] = LENS_SITES,
) -> dict[str, tuple[int, ...]]:
    """The ``{site: sorted layers}`` a fit of ``kind`` covers.

    Each site draws from its own pool (:func:`~drowse.lens.key.site_layers`:
    the output kinds' ``out`` stops below the last block).  ``None``/``all``
    is the whole pool, ``workspace`` its 40–90% band, an index list the
    listed layers each site holds.  ``sample`` is refused (it is a readout
    mode, not a wall-time lever); so is a layer outside the model, a layer
    no requested site holds, or a site left with no layer.
    """
    from drowse.model.arch import workspace_layer_indices

    chosen = [site for site in LENS_SITES if site in set(sites)]
    if not chosen or len(chosen) != len(tuple(sites)):
        raise RequestError(
            f"sites must be a non-empty subset of {', '.join(LENS_SITES)}, got {list(sites)}"
        )
    LensKey(kind)
    explicit: list[int] | None = None
    if isinstance(selector, str) and selector.lower() not in ("all", "workspace"):
        raise RequestError(
            f"lens fit layers must be all, workspace or an index list, not {selector!r}"
        )
    if selector is not None and not isinstance(selector, str):
        explicit = sorted({int(layer) for layer in selector})
        outside = [layer for layer in explicit if not 0 <= layer < n_layers]
        if outside or not explicit:
            raise RequestError(
                f"lens fit layers must lie in [0, {n_layers}); got {list(selector)}"
            )
    out: dict[str, tuple[int, ...]] = {}
    for site in chosen:
        pool = site_layers(kind, site, n_layers)
        if explicit is not None:
            held = set(pool)
            wanted = [layer for layer in explicit if layer in held]
        elif isinstance(selector, str) and selector.lower() == "workspace":
            wanted = workspace_layer_indices(pool, n_layers)
        else:
            wanted = pool
        if not wanted:
            raise RequestError(
                f"no requested layer is a {kind}:{site} layer (it holds 0..{len(pool) - 1}"
                + ("; the last block is the output kinds' target" if len(pool) < n_layers else "")
                + ")"
            )
        out[site] = tuple(sorted(wanted))
    if explicit is not None:
        covered = {layer for layers in out.values() for layer in layers}
        orphans = [layer for layer in explicit if layer not in covered]
        if orphans:
            raise RequestError(
                f"layers {orphans} belong to no requested {kind} site "
                f"({', '.join(chosen)}): the last block is the output kinds' "
                "target, a layer only their mid site holds"
            )
    return out


def _site_layers_of(sidecar: Mapping[str, Any]) -> dict[str, frozenset[int]]:
    sites = sidecar.get("sites")
    if not isinstance(sites, Mapping):
        return {}
    return {site: frozenset(lens_layers(sidecar, site)) for site in sites}


def _covers(have: Mapping[str, frozenset[int]], want: SiteLayers) -> bool:
    return all(set(layers) <= have.get(site, frozenset()) for site, layers in want.items())


@dataclass(frozen=True)
class LensFitIdentity:
    """Everything a sidecar must match before its payload is reused."""

    kind: str
    corpus: LensFitCorpus
    layers: SiteLayers
    seq_len: int
    model_fingerprint: str
    policy: Mapping[str, Any]

    @property
    def name(self) -> str:
        """The local artifact directory (``local:<kind>``)."""
        return self.kind.lower()

    def _semantics_match(self, sidecar: Mapping[str, Any]) -> bool:
        return (
            sidecar.get("corpus_hash_kind") == CORPUS_HASH_KIND
            and sidecar.get("seq_len") == self.seq_len
            and sidecar.get("model_fingerprint") == self.model_fingerprint
            and sidecar.get("estimator_policy") == self.policy
        )

    def checkpoint_matches(
        self, sidecar: Mapping[str, Any] | None, layers: SiteLayers,
    ) -> bool:
        """A checkpoint over this corpus (or a prefix of it) for exactly
        ``layers`` (``{site: layers}``)."""
        if sidecar is None:
            return False
        progress = int(sidecar["n_prompts"])
        corpus = self.corpus
        corpus_matches = sidecar.get("corpus_sha256") == corpus.sha256 or (
            0 < progress <= corpus.n_usable
            and sidecar.get("consumed_prefix_sha256") == corpus.prefix_sha256(progress)
        )
        return bool(
            corpus_matches
            and progress > 0
            and self._semantics_match(sidecar)
            and _site_layers_of(sidecar) == {
                site: frozenset(wanted) for site, wanted in layers.items()
            }
        )

    def durable_matches(self, sidecar: Mapping[str, Any] | None) -> bool:
        """A durable artifact fitted on this corpus or a prefix of it."""
        if sidecar is None or not self._semantics_match(sidecar):
            return False
        saved_n = int(sidecar.get("n_prompts", 0))
        saved_sha = sidecar.get("corpus_sha256")
        return saved_sha == self.corpus.sha256 or (
            0 < saved_n <= self.corpus.n_usable
            and saved_sha == self.corpus.prefix_sha256(saved_n)
        )


@dataclass(frozen=True)
class ResidentLens:
    """What the planner needs to know about the session's resident lens."""

    n_prompts: int
    layers: Mapping[str, frozenset[int]]
    identity: str | None


@dataclass(frozen=True)
class LensFitPlan:
    """One fit decision.

    ``durable`` names how the durable artifact is obtained (``resident``
    reuses the session's resident lens, ``selective`` loads only the
    requested shards, ``full`` loads every shard).  ``checkpoint`` loads the
    checkpoint payload: the resume base, or for a ``topup`` the missing
    layers' partial accumulator.
    """

    action: FitAction
    durable: DurableUse = "none"
    checkpoint: bool = False
    #: ``{site: layers}`` a ``topup`` fits (only sites with missing layers).
    missing: SiteLayers = field(default_factory=dict)


def plan_lens_fit(
    identity: LensFitIdentity,
    *,
    durable: Mapping[str, Any] | None,
    checkpoint: Mapping[str, Any] | None,
    resident: ResidentLens | None,
    force: bool = False,
) -> LensFitPlan:
    """Decide a fit from metadata alone (see the module docstring).

    Raises :class:`JacobianLensError` when extending the corpus would have
    to drop durable layers the request does not name: one artifact has one
    corpus identity, so a subset extension cannot keep the other layers.
    """
    if force:
        return LensFitPlan("fresh")
    requested = identity.layers
    n_usable = identity.corpus.n_usable
    checkpoint_ok = identity.checkpoint_matches(checkpoint, requested)
    checkpoint_n = int(checkpoint["n_prompts"]) if checkpoint_ok and checkpoint else -1
    if durable is not None and identity.durable_matches(durable):
        saved_n = int(durable.get("n_prompts", 0))
        durable_layers = _site_layers_of(durable)
        extra = {
            site: sorted(layers - set(requested.get(site, ())))
            for site, layers in durable_layers.items()
            if layers - set(requested.get(site, ()))
        }
        if saved_n < n_usable and extra:
            raise JacobianLensError(
                f"cannot extend only a subset of an existing {identity.kind} lens: "
                f"it holds layers the request omits ({extra}). Request every "
                "site and layer it holds to extend its corpus, or pass "
                "force=True to replace it with the requested subset."
            )
        if checkpoint_ok and checkpoint_n > saved_n:
            # The checkpoint is farther ahead than the durable artifact.
            return LensFitPlan("resume", checkpoint=True)
        covers = _covers(durable_layers, requested)
        resident_ok = bool(
            resident is not None
            and resident.n_prompts == saved_n
            # A resident stamped with the durable pointer must represent
            # that whole artifact, not a subset view from an earlier call.
            and dict(resident.layers) == durable_layers
            and covers
            and resident.identity == lens_identity(durable)
        )
        if saved_n >= n_usable:
            if covers:
                return LensFitPlan(
                    "noop", durable="resident" if resident_ok else "selective",
                )
            missing = {
                site: tuple(sorted(set(layers) - durable_layers.get(site, frozenset())))
                for site, layers in requested.items()
                if set(layers) - durable_layers.get(site, frozenset())
            }
            return LensFitPlan(
                "topup",
                durable="full",
                missing=missing,
                checkpoint=identity.checkpoint_matches(checkpoint, missing),
            )
        if covers:
            return LensFitPlan(
                "resume", durable="resident" if resident_ok else "full",
            )
    if checkpoint_ok:
        return LensFitPlan("resume", checkpoint=True)
    return LensFitPlan("fresh")


def _holds_artifact(lens: JacobianLens, sidecar: Mapping[str, Any]) -> bool:
    """Whether ``lens`` is the whole artifact ``sidecar`` names: every site,
    every layer (a selective load or selection is a view, not the artifact)."""
    sites = tuple(sidecar.get("sites", {}))
    return set(lens.sites) == set(sites) and all(
        set(lens.layers(site)) == set(lens_layers(sidecar, site)) for site in sites
    )

# ---------------------------------------------------------------------------
# Weight-free no-op proof
# ---------------------------------------------------------------------------

def fit_covers_layers(
    sidecar: Mapping[str, Any],
    requested: LayerSelector,
    *,
    sites: Sequence[str] = LENS_SITES,
) -> bool:
    """Whether a fitted sidecar covers a fit request at every requested site."""
    n_layers = sidecar.get("model_layer_count")
    kind = sidecar.get("kind")
    if not isinstance(n_layers, int) or n_layers < 1 or kind not in FITTABLE_KINDS:
        return False
    try:
        wanted = resolve_fit_layers(requested, kind=str(kind), n_layers=n_layers, sites=sites)
    except (ValueError, RequestError):
        return False
    return _covers(_site_layers_of(sidecar), wanted)


def loaded_source_fingerprint(model: Any) -> str | None:
    """A loaded model's ``model_source_fingerprint`` (``None`` when the
    loader stamped none), through a compile wrapper."""
    return getattr(
        getattr(model, "_orig_mod", model), "_drowse_source_fingerprint", None,
    )


def prove_lens_fit_noop(
    model_id: str,
    name: str,
    *,
    requested: LayerSelector,
    sites: Sequence[str] = LENS_SITES,
    seq_len: int | None,
    source_fingerprint: Callable[[], str | None],
    docs: Sequence[str] | None = None,
    corpus_spec: str | None = None,
    prompt_count: int | None = None,
) -> dict[str, Any] | None:
    """Prove a fit of ``local:<name>`` would be a no-op, without the model.

    The corpus identity is ``docs`` (a file corpus: the raw document
    digest) or ``corpus_spec`` plus ``prompt_count`` (a pinned Hub corpus,
    before streaming it).  ``source_fingerprint`` computes the exact model
    source identity and runs only once the cheap checks pass.  Runs under
    the fit lock, reaping a crash-left checkpoint the proven artifact
    subsumes.  Returns the proven sidecar, or ``None``.
    """
    from drowse.lens.key import local_lens_kind
    from drowse.lens.store import (
        lens_fit_lock,
        lens_payloads_match,
        load_local_lens_sidecar,
        remove_subsumed_lens_checkpoint,
    )

    with lens_fit_lock(model_id, name):
        sidecar = load_local_lens_sidecar(model_id, name)
        if sidecar is None or sidecar.get("estimator_policy") != lens_estimator_policy(
            local_lens_kind(name),
        ):
            return None
        if (
            sidecar.get("seq_len") != (seq_len or DEFAULT_SEQ_LEN)
            or not fit_covers_layers(sidecar, requested, sites=sites)
        ):
            return None
        if docs is not None:
            if (
                sidecar.get("raw_corpus_sha256") != raw_corpus_sha256(docs)
                or sidecar.get("raw_prompt_count") != len(docs)
            ):
                return None
        elif corpus_spec is None or (
            sidecar.get("corpus_spec") != corpus_spec
            or sidecar.get("raw_prompt_count") != prompt_count
        ):
            return None
        usable = int(sidecar.get("usable_prompt_count", -1))
        if usable < 0 or int(sidecar.get("n_prompts", -1)) < usable:
            return None
        source_fp = source_fingerprint()
        if source_fp is None or sidecar.get("model_source_fingerprint") != source_fp:
            return None
        if not lens_payloads_match(model_id, sidecar, name=name):
            return None
        remove_subsumed_lens_checkpoint(
            model_id, name=name, verified_final_sidecar=sidecar,
        )
        return sidecar


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
#: A live-readout configuration: ``(layers, uses_all_layers)``.
LiveConfig = tuple[list[int], bool]


class LensResidency(Protocol):
    """The resident-lens transactions a fit drives (the lens instrument)."""

    @property
    def resident_lens(self) -> JacobianLens | None: ...

    @property
    def resident_identity(self) -> str | None: ...

    def live_config(self) -> LiveConfig | None: ...

    def evict(self) -> None: ...

    def adopt(
        self,
        lens: JacobianLens,
        sidecar: Mapping[str, Any] | None,
        *,
        live: LiveConfig | None = None,
        restore_live: bool = False,
    ) -> JacobianLens: ...


class _NoResidency:
    """The residency of a side nothing holds resident (input kinds): a fit
    publishes and selects, and adopts nothing."""

    resident_lens: JacobianLens | None = None
    resident_identity: str | None = None

    def live_config(self) -> LiveConfig | None:
        return None

    def evict(self) -> None:
        return None

    def adopt(
        self,
        lens: JacobianLens,
        sidecar: Mapping[str, Any] | None,
        *,
        live: LiveConfig | None = None,
        restore_live: bool = False,
    ) -> JacobianLens:
        del sidecar, live, restore_live
        return lens


@dataclass(frozen=True)
class LensFitRequest:
    """The knobs of one fit call, validated."""

    kind: str = "J"
    corpus_spec: str = "custom"
    layers: LayerSelector = None
    sites: tuple[str, ...] = LENS_SITES
    dim_batch: int = DEFAULT_DIM_BATCH
    seq_len: int = DEFAULT_SEQ_LEN
    force: bool = False
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    split_half: bool = True

    def __post_init__(self) -> None:
        if self.kind not in FITTABLE_KINDS:
            raise RequestError(
                f"lens kind must be one of {', '.join(FITTABLE_KINDS)}, "
                f"got {self.kind!r}"
            )
        sites = tuple(self.sites)
        if not sites or len(set(sites)) != len(sites) or set(sites) - set(LENS_SITES):
            raise RequestError(
                f"sites must be a non-empty subset of {', '.join(LENS_SITES)}, got {list(sites)}"
            )
        object.__setattr__(self, "sites", tuple(s for s in LENS_SITES if s in sites))
        if isinstance(self.layers, str) and self.layers.lower() == "sample":
            raise RequestError(
                "layers 'sample' is a readout mode, not a fit lever; use "
                "'workspace' or an index list"
            )
        if self.dim_batch <= 0:
            raise RequestError("dim_batch must be > 0")
        if self.seq_len <= 0:
            raise RequestError("seq_len must be > 0")
        if self.checkpoint_every <= 0:
            raise RequestError("checkpoint_every must be > 0")


class LensFitter:
    """Runs a planned fit against the store and the resident lens.

    The caller holds the fit lock and the model-exclusive guard, unsteered:
    the fit verifies the model's block description and runs transport
    passes through it.  ``residency`` is the output side's resident lens; an
    input-kind fit adopts nothing.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        model_id: str,
        residency: LensResidency,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._residency = residency

    def source_fingerprint(self) -> str | None:
        """The loaded model's exact source identity, as a fit records it."""
        return loaded_source_fingerprint(self._model)

    def run(
        self,
        prompts: Sequence[str],
        request: LensFitRequest,
        *,
        on_progress: Callable[[str], None] | None = None,
        cancel_event: Any | None = None,
    ) -> JacobianLens:
        from drowse.lens.store import (
            load_lens_checkpoint_sidecar,
            load_local_lens_sidecar,
            remove_lens_checkpoint,
        )
        from drowse.model.blocks import verify_model_spec
        from drowse.model.fingerprint import loaded_model_fingerprint

        if cancel_event is not None and cancel_event.is_set():
            raise JacobianLensCancelled("lens fit cancelled before start")
        model = getattr(self._model, "_orig_mod", self._model)
        # Refuse an undescribable block before any work, and hold every block
        # to its description once (the maps are written from it).
        spec = verify_model_spec(model)
        layers = resolve_fit_layers(
            request.layers, kind=request.kind, n_layers=spec.n_layers, sites=request.sites,
        )
        corpus = tokenize_lens_corpus(self._tokenizer, prompts, seq_len=request.seq_len)
        if corpus.n_usable < corpus.raw_count and on_progress is not None:
            on_progress(
                f"dropped {corpus.raw_count - corpus.n_usable} too-short prompts "
                f"({corpus.n_usable} usable)"
            )
        if not corpus.n_usable:
            raise JacobianLensError(
                "no usable prompts: the lens estimator needs prompts "
                f"longer than {SKIP_FIRST_POSITIONS + 1} tokens"
            )
        identity = LensFitIdentity(
            kind=request.kind,
            corpus=corpus,
            layers=layers,
            seq_len=request.seq_len,
            model_fingerprint=loaded_model_fingerprint(model, self._model_id),
            policy=lens_estimator_policy(request.kind),
        )
        residency = (
            self._residency if LensKey(request.kind).side == "output" else _NoResidency()
        )
        run = _FitRun(
            self, model, spec, identity, request, residency,
            on_progress=on_progress, cancel_event=cancel_event,
        )
        name = identity.name
        if request.force:
            remove_lens_checkpoint(self._model_id, name)
            return run.execute(durable=None, checkpoint=None)
        return run.execute(
            durable=load_local_lens_sidecar(self._model_id, name),
            checkpoint=load_lens_checkpoint_sidecar(self._model_id, name),
        )


def _merge_split_half(*parts: Mapping[str, Mapping[int, float]]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for part in parts:
        for site, values in part.items():
            out.setdefault(site, {}).update(values)
    return out


class _FitRun:
    """One fit call's executor state (the early-eviction live snapshot)."""

    def __init__(
        self,
        fitter: LensFitter,
        model: Any,
        spec: Any,
        identity: LensFitIdentity,
        request: LensFitRequest,
        residency: LensResidency,
        *,
        on_progress: Callable[[str], None] | None,
        cancel_event: Any | None,
    ) -> None:
        self.model_id = fitter._model_id
        self.tokenizer = fitter._tokenizer
        self.residency = residency
        self.model = model
        self.spec = spec
        self.identity = identity
        self.request = request
        self.on_progress = on_progress
        self.cancel_event = cancel_event
        #: Set once the resident lens was evicted ahead of a checkpoint load,
        #: with the live configuration it had (restored on adoption).
        self.evicted_early = False
        self.evicted_live: LiveConfig | None = None

    # -- helpers -------------------------------------------------------------

    def note(self, message: str) -> None:
        if self.on_progress is not None:
            self.on_progress(message)

    def provenance(self) -> LensProvenance:
        corpus = self.identity.corpus
        return LensProvenance(
            corpus_spec=self.request.corpus_spec,
            corpus_sha256=corpus.sha256,
            corpus_hash_kind=CORPUS_HASH_KIND,
            seq_len=self.request.seq_len,
            dim_batch=self.request.dim_batch,
            skip_first=SKIP_FIRST_POSITIONS,
            raw_corpus_sha256=corpus.raw_sha256,
            raw_prompt_count=corpus.raw_count,
            usable_prompt_count=corpus.n_usable,
            model_layer_count=int(self.spec.n_layers),
            model_fingerprint=self.identity.model_fingerprint,
            model_source_fingerprint=loaded_source_fingerprint(self.model),
        )

    def adopt(
        self,
        lens: JacobianLens,
        sidecar: Mapping[str, Any] | None,
        *,
        live: LiveConfig | None = None,
        restore_live: bool = False,
    ) -> JacobianLens:
        """Make ``lens`` resident; a lens without the ``out`` site, which
        the resident readout reads, leaves nothing resident instead."""
        if "out" not in lens.sites:
            self.residency.evict()
            return lens
        return self.residency.adopt(lens, sidecar, live=live, restore_live=restore_live)

    def resident_view(self) -> ResidentLens | None:
        lens = self.residency.resident_lens
        if lens is None:
            return None
        return ResidentLens(
            n_prompts=lens.n_prompts,
            layers={site: frozenset(lens.layers(site)) for site in lens.sites},
            identity=self.residency.resident_identity,
        )

    def evict_early(self) -> None:
        """Evict the resident before paging a checkpoint payload in, keeping
        its live configuration for the adoption that follows."""
        if self.evicted_early or self.residency.resident_lens is None:
            return
        self.evicted_live = self.residency.live_config()
        self.residency.evict()
        self.evicted_early = True

    def estimate(
        self,
        prompts: Sequence[str],
        layers: SiteLayers,
        *,
        start: int,
        initial: "LensSums | None",
        proof_out: list[Any] | None = None,
    ) -> "LensFit":
        from drowse.lens.fit import fit_lens

        request = self.request
        return fit_lens(
            self.model, self.tokenizer, list(prompts),
            kind=request.kind,
            layers={site: list(wanted) for site, wanted in layers.items()},
            dim_batch=request.dim_batch,
            max_seq_len=request.seq_len,
            split_half=request.split_half,
            on_checkpoint=self.checkpoint_writer(proof_out),
            checkpoint_every=request.checkpoint_every,
            on_progress=self.on_progress,
            progress_base=start,
            input_id_rows=self.identity.corpus.id_rows(start),
            cancel_event=self.cancel_event,
            initial=initial,
            spec=self.spec,
        )

    def checkpoint_writer(
        self, proof_out: list[Any] | None = None,
    ) -> "Callable[[LensSums], None]":
        from drowse.lens.store import save_lens_checkpoint

        def save(sums: "LensSums") -> None:
            # The estimator owns the sums; any resume prefix is already
            # folded in.
            out: list[Any] | None = None if proof_out is None else []
            save_lens_checkpoint(
                sums.total, sums.n_prompts, sums.d_model, self.model_id, self.provenance(),
                kind=self.request.kind,
                consumed_prefix_sha256=self.identity.corpus.prefix_sha256(sums.n_prompts),
                odd_sums=sums.odd,
                split_half=sums.split_half(),
                _proof_out=out,
            )
            if proof_out is not None:
                # Non-empty marks "a checkpoint was written" even when the
                # store returned no proof.
                proof_out[:] = out or [None]

        return save

    def resume_base(
        self, lens: JacobianLens, sidecar: Mapping[str, Any] | None,
    ) -> "LensSums":
        """The estimator's accumulator from an averaged base lens (taken over
        in place) and, from a checkpoint, its odd-half means."""
        from drowse.lens.fit import LensSums
        from drowse.lens.store import load_lens_checkpoint_odd

        odd = None
        if sidecar is not None and self.request.split_half:
            odd = load_lens_checkpoint_odd(self.model_id, self.identity.name, sidecar)
        return LensSums.from_means(lens, odd)

    def obtain_durable(
        self, use: DurableUse, sidecar: Mapping[str, Any],
    ) -> tuple[JacobianLens, Mapping[str, Any], Any] | None:
        """The durable lens for ``use``: ``(lens, sidecar, reuse proof)``."""
        from drowse.lens.store import _load_lens_verified

        if use == "resident":
            lens = self.residency.resident_lens
            return None if lens is None else (lens, sidecar, None)
        return _load_lens_verified(
            self.model_id,
            name=self.identity.name,
            requested_layers=dict(self.identity.layers) if use == "selective" else None,
        )

    # -- execution -----------------------------------------------------------

    def execute(
        self,
        *,
        durable: Mapping[str, Any] | None,
        checkpoint: Mapping[str, Any] | None,
    ) -> JacobianLens:
        from drowse.lens.store import load_lens_checkpoint

        while True:
            plan = plan_lens_fit(
                self.identity, durable=durable, checkpoint=checkpoint,
                resident=self.resident_view(), force=self.request.force,
            )
            existing = None
            if plan.durable != "none":
                assert durable is not None
                existing = self.obtain_durable(plan.durable, durable)
                if existing is None:
                    durable = None
                    continue
            if plan.action == "noop":
                assert existing is not None
                return self.noop(existing[0], existing[1])
            if plan.action == "topup":
                assert existing is not None
                return self.topup(plan, *existing)
            base: JacobianLens | None = None
            base_sidecar: Mapping[str, Any] | None = None
            if existing is not None:
                base = existing[0].select_layers(self.identity.layers)
            elif plan.checkpoint:
                self.evict_early()
                loaded = load_lens_checkpoint(self.model_id, self.identity.name)
                if loaded is None:
                    checkpoint = None
                    continue
                base, base_sidecar = loaded
                self.note(f"resuming from checkpoint at {base.n_prompts} prompts")
            # Drop the source containers so an all-layer durable lens does
            # not keep unrequested matrices alive beside the selected base.
            existing = None
            return self.resume(base, base_sidecar)

    def noop(
        self, lens: JacobianLens, sidecar: Mapping[str, Any],
    ) -> JacobianLens:
        from drowse.lens.store import remove_subsumed_lens_checkpoint

        self.note(f"lens already fitted on {lens.n_prompts} prompts — nothing to do")
        # A kill after durable publication but before the checkpoint unlink
        # leaves a second full shard set; reap it only when the final
        # artifact proves it is not farther ahead or different.
        remove_subsumed_lens_checkpoint(self.model_id, name=self.identity.name)
        selected = lens.select_layers(self.identity.layers)
        if _holds_artifact(lens, sidecar):
            self.adopt(lens, sidecar)
        else:
            # A selective no-op is a return view, not the complete artifact
            # its sidecar names: never stamp it with the durable identity.
            self.residency.evict()
        return selected

    def topup(
        self,
        plan: LensFitPlan,
        lens: JacobianLens,
        sidecar: Mapping[str, Any],
        proof: Any,
    ) -> JacobianLens:
        from drowse.lens.store import (
            load_lens_checkpoint,
            load_local_lens_sidecar,
            remove_lens_checkpoint,
            save_lens,
        )

        missing = dict(plan.missing)
        self.note(
            f"reusing existing fitted layers; fitting missing {self.request.kind} "
            f"layers {', '.join(f'{site} {list(layers)}' for site, layers in missing.items())}"
        )
        kept = {site: lens_split_half(sidecar, site) for site in lens.sites}
        partial = None
        if plan.checkpoint:
            loaded = load_lens_checkpoint(self.model_id, self.identity.name)
            if loaded is not None and {
                site: tuple(loaded[0].layers(site)) for site in loaded[0].sites
            } == {site: tuple(layers) for site, layers in missing.items()}:
                partial = loaded
                self.note(
                    f"resuming missing-layer checkpoint at {loaded[0].n_prompts} prompts"
                )
        start = partial[0].n_prompts if partial is not None else 0
        prompts = self.identity.corpus.prompts[start:]
        if prompts:
            fitted = self.estimate(
                prompts, missing, start=start,
                initial=None if partial is None else self.resume_base(*partial),
            )
            fitted_lens, fitted_split = fitted.lens, fitted.split_half
        else:
            assert partial is not None
            fitted_lens = partial[0]
            fitted_split = {
                site: lens_split_half(partial[1], site) for site in fitted_lens.sites
            }
        merged = JacobianLens.union_layers([lens, fitted_lens])
        save_lens(
            merged, self.model_id, self.provenance(),
            reuse_layers={layer for site in lens.sites for layer in lens.layers(site)},
            split_half=_merge_split_half(kept, fitted_split),
            _verified_reuse_proof=proof,
        )
        remove_lens_checkpoint(self.model_id, self.identity.name)
        selected = merged.select_layers(self.identity.layers)
        self.adopt(merged, load_local_lens_sidecar(self.model_id, self.identity.name))
        return selected

    def resume(
        self, base: JacobianLens | None, base_sidecar: Mapping[str, Any] | None,
    ) -> JacobianLens:
        from drowse.lens.store import (
            load_local_lens,
            load_local_lens_sidecar,
            promote_lens_checkpoint,
            remove_lens_checkpoint,
            save_lens,
        )

        name = self.identity.name
        start = base.n_prompts if base is not None else 0
        prompts = self.identity.corpus.prompts[start:]
        if base is not None:
            self.note(f"resuming from {start} prompts ({len(prompts)} remaining)")

        def publish(lens: JacobianLens, split_half: Mapping[str, Mapping[int, float]]) -> JacobianLens:
            save_lens(lens, self.model_id, self.provenance(), split_half=split_half)
            remove_lens_checkpoint(self.model_id, name)
            return lens

        def restore(live: LiveConfig | None) -> None:
            restored = load_local_lens(self.model_id, name)
            if restored is not None:
                self.adopt(restored[0], restored[1], live=live, restore_live=True)

        if base is not None and not prompts:
            # The base already covers every usable document: publish it.
            try:
                kept = (
                    {} if base_sidecar is None
                    else {site: lens_split_half(base_sidecar, site) for site in base.sites}
                )
                publish(base, kept)
                return self.adopt(
                    base, load_local_lens_sidecar(self.model_id, name),
                    live=self.evicted_live, restore_live=self.evicted_early,
                )
            except BaseException:
                if self.evicted_early:
                    restore(self.evicted_live)
                raise

        had_resident = self.residency.resident_lens is not None or self.evicted_early
        resume_live = (
            self.evicted_live if self.evicted_early else self.residency.live_config()
        )
        if self.residency.resident_lens is not None:
            # The estimator takes ownership of ``base`` and converts averages
            # to sums; a resident that aliases it, or a stale replaced lens
            # with its device stacks, would break the one-matrix-set bound.
            self.residency.evict()
        proof_out: list[Any] = []
        try:
            initial = None if base is None else self.resume_base(base, base_sidecar)
            base = None
            fitted = self.estimate(
                prompts, self.identity.layers, start=start, initial=initial,
                proof_out=proof_out,
            )
            merged = fitted.lens
            corpus = self.identity.corpus
            if proof_out and promote_lens_checkpoint(
                self.model_id,
                name=name,
                n_prompts=merged.n_prompts,
                site_layers={site: merged.layers(site) for site in merged.sites},
                corpus_sha256=corpus.sha256,
                corpus_hash_kind=CORPUS_HASH_KIND,
                seq_len=self.request.seq_len,
                d_model=merged.d_model,
                model_fingerprint=self.identity.model_fingerprint,
                _verified_proof=proof_out[0],
            ):
                sidecar = load_local_lens_sidecar(self.model_id, name)
            else:
                merged = publish(merged, fitted.split_half)
                sidecar = load_local_lens_sidecar(self.model_id, name)
            return self.adopt(merged, sidecar, live=resume_live, restore_live=True)
        except BaseException:
            if had_resident:
                restore(resume_live)
            raise


__all__ = [
    "CORPUS_HASH_KIND",
    "LensFitCorpus",
    "LensFitIdentity",
    "LensFitPlan",
    "LensFitRequest",
    "LensFitter",
    "LensResidency",
    "LiveConfig",
    "ResidentLens",
    "SiteLayers",
    "fit_covers_layers",
    "loaded_source_fingerprint",
    "plan_lens_fit",
    "prove_lens_fit_noop",
    "raw_corpus_sha256",
    "resolve_fit_layers",
    "tokenize_lens_corpus",
]
