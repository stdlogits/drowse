"""NeutralBaseline — the per-model neutral means, Mahalanobis whitener and
unit statistics.

All read the one neutral cache
(:func:`~drowse.fit.neutral.load_or_compute_neutral_cache`): the
probe-centering mean is its per-layer ``X.mean(0)``, the whitener's
covariance is built from the same stack, and the unit statistics (a neuron's
or head's neutral mean and scale) and the mid means (the residual after each
block's attention, averaged at the same positions) ride in the same shards,
so one load produces all of them.  :meth:`NeutralBaseline.site_means` is the
centring mean of an S-family read of a residual at a lens site.

Two entry points:

- :meth:`NeutralBaseline.ensure` is unguarded.  A caller that already holds the
  exclusive model lock (steering composition, generation preflight, session
  construction) builds the baseline through it.  It still refuses to *compute*
  while steering hooks are installed, since the neutral forward pass would read
  steered activations and persist them.
- :meth:`layer_means` and :meth:`whitener` are the guarded public getters: a
  build from another thread takes the exclusive lock non-blocking and refuses
  during a generation or inside a steering scope.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from drowse.base.exclusive import ExclusiveGuard
from drowse.errors import ConcurrentExtractionError, DrowseError, NeutralMeansError
from drowse.units.stats import UnitStats

if TYPE_CHECKING:
    from drowse.geometry.whitener import LayerWhitener

_log = logging.getLogger(__name__)

class NeutralBaseline:
    """Lazily built neutral means + whitener for one loaded model."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        layers: Any,
        model_id: str,
        exclusive: ExclusiveGuard,
        steered: Callable[[], bool],
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._layers = layers
        self._model_id = model_id
        self._exclusive = exclusive
        self._steered = steered
        self._means: dict[int, torch.Tensor] = {}
        self._whitener: LayerWhitener | None = None
        self._load_error: BaseException | None = None
        self._on_whitener: list[Callable[["LayerWhitener"], None]] = []
        # The unit statistics of the same load: the stats, or why there are
        # none; both ``None`` until a load (or ``adopt``) settled them.
        self._unit_stats: UnitStats | None = None
        self._units_reason: str | None = None
        # The mid means of the same load, or why there are none.
        self._mid_means: dict[int, torch.Tensor] | None = None
        self._mid_reason: str | None = None

    # ------------------------------------------------------------- state

    @property
    def means(self) -> dict[int, torch.Tensor]:
        """The resident means (``{}`` until built).  No build, no guard."""
        return self._means

    @property
    def resident_whitener(self) -> "LayerWhitener | None":
        """The resident whitener (``None`` until built).  No build, no guard."""
        return self._whitener

    @property
    def unit_stats(self) -> "UnitStats | None":
        """The resident unit statistics (``None`` until built, or when the
        model has none: :attr:`units_unavailable` says why).  No build."""
        return self._unit_stats

    @property
    def units_unavailable(self) -> str | None:
        """Why the loaded cache has no unit statistics (``None`` when it has
        them, or before a load)."""
        return self._units_reason

    @property
    def mid_means(self) -> dict[int, torch.Tensor] | None:
        """The resident mid means (``None`` until built, or when the cache
        has none: :attr:`mid_unavailable` says why).  No build."""
        return self._mid_means

    @property
    def mid_unavailable(self) -> str | None:
        """Why the loaded cache has no mid means (``None`` when it has them,
        or before a load)."""
        return self._mid_reason

    @property
    def _units_settled(self) -> bool:
        return (
            (self._unit_stats is not None or self._units_reason is not None)
            and (self._mid_means is not None or self._mid_reason is not None)
        )

    @property
    def ready(self) -> bool:
        return bool(self._means) and self._whitener is not None

    def on_whitener(self, callback: Callable[["LayerWhitener"], None]) -> None:
        """Call ``callback(whitener)`` whenever a whitener is built."""
        self._on_whitener.append(callback)

    def adopt(
        self, means: dict[int, torch.Tensor], whitener: "LayerWhitener | None",
        *, unit_stats: "UnitStats | None" = None,
        mid_means: "dict[int, torch.Tensor] | None" = None,
    ) -> None:
        """Install an externally computed baseline (means + whitener, and the
        unit statistics and mid means when given; without them each is
        unavailable)."""
        self._means = dict(means)
        self._whitener = whitener
        self._load_error = None
        self._unit_stats = unit_stats
        self._units_reason = (
            None if unit_stats is not None
            else "the adopted neutral baseline carries no unit statistics"
        )
        self._mid_means = None if mid_means is None else dict(mid_means)
        self._mid_reason = (
            None if mid_means is not None
            else "the adopted neutral baseline carries no mid means"
        )
        if whitener is not None:
            for callback in self._on_whitener:
                callback(whitener)

    def reset(self) -> None:
        """Drop the resident baseline (session teardown)."""
        self._means = {}
        self._whitener = None
        self._load_error = None
        self._unit_stats = None
        self._units_reason = None
        self._mid_means = None
        self._mid_reason = None

    # ------------------------------------------------------------ build

    def ensure(self, *, on_progress: Callable[[str], None] | None = None) -> None:
        """Make the means and the whitener resident with one neutral load.

        Unguarded: the caller holds the exclusive model lock.  A failed load
        leaves the means empty (``require_means`` raises) and the whitener
        ``None``; a whitener that cannot be built from loaded activations is
        logged and left ``None``, since every Mahalanobis consumer raises its
        own ``WhitenerError`` without one.  The unit statistics settle from
        the same load (:attr:`unit_stats`, or :attr:`units_unavailable`), and
        so do the mid means (:attr:`mid_means`, or :attr:`mid_unavailable`).
        """
        if self.ready and self._units_settled:
            return
        if self._steered():
            raise ConcurrentExtractionError(
                "artifact-producing model operations cannot run inside an "
                "active steering scope; exit session.steering() first"
            )
        from drowse.geometry.whitener import LayerWhitener
        from drowse.fit.neutral import load_or_compute_neutral_cache

        try:
            cache = load_or_compute_neutral_cache(
                self._model, self._tokenizer, self._layers,
                model_id=self._model_id, on_progress=on_progress,
            )
        except Exception as exc:
            self._load_error = exc
            _log.warning(
                "neutral baseline: load failed (%s); Mahalanobis-only fitting, "
                "projection, and probe reads are unavailable. Error: %s",
                type(exc).__name__, exc,
            )
            return
        self._load_error = None
        acts = cache.activations
        if self._unit_stats is None and self._units_reason is None:
            if cache.units.tensors is None:
                self._units_reason = cache.units.reason or "unavailable"
            else:
                self._unit_stats = UnitStats.from_tensors(
                    cache.units.tensors, n_documents=cache.units.n_documents,
                )
        if self._mid_means is None and self._mid_reason is None:
            if cache.mid.means is None:
                self._mid_reason = cache.mid.reason or "unavailable"
            else:
                self._mid_means = dict(cache.mid.means)
        if not self._means:
            self._means = {idx: X.mean(dim=0) for idx, X in acts.items()}
        if self._whitener is not None:
            return
        try:
            whitener = LayerWhitener.from_neutral_activations(acts, self._means)
        except Exception as exc:
            _log.warning(
                "whitener: build failed (%s); Mahalanobis-only fitting, "
                "projection, and probe reads are unavailable. Error: %s",
                type(exc).__name__, exc,
            )
            return
        self._whitener = whitener
        for callback in self._on_whitener:
            callback(whitener)

    def require_means(self) -> dict[int, torch.Tensor]:
        """The means, built if absent (unguarded); raises when they cannot be."""
        if not self._means:
            self.ensure()
        if not self._means:
            if self._load_error is not None:
                raise DrowseError(
                    "failed to build neutral layer means"
                ) from self._load_error
            raise DrowseError("neutral layer-mean bootstrap returned empty")
        return self._means

    def site_means(self, site: str) -> dict[int, torch.Tensor]:
        """The neutral mean at a lens site, ``{layer: [D]}`` fp32 CPU: the
        centring mean of an S-family read of a residual there.

        ``out`` is the block-output mean (:attr:`means`, the whitener's and
        the monitor's centring mean); ``mid`` the residual after each
        block's attention at the same positions.  Built on first use under
        the exclusive section (reentrant: callers inside it, a generation's
        own thread included, build in place; a build still refuses inside a
        steering scope).  Raises :class:`~drowse.errors.NeutralMeansError`
        (422) when the site has no means: the baseline cannot be built, or
        the model's blocks are outside the description (no ``mid``).  There
        is no uncentred fallback.
        """
        if site not in ("out", "mid"):
            raise ValueError(f"site must be 'out' or 'mid', got {site!r}")
        if not self._means or not self._units_settled:
            with self._exclusive(
                "neutral means requested while another model use is in flight",
            ):
                self.ensure()
        if not self._means:
            detail = f": {self._load_error}" if self._load_error is not None else ""
            raise NeutralMeansError(
                f"the neutral baseline for {self._model_id} did not build{detail}; "
                "residual reads are centred on its means and refuse without them"
            )
        if site == "out":
            return self._means
        if self._mid_means is None:
            raise NeutralMeansError(
                f"no neutral mid means for {self._model_id} "
                f"({self._mid_reason or 'the neutral cache has none'}); an S read "
                "of a mid residual is centred on them and refuses without them"
            )
        return self._mid_means

    # ---------------------------------------------------- guarded getters

    def layer_means(self) -> dict[int, torch.Tensor]:
        """Per-layer neutral means, built on first use under the exclusive lock.

        Raises :class:`~drowse.errors.DrowseError` when the neutral cache
        cannot be built, and :class:`~drowse.errors.ConcurrentExtractionError`
        when a build is needed while a model op, a generation, or a steering
        scope is active.
        """
        if not self._means:
            with self._exclusive(
                "layer_means requested while another model use is in flight",
                phase_msg="layer_means requested while generation is in flight",
            ):
                return self.require_means()
        return self._means

    def whitener(self) -> "LayerWhitener | None":
        """The Mahalanobis whitener, built on first use under the exclusive lock.

        ``None`` when it cannot be built (the consumers raise ``WhitenerError``);
        :class:`~drowse.errors.ConcurrentExtractionError` when a build is needed
        while a model op, a generation, or a steering scope is active.
        """
        if self._whitener is None:
            with self._exclusive(
                "whitener requested while another model use is in flight",
                phase_msg="whitener requested while generation is in flight",
            ):
                self.ensure()
        return self._whitener
