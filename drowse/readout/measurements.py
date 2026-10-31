"""The measurement envelope — the one wire record for read-side data.

Every surface that ships per-token, aggregate, or replayed instrument
data — the native WS ``token``/``done`` frames, loom token rows, the
token-replay endpoints — carries this single versioned shape.

Shape::

    {
      "version": 1,
      "scope": "token" | "aggregate" | "replay",
      "provenance": "captured" | "replayed",
      "scores": {probe: axis0, ...},              # flat cross-family view
      "per_layer_scores": {layer: {probe: v}},    # optional heatmap view
      "instruments": {
        "geometry": {"readings": {name: ProbeReading.to_dict()}},
        "lens": {
          "binding": {"sources": {"R": ..., "S": ...}, "steering": ...},
          "readings": {name: ...},                # attached jlens/ and slens/ probes
          "readout": {"layers": [...], "aggregate": [...]}   # native top-k (R)
        },
        "sae": {
          "binding": {"source": ..., "steering": ..., "layer": N},
          "readings": {name: ...},                # attached sae/<id> probes
          "readout": {"features": [...]}          # native top-k discovery
        },
        "units": {
          "binding": {"source": null, "steering": ...},
          "readings": {name: ...}                 # attached neuron/… head/… probes
        }
      },
      "anatomy": {...}                            # replay only: token anatomy
    }

A family's ``readings`` are its *attached probes'* values; its
``readout`` is the *native discovery* surface (per-layer top-k matrix / feature list).  ``scores``
and ``per_layer_scores`` stay envelope-level because their consumers
(transcript tinting, the loom heatmap) key probes across families by
name — they are flat views derived from the same readings, never extra
data.  ``binding`` records what the family was measuring (source
identity + recipe steering) so historical rows stay interpretable after
a source switch.  The lens channel carries two lens families, so its
binding names a source per family that produced its data
(``sources.R``: the ``jlens/`` readings and the native readout;
``sources.S``: the ``slens/`` readings), told apart by the reading's unit.

``anatomy`` rides only a ``replay`` envelope (``session.token_anatomy``,
:mod:`drowse.loom.node_replay`): one token's layers read through both lens
families (``reads_as``, ``attention``, ``mlp``, ``steering``, each a lens
read with its halves), the units that pushed the token and the top heads'
attention.  It is a record of the replay forward, not of a probe, so it sits
beside ``instruments`` rather than in a family channel; its ``binding``
names the lens sources it read through.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, cast

from typing_extensions import NotRequired, TypedDict

from drowse.readout.types import (
    UNIT_MEAN_FOOTPRINT_Z,
    UNIT_MEAN_TOKEN_PROBABILITY,
    ProbeReading,
    Reading,
    ScalarReading,
    reading_axis0,
)

MEASUREMENTS_VERSION = 2

MeasurementScope = Literal["token", "aggregate", "replay"]
MeasurementProvenance = Literal["captured", "replayed"]


# --------------------------------------------------------------------------
# The envelope, declared
# --------------------------------------------------------------------------
# The Python-side declaration of the wire contract (the dashboard types are
# generated from it); ``tests/readout/test_measurements_envelope.py`` pins the exact
# key sets of a token frame, a done aggregate, and all three replay envelopes,
# so wire drift is a test failure rather than a dashboard bug.

class MeasurementBinding(TypedDict):
    """What a family was measuring: source identity + recipe steering."""

    source: str | None
    steering: str | None
    layer: NotRequired[int | None]


class LensBinding(TypedDict):
    """What the lens channel was measuring: the source of each lens family
    that produced its data, keyed ``R`` (``jlens/`` readings, the native
    readout) and ``S`` (``slens/`` readings), present only for a family that
    did; and the recipe steering."""

    sources: dict[str, str | None]
    steering: str | None


#: Which lens family produced a lens-channel reading, by its unit.
LENS_FAMILY_BY_UNIT = {
    UNIT_MEAN_TOKEN_PROBABILITY: "R",
    UNIT_MEAN_FOOTPRINT_Z: "S",
}


class DepthSummaryDict(TypedDict):
    """Wire form of ``instruments.types.DepthSummary``.

    ``basis`` travels with the numbers because ``center`` means three
    mathematically unrelated things across the families.
    """

    center: list[float]
    spread: list[float]
    basis: str


class ScalarReadingDict(TypedDict):
    """Wire form of ``ScalarReading.to_dict()`` — the lens/SAE reading.

    One value with an explicit ``unit``, its per-layer trace, and a depth
    summary.  No geometry fields: a readout channel has no subspace behind
    it.  ``meta`` is emitted only when the producer attached one.
    """

    value: float
    unit: str
    per_layer: dict[str, float]
    depth: DepthSummaryDict | None
    meta: NotRequired[dict[str, Any]]


class ProbeReadingDict(TypedDict):
    """Wire form of ``ProbeReading.to_dict()`` — the geometry reading.

    The complete key set that serializer emits, in its order.  Declaring it
    here (rather than as an anonymous ``dict[str, Any]`` on the channel)
    is what lets the generated dashboard types name the geometry reading
    instead of falling back to an opaque record.
    """

    fraction: float
    nearest: list[tuple[str, float]]
    coords: list[float]
    residual: float
    fraction_per_layer: dict[str, float]
    coords_per_layer: dict[str, list[float]]
    residual_per_layer: dict[str, float]
    # The live serializer always emits these five, but a *persisted* loom
    # row written before they existed does not, and the same wire type
    # covers both — so they are optional on read.
    assignment: NotRequired[list[tuple[str, float]]]
    membership: NotRequired[float]
    depth_com: NotRequired[list[float]]
    depth_spread: NotRequired[list[float]]
    subspace_coords_per_layer: NotRequired[dict[str, list[float]]]


class LensReadoutToken(TypedDict):
    token: str
    id: int
    logprob: float


class LensReadoutLayer(TypedDict):
    layer: int
    tokens: list[LensReadoutToken]


class LensAggregateToken(TypedDict):
    token: str
    strength: float
    com: float
    spread: float


class LensReadout(TypedDict):
    layers: list[LensReadoutLayer]
    aggregate: list[LensAggregateToken]


class SaeFeature(TypedDict):
    id: int
    activation: float
    label: str | None
    max_act: float | None


class SaeReadout(TypedDict):
    features: list[SaeFeature]


class GeometryChannel(TypedDict):
    # Never null: ``build_measurements`` creates the channel only when there
    # are geometry readings to put in it, so the historical ``| None`` was
    # defensive over-typing that every consumer then had to defend against.
    readings: dict[str, ProbeReadingDict]
    binding: NotRequired[MeasurementBinding]


class LensChannel(TypedDict):
    binding: LensBinding
    readings: NotRequired[dict[str, ScalarReadingDict]]
    readout: NotRequired[LensReadout]


class SaeChannel(TypedDict):
    binding: MeasurementBinding
    readings: NotRequired[dict[str, ScalarReadingDict]]
    readout: NotRequired[SaeReadout]


class UnitsChannel(TypedDict):
    """The units family: attached ``neuron/…`` / ``head/…`` probe readings.
    ``binding.source`` is always ``None`` (units are the model's own parts)."""

    binding: MeasurementBinding
    readings: dict[str, ScalarReadingDict]


class TokenScoreDict(TypedDict):
    """One ranked token of a lens read half (``lens.read.TokenScore``)."""

    token_id: int
    token: str
    value: float


class FamilyReadDict(TypedDict):
    """One family's half of a lens read (``lens.read.FamilyRead``):
    ``weight`` is the depth rule's, ``centred`` whether it read ``x − μ``
    (the S half of a residual), ``absent`` why it was not read."""

    family: Literal["R", "S"]
    weight: float
    unit: Literal["probability", "footprint_z"]
    lens: str | None
    centred: bool
    tokens: list[TokenScoreDict]
    absent: str | None


class LensReadDict(TypedDict):
    """The read of one vector at ``(layer, site)`` (``lens.read.LensRead``);
    ``of`` is ``residual`` or ``direction``."""

    layer: int
    site: Literal["out", "mid"]
    depth: float
    of: Literal["residual", "direction"]
    r: FamilyReadDict
    s: FamilyReadDict


class AnatomyUnit(TypedDict):
    """A unit's direct contribution to the anatomy token's logit through the
    R lens; ``activation`` is a neuron's raw activation or a head's write
    norm, and ``activation_over_scale`` the same over the unit's soft neutral
    scale (the units family's unit: the number its probe, unit activity and
    the unit view show), ``None`` without unit statistics.  Both are ``None``
    in the cross-layer list."""

    unit: str
    kind: Literal["neuron", "head"]
    layer: int
    index: int
    contribution: float
    activation: float | None
    activation_over_scale: float | None


class AnatomyLayer(TypedDict):
    """One layer of a token's anatomy: the residual at ``out`` (``reads_as``,
    S centred), the attention write at ``mid``, the MLP write and the
    steering injection (``None`` when none) at ``out``, read as directions;
    the layer's top neurons and heads by contribution."""

    layer: int
    depth: float
    reads_as: LensReadDict
    attention: LensReadDict
    mlp: LensReadDict
    steering: LensReadDict | None
    neurons: list[AnatomyUnit]
    heads: list[AnatomyUnit]


class AnatomyPattern(TypedDict):
    """A top head's attention from the token's producing position over the
    context (``weights[j]`` for position ``j``; position 0 is the sink)."""

    unit: str
    layer: int
    head: int
    contribution: float
    weights: list[float]


class AnatomyToken(TypedDict):
    """The anatomy's token: raw index, id and piece, its producing
    ``position`` in the replayed sequence, the prompt render's length and the
    model's probability of it there."""

    raw_index: int
    token_id: int
    token: str
    position: int
    prompt_length: int
    probability: float


class TokenAnatomy(TypedDict):
    """One token, every layer (``session.token_anatomy``)."""

    token: AnatomyToken
    binding: LensBinding
    layers: list[AnatomyLayer]
    units: list[AnatomyUnit]
    units_absent: str | None
    patterns: list[AnatomyPattern] | None
    context: list[str] | None


class Instruments(TypedDict):
    geometry: NotRequired[GeometryChannel]
    lens: NotRequired[LensChannel]
    sae: NotRequired[SaeChannel]
    units: NotRequired[UnitsChannel]


class Measurements(TypedDict):
    """The one versioned read-side wire record."""

    version: int
    scope: MeasurementScope
    provenance: MeasurementProvenance
    instruments: Instruments
    scores: NotRequired[dict[str, float]]
    per_layer_scores: NotRequired[dict[str, dict[str, float]]]
    anatomy: NotRequired[TokenAnatomy]


class MeasurementsEnvelope(TypedDict):
    """The replay endpoints' response body."""

    measurements: Measurements | None


def _axis0(readings: Mapping[str, Reading]) -> dict[str, float]:
    return {
        str(name): round(reading_axis0(reading), 6)
        for name, reading in readings.items()
    }


def _readings_dict(
    readings: Mapping[str, Reading] | None,
) -> dict[str, Any] | None:
    """Serialize a family's attached-probe readings in ITS OWN shape.

    Geometry emits the full ``ProbeReading``; the single-axis families emit
    the native ``ScalarReading`` (``{value, unit, per_layer, depth}``).  The
    eight constant geometry fields a synthesized ``ProbeReading`` used to
    carry for lens/SAE probes — ``fraction`` 0, ``residual`` 0, empty
    ``nearest``/``assignment``, ``membership`` 1.0 and three empty per-layer
    maps — are off the wire: they were constants masquerading as
    measurements, and the client had to infer the unit they omitted.
    """
    if not readings:
        return None
    return {str(name): reading.to_dict() for name, reading in readings.items()}


def build_measurements(
    *,
    scope: str,
    provenance: str = "captured",
    geometry_readings: Mapping[str, ProbeReading] | None = None,
    geometry_binding: Mapping[str, Any] | None = None,
    lens_readings: Mapping[str, ScalarReading] | None = None,
    sae_readings: Mapping[str, ScalarReading] | None = None,
    per_layer_scores: Mapping[str, Mapping[str, float]] | None = None,
    lens_readout: Mapping[int, list[tuple[str, float]]] | None = None,
    lens_aggregate: list[tuple[str, float, float, float]] | None = None,
    lens_token_ids: Mapping[int, list[int]] | None = None,
    lens_sources: Mapping[str, str | None] | None = None,
    sae_features: list[tuple[int, float, str | None, float | None]] | None = None,
    sae_source: str | None = None,
    sae_layer: int | None = None,
    units_readings: Mapping[str, ScalarReading] | None = None,
    anatomy: TokenAnatomy | None = None,
    steering: str | None = None,
) -> Measurements | None:
    """Build one measurement envelope; ``None`` when nothing measured.

    JSON-safe throughout (floats rounded/floored exactly as the historical
    ``captured`` serializers did, so persisted loom rows stay comparable
    across the 5.x boundary).  ``lens_sources`` is ``{"R": label, "S":
    label}``; the lens binding keeps the families that produced data.
    ``anatomy`` (built by :mod:`drowse.loom.node_replay`) rides a replay
    envelope only.
    """
    if anatomy is not None and scope != "replay":
        raise ValueError("a token anatomy rides a replay envelope only")
    instruments: dict[str, Any] = {}
    scores: dict[str, float] = {}
    lens_families: set[str] = set()

    if geometry_readings:
        scores.update(_axis0(geometry_readings))
        geometry_channel: dict[str, Any] = {
            "readings": _readings_dict(geometry_readings),
        }
        # Opt-in only (the replay envelope records its applied steering,
        # mirroring lens/sae): live token/aggregate envelopes stay
        # binding-less on the geometry channel — the recipe already lives
        # on the loom node.
        if geometry_binding is not None:
            geometry_channel["binding"] = dict(geometry_binding)
        instruments["geometry"] = geometry_channel

    lens_channel: dict[str, Any] = {}
    if lens_readings:
        scores.update(_axis0(lens_readings))
        lens_channel["readings"] = _readings_dict(lens_readings)
        for reading in lens_readings.values():
            family = LENS_FAMILY_BY_UNIT.get(reading.unit)
            if family is None:
                raise ValueError(f"no lens family reads unit {reading.unit!r}")
            lens_families.add(family)
    if lens_readout or lens_aggregate:
        lens_families.add("R")
        id_rows = lens_token_ids or {}
        layers: list[dict[str, Any]] = []
        for layer, row in sorted(lens_readout.items() if lens_readout else ()):
            token_ids = id_rows.get(int(layer), [])
            tokens: list[dict[str, Any]] = []
            for index, (token, probability) in enumerate(row):
                token_id = token_ids[index] if index < len(token_ids) else -1
                # Every producer holds the per-layer probability ``p_l`` (the
                # readout's one unit) and every consumer wants it back; this
                # is the ONE conversion into the wire's logprob key, and the
                # floor keeps strict JSON finite.
                logprob = math.log(max(float(probability), 1e-45))
                tokens.append({
                    "token": str(token),
                    "id": int(token_id),
                    "logprob": float(logprob),
                })
            layers.append({"layer": int(layer), "tokens": tokens})
        lens_channel["readout"] = {
            "layers": layers,
            "aggregate": [
                {
                    "token": str(token),
                    "strength": float(strength),
                    "com": float(com),
                    "spread": float(spread),
                }
                for token, strength, com, spread in (lens_aggregate or ())
            ],
        }
    if lens_channel:
        known = lens_sources or {}
        lens_channel["binding"] = {
            "sources": {family: known.get(family) for family in sorted(lens_families)},
            "steering": steering,
        }
        instruments["lens"] = lens_channel

    sae_channel: dict[str, Any] = {}
    if sae_readings:
        scores.update(_axis0(sae_readings))
        sae_channel["readings"] = _readings_dict(sae_readings)
    if sae_features:
        sae_channel["readout"] = {
            "features": [
                {
                    "id": int(row[0]),
                    "activation": float(row[1]),
                    "label": row[2],
                    "max_act": (
                        float(row[3])
                        if len(row) > 3 and row[3] is not None
                        else None
                    ),
                }
                for row in sae_features
            ],
        }
    if sae_channel:
        sae_channel["binding"] = {
            "source": sae_source,
            "steering": steering,
            "layer": int(sae_layer) if sae_layer is not None else None,
        }
        instruments["sae"] = sae_channel

    if units_readings:
        scores.update(_axis0(units_readings))
        instruments["units"] = {
            "binding": {"source": None, "steering": steering},
            "readings": _readings_dict(units_readings),
        }

    if not instruments and anatomy is None:
        return None

    envelope: Measurements = {
        "version": MEASUREMENTS_VERSION,
        "scope": cast(MeasurementScope, scope),
        "provenance": cast(MeasurementProvenance, provenance),
        "instruments": cast(Instruments, instruments),
    }
    if scores:
        envelope["scores"] = scores
    if per_layer_scores:
        envelope["per_layer_scores"] = {
            str(layer): dict(row) for layer, row in per_layer_scores.items()
        }
    if anatomy is not None:
        envelope["anatomy"] = anatomy
    return envelope


class EnvelopeSources(TypedDict):
    """The source fields an envelope binds: :func:`build_measurements`'
    ``lens_sources`` / ``sae_source`` / ``sae_layer`` keywords."""

    lens_sources: dict[str, str | None]
    sae_source: str | None
    sae_layer: int | None


def envelope_sources(
    lens_sources: Mapping[str, str | None] | None,
    sae_live: Mapping[str, Any] | None,
    *,
    lens: bool = True,
    sae: bool = True,
) -> EnvelopeSources:
    """The envelope source fields of the lens families and the live SAE.

    ``lens_sources`` is the lens instrument's
    ``measurement_sources()`` (``{"R": label, "S": label}``, the lenses its
    run pinned); ``sae_live`` the SAE's live config (``None`` while off).
    ``lens`` / ``sae`` gate each family: the aggregate envelope binds a
    family's sources only when it carries that family's readings.  An
    inactive or gated-off family binds nothing.
    """
    sae_cfg = sae_live if sae and isinstance(sae_live, Mapping) else None
    layer = sae_cfg.get("layer") if sae_cfg is not None else None
    return {
        "lens_sources": dict(lens_sources or {}) if lens else {},
        "sae_source": sae_cfg.get("source") if sae_cfg is not None else None,
        "sae_layer": int(layer) if layer is not None else None,
    }


__all__ = [
    "AnatomyLayer",
    "AnatomyPattern",
    "AnatomyToken",
    "AnatomyUnit",
    "DepthSummaryDict",
    "FamilyReadDict",
    "EnvelopeSources",
    "GeometryChannel",
    "Instruments",
    "LENS_FAMILY_BY_UNIT",
    "LensAggregateToken",
    "LensBinding",
    "LensChannel",
    "LensReadDict",
    "LensReadout",
    "LensReadoutLayer",
    "LensReadoutToken",
    "MEASUREMENTS_VERSION",
    "MeasurementBinding",
    "MeasurementProvenance",
    "MeasurementScope",
    "Measurements",
    "MeasurementsEnvelope",
    "ProbeReadingDict",
    "SaeChannel",
    "SaeFeature",
    "SaeReadout",
    "ScalarReadingDict",
    "TokenAnatomy",
    "TokenScoreDict",
    "UnitsChannel",
    "build_measurements",
    "envelope_sources",
]
