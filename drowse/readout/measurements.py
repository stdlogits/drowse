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
stays envelope-level because its consumers (transcript tinting, the tree
filter, the reply trace) key probes across families by name — a flat view
of each reading's primary scalar, never extra data.  A reading's per-layer
trace lives in the reading alone (``coords_per_layer`` / ``per_layer``):
there is no second, layer-major copy.  ``binding`` records what the family was measuring (source
identity + recipe steering) so historical rows stay interpretable after
a source switch.  The lens channel carries two lens families, so its
binding names a source per family that produced its data
(``sources.R``: the ``jlens/`` readings and the native readout;
``sources.S``: the ``slens/`` readings), told apart by the reading's unit.

A loom token row persists the token envelope through
:func:`token_row_measurements`, the one row projection: each geometry
reading's summary (no per-layer traces, never the inspector's subspace
coordinates), the scalar readings whole, the lens readout's layer-aggregated
chips without its per-layer matrix, and every float rounded to
:data:`ROW_DECIMALS` places (the views show three).  A generation that sets
``persist_per_layer_scores`` keeps the geometry traces and the readout's
layer rows on its rows too; otherwise the per-layer views replay the token
(``token_readout``, ``token_anatomy``) while a model is loaded.  The live
``token`` frame carries the whole envelope.

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

#: One layer-aggregated lens chip, ``(token, strength, com, spread, token_id)``:
#: the decoded token and its vocabulary id (two ids can decode to the same
#: text, e.g. byte pieces that both render as U+FFFD; consumers key by the id).
LensAggregateRow = tuple[str, float, float, float, int]
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

    A loom token row keeps the summary (:func:`token_row_measurements`): the
    three per-layer traces only when its generation set
    ``persist_per_layer_scores``, and ``subspace_coords_per_layer`` (the
    probe inspector's live trail) never.
    """

    fraction: float
    nearest: list[tuple[str, float]]
    coords: list[float]
    residual: float
    fraction_per_layer: NotRequired[dict[str, float]]
    coords_per_layer: NotRequired[dict[str, list[float]]]
    residual_per_layer: NotRequired[dict[str, float]]
    assignment: list[tuple[str, float]]
    membership: float
    depth_com: list[float]
    depth_spread: list[float]
    subspace_coords_per_layer: NotRequired[dict[str, list[float]]]


class LensReadoutToken(TypedDict):
    token: str
    id: int
    logprob: float


class LensReadoutLayer(TypedDict):
    layer: int
    tokens: list[LensReadoutToken]


class LensAggregateToken(TypedDict):
    """One layer-aggregated chip; ``id`` is its vocabulary id (two ids may
    decode to the same ``token`` text)."""

    token: str
    id: int
    strength: float
    com: float
    spread: float


class LensReadout(TypedDict):
    """The native readout: the per-layer top-k matrix and its
    layer-aggregated chips.  A loom token row keeps ``layers`` only when its
    generation set ``persist_per_layer_scores``."""

    layers: NotRequired[list[LensReadoutLayer]]
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
    lens_readout: Mapping[int, list[tuple[str, float]]] | None = None,
    lens_aggregate: list[LensAggregateRow] | None = None,
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
                    "id": int(token_id),
                    "strength": float(strength),
                    "com": float(com),
                    "spread": float(spread),
                }
                for token, strength, com, spread, token_id in (lens_aggregate or ())
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
    if anatomy is not None:
        envelope["anatomy"] = anatomy
    return envelope


# --------------------------------------------------------------------------
# The loom token row's projection
# --------------------------------------------------------------------------

#: Decimal places a loom token row keeps for every float it persists: one
#: more than any view shows (coordinates, shares and distances print three).
ROW_DECIMALS = 4

#: A geometry reading's per-layer traces, kept on a row only on request.
GEOMETRY_TRACE_KEYS = ("fraction_per_layer", "coords_per_layer", "residual_per_layer")

#: The keys a row's geometry reading carries without its traces.
GEOMETRY_SUMMARY_KEYS = (
    "fraction", "nearest", "coords", "residual", "assignment", "membership",
    "depth_com", "depth_spread",
)


def _r(value: Any) -> float:
    return round(float(value), ROW_DECIMALS)


def _row_ranked(pairs: list[Any]) -> list[list[Any]]:
    return [[str(label), _r(value)] for label, value in pairs]


def _row_probe_reading(reading: Mapping[str, Any], per_layer: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "fraction": _r(reading["fraction"]),
        "nearest": _row_ranked(reading["nearest"]),
        "coords": [_r(c) for c in reading["coords"]],
        "residual": _r(reading["residual"]),
        "assignment": _row_ranked(reading["assignment"]),
        "membership": _r(reading["membership"]),
        "depth_com": [_r(c) for c in reading["depth_com"]],
        "depth_spread": [_r(s) for s in reading["depth_spread"]],
    }
    if per_layer:
        out["fraction_per_layer"] = {
            layer: _r(v) for layer, v in reading["fraction_per_layer"].items()
        }
        out["coords_per_layer"] = {
            layer: [_r(c) for c in v] for layer, v in reading["coords_per_layer"].items()
        }
        out["residual_per_layer"] = {
            layer: _r(v) for layer, v in reading["residual_per_layer"].items()
        }
    return out


def _row_scalar_reading(reading: Mapping[str, Any]) -> dict[str, Any]:
    # A scalar reading's trace is a handful of values (a lens probe's band, a
    # unit's or SAE feature's one layer), and no replay returns a lens
    # probe's: the row keeps it.
    depth = reading["depth"]
    out: dict[str, Any] = {
        "value": _r(reading["value"]),
        "unit": reading["unit"],
        "per_layer": {layer: _r(v) for layer, v in reading["per_layer"].items()},
        "depth": None if depth is None else {
            "center": [_r(c) for c in depth["center"]],
            "spread": [_r(s) for s in depth["spread"]],
            "basis": depth["basis"],
        },
    }
    if "meta" in reading:
        out["meta"] = {
            key: _r(v) if isinstance(v, float) else v
            for key, v in reading["meta"].items()
        }
    return out


def token_row_measurements(
    envelope: Measurements, *, per_layer: bool,
) -> Measurements:
    """The loom token row's form of a token envelope: the one row projection.

    A row keeps what the transcript, the reply trace and the token view's
    summary read, once: each geometry reading's summary (without its
    per-layer traces), each scalar reading whole (its trace is a few values),
    the lens readout's layer-aggregated chips, the SAE features, the bindings
    and the flat ``scores``, every float at :data:`ROW_DECIMALS` places.
    ``per_layer`` (the generation's ``persist_per_layer_scores``) also keeps
    the geometry traces and the lens readout's ``layers``.  The probe
    inspector's ``subspace_coords_per_layer`` is a live trail and is never
    kept.  Without the traces, the per-layer views replay the token while a
    model is loaded.
    """
    instruments = envelope["instruments"]
    out: dict[str, Any] = {}
    geometry = instruments.get("geometry")
    if geometry is not None:
        channel: dict[str, Any] = {
            "readings": {
                name: _row_probe_reading(reading, per_layer)
                for name, reading in geometry["readings"].items()
            },
        }
        if "binding" in geometry:
            channel["binding"] = dict(geometry["binding"])
        out["geometry"] = channel
    lens = instruments.get("lens")
    if lens is not None:
        lens_row: dict[str, Any] = {
            "binding": {
                "sources": dict(lens["binding"]["sources"]),
                "steering": lens["binding"]["steering"],
            },
        }
        if "readings" in lens:
            lens_row["readings"] = {
                name: _row_scalar_reading(reading)
                for name, reading in lens["readings"].items()
            }
        if "readout" in lens:
            readout = lens["readout"]
            readout_row: dict[str, Any] = {
                "aggregate": [
                    {
                        "token": chip["token"],
                        "id": chip["id"],
                        "strength": _r(chip["strength"]),
                        "com": _r(chip["com"]),
                        "spread": _r(chip["spread"]),
                    }
                    for chip in readout["aggregate"]
                ],
            }
            if per_layer and "layers" in readout:
                readout_row["layers"] = [
                    {
                        "layer": row["layer"],
                        "tokens": [
                            {"token": t["token"], "id": t["id"], "logprob": _r(t["logprob"])}
                            for t in row["tokens"]
                        ],
                    }
                    for row in readout["layers"]
                ]
            lens_row["readout"] = readout_row
        out["lens"] = lens_row
    sae = instruments.get("sae")
    if sae is not None:
        sae_row: dict[str, Any] = {"binding": dict(sae["binding"])}
        if "readings" in sae:
            sae_row["readings"] = {
                name: _row_scalar_reading(reading)
                for name, reading in sae["readings"].items()
            }
        if "readout" in sae:
            sae_row["readout"] = {
                "features": [
                    {
                        "id": feature["id"],
                        "activation": _r(feature["activation"]),
                        "label": feature["label"],
                        "max_act": (
                            None if feature["max_act"] is None
                            else _r(feature["max_act"])
                        ),
                    }
                    for feature in sae["readout"]["features"]
                ],
            }
        out["sae"] = sae_row
    units = instruments.get("units")
    if units is not None:
        out["units"] = {
            "binding": dict(units["binding"]),
            "readings": {
                name: _row_scalar_reading(reading)
                for name, reading in units["readings"].items()
            },
        }
    row: Measurements = {
        "version": envelope["version"],
        "scope": envelope["scope"],
        "provenance": envelope["provenance"],
        "instruments": cast(Instruments, out),
    }
    if "scores" in envelope:
        row["scores"] = {name: _r(v) for name, v in envelope["scores"].items()}
    return row


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
    "GEOMETRY_SUMMARY_KEYS",
    "GEOMETRY_TRACE_KEYS",
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
    "ROW_DECIMALS",
    "SaeChannel",
    "SaeFeature",
    "SaeReadout",
    "ScalarReadingDict",
    "TokenAnatomy",
    "TokenScoreDict",
    "UnitsChannel",
    "build_measurements",
    "envelope_sources",
    "token_row_measurements",
]
