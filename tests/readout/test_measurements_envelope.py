"""Golden payloads for the one read-side wire record.

The measurement envelope (``readout/measurements.py``) is carried by the native
WS ``token`` and ``done`` frames, persisted loom token rows, and all three
replay endpoints.  Its contract used to live only in hand-mirrored
TypeScript, which is how a never-read aggregate block and a unit no producer
held both survived unnoticed.

These tests assert **exact key sets**, so wire drift is a test failure rather
than a dashboard bug.  They also pin the 5.x/6.x reading split: geometry
carries the full whitened ``ProbeReading``, the single-axis families carry
their native ``ScalarReading`` (``value``/``unit``/``per_layer``/``depth``)
and NOT eight constant geometry fields.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.lens import lens_kwargs

from drowse.readout.types import DepthSummary, ProbeReading, ScalarReading
from drowse.readout.measurements import (
    MEASUREMENTS_VERSION,
    build_measurements,
    envelope_sources,
)

# The eight fields a synthesized ``ProbeReading`` used to ship for every
# lens/SAE probe on every token — constants masquerading as measurements.
_FAKE_GEOMETRY_FIELDS = frozenset({
    "fraction", "residual", "nearest", "assignment", "membership",
    "fraction_per_layer", "residual_per_layer", "subspace_coords_per_layer",
})


def _geometry_reading() -> ProbeReading:
    return ProbeReading(
        fraction=0.42,
        nearest=[("formal", 1.25)],
        coords=(0.61, -0.2),
        residual=0.03,
        coords_per_layer={12: (0.5, -0.1), 18: (0.7, -0.3)},
        depth_com=(0.55, 0.5),
        depth_spread=(0.1, 0.2),
    )


def _lens_reading() -> ScalarReading:
    return ScalarReading(
        value=0.0125,
        unit="mean_token_probability",
        per_layer={12: 0.01, 18: 0.015},
        depth=DepthSummary(
            center=(0.62,), spread=(0.08,), basis="readout_probability_mass",
        ),
    )


def _sae_reading() -> ScalarReading:
    return ScalarReading(
        value=0.84,
        unit="activation_over_max",
        per_layer={14: 0.84},
        depth=DepthSummary(
            center=(0.56,), spread=(0.0,), basis="single_layer",
        ),
    )


# ---------------------------------------------------------------------------
# The native reading shapes
# ---------------------------------------------------------------------------

class TestReadingShapes:
    def test_scalar_reading_wire_keys(self) -> None:
        wire = _lens_reading().to_dict()
        assert set(wire) == {"value", "unit", "per_layer", "depth"}
        assert wire["value"] == pytest.approx(0.0125)
        assert wire["unit"] == "mean_token_probability"
        assert wire["per_layer"] == {"12": 0.01, "18": 0.015}
        assert set(wire["depth"]) == {"center", "spread", "basis"}
        assert wire["depth"]["basis"] == "readout_probability_mass"

    def test_scalar_reading_carries_no_geometry_fields(self) -> None:
        """The 171 bytes of constants per probe per token are off the wire."""
        wire = _lens_reading().to_dict()
        assert not (_FAKE_GEOMETRY_FIELDS & set(wire))

    def test_depth_basis_is_family_specific(self) -> None:
        """``depth_com`` means three unrelated things, so the mass names
        itself — a consumer must never compare centers across bases."""
        assert (
            _lens_reading().to_dict()["depth"]["basis"]
            != _sae_reading().to_dict()["depth"]["basis"]
        )

    def test_compat_projection_reintroduces_them_deliberately(self) -> None:
        """``to_probe_reading`` survives ONLY at the two compatibility
        boundaries; there it must still produce the historical shape."""
        compat = _lens_reading().to_probe_reading().to_dict()
        assert _FAKE_GEOMETRY_FIELDS <= set(compat)
        assert compat["coords"] == [pytest.approx(0.0125)]
        assert compat["fraction"] == 0.0
        assert compat["membership"] == 1.0


# ---------------------------------------------------------------------------
# Golden envelopes
# ---------------------------------------------------------------------------

class TestTokenFrame:
    @staticmethod
    def _envelope() -> dict[str, Any]:
        env = build_measurements(
            scope="token",
            geometry_readings={"formal.casual": _geometry_reading()},
            lens_readings={"jlens/fake": _lens_reading()},
            sae_readings={"sae/548": _sae_reading()},
            lens_readout={12: [(" a", 0.78)], 18: [(" b", 0.6)]},
            lens_aggregate=[(" a", 0.41, 0.31, 0.05, 7)],
            lens_token_ids={12: [5], 18: [7]},
            lens_sources={"R": "local:default", "S": "rs-s"},
            sae_features=[(42, 3.5, "fruit", 121.1)],
            sae_source="saelens:scope",
            sae_layer=14,
            steering="0.3 formal.casual",
        )
        assert env is not None
        return dict(env)

    def test_top_level_keys(self) -> None:
        env = self._envelope()
        assert set(env) == {"version", "scope", "provenance", "instruments", "scores"}
        assert env["version"] == MEASUREMENTS_VERSION
        assert env["scope"] == "token"
        assert env["provenance"] == "captured"

    def test_families_and_channel_keys(self) -> None:
        inst = self._envelope()["instruments"]
        assert set(inst) == {"geometry", "lens", "sae"}
        assert set(inst["geometry"]) == {"readings"}
        assert set(inst["lens"]) == {"readings", "readout", "binding"}
        assert set(inst["sae"]) == {"readings", "readout", "binding"}

    def test_bindings(self) -> None:
        inst = self._envelope()["instruments"]
        # The S source is known but no slens/ reading came out: R only.
        assert inst["lens"]["binding"] == {
            "sources": {"R": "local:default"}, "steering": "0.3 formal.casual",
        }
        assert inst["sae"]["binding"] == {
            "source": "saelens:scope",
            "steering": "0.3 formal.casual",
            "layer": 14,
        }

    def test_readings_are_family_native(self) -> None:
        inst = self._envelope()["instruments"]
        geo = inst["geometry"]["readings"]["formal.casual"]
        assert "coords" in geo and "fraction" in geo
        lens = inst["lens"]["readings"]["jlens/fake"]
        assert set(lens) == {"value", "unit", "per_layer", "depth"}
        sae = inst["sae"]["readings"]["sae/548"]
        assert sae["unit"] == "activation_over_max"

    def test_flat_scores_join_all_three_families(self) -> None:
        env = self._envelope()
        assert env["scores"] == {
            "formal.casual": pytest.approx(0.61),
            "jlens/fake": pytest.approx(0.0125),
            "sae/548": pytest.approx(0.84),
        }

    def test_lens_readout_rows(self) -> None:
        readout = self._envelope()["instruments"]["lens"]["readout"]
        assert set(readout) == {"layers", "aggregate"}
        assert [row["layer"] for row in readout["layers"]] == [12, 18]
        assert set(readout["layers"][0]["tokens"][0]) == {
            "token", "id", "logprob",
        }
        assert set(readout["aggregate"][0]) == {
            "token", "id", "strength", "com", "spread",
        }

    def test_sae_readout_rows(self) -> None:
        features = self._envelope()["instruments"]["sae"]["readout"]["features"]
        assert set(features[0]) == {"id", "activation", "label", "max_act"}


class TestDoneAggregate:
    """The ``done`` frame's envelope — built once by the engine at finalize
    (``GenerationResult.measurements``), forwarded verbatim by the server."""

    @staticmethod
    def _envelope() -> dict[str, Any]:
        env = build_measurements(
            scope="aggregate",
            geometry_readings={"formal.casual": _geometry_reading()},
            lens_readings={"jlens/fake": _lens_reading()},
            sae_readings={"sae/548": _sae_reading()},
            lens_sources={"R": "local:default"},
            sae_source="saelens:scope",
            sae_layer=14,
            steering="0.3 formal.casual",
        )
        assert env is not None
        return dict(env)

    def test_top_level_keys(self) -> None:
        env = self._envelope()
        assert set(env) == {
            "version", "scope", "provenance", "instruments", "scores",
        }
        assert env["scope"] == "aggregate"

    def test_no_readout_channel_without_a_discovery_surface(self) -> None:
        """An aggregate has attached-probe ``readings`` only — the native
        top-k ``readout`` is a per-step discovery surface."""
        inst = self._envelope()["instruments"]
        assert set(inst["lens"]) == {"readings", "binding"}
        assert set(inst["sae"]) == {"readings", "binding"}

    def test_none_when_nothing_measured(self) -> None:
        assert build_measurements(scope="aggregate") is None

    def test_the_result_carries_it(self) -> None:
        """The finalize path — not the server — owns the aggregate
        envelope, so the lens/SAE channels keep their native shape and the
        ``done`` frame is a forward, not a re-split."""
        from drowse.generation.results import GenerationResult

        result = GenerationResult(
            text="", tokens=[], token_count=0, tok_per_sec=0.0, elapsed=0.0,
            measurements=self._envelope(),
        )
        assert result.to_dict()["measurements"]["scope"] == "aggregate"


class TestReplayEnvelopes:
    """One ``{"measurements": …}`` body per family, built by the family's
    own ``token_readout`` — the route no longer reshapes anything."""

    def test_geometry(self) -> None:
        env = build_measurements(
            scope="replay",
            provenance="replayed",
            geometry_readings={"formal.casual": _geometry_reading()},
            geometry_binding={"source": None, "steering": "0.5 formal.casual"},
        )
        assert env is not None
        assert set(env) == {
            "version", "scope", "provenance", "instruments", "scores",
        }
        assert env["scope"] == "replay"
        assert env["provenance"] == "replayed"
        assert set(env["instruments"]) == {"geometry"}
        assert "geometry" in env["instruments"]
        assert set(env["instruments"]["geometry"]) == {"readings", "binding"}
        assert "binding" in env["instruments"]["geometry"]
        assert env["instruments"]["geometry"]["binding"] == {
            "source": None, "steering": "0.5 formal.casual",
        }

    def test_lens(self) -> None:
        env = build_measurements(
            scope="replay",
            provenance="replayed",
            lens_readout={12: [(" a", 0.78)]},
            lens_aggregate=[(" a", 0.41, 0.31, 0.05, 7)],
            lens_token_ids={12: [5]},
            lens_sources={"R": "local:default"},
            steering=None,
        )
        assert env is not None
        assert set(env) == {"version", "scope", "provenance", "instruments"}
        assert set(env["instruments"]) == {"lens"}
        assert "lens" in env["instruments"]
        assert set(env["instruments"]["lens"]) == {"readout", "binding"}
        assert env["instruments"]["lens"]["binding"] == {
            "sources": {"R": "local:default"}, "steering": None,
        }

    def test_sae(self) -> None:
        env = build_measurements(
            scope="replay",
            provenance="replayed",
            sae_features=[(42, 3.5, "fruit", 121.1)],
            sae_source="saelens:scope",
            sae_layer=14,
            steering="0.2 sae/42",
        )
        assert env is not None
        assert set(env["instruments"]) == {"sae"}
        assert "sae" in env["instruments"]
        assert set(env["instruments"]["sae"]) == {"readout", "binding"}
        assert "binding" in env["instruments"]["sae"]
        assert "layer" in env["instruments"]["sae"]["binding"]
        assert env["instruments"]["sae"]["binding"]["layer"] == 14

    def test_every_family_returns_the_same_envelope_wrapper(self) -> None:
        """``Instrument.token_readout`` is uniform: ``{"measurements": …}``
        from all three families, so the route dispatches without a branch."""
        from drowse.lens.instrument import LensInstrument
        from tests.support.geometry import stub_geometry_instrument
        from tests.support.sae import make_resident, sae_instrument

        session = SimpleNamespace(
            model_id="test/model",
            _monitor=SimpleNamespace(
                probe_names=["formal.casual"], probe_layers=lambda: {0},
            ),
        )
        geometry = stub_geometry_instrument(
            session, replay=_canned_replay({"formal.casual": _geometry_reading()}),
        )
        lens = LensInstrument(
            **lens_kwargs(replay=_canned_replay((
            {0: [("a", 0.5, 7)]}, [("a", 0.5, 0.0, 0.0, 7)],
        ))),
        )
        # No fitted lens on the stub: resolve a stand-in over one layer.
        lens._require = lambda: object()  # type: ignore[method-assign]
        lens.runtime.resolve_layers = lambda _lens, _layers: [0]  # type: ignore[method-assign]
        sae = sae_instrument(session, replay=_canned_replay([(3, 1.5)]))
        make_resident(sae.runtime, layer=14)
        for instrument in (geometry, lens, sae):
            out = _token_readout(instrument, instrument.family)
            assert set(out) == {"measurements"}
            binding = out["measurements"]["instruments"][instrument.family]["binding"]
            assert binding["steering"] == "0.3 formal.casual"


def _canned_replay(result: Any) -> Any:
    """A token replay that skips the forward and hands back ``result`` as
    the family read's output, under a steered replayed token."""
    from drowse.readout.replay import ReplayedToken

    def replay(
        node_id: str, raw_index: int, *, apply_steering: bool = True, **_kw: Any,
    ) -> tuple[ReplayedToken, Any]:
        token = ReplayedToken(
            node_id=node_id, raw_index=raw_index, token_id=1, token_text="x",
            steering="0.3 formal.casual" if apply_steering else None,
        )
        return token, result

    return replay


def _token_readout(instrument: Any, family: str) -> dict[str, Any]:
    """Call ``token_readout`` with only the knobs the family accepts."""
    if family == "lens":
        return instrument.token_readout("n1", 0, top_k=4, layers="all")
    if family == "sae":
        return instrument.token_readout("n1", 0, top_k=4)
    return instrument.token_readout("n1", 0)


class TestRejectedKnobs:
    """A family that cannot honor a knob rejects it (400 at the route)
    instead of dropping it silently — the route used to ignore ``top_k``
    and ``layers`` on geometry with only a comment to say so."""

    @staticmethod
    def _instrument(family: str) -> Any:
        from tests.support.geometry import stub_geometry_instrument
        from tests.support.sae import sae_instrument

        # The knob is refused before any replay work: the replay is unwired.
        session = SimpleNamespace(model_id="test/model")
        if family == "geometry":
            return stub_geometry_instrument(session)
        return sae_instrument(session)

    def test_geometry_rejects_top_k(self) -> None:
        with pytest.raises(ValueError, match="no top_k"):
            self._instrument("geometry").token_readout("n1", 0, top_k=8)

    def test_geometry_rejects_layers(self) -> None:
        with pytest.raises(ValueError, match="no layers"):
            self._instrument("geometry").token_readout("n1", 0, layers=[1, 2])

    def test_sae_rejects_layers(self) -> None:
        with pytest.raises(ValueError, match="no layers"):
            self._instrument("sae").token_readout("n1", 0, layers=[1])


# ---------------------------------------------------------------------------
# The envelope source binding is one expression
# ---------------------------------------------------------------------------

class TestEnvelopeSources:
    """``envelope_sources`` is the one reading of the lens families' sources
    and the live SAE config into the envelope's source fields (token rows,
    authored rows, aggregates)."""

    _LENS = {"R": "local:j", "S": "rs-s"}
    _SAE = {"source": "saelens:scope", "layer": 14}

    def test_binds_both_families(self) -> None:
        assert envelope_sources(self._LENS, self._SAE) == {
            "lens_sources": {"R": "local:j", "S": "rs-s"},
            "sae_source": "saelens:scope", "sae_layer": 14,
        }

    def test_an_inactive_readout_binds_nothing(self) -> None:
        assert envelope_sources(None, None) == {
            "lens_sources": {}, "sae_source": None, "sae_layer": None,
        }

    def test_gates_each_family(self) -> None:
        assert envelope_sources(self._LENS, self._SAE, lens=False) == {
            "lens_sources": {}, "sae_source": "saelens:scope", "sae_layer": 14,
        }
        assert envelope_sources(self._LENS, self._SAE, sae=False) == {
            "lens_sources": {"R": "local:j", "S": "rs-s"},
            "sae_source": None, "sae_layer": None,
        }

    def test_feeds_the_envelope_binding(self) -> None:
        env = build_measurements(
            scope="token",
            lens_readout={0: [("a", 0.5)]},
            sae_features=[(3, 1.5, None, None)],
            **envelope_sources(self._LENS, self._SAE),
        )
        assert env is not None
        lens, sae = env["instruments"].get("lens"), env["instruments"].get("sae")
        assert lens is not None and sae is not None
        assert lens["binding"]["sources"] == {"R": "local:j"}
        binding = sae["binding"]
        assert (binding["source"], binding.get("layer")) == ("saelens:scope", 14)


class TestLensFamilyAttribution:
    """Regression: the lens channel's binding named only the R source, so an
    ``slens/`` reading was attributed to the R lens (and to nothing with the
    R live readout off).  Each family that produced data is named."""

    @staticmethod
    def _s_reading() -> ScalarReading:
        return ScalarReading(
            value=2.5,
            unit="mean_footprint_z",
            per_layer={0: 3.0, 9: 1.0},
            depth=DepthSummary(center=(0.0,), spread=(0.0,), basis="footprint_z"),
        )

    def test_s_only(self) -> None:
        env = build_measurements(
            scope="aggregate",
            lens_readings={"slens/owl": self._s_reading()},
            lens_sources={"R": "rs-r", "S": "rs-s"},
        )
        assert env is not None
        lens = env["instruments"].get("lens")
        assert lens is not None
        assert lens["binding"] == {"sources": {"S": "rs-s"}, "steering": None}

    def test_mixed(self) -> None:
        env = build_measurements(
            scope="token",
            lens_readings={"slens/owl": self._s_reading(), "jlens/fake": _lens_reading()},
            lens_sources={"R": None, "S": "local:s"},
        )
        assert env is not None
        lens = env["instruments"].get("lens")
        assert lens is not None
        assert lens["binding"]["sources"] == {"R": None, "S": "local:s"}

    def test_an_unknown_lens_unit_is_a_broken_invariant(self) -> None:
        with pytest.raises(ValueError, match="no lens family"):
            build_measurements(
                scope="token",
                lens_readings={"x": ScalarReading(value=1.0, unit="raw_activation")},
            )


class TestUnitsChannel:
    """The units channel: readings plus a source-less binding, nothing else."""

    @staticmethod
    def _reading() -> ScalarReading:
        return ScalarReading(
            value=-1.5,
            unit="activation_over_scale",
            per_layer={8: -1.5},
            depth=DepthSummary(center=(0.25,), spread=(0.0,), basis="single_layer"),
            meta={"unit": "neuron/L8n344", "activation": -0.75},
        )

    def test_channel_keys_and_scores(self) -> None:
        env = build_measurements(
            scope="aggregate",
            units_readings={"neuron/L8n344": self._reading()},
            steering="!head/L2h1",
        )
        assert env is not None
        assert set(env["instruments"]) == {"units"}
        channel = env["instruments"].get("units")
        assert channel is not None
        assert set(channel) == {"binding", "readings"}
        assert channel["binding"] == {"source": None, "steering": "!head/L2h1"}
        reading = channel["readings"]["neuron/L8n344"]
        assert set(reading) == {"value", "unit", "per_layer", "depth", "meta"}
        assert reading.get("meta") == {"unit": "neuron/L8n344", "activation": -0.75}
        assert env.get("scores") == {"neuron/L8n344": -1.5}

    def test_no_units_readings_no_channel(self) -> None:
        env = build_measurements(scope="token", units_readings={})
        assert env is None


# ---------------------------------------------------------------------------
# The loom token row's projection
# ---------------------------------------------------------------------------

class TestTokenRow:
    """``token_row_measurements``: what a loom token row persists of a token
    envelope (summaries once, rounded; traces only on request; the inspector's
    subspace coordinates never)."""

    @staticmethod
    def _envelope() -> Any:
        geometry = _geometry_reading()
        geometry.fraction_per_layer = {12: 0.123456789, 18: 0.5}
        geometry.residual_per_layer = {12: 0.0, 18: 0.0}
        geometry.subspace_coords_per_layer = {12: (0.1, 0.2), 18: (0.3, 0.4)}
        geometry.assignment = [("formal", 0.987654321)]
        env = build_measurements(
            scope="token",
            geometry_readings={"formal.casual": geometry},
            lens_readings={"jlens/fake": _lens_reading()},
            sae_readings={"sae/548": _sae_reading()},
            lens_readout={12: [(" a", 0.78)], 18: [(" b", 0.6)]},
            lens_aggregate=[(" a", 0.41234567, 0.31, 0.05, 7)],
            lens_token_ids={12: [5], 18: [7]},
            lens_sources={"R": "local:default"},
            sae_features=[(42, 3.5123456, "fruit", 121.1)],
            sae_source="saelens:scope",
            sae_layer=14,
            units_readings={"neuron/L8n344": TestUnitsChannel._reading()},
            steering="0.3 formal.casual",
        )
        assert env is not None
        return env

    def test_summary_row_keys(self) -> None:
        from drowse.readout.measurements import (
            GEOMETRY_SUMMARY_KEYS,
            token_row_measurements,
        )

        row: Any = token_row_measurements(self._envelope(), per_layer=False)
        assert set(row) == {"version", "scope", "provenance", "instruments", "scores"}
        inst = row["instruments"]
        assert set(inst) == {"geometry", "lens", "sae", "units"}
        assert set(inst["geometry"]["readings"]["formal.casual"]) == set(GEOMETRY_SUMMARY_KEYS)
        # A scalar reading's trace is a few values; it stays whole.
        assert set(inst["lens"]["readings"]["jlens/fake"]) == {
            "value", "unit", "per_layer", "depth",
        }
        assert inst["lens"]["readings"]["jlens/fake"]["per_layer"] == {
            "12": 0.01, "18": 0.015,
        }
        assert set(inst["lens"]["readout"]) == {"aggregate"}
        assert set(inst["sae"]["readings"]["sae/548"]) == {
            "value", "unit", "per_layer", "depth",
        }
        assert set(inst["units"]["readings"]["neuron/L8n344"]) == {
            "value", "unit", "per_layer", "depth", "meta",
        }
        assert inst["lens"]["binding"] == {
            "sources": {"R": "local:default"}, "steering": "0.3 formal.casual",
        }
        assert inst["sae"]["binding"]["layer"] == 14

    def test_traces_on_request_and_never_the_subspace_coordinates(self) -> None:
        from drowse.readout.measurements import (
            GEOMETRY_SUMMARY_KEYS,
            GEOMETRY_TRACE_KEYS,
            token_row_measurements,
        )

        row: Any = token_row_measurements(self._envelope(), per_layer=True)
        inst = row["instruments"]
        reading = inst["geometry"]["readings"]["formal.casual"]
        assert set(reading) == set(GEOMETRY_SUMMARY_KEYS) | set(GEOMETRY_TRACE_KEYS)
        assert reading.get("fraction_per_layer") == {"12": 0.123457, "18": 0.5}
        assert reading.get("coords_per_layer") == {"12": [0.5, -0.1], "18": [0.7, -0.3]}
        readout = inst["lens"]["readout"]
        assert [layer["layer"] for layer in readout.get("layers", [])] == [12, 18]
        assert readout.get("layers", [])[0]["tokens"][0]["logprob"] == float(
            f"{math.log(0.78):.6g}",
        )

    def test_floats_round_to_row_digits(self) -> None:
        from drowse.readout.measurements import READING_DIGITS, token_row_measurements

        assert READING_DIGITS == 6
        row: Any = token_row_measurements(self._envelope(), per_layer=False)
        inst = row["instruments"]
        reading = inst["geometry"]["readings"]["formal.casual"]
        assert reading["assignment"] == [["formal", 0.987654]]
        assert reading["nearest"] == [["formal", 1.25]]
        assert inst["lens"]["readout"]["aggregate"][0]["strength"] == 0.412346
        assert inst["sae"]["readout"]["features"][0]["activation"] == 3.51235
        assert inst["units"]["readings"]["neuron/L8n344"].get("meta") == {
            "unit": "neuron/L8n344", "activation": -0.75,
        }

    def test_the_live_frame_keeps_the_whole_envelope(self) -> None:
        """The projection copies: the token frame's envelope is untouched."""
        from drowse.readout.measurements import token_row_measurements

        env = self._envelope()
        before = repr(env)
        token_row_measurements(env, per_layer=False)
        assert repr(env) == before
        reading = env["instruments"]["geometry"]["readings"]["formal.casual"]
        assert reading.get("subspace_coords_per_layer") == {"12": [0.1, 0.2], "18": [0.3, 0.4]}


# ---------------------------------------------------------------------------
# The family enumeration is one list
# ---------------------------------------------------------------------------

def test_payload_family_slots_match_the_registry(make_tiny_session: Any) -> None:
    """``token_payloads``' per-family slots are pinned to the registry, so a
    fourth read family cannot land a slot-less payload."""
    from drowse.generation.token_payloads import _FAMILY_SLOTS

    session = make_tiny_session(warm=False)
    assert set(_FAMILY_SLOTS) == set(session.instruments)


def test_capability_table_covers_every_family(make_tiny_session: Any) -> None:
    from drowse.server.instrument_routes import CAPABILITIES

    session = make_tiny_session(warm=False)
    assert set(CAPABILITIES) == set(session.instruments)


def test_a_token_anatomy_rides_a_replay_envelope_only() -> None:
    """The anatomy key set, pinned: the record sits beside ``instruments`` on
    a replay envelope (which it alone may fill) and nowhere else."""
    from drowse.loom.node_replay import token_anatomy_record

    record = token_anatomy_record(
        raw_index=0, token_id=1, token="a", position=3, prompt_length=4,
        probability=0.5, sources={"R": "local:r", "S": None}, steering=None,
        layers=[], units=[], units_absent="no R-family lens", units_scale=None,
        patterns=None, context=None,
    )
    assert set(record) == {
        "token", "binding", "layers", "units", "units_absent", "units_scale", "patterns",
        "context",
    }
    assert set(record["token"]) == {
        "raw_index", "token_id", "token", "position", "prompt_length", "probability",
    }
    envelope = build_measurements(scope="replay", provenance="replayed", anatomy=record)
    assert envelope is not None
    assert set(envelope) == {"version", "scope", "provenance", "instruments", "anatomy"}
    assert envelope["instruments"] == {} and envelope.get("anatomy") == record
    with pytest.raises(ValueError, match="replay envelope only"):
        build_measurements(scope="token", anatomy=record)


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__]))
