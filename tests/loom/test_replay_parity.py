"""One token, one number: the token replay and the node replay agree.

Every family's ``token_readout`` replays a token through the loom's token
replay; anatomy and unit activity replay the node through the node replay.
Both drive the forward with :func:`drowse.loom.replay.replay_segments`, so a
triggered recipe fires in a replay where it fired in generation (a probe gate
steps one position at a time, scoring the gates as decode did), and a recipe
whose firing the node does not record (a thinking-phase window) refuses in both.
"""

from __future__ import annotations

from typing import Any

import pytest

from drowse.errors import SteeringCompositionError
from drowse.generation.sampling import SamplingConfig

pytestmark = pytest.mark.usefixtures("isolated_home")

_UNIT = "neuron/L1n3"


def _captured(node: Any, raw_index: int, unit: str = _UNIT) -> float:
    return next(
        row["measurements"]["scores"][unit]
        for row in node.tokens if row["raw_index"] == raw_index
    )


@pytest.mark.parametrize("steering", [
    f"1 !{_UNIT}@prompt",
    f"1 !{_UNIT}@response",
    f"1 !{_UNIT}@first:1",
    f"0.5 {_UNIT}@prompt",
])
def test_unit_token_readout_matches_generation_and_activity(
    make_tiny_session: Any, steering: str,
) -> None:
    """Regression: the token replay ran a triggered recipe at its rest state
    (``@prompt`` steering everywhere), so the units readout of raw token 0
    disagreed with the live reading and the activity trace."""
    session, _script = make_tiny_session.scripted("abc")
    session.add_probe(_UNIT)
    session.generate("hello", steering=steering, sampling=SamplingConfig(max_tokens=3, seed=0))
    node = session.tree.nodes[session.tree.active_node_id]
    activity = session.unit_activity(node.id, units=[_UNIT])["traces"][_UNIT]
    for raw_index in range(len(node.raw_token_ids)):
        replayed = session.units.token_readout(node.id, raw_index)[
            "measurements"]["instruments"]["units"]["readings"][_UNIT]["value"]
        captured = _captured(node, raw_index)
        # The row keeps READING_DIGITS significant digits: half a unit in the last one.
        assert replayed == pytest.approx(captured, rel=1e-5, abs=1e-6), raw_index
        assert activity[raw_index] == pytest.approx(captured, rel=1e-3, abs=1e-4), raw_index


@pytest.mark.parametrize("threshold", ["0", "-1000", "1000"])
def test_a_gated_reply_replays_stepwise_as_generation_fired(
    make_tiny_session: Any, threshold: str,
) -> None:
    """Regression: a probe-gated term fires on the generation's live scores,
    which the node does not keep, so every replay of a gated reply refused
    (422) and its slim rows lost the token view's details for good.  The
    replay now steps one decode position at a time and scores the gates
    after each forward as the decode loop does, so the token readout and the
    activity trace read what generation read.  The layer-2 unit gates a push
    on the layer-1 unit, which moves it: at ``0`` the gate opens mid-reply."""
    read, push = "neuron/L2n3", "neuron/L1n3"
    session, _script = make_tiny_session.scripted("abcd")
    session.add_probe(read)
    session.generate(
        "hello", steering=f"4 {push}@when:{read}>{threshold}",
        sampling=SamplingConfig(max_tokens=4, seed=0),
    )
    node = session.tree.nodes[session.tree.active_node_id]
    activity = session.unit_activity(node.id, units=[read])["traces"][read]
    for raw_index in range(len(node.raw_token_ids)):
        replayed = session.units.token_readout(node.id, raw_index)[
            "measurements"]["instruments"]["units"]["readings"][read]["value"]
        captured = _captured(node, raw_index, read)
        assert replayed == pytest.approx(captured, rel=1e-5, abs=1e-6), raw_index
        assert activity[raw_index] == pytest.approx(captured, rel=1e-3, abs=1e-4), raw_index
        # The anatomy (the token view's other replay) injects exactly where the gate was open:
        # after a step whose reading passed the threshold.
        fired = raw_index >= 1 and _captured(node, raw_index - 1, read) > float(threshold)
        layers = session.token_anatomy(node.id, raw_index, patterns=0)["measurements"]["anatomy"]["layers"]
        assert any(layer["steering"] is not None for layer in layers) == fired, raw_index
    # Without the node's own steering the replay runs unsteered, as before.
    unsteered = session.units.token_readout(node.id, len(node.raw_token_ids) - 1, apply_steering=False)
    assert unsteered["measurements"]["instruments"]["units"]["readings"][read]["value"] != (
        pytest.approx(_captured(node, len(node.raw_token_ids) - 1, read), abs=1e-3)
    ) or threshold == "1000"


def test_a_thinking_phase_window_still_refuses_the_replay() -> None:
    """The thinking state follows the decode loop's delimiter machine, which
    the node does not record: a phase window on a thinking family refuses."""
    from drowse.loom.replay import replay_segments
    from drowse.steering.triggers import Trigger

    for phase in (Trigger.AFTER_THINKING, Trigger.THINKING_ONLY):
        with pytest.raises(SteeringCompositionError, match="thinking-phase"):
            replay_segments(5, 9, [phase], can_think=True, gate=lambda _step: {})
