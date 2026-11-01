"""One token, one number: the token replay and the node replay agree.

Every family's ``token_readout`` replays a token through the loom's token
replay; anatomy and unit activity replay the node through the node replay.
Both drive the forward with :func:`drowse.loom.replay.replay_segments`, so a
triggered recipe fires in a replay where it fired in generation, and a recipe
whose firing the node does not record refuses in both.
"""

from __future__ import annotations

from typing import Any

import pytest

from drowse.errors import SteeringCompositionError
from drowse.generation.sampling import SamplingConfig

pytestmark = pytest.mark.usefixtures("isolated_home")

_UNIT = "neuron/L1n3"


def _captured(node: Any, raw_index: int) -> float:
    return next(
        row["measurements"]["instruments"]["units"]["readings"][_UNIT]["value"]
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
        assert replayed == pytest.approx(captured, rel=1e-4, abs=1e-5), raw_index
        assert activity[raw_index] == pytest.approx(captured, rel=1e-3, abs=1e-4), raw_index


def test_token_readout_refuses_a_gate_the_node_does_not_record(make_tiny_session: Any) -> None:
    """A probe-gated term fires on the generation's live scores, which the node
    does not keep: every replay refuses it (422) rather than guess."""
    session, _script = make_tiny_session.scripted("abc")
    session.add_probe(_UNIT)
    session.generate(
        "hello", steering=f"0.5 {_UNIT}@when:{_UNIT}>0.1",
        sampling=SamplingConfig(max_tokens=2, seed=0),
    )
    node = session.tree.nodes[session.tree.active_node_id]
    with pytest.raises(SteeringCompositionError):
        session.units.token_readout(node.id, 0)
    with pytest.raises(SteeringCompositionError):
        session.unit_activity(node.id, units=[_UNIT])
    # Without the node's own steering the replay runs.
    assert session.units.token_readout(node.id, 0, apply_steering=False)
