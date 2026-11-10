"""Layer-scoped steering terms (``0.5 honest{12-20}``) through composition.

A ``{…}`` clause restricts a term's per-layer fragments (or a curved term's
layers) before grouping and synthesis.  Pinned here on a real session and on a
model-free :class:`SteeringRig`:

- a scoped term steers exactly its layers, for every term kind;
- an ablation keeps its exact coefficient under any selection;
- the mean-1 share normalization runs over the selected layers, so a one-layer
  selection lands where a one-layer fit of the same direction does;
- a layer the term is not fitted at, or an empty workspace intersection,
  raises ``SteeringCompositionError`` (422);
- curved ``%`` terms restrict their layers and shares;
- the compiled offsets and the transient ``AffineOp`` agree on scoped plans;
- the plan cache keys on the clause.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from drowse.errors import OverlappingManifoldError, SteeringCompositionError
from drowse.steering.composer import TermLayers, resolve_scope_role
from drowse.steering.hooks import (
    _SUBSPACE_GAIN,
    AffineGroup,
    CurvedGroup,
    SteeringManager,
    SteeringPlan,
    _lower_affine_subspaces,
    install_persistent_offset_hooks,
)
from drowse.steering.terms import Steering
from drowse.steering.triggers import Trigger
from tests.support.composer import SteeringRig
from tests.support.manifolds import loop_manifold, open_manifold
from tests.support.sessions import forward_logits, register_vectors
from tests.support.whitener import synthetic_means, synthetic_whitener

_D = 8
_N_LAYERS = 4


def _units(seed: int, layers: range | tuple[int, ...]) -> dict[int, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for L in layers:
        v = torch.randn(_D, generator=g)
        out[L] = v / v.norm()
    return out


def _rig(*, compiled: bool = False) -> tuple[SteeringRig, dict[int, torch.Tensor]]:
    """A composer over four identity layers: ``a``/``b`` fitted at every layer,
    ``c`` at 1-2, ``d`` at 0 only; optionally with compiled offset buffers."""
    layers = nn.ModuleList([nn.Identity() for _ in range(_N_LAYERS)])
    manager = SteeringManager()
    buffers: dict[int, torch.Tensor] = {}
    if compiled:
        buffers, handles = install_persistent_offset_hooks(
            layers, _D, torch.device("cpu"), torch.float32,
        )
        manager.adopt_compiled_offsets(buffers, handles)
    means = synthetic_means(range(_N_LAYERS), _D)
    rig = SteeringRig(
        layers=layers,
        manager=manager,
        profiles={
            "a": _units(1, range(_N_LAYERS)),
            "b": _units(2, range(_N_LAYERS)),
            "c": _units(3, (1, 2)),
            "d": _units(4, (0,)),
        },
        layer_means=means,
        whitener=synthetic_whitener(range(_N_LAYERS), _D, means=means),
    )
    return rig, buffers


def _lowered(rig: SteeringRig) -> dict[int, tuple[Any, ...]]:
    plan = rig.manager.plan
    assert plan is not None
    return plan.lower(torch.device("cpu"), torch.float32)


# ------------------------------------------------------------ real session ---


def _capture(session: Any) -> dict[int, torch.Tensor]:
    """``{layer: stack(pre, post)}`` around the steering hooks of each layer.

    Registered after a scope is entered: a prepended hook snapshots the layer
    output before the steering hook runs, an appended one reads it after.
    """
    seen: dict[int, list[torch.Tensor]] = {}
    handles = []

    def _hidden(out: Any) -> torch.Tensor:
        return out if isinstance(out, torch.Tensor) else out[0]

    for idx, layer in enumerate(session.layers):
        def pre(_m: Any, _i: Any, out: Any, idx: int = idx) -> None:
            seen[idx] = [_hidden(out).clone()]

        def post(_m: Any, _i: Any, out: Any, idx: int = idx) -> None:
            seen[idx].append(_hidden(out).clone())

        handles.append(layer.register_forward_hook(pre, prepend=True))
        handles.append(layer.register_forward_hook(post))
    try:
        forward_logits(session)
    finally:
        for handle in handles:
            handle.remove()
    return {L: torch.stack(pair) for L, pair in seen.items()}


def _touched(captured: dict[int, torch.Tensor]) -> set[int]:
    """The layers whose steering hooks changed the residual."""
    return {L for L, pair in captured.items() if not torch.equal(pair[0], pair[1])}


@pytest.mark.parametrize(("expr", "layers"), [
    ("0.5 a", {0, 1, 2, 3}),
    ("0.5 a{1}", {1}),
    ("0.5 a{1,3}", {1, 3}),
    ("0.5 a{0-2}", {0, 1, 2}),
    ("0.5 a{workspace}", {2}),
    ("0.5 a{all}", {0, 1, 2, 3}),
    ("0.4 !a{2}", {2}),
    ("0.3 a~b{0}", {0}),
    ("0.3 a|b{3}", {3}),
    ("0.5 a{1} + 0.3 b{3}", {1, 3}),
])
@pytest.mark.usefixtures("isolated_home")
def test_a_scoped_term_steers_exactly_its_layers(
    tiny_session: Any, expr: str, layers: set[int],
) -> None:
    register_vectors(tiny_session, "a", "b")
    with tiny_session.steering(expr):
        assert _touched(_capture(tiny_session)) == layers


@pytest.mark.usefixtures("isolated_home")
def test_a_scoped_ablation_keeps_its_exact_coefficient(tiny_session: Any) -> None:
    """``0.4 !a{1,3}`` keeps 60% of the neutral-centered reading along ``a``'s
    covector ``Σ⁻¹a`` at layers 1 and 3, exactly as the unscoped ablation does
    everywhere."""
    profile = register_vectors(tiny_session, "a")["a"]
    means = tiny_session.layer_means
    whitener = tiny_session.whitener
    for expr, layers in (("0.4 !a{1,3}", (1, 3)), ("0.4 !a", (0, 1, 2, 3))):
        with tiny_session.steering(expr):
            captured = _capture(tiny_session)
        assert _touched(captured) == set(layers)
        for L in layers:
            d = whitener.apply_inv(L, profile[L].to(torch.float32)).to(torch.float32)
            mu = means[L].to(torch.float32) @ d
            before, after = (captured[L][i].to(torch.float32) @ d - mu for i in (0, 1))
            assert torch.allclose(after, 0.6 * before, rtol=1e-4, atol=1e-4), (expr, L)


@pytest.mark.usefixtures("isolated_home")
def test_unfitted_layers_raise_422_on_entry(tiny_session: Any) -> None:
    from drowse.geometry.profile import Profile

    register_vectors(tiny_session, "a")
    tensors = register_vectors(tiny_session, "p")["p"]
    tiny_session.registry.put_profile("p", Profile({1: tensors[1], 2: tensors[2]}))
    tiny_session.registry.put_profile("q", Profile({0: tensors[0]}))
    baseline = forward_logits(tiny_session)
    cases = [
        ("0.5 a{7}", r"'a\{7\}' selects layer\(s\) 7 it is not fitted at; fitted layers: 0-3"),
        ("0.5 p{0-1}", r"'p\{0-1\}' selects layer\(s\) 0 it is not fitted at; fitted layers: 1-2"),
        ("0.5 !p{3}", r"selects layer\(s\) 3 .* fitted layers: 1-2"),
        ("0.5 a~p{0}", r"selects layer\(s\) 0 .* fitted layers: 1-2"),
        ("0.5 q{workspace}", r"workspace band \(layers 2\) but is fitted at layers 0"),
    ]
    for expr, message in cases:
        scope = tiny_session.steering(expr)
        with pytest.raises(SteeringCompositionError, match=message) as exc:
            scope.__enter__()
        assert exc.value.user_message()[0] == 422
        assert torch.equal(forward_logits(tiny_session), baseline)


# ---------------------------------------------------------------- the rig ---


def test_shares_are_mean_one_over_the_selection() -> None:
    rig, _ = _rig()
    with rig.steering("0.5 a{1,3}"):
        plan = rig.manager.plan
        assert plan is not None
        (term,) = plan.affine
        lowered = _lower_affine_subspaces(term.synth)
        assert set(lowered) == {1, 3}
        total = sum(low.eff_along for low in lowered.values())
        assert total == pytest.approx(2 * _SUBSPACE_GAIN)


def test_a_one_layer_selection_matches_a_one_layer_fit() -> None:
    """Scoping a 4-layer vector to layer 2 lands in the same α band as a vector
    fitted only at layer 2: the two push identical offsets."""
    rig, _ = _rig()
    rig.registry.put_profile("a2", {2: rig.registry.profiles["a"][2]})
    offsets = {}
    for expr in ("0.5 a{2}", "0.5 a2"):
        with rig.steering(expr):
            plan = rig.manager.plan
            assert plan is not None
            offsets[expr] = plan.static_offsets()
    scoped, fitted = offsets["0.5 a{2}"], offsets["0.5 a2"]
    assert scoped is not None and fitted is not None
    assert set(scoped) == set(fitted) == {2}
    assert torch.allclose(scoped[2], fitted[2], atol=1e-6)


def test_a_scoped_and_an_unscoped_term_share_one_trigger_group() -> None:
    rig, _ = _rig()
    with rig.steering("0.5 a{1} + 0.3 b"):
        plan = rig.manager.plan
        assert plan is not None
        (term,) = plan.affine
        assert set(term.synth.layers) == {0, 1, 2, 3}
        assert set(_lowered(rig)) == {0, 1, 2, 3}
    with rig.steering("0.5 a{1}@after + 0.3 b{2}"):
        lowered = _lowered(rig)
        assert set(lowered) == {1, 2}
        assert lowered[1][0].trigger == Trigger.AFTER_THINKING
        assert lowered[2][0].trigger is Trigger.BOTH


@pytest.mark.parametrize(("expr", "message"), [
    ("0.5 c{0}", r"'c\{0\}' selects layer\(s\) 0 .* fitted layers: 1-2"),
    ("0.5 a{2-5}", r"selects layer\(s\) 4-5 .* fitted layers: 0-3"),
    ("0.5 d{workspace}", r"workspace band \(layers 2\) but is fitted at layers 0"),
    ("0.5 mood%n1{3}", r"'mood%n1\{3\}' selects layer\(s\) 3 .* fitted layers: 0-2"),
    ("0.5 mood%n1{workspace} + 0.5 d", None),
])
def test_scoped_composition_errors(expr: str, message: str | None) -> None:
    rig, _ = _rig()
    rig.registry.put_manifold("mood", loop_manifold(layers=(0, 1, 2)))
    if message is None:
        with rig.steering(expr):
            assert set(_lowered(rig)) == {0, 2}
        return
    scope = rig.steering(expr)
    with pytest.raises(SteeringCompositionError, match=message):
        scope.__enter__()
    assert rig.manager.plan is None


def test_workspace_scope_intersects_the_fitted_layers() -> None:
    rig, _ = _rig()
    with rig.steering("0.5 c{workspace} + 0.2 !b{workspace}"):
        assert set(_lowered(rig)) == {2}


def test_a_curved_term_restricts_its_layers_and_shares() -> None:
    rig, _ = _rig()
    manifold = open_manifold(layers=(0, 1, 2))
    rig.registry.put_manifold("mood", manifold)
    with rig.steering("0.5 mood%n1{0,2}"):
        plan = rig.manager.plan
        assert plan is not None
        (curved,) = plan.curved
        assert set(curved.shares) == {0, 2}
        assert sum(curved.shares.values()) == pytest.approx(2.0)
        baked = manifold.mahalanobis_share
        assert curved.shares[0] / curved.shares[2] == pytest.approx(baked[0] / baked[2])
        lowered = _lowered(rig)
        assert set(lowered) == {0, 2}
        assert all(isinstance(g, CurvedGroup) for gs in lowered.values() for g in gs)
        assert set(rig.manager.hooks) == {0, 2}


@pytest.mark.parametrize("layers", [(), (2,), (0, 2)])
def test_add_manifold_refuses_layers_outside_the_fit(layers: tuple[int, ...]) -> None:
    plan = SteeringPlan()
    with pytest.raises(SteeringCompositionError, match="cannot act on layers"):
        plan.add_manifold(
            "mood", loop_manifold(layers=(0, 1)), position=(0.5,),
            along=0.5, onto=0.0, layers=layers,
        )
    assert not plan


def test_disjoint_layer_clauses_let_one_curved_manifold_steer_twice() -> None:
    rig, _ = _rig()
    rig.registry.put_manifold("mood", loop_manifold(layers=(0, 1)))
    with pytest.raises(OverlappingManifoldError):
        rig.steering("0.5 mood%n1 + 0.5 mood%n3").__enter__()
    with rig.steering("0.5 mood%n1{0} + 0.5 mood%n3{1}"):
        lowered = _lowered(rig)
        assert set(lowered) == {0, 1}
        assert all(len(groups) == 1 for groups in lowered.values())


def test_an_affine_manifold_term_restricts_its_push() -> None:
    from drowse.geometry.fold import fold_directions_to_subspace

    rig, _ = _rig()
    ray = fold_directions_to_subspace(
        "ray", rig.registry.profiles["a"], rig.neutral.means,
        whitener=rig.neutral.resident_whitener, label="pole",
    )
    rig.registry.put_manifold("ray", ray)
    with rig.steering("0.5 ray%pole{1-2}"):
        assert set(_lowered(rig)) == {1, 2}
        plan = rig.manager.plan
        assert plan is not None
        total = sum(
            low.eff_along
            for low in _lower_affine_subspaces(plan.affine[0].synth).values()
        )
        assert total == pytest.approx(2 * _SUBSPACE_GAIN)


@pytest.mark.parametrize("expr", [
    "0.5 a{1,3} + 0.3 b{2}",
    "0.5 a{workspace} - 0.2 c",
    "0.4 a~b{0-1} + 0.3 b{3}",
])
def test_compiled_offsets_match_the_transient_ops_for_scoped_plans(expr: str) -> None:
    compiled, buffers = _rig(compiled=True)
    eager, _ = _rig()
    with compiled.steering(expr), eager.steering(expr):
        assert compiled.manager.uses_compiled_offsets
        assert not compiled.manager.hooks
        hooks = eager.manager.hooks
        assert set(hooks) == {L for L, buf in buffers.items() if bool(buf.any())}
        for L, buf in buffers.items():
            if L not in hooks:
                assert not bool(buf.any())
                continue
            const = torch.zeros(_D)
            for group in hooks[L].groups:
                assert isinstance(group, AffineGroup)
                assert group.op.const is not None
                const += group.op.const
            assert torch.allclose(buf, const, atol=1e-6)
            hidden = torch.randn(1, 2, _D)
            via_buffer = hidden + buf
            via_hook = hidden.clone()
            for group in hooks[L].groups:
                assert isinstance(group, AffineGroup)
                group.op.apply(via_hook)
            assert torch.allclose(via_buffer, via_hook, atol=1e-6)


def test_the_plan_cache_keys_on_the_clause() -> None:
    rig, _ = _rig()
    composer = rig.composer

    def plan(expr: str) -> Any:
        steering = Steering.from_value(expr)
        assert steering is not None
        return composer.plan_for(composer.resolve(steering))

    one, two, every = plan("0.5 a{1}"), plan("0.5 a{2}"), plan("0.5 a")
    assert len({id(one), id(two), id(every)}) == 3
    assert plan("0.5 a{1}") is one
    assert plan("0.5 a{all}") is every
    # The projection itself is shared across clauses on the same operands.
    plan("0.3 a~b{1}")
    plan("0.3 a~b{2}")
    projected = composer.projection("a~b{1}")
    assert projected is not None
    assert composer.projection("a~b{2}") is projected
    assert composer.projection("a~b") is projected


def test_scope_role_reads_the_variant_under_a_clause() -> None:
    entries: dict[str, Any] = {"a:role-pirate{1-2}": (0.5, Trigger.BOTH)}
    assert resolve_scope_role(entries, {}) == ("pirate", [])


# ---------------------------------------------------------- term layers ---


def test_term_layers_report_fit_share_and_selection() -> None:
    """``term_layers`` (the term editor's depth strip): each term's fitted
    layers, its share there (mean 1), and the layers its clause selects."""
    rig, _ = _rig()
    rig.registry.put_manifold("mood", open_manifold(layers=(0, 1, 2)))
    expr = "0.5 a{1,3} + 0.4 !c + 0.2 b{workspace} + 0.3 a~b{0} + 0.5 mood%n1{0,2}"
    with rig.steering(expr) as scope:
        where = scope.term_layers()
    assert set(where) == {"a{1,3}", "!c", "b{workspace}", "a~b{0}", "mood%n1{0,2}"}

    a = where["a{1,3}"]
    assert (a.fitted, a.layers) == ((0, 1, 2, 3), (1, 3))
    assert sum(a.share) == pytest.approx(len(a.fitted))
    whitener = rig.neutral.resident_whitener
    assert whitener is not None
    raw = [
        float(whitener.mahalanobis_norms(L, rig.registry.profiles["a"][L].reshape(1, -1))[0])
        for L in a.fitted
    ]
    for L in range(1, len(raw)):
        assert a.share[L] / a.share[0] == pytest.approx(raw[L] / raw[0], rel=1e-4)

    assert where["!c"] == TermLayers((1, 2), (1.0, 1.0), (1, 2))
    assert where["b{workspace}"].layers == (2,)
    assert where["a~b{0}"].fitted == (0, 1, 2, 3)
    assert where["a~b{0}"].layers == (0,)

    curved = where["mood%n1{0,2}"]
    baked = rig.registry.manifolds["mood"].mahalanobis_share
    assert (curved.fitted, curved.layers) == ((0, 1, 2), (0, 2))
    assert curved.share[0] / curved.share[2] == pytest.approx(baked[0] / baked[2])
    assert sum(curved.share) == pytest.approx(3.0)


def test_term_layers_weigh_an_affine_position_by_its_whitened_push() -> None:
    """A flat ``%`` term (every flat concept chip: ``a.b%a``) is weighed by
    the whitened displacement of its push, ``‖c_L B_L‖_M``, at each layer, in
    both position forms (they agree at a node)."""
    from drowse.geometry.domains import CustomDomain
    from drowse.geometry.manifold import Manifold
    from drowse.geometry.subspace import fit_affine_subspace

    rig, _ = _rig()
    whitener = rig.neutral.resident_whitener
    assert whitener is not None
    means = rig.neutral.means
    spread = torch.randn(3, _D, generator=torch.Generator().manual_seed(11))
    scales = (1.0, 3.0, 2.0, 0.5)
    layers = {
        L: fit_affine_subspace(
            means[L] + scale * spread, neutral_mean=means[L], n_components=2,
            whitener=whitener, layer=L,
        ).subspace
        for L, scale in enumerate(scales)
    }
    fan = Manifold(
        name="fan", domain=CustomDomain(2), node_labels=["n0", "n1", "n2"],
        node_coords=torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
        layers=layers, feature_space="raw", node_roles=[None] * 3,
        node_kinds=[None] * 3, mahalanobis_share=dict.fromkeys(layers, 1.0),
    )
    assert all(sub.is_affine for sub in fan.layers.values())
    rig.registry.put_manifold("fan", fan)
    with rig.steering("0.5 fan%n1{1-2} + 0.5 fan%0,1{0,3}") as scope:
        where = scope.term_layers()

    raw = []
    for L in range(_N_LAYERS):
        node_coords = layers[L].node_coords
        assert node_coords is not None
        push = node_coords[1] @ layers[L].basis
        raw.append(float(whitener.mahalanobis_norms(L, push.reshape(1, -1))[0]))
    expected = [r / sum(raw) * _N_LAYERS for r in raw]
    assert max(expected) / min(expected) > 2.0, "the fixture must weigh layers unevenly"
    label, coords = where["fan%n1{1-2}"], where["fan%0,1{0,3}"]
    assert (label.fitted, label.layers) == ((0, 1, 2, 3), (1, 2))
    assert coords.layers == (0, 3)
    assert label.share == pytest.approx(expected, rel=1e-4)
    assert coords.share == pytest.approx(expected, rel=1e-4)


def test_term_layers_refuse_a_clause_composition_refuses() -> None:
    rig, _ = _rig()
    scope = rig.steering("0.5 c{0}")
    with pytest.raises(SteeringCompositionError, match=r"fitted layers: 1-2"):
        scope.term_layers()
    assert rig.manager.plan is None


@pytest.mark.usefixtures("isolated_home")
def test_term_layers_place_a_unit_at_its_layer(tiny_session: Any) -> None:
    where = tiny_session.steering("0.5 !neuron/L1n3 + !head/L2h0").term_layers()
    assert where == {
        "!neuron/L1n3": TermLayers((1,), (1.0,), (1,)),
        "!head/L2h0": TermLayers((2,), (1.0,), (2,)),
    }
