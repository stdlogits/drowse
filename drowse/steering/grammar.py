"""Unified steering-expression grammar and IR compiler.

One parser + formatter for the steering expression language shared across
every drowse input surface (Python, YAML, HTTP, CLI). Every surface
turns a user-supplied string into the same :class:`Steering` IR.

Grammar::

    expr        := term (("+" | "-") term)*
    term        := [coeff ["*"]] ["!"] selector ["{" layers "}"] ["@" trigger]
    layers      := the layer-selector language of drowse.model.layers:
                   indices and a-b ranges ("4-8,20"), "all" or "workspace"
    selector    := atom (("~" | "|") atom | "%" position)?
    position    := signed_num ("," signed_num)* | label   # coord list | node label
    label       := NAME                                # a manifold node label
    atom        := [ns "/"] NAME ["." NAME] [":" variant]
                 | "sae" "/" INT
                 | ("jlens" | "slens") "/" "#" INT      # exact vocab id
                 | ("neuron" | "head") "/" UNIT         # L8n344 / L20h17
    trigger     := phase ["&" gate] | gate
    phase       := preset | ("first" | "after") ":" INT
    preset      := "before" | "after" | "both" | "thinking" | "response"
                 | "prompt" | "generated"
    gate        := "when" ":" probe_atom op NUM
    probe_atom  := [ns "/"] NAME ["." NAME] ["[" INT "]"]  # vector probe (e.g.
                                                       # "angry.calm"), optional
                                                       # coord-axis index ("personas[3]")
                 | [ns "/"] NAME ":" ("fraction" | "membership")  # subspace
                                                       # fraction / tube-fit density
                 | [ns "/"] NAME "@" NAME              # manifold label similarity (−dist)
                 | [ns "/"] NAME "~" NAME              # soft-assignment prob ∈ [0,1]
                 | "sae" "/" INT                       # resident SAE feature
                 | ("jlens" | "slens") "/" "#" INT     # token probe by vocab id
                 | ("neuron" | "head") "/" UNIT        # unit probe (one scalar:
                                                       # no channel suffix)
    op          := ">" | ">=" | "<" | "<="
    coeff       := signed_float ("," signed_float)?  # comma-run of ≤2;
                                                       # >1 value is valid
                                                       # ONLY on a "%" term.
                                                       # 1→along (onto off),
                                                       # 2→along,onto.
                                                       # Optional; defaults
                                                       # to DEFAULT_COEFF = 0.5
    variant     := "raw" | "sae" | "sae-" ID
                 | "role" | "role-" ID | "from" | "from-" ID

Reserved instrument namespaces (:data:`RESERVED_NAMESPACES`: ``jlens``,
``slens``, ``sae``, ``neuron``, ``head``) name instrument units, not fitted
concepts: the name after the ``/`` is kept verbatim (case-preserving, so
``jlens/Paris`` and ``@when:jlens/Paris`` agree and differ from
``jlens/paris``), takes no ``:variant``, and never routes through the
manifold tiers.

Unit atoms ``neuron/L8n344`` and ``head/L20h17`` name one MLP neuron or one
attention head (:func:`~drowse.naming.parse_unit_ref`: case-insensitive input,
canonical ``L8n344``).  ``α neuron/…`` is a plain term keyed ``neuron/L8n344``
(a push along the neuron's write; the sign of ``α`` picks the pole).  A head
has no single write direction, so ``α head/…`` is a parse error;
``!neuron/…`` and ``!head/…`` are :class:`UnitAblationTerm` s (the unit's
activation replaced by its neutral mean, not a residual direction ablation).
Unit atoms do not compose with ``~``, ``|`` or ``%`` and take no layer clause.

A ``{…}`` layer clause scopes the term to exact layers (``0.5 honest{12-20}``,
``!jlens/Paris{24-30}@response``, ``0.4 personas%hacker{workspace}``).  It
follows the whole selector, precedes ``@``, and is part of the term's
identity: the ``Steering.alphas`` key carries the canonical clause, so the
same selector under two clauses is two terms.  ``neuron/`` and ``head/``
atoms carry their own layer and take no clause.

``!`` mean-ablates the selector (``h' = h − α(h·d̂ − μ·d̂)d̂``; bare
``!x`` is α=1.0); it does not compose with ``~`` / ``|`` / ``%``.  The
``%`` operator places a generation on a fitted manifold — the
``position`` is either a comma-separated list of authoring coordinates
(one per intrinsic dimension) or a single node-label string (sugar for
that node's coords); the parser only collects the payload, arity /
label-existence is validated at manifold-load time.

Probe gates: ``@when:<probe><op><threshold>`` fires the term only on
decode steps where the named probe's last reading satisfies the
comparison.  Implicit ``prompt=False`` (no probe reading during prefill).
A gate composes with a phase window as ``@<phase>&when:...``
(``@after&when:angry.calm>0.4`` = post-thinking *and* gated).

Counted decode windows: ``@first:N`` / ``@after:N`` are the grammar forms
of :meth:`Trigger.first` / :meth:`Trigger.after` — apply only to the first
``N`` generated tokens, or only from token ``N`` on.  The ``:N`` is
required, and is what separates ``@after:5`` (token window) from the bare
``@after`` preset (post-thinking).

Between them these cover every ``Trigger`` a public factory builds, so
``parse_expr(format_expr(s)) == s`` holds for any ``Steering`` whose
triggers came from the public surface.  A ``Trigger`` hand-built outside
that set has no grammar form and :func:`format_expr` raises rather than
emitting an unparseable string.

Manifold-probe gates extend the same shape over the two scalar channels
:class:`Monitor` exposes: ``@when:<manifold>:fraction <op> N``
fires on the subspace-fraction reading (the share of the centered
activation living in the manifold's PCA subspace), and
``@when:<manifold>@<label> <op> N`` fires on the negated distance to a
named node (larger = closer; label-similarity gates routinely use
negative thresholds).  The gate's probe string is stored verbatim
(``"emotions:fraction"``, ``"emotions@happy"``) so it matches the
key ``Monitor.flat_scalars`` already merges into
``TriggerContext.probe_scores``; no runtime gate machinery changes.

Coordinate-axis gates: a vector probe now reads out a coordinate per
intrinsic axis, so ``@when:<probe>[<i>] <op> N`` gates on one axis of the
probe's coordinate vector (``@when:personas[3] > 0.4``).  The index is
appended to the verbatim probe string (``"personas[3]"``) and the session
writes a matching per-axis scalar channel into ``probe_scores`` for every
axis, aliasing the bare ``<probe>`` to axis 0 — so an un-indexed gate on a
2-node concept (``@when:angry.calm > 0.4``) reads its single coordinate
exactly as before.

Concept names are ASCII identifiers: letter followed by any of
``[a-z0-9_-]``.  Multi-word concepts use underscores
(``artificial_intelligence``) — spaces separate tokens, so
``artificial intelligence`` errors with an underscore hint.  Quoted
identifiers are rejected.  Bipolar pairs join with ``.``
(``human.artificial_intelligence``).

A bare atom resolves through :func:`drowse.store.selectors.resolve_bare_atom`
(manifold-label tier, then 2-node manifold name, then plain canonical concept)
before the term lands in ``Steering.alphas``; a bipolar pole is itself a node
label, lowered to an affine ``%`` push.  Projection terms produce
:class:`ProjectedTerm` values that the composer materializes on scope entry.

Manifold steering: the ``%`` infix operator places a generation at a
point of a fitted manifold — ``manifold % coord_list``, e.g.
``0.7 emotions%0.3,0.8,0.0@response``.  The left operand is a manifold name
(not a concept; no pole resolution), the right is a comma-separated list
of authoring coordinates — one per intrinsic dimension of the manifold's
domain.  The parser only collects the coordinate tuple; arity and range
are validated at manifold-load time, when the domain is known.  A ``%``
selector does not compose with the ``~`` / ``|`` projection operators or
with ``!`` ablation.  Manifold terms produce :class:`ManifoldTerm` values;
the composer loads the manifold artifact and lowers it into the steering plan.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Literal, Optional, TYPE_CHECKING, cast

from drowse.errors import SteeringExprError, UnitRefError
from drowse.model.layers import (
    LayerSelectorError,
    format_layer_indices,
    parse_layer_selector,
)
from drowse.naming import RESERVED_NAMESPACES, UNIT_NAMESPACES, UnitRef, parse_unit_ref
from drowse.steering.triggers import Trigger

if TYPE_CHECKING:
    from drowse.steering.terms import AlphaEntry, Steering


# Namespaces whose atom also takes the exact-vocab-id form ``#<int>``.
_TOKEN_ID_NAMESPACES = frozenset({"jlens", "slens"})

#: A term's layer scope in the IR.  ``None`` acts on every layer the term is
#: fitted at; ``"workspace"`` on the model's workspace band intersected with
#: them; a sorted tuple on exactly those layers.  ``{all}`` canonicalizes to
#: ``None``.
LayerScope = tuple[int, ...] | Literal["workspace"] | None

# Default coefficient when a term omits the explicit number — the observed
# coherent-α sweet spot post-share-baking.
DEFAULT_COEFF = 0.5

# Shared message for the two structurally-parallel "manifold doesn't
# compose" parse guards (the ``%``-with-projection / second-``%`` guard
# and the ``!``-with-``%`` guard).  Consolidated so both surfaces report
# the same constraint in the same words.
_MANIFOLD_COMPOSE_MSG = (
    "a manifold term does not compose with projection ('~'/'|') or "
    "ablation ('!')"
)


_UNIT_COMPOSE_MSG = (
    "a unit atom ('neuron/…', 'head/…') does not compose with projection "
    "('~'/'|') or a manifold position ('%')"
)


# Default coefficient for ablation terms.  Bare ``!x`` means "fully
# replace the component along d̂ with the neutral-baseline mean" — the
# intuitive "ablation" semantics from the literature.  Partial ablation
# (``0.5 !x``) blends the original component toward the mean.  This
# intentionally differs from ``DEFAULT_COEFF`` (0.5), which applies to
# additive terms.
DEFAULT_ABLATION_COEFF = 1.0


_TRIGGER_PRESETS: dict[str, Trigger] = {
    "both": Trigger.BOTH,
    "after": Trigger.AFTER_THINKING,
    "before": Trigger.PROMPT_ONLY,
    "thinking": Trigger.THINKING_ONLY,
    "response": Trigger.GENERATED_ONLY,
    "prompt": Trigger.PROMPT_ONLY,
    "generated": Trigger.GENERATED_ONLY,
}

# Preferred render string per preset.  Multiple grammar tokens alias onto
# the same Trigger (``@before`` == ``@prompt``); we pick one canonical form
# so round-tripping is deterministic.
_TRIGGER_CANONICAL: dict[Trigger, str] = {
    Trigger.BOTH: "both",
    Trigger.AFTER_THINKING: "after",
    Trigger.PROMPT_ONLY: "before",
    Trigger.THINKING_ONLY: "thinking",
    Trigger.GENERATED_ONLY: "response",
}

# Counted decode windows: ``@first:N`` / ``@after:N`` are the grammar forms of
# :meth:`Trigger.first` / :meth:`Trigger.after`.  Both require the ``:N``, which
# is also what disambiguates ``@after:5`` (the token window) from the bare
# ``@after`` preset (post-thinking).  Rendered by :func:`_phase_name`.
_COUNTED_TRIGGER_KEYWORDS = ("first", "after")

# One human-readable list of everything ``@`` accepts, shared by the parse
# errors so the grammar surface is described in exactly one place.
_TRIGGER_FORMS_HELP = (
    "valid: " + ", ".join(sorted(_TRIGGER_PRESETS)) + "; "
    "'first:<n>' / 'after:<n>' token windows; "
    "'when:<probe><op><threshold>' probe gates; "
    "or a compound '<preset>&when:<probe><op><threshold>'"
)


@dataclass(frozen=True)
class ProjectedTerm:
    """Runtime projection entry in ``Steering.alphas``.

    Keyed by ``"<base><op><onto>[{clause}]"`` so duplicate references
    compose.  The steering composer projects the ``base`` profile against
    ``onto`` (``project_profile``) at lowering time and folds the result like
    a plain vector; the derived direction is never registered as a profile.
    ``layers`` is the term's :data:`LayerScope`.
    """
    coeff: float
    trigger: Trigger
    operator: Literal["~", "|"]
    base: str
    onto: str
    layers: LayerScope = None


@dataclass(frozen=True)
class AblationTerm:
    """Mean-replacement ablation entry in ``Steering.alphas``.

    ``SteeringComposer.compose`` lowers it to an ablate fragment that
    ``synthesize_subspace`` folds into the merged affine subspace (per-axis
    ``κ`` on the ablation axes). Stored as a value inside
    ``Steering.alphas``; the key is ``"!<target>[{clause}]"`` so ablation
    entries never collide with plain-term entries on the same concept.
    ``layers`` is the term's :data:`LayerScope`.
    """
    coeff: float
    trigger: Trigger
    target: str
    layers: LayerScope = None


@dataclass(frozen=True)
class UnitAblationTerm:
    """Unit mean-ablation entry in ``Steering.alphas``: ``!neuron/…`` / ``!head/…``.

    Silences one unit: its activation (a neuron's ``act``, a head's ``z_h``)
    is replaced by its neutral mean, blended by ``coeff`` (bare ``!`` is 1.0).
    Lowered as a unit hook, never as a residual direction ablation, so it is
    not an :class:`AblationTerm`.  Keyed ``"!neuron/L8n344"``; a unit carries
    its layer, so there is no layer clause.
    """
    unit: UnitRef
    coeff: float
    trigger: Trigger


@dataclass(frozen=True)
class ManifoldTerm:
    """Manifold-steering entry in ``Steering.alphas``.

    Produced by a ``manifold % position`` term.  The composer loads the
    named :class:`~drowse.geometry.manifold.Manifold` artifact on scope entry
    and lowers the term into the steering plan: an affine (flat) fit joins
    the merged push subspace, a curved fit gets its own along/onto injection
    (:func:`~drowse.geometry.inject.subspace_inject`).  Stored as a value
    inside ``Steering.alphas`` under the key
    ``"<manifold>%<position>[{clause}]"`` so two positions on the same
    manifold compose as distinct entries and never collide with plain /
    projected / ablation keys.  ``layers`` is the term's :data:`LayerScope`.

    The two coefficients:

    - ``along`` — slide the projected foot toward ``position`` geodesically
      in coordinate space (the principled "directional" op).  On an affine
      fit it is the signed, unclamped push magnitude (negative pushes away);
      on a curved fit it is a fraction of the way to the position — values
      above 1 cap at 1 and a negative value raises
      :class:`~drowse.errors.SteeringCompositionError`, since a curved
      position has no opposite.
    - ``onto`` — collapse the off-manifold, in-subspace residual onto the
      surface, a fraction in ``[0, 1]`` (vacuous when the surface fills its
      subspace, so ignored on an affine fit).

    The off-*subspace* residual is always kept verbatim, which is what lets a
    manifold term compose with neighbors at zero cross-talk; ``~`` / ``|``
    semantics live inside the subspace, as push/ablation axes of the merged
    affine basis.

    The grammar's coefficient slot expands to a comma-run of ≤ 2: one coeff
    ⇒ ``along`` (``onto`` off); two ⇒ ``along, onto``.  ``manifold`` is the
    registry key — namespace-qualified when the user typed a namespace,
    variant-suffixed when not ``raw``.

    ``position`` is either:

    - A tuple of authoring coordinates — one float per intrinsic
      dimension of the manifold's domain (the historical shape, e.g.
      ``(0.3, 0.8)`` on a 2-D affect manifold).
    - A node label string — sugar for "the coords of the node labeled
      <s> on this manifold", resolved at scope entry via
      :meth:`Manifold.resolve_position`.  The label form makes
      ``persona%pirate`` a first-class steering term parallel to
      ``persona%0.3,0.8``.

    Round-trip via :func:`format_expr` preserves the authored form.
    """
    along: float
    onto: float
    trigger: Trigger
    manifold: str
    position: tuple[float, ...] | str
    layers: LayerScope = None

    @property
    def coeff(self) -> float:
        """Representative scalar — the ``along`` (directional) coefficient.

        Read by the uniform telemetry / snapshot / role-aggregation sites
        that want one "how strongly is this term steering" number; ``along``
        is the directional strength that picks the node, so it is the right
        representative.  Not a stored field — there is no dual shape.
        """
        return self.along


# ``SteeringExprError`` lives in :mod:`drowse.errors`; it is imported
# above and re-exported here so
# ``from drowse.steering.grammar import SteeringExprError`` keeps working.


# ---------------------------------------------------------------- lexer ---

# ASCII digits only (``re.ASCII``): ``\d`` would also take Arabic-Indic or
# fullwidth digits, which a JavaScript ``\d`` refuses.
_NUM_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
_DIGITS = frozenset("0123456789")
_IDENT_START_RE = re.compile(r"[A-Za-z]")
_IDENT_CHAR_RE = re.compile(r"[A-Za-z0-9_]")

_SINGLE_CHAR_TOKENS = {
    ".": "DOT", "/": "SLASH", ":": "COLON", "*": "STAR",
    "+": "PLUS", "-": "MINUS", "@": "AT", "~": "TILDE",
    "|": "ORTHO", "!": "BANG", "%": "PERCENT", ",": "COMMA",
    "[": "LBRACK", "]": "RBRACK", "&": "AMP", "#": "HASH",
}

# Comparison-op tokens.  Two-char ops (``>=``, ``<=``) take precedence
# over the single-char ones; the lexer peeks one char ahead before
# emitting GT / LT.  ``=`` alone is not a token — equality (``==``) is
# not part of the gate grammar in v2.1 (probe scores are continuous
# floats; equality is meaningless).
_COMPARE_OPS: dict[str, str] = {
    ">=": "GTE", "<=": "LTE",
    ">": "GT", "<": "LT",
}


@dataclass
class _Tok:
    kind: str
    value: str | float
    col: int


def _lex(text: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        # Number detection must precede the single-char DOT branch, since
        # ``.25`` starts with a DOT that is actually a decimal point.
        if c in _DIGITS or (c == "." and i + 1 < n and text[i + 1] in _DIGITS):
            m = _NUM_RE.match(text, i)
            if m is None:  # pragma: no cover — digit guard
                raise SteeringExprError(f"malformed number at {c!r}", col=i)
            toks.append(_Tok("NUM", float(m.group()), i))
            i = m.end()
            continue
        # Comparison ops: two-char first, then single-char.
        if c in (">", "<"):
            two = text[i:i + 2]
            if two in _COMPARE_OPS:
                toks.append(_Tok(_COMPARE_OPS[two], two, i))
                i += 2
                continue
            toks.append(_Tok(_COMPARE_OPS[c], c, i))
            i += 1
            continue
        if c in _SINGLE_CHAR_TOKENS:
            toks.append(_Tok(_SINGLE_CHAR_TOKENS[c], c, i))
            i += 1
            continue
        if c == "{":
            # A layer clause: its body is the layer-selector language of
            # ``drowse.model.layers``, handed to that parser verbatim.
            close = text.find("}", i + 1)
            if close < 0:
                raise SteeringExprError(
                    "unterminated layer clause; close it with '}'", col=i,
                )
            toks.append(_Tok("LAYERS", text[i + 1:close], i))
            i = close + 1
            continue
        if _IDENT_START_RE.match(c):
            start = i
            i += 1
            while i < n:
                ch = text[i]
                if _IDENT_CHAR_RE.match(ch):
                    i += 1
                    continue
                # Dash-joined segments inside an ident (e.g. sae-release)
                # require a valid ident-char on each side; otherwise the
                # dash is a MINUS operator.
                if (
                    ch == "-"
                    and i + 1 < n
                    and _IDENT_CHAR_RE.match(text[i + 1])
                ):
                    i += 1
                    continue
                break
            toks.append(_Tok("IDENT", text[start:i], start))
            continue
        if c in ('"', "'"):
            raise SteeringExprError(
                "quoted identifiers are not supported; use underscores "
                "for multi-word concept names "
                "(e.g. 'artificial_intelligence') and '.' for bipolar "
                "pairs (e.g. 'human.artificial_intelligence')",
                col=i,
            )
        raise SteeringExprError(f"unexpected character {c!r}", col=i)
    toks.append(_Tok("EOF", "", n))
    return toks


# ------------------------------------------------------------------ ast ---

@dataclass
class _Atom:
    namespace: Optional[str]
    concept: str  # may contain a single '.' joining two poles
    variant: str  # 'raw' (default) or an store.selectors tensor variant
    col: int


@dataclass
class _Selector:
    base: _Atom
    operator: Optional[str]  # None | '~' | '|'
    onto: Optional[_Atom]
    # Set when the selector is a manifold position term
    # (``base % NUM (, NUM)*``); mutually exclusive with
    # ``operator`` / ``onto``.
    # ``tuple[float, ...]`` for coord-form (``%0.3,0.8``); ``str`` for
    # label-form (``%pirate``).  ``None`` when this selector is not a
    # manifold term.
    manifold_position: Optional[tuple[float, ...] | str] = None


@dataclass
class _Term:
    coeff: float
    selector: _Selector
    # Pre-resolved Trigger object (or ``None`` to fall through to the
    # session/Steering default).  Preset triggers map through
    # :data:`_TRIGGER_PRESETS`; probe gates (``@when:angry>0.4``)
    # produce a ``Trigger(prompt=False, gate=ProbeGate(...))`` directly
    # at parse time.  Rendered back to canonical form by ``_fmt_*``.
    trigger: Optional[Trigger]
    ablation: bool = False  # True iff term was prefixed with `!`
    # The full comma-run of coefficients the user typed.  A plain term
    # carries a 1-tuple; a manifold ``%`` term may carry up to 2 mapping
    # to (along, onto) via :func:`_expand_along_onto_coeffs`.
    # ``coeff`` always equals ``coeffs[0]`` (the representative scalar all
    # non-manifold code paths read), so existing single-coeff behavior is
    # bit-identical; ``> 1`` on a non-manifold term is a parse-fold error.
    coeffs: tuple[float, ...] = ()
    # The ``{…}`` layer clause as written: ``None`` without one, else
    # ``"all"``, ``"workspace"`` or the sorted index tuple.  ``_fold``
    # canonicalizes ``"all"`` away.
    layers: tuple[int, ...] | Literal["all", "workspace"] | None = None

    def __post_init__(self) -> None:
        if not self.coeffs:
            self.coeffs = (self.coeff,)


# --------------------------------------------------------------- parser ---

class _Parser:
    def __init__(self, toks: list[_Tok]) -> None:
        self._toks = toks
        self._pos = 0

    def _peek(self, off: int = 0) -> _Tok:
        return self._toks[self._pos + off]

    def _consume(self) -> _Tok:
        t = self._toks[self._pos]
        self._pos += 1
        return t

    def _expect(self, kind: str) -> _Tok:
        t = self._peek()
        if t.kind != kind:
            raise SteeringExprError(
                f"expected {kind}, got {t.kind} ({t.value!r})", col=t.col,
            )
        return self._consume()

    def parse(self) -> list[_Term]:
        sign = +1
        if self._peek().kind in ("PLUS", "MINUS"):
            sign = -1 if self._consume().kind == "MINUS" else +1
        terms = [self._term(sign)]
        while self._peek().kind in ("PLUS", "MINUS"):
            op_sign = -1 if self._consume().kind == "MINUS" else +1
            terms.append(self._term(op_sign))
        if self._peek().kind != "EOF":
            t = self._peek()
            if t.kind == "IDENT":
                raise SteeringExprError(
                    f"unexpected identifier {t.value!r} after a complete "
                    f"term; multi-word concept names use underscores "
                    f"(e.g. 'artificial_intelligence', not "
                    f"'artificial intelligence'), and bipolar pairs join "
                    f"with '.' (e.g. 'human.artificial_intelligence')",
                    col=t.col,
                )
            raise SteeringExprError(
                f"unexpected token {t.kind} ({t.value!r})", col=t.col,
            )
        return terms

    def _term(self, sign: int) -> _Term:
        explicit = False
        coeff = float(sign) * DEFAULT_COEFF
        coeffs: tuple[float, ...] = (coeff,)
        if self._peek().kind == "NUM":
            coeff = float(sign) * float(self._consume().value)
            explicit = True
            # Comma-run of coefficients: the TERM-level coeff slot may carry
            # up to 2 values mapping to (along, onto) on a manifold ``%``
            # term.  This run is lexically unambiguous from the post-``%``
            # position commas — those follow the manifold IDENT and ``%``,
            # this precedes the selector.  Each subsequent value carries the
            # term's leading sign (so ``-0.6,0.3`` is ``(-0.6, -0.3)``, the
            # natural read of a signed term).  A 3rd value is a parse error;
            # whether a comma-run is even legal for the selector kind
            # (manifold-only) is enforced in ``_fold``.
            run = [coeff]
            while self._peek().kind == "COMMA":
                self._consume()
                run.append(float(sign) * self._signed_num())
                if len(run) > 2:
                    raise SteeringExprError(
                        "a manifold coefficient slot takes at most 2 "
                        "comma-separated values (along, onto); got more",
                        col=self._peek(-1).col,
                    )
            coeffs = tuple(run)
            if self._peek().kind == "STAR":
                self._consume()
        ablation = False
        if self._peek().kind == "BANG":
            self._consume()
            ablation = True
            if not explicit:
                # Bare `!x` defaults to fully-replace (coeff=1.0).
                coeff = float(sign) * DEFAULT_ABLATION_COEFF
                coeffs = (coeff,)
        selector = self._selector()
        if selector.base.namespace == "head" and not ablation:
            raise SteeringExprError(
                "a head has no single write direction; silence it with "
                f"!head/{selector.base.concept}",
                col=selector.base.col,
            )
        layers: tuple[int, ...] | Literal["all", "workspace"] | None = None
        if self._peek().kind == "LAYERS":
            layers = self._layer_clause(selector)
        trigger: Trigger | None = None
        if self._peek().kind == "AT":
            self._consume()
            trigger = self._parse_trigger()
            if self._peek().kind == "LAYERS":
                raise SteeringExprError(
                    "the layer clause goes before the '@' trigger "
                    "(e.g. 'x{4-8}@response'); gate probes take no layer clause",
                    col=self._peek().col,
                )
        return _Term(
            coeff=coeff, selector=selector, trigger=trigger,
            ablation=ablation, coeffs=coeffs, layers=layers,
        )

    def _layer_clause(
        self, selector: _Selector,
    ) -> tuple[int, ...] | Literal["all", "workspace"]:
        """Parse a ``{…}`` layer clause through the one layer-selector parser.

        Returns ``"all"``, ``"workspace"`` or the sorted index tuple.  An empty
        clause, ``sample``, any selector error, a clause on a ``neuron/`` or
        ``head/`` atom, a second clause, or a selector operator after the
        clause raises with the failing column.
        """
        tok = self._consume()
        for atom in (selector.base, selector.onto):
            if atom is not None and atom.namespace in UNIT_NAMESPACES:
                raise SteeringExprError(
                    f"'{atom.namespace}/' atoms carry their own layer and take "
                    f"no layer clause",
                    col=tok.col,
                )
        body = str(tok.value)
        try:
            parsed = parse_layer_selector(body, allow_sample=False)
        except LayerSelectorError as exc:
            offset = 0 if exc.offset is None else exc.offset
            raise SteeringExprError(
                f"layer clause: {exc}", col=tok.col + 1 + offset,
            ) from None
        if parsed is None:
            raise SteeringExprError(
                "empty layer clause; name layers ('{4-8,20}'), 'workspace' "
                "or 'all'",
                col=tok.col,
            )
        nxt = self._peek()
        if nxt.kind == "LAYERS":
            raise SteeringExprError(
                "a term takes one layer clause", col=nxt.col,
            )
        if nxt.kind in ("TILDE", "ORTHO", "PERCENT"):
            raise SteeringExprError(
                "the layer clause follows the whole selector "
                "(e.g. 'a~b{4-8}', 'm%x{4-8}')",
                col=nxt.col,
            )
        if isinstance(parsed, list):
            return tuple(sorted(parsed))
        return cast(Literal["all", "workspace"], parsed)

    def _parse_trigger(self) -> Trigger:
        """Parse the whole ``@`` clause into a :class:`Trigger`.

        Already consumed: the ``@``.  Grammar::

            trigger := (phase ["&" gate] | gate)
            phase   := preset | ("first" | "after") ":" INT

        Every publicly-constructible ``Trigger`` has a form here — the five
        presets, the counted decode windows :meth:`Trigger.first` /
        :meth:`Trigger.after`, the bare probe gate :meth:`Trigger.when`, and a
        gate narrowed to a phase (``@after&when:x>0.4``, the shape
        ``Trigger.when``'s own docstring points callers at).  That totality is
        what keeps ``parse_expr(format_expr(s)) == s``: without it,
        ``format_expr`` would have to emit an unparseable sentinel and every
        Recipe / ``applied_steering`` string stamped from such a Trigger would
        poison the regen / transcript / loom replay that re-parses it.
        """
        tok = self._expect("IDENT")
        kw = str(tok.value)
        if kw == "when":
            # A bare gate: no phase clause, so it can't be followed by "&".
            return self._parse_when_gate_clause(tok.col)
        base = self._parse_phase(tok)
        if self._peek().kind != "AMP":
            return base
        self._consume()  # AMP
        gate_tok = self._expect("IDENT")
        if str(gate_tok.value) != "when":
            raise SteeringExprError(
                f"a compound trigger's second clause must be a probe gate "
                f"('&when:<probe><op><threshold>'); got '{gate_tok.value}'",
                col=gate_tok.col,
            )
        gated = self._parse_when_gate_clause(gate_tok.col)
        # The phase clause supplies the window verbatim, the gate clause the
        # condition — so ``@<preset>&when:…`` is exactly "that preset, gated",
        # which is what makes the compound form an exact round-trip of a
        # ``replace(preset, gate=…)`` Trigger.  The phase's ``prompt`` flag is
        # left alone: a gate can never fire during prefill anyway (there is no
        # post-forward probe reading yet), so ``Trigger.active`` ignores it.
        return replace(base, gate=gated.gate)

    def _parse_phase(self, tok: _Tok) -> Trigger:
        """Parse a preset name or a ``first:N`` / ``after:N`` token window."""
        kw = str(tok.value)
        if kw in _COUNTED_TRIGGER_KEYWORDS and self._peek().kind == "COLON":
            self._consume()  # COLON
            n_tok = self._expect("NUM")
            n_val = float(n_tok.value)
            n = int(n_val)
            if n_val != n or n < 0:
                raise SteeringExprError(
                    f"trigger '@{kw}:' takes a non-negative integer token "
                    f"count, got '{n_tok.value}'",
                    col=n_tok.col,
                )
            return Trigger.first(n) if kw == "first" else Trigger.after(n)
        if kw == "first":
            raise SteeringExprError(
                "trigger '@first' requires ':' followed by a token count "
                "(e.g. '@first:5')",
                col=tok.col,
            )
        if kw in _TRIGGER_PRESETS:
            return _TRIGGER_PRESETS[kw]
        raise SteeringExprError(
            f"unknown trigger '@{kw}'; {_TRIGGER_FORMS_HELP}.  Note: '@' is "
            f"for triggers only; HF revisions are not accepted inside "
            f"steering expressions.",
            col=tok.col,
        )

    def _parse_when_gate_clause(self, when_col: int) -> Trigger:
        """Consume the ``:`` after a ``when`` keyword, then its gate payload."""
        if self._peek().kind != "COLON":
            raise SteeringExprError(
                f"trigger '@when' requires ':' followed by a "
                f"probe gate (e.g. '@when:angry.calm>0.4'); "
                f"got {self._peek().kind}",
                col=self._peek().col,
            )
        self._consume()  # COLON
        return self._parse_when_gate(when_col)

    def _parse_when_gate(self, when_col: int) -> Trigger:
        """Parse the ``when:`` payload into a probe-gated :class:`Trigger`.

        Already consumed: ``@`` ``when`` ``:``.  Expects a probe
        identifier, one of ``>``/``>=``/``<``/``<=``, then a numeric
        threshold.

        Three probe identifier shapes are accepted:

        - Vector probe: ``IDENT`` optionally followed by ``.IDENT`` for
          a bipolar concept (e.g. ``angry.calm``), and an optional
          ``[<i>]`` coordinate-axis index (e.g. ``personas[3]``,
          ``angry.calm[0]``).  No variant suffix.  The index selects
          one axis of the probe's coordinate readout; an un-indexed gate
          reads axis 0 (the only axis of a 2-node concept), so existing
          ``@when:angry.calm>0.4`` gates are unchanged.
        - Manifold subspace-fraction probe:
          ``<manifold>:fraction`` — fires on the fraction of the
          centered activation that lives in the manifold's PCA
          subspace.  Stored verbatim as the gate's probe string
          (e.g. ``"emotions:fraction"``); the session's
          :class:`Monitor.flat_scalars` already emits a
          matching namespaced key, so ``Trigger.active`` looks it up
          unchanged.
        - Manifold label-similarity probe:
          ``<manifold>@<label>`` — fires on the negated distance to
          the named node (larger = closer).  Stored verbatim as the
          gate's probe string (e.g. ``"emotions@happy"``); same
          ``flat_scalars`` correspondence.

        Every shape also accepts a leading ``<ns>/`` namespace segment
        (``jlens/fake``, ``default/emotions@happy``) — consumed before
        the channel discriminators, stored verbatim.

        The discriminator on the trailing IDENT: a ``COLON`` after the
        manifold name routes to the fraction form, an ``AT`` to the
        label form, a ``DOT`` to the bipolar vector form, anything
        else is a plain vector probe.  Within the ``when:`` body the
        ``AT`` token cannot be a fresh trigger marker (one ``@`` clause
        per term, already consumed), so the disambiguation is
        unambiguous.
        """
        probe_tok = self._expect("IDENT")
        probe = str(probe_tok.value)
        if self._peek().kind == "SLASH":
            # Namespaced probe: ``<ns>/<name>`` — an instrument probe
            # (``jlens/fake``, ``jlens/#1234``, ``sae/12``) or a probe attached
            # under a qualified selector (``default/emotions``).  Consumed
            # before the channel discriminators so every channel shape
            # composes with it; stored verbatim, matching the scalar keys the
            # instruments emit.
            self._consume()
            namespace = probe
            probe = f"{probe}/{self._namespaced_name(probe)}"
            if namespace in UNIT_NAMESPACES and self._peek().kind in (
                "COLON", "AT", "TILDE", "LBRACK",
            ):
                # A unit emits one scalar under its bare name: a channel
                # suffix would name a key no instrument writes, and the gate
                # would sit inactive forever.
                raise SteeringExprError(
                    f"a unit probe ('{probe}') is one scalar and takes no "
                    f"channel suffix ('{self._peek().value}')",
                    col=self._peek().col,
                )
        nxt = self._peek().kind
        if nxt == "COLON":
            # Manifold subspace-fraction gate: ``<manifold>:fraction``.
            # The suffix after the colon must be the literal slug
            # ``fraction`` — the only fraction-channel scalar
            # ``Monitor.flat_scalars`` emits.  Any other slug
            # would silently never match a probe score, so we surface
            # the typo here rather than letting the gate report
            # inactive forever.
            self._consume()
            suffix_tok = self._expect("IDENT")
            suffix = str(suffix_tok.value)
            # ``fraction`` (subspace energy share) and ``membership`` (the
            # fuzzy-manifold tube-fit density ∈ [0,1]) are the two scalar
            # ``:``-channels ``Monitor.flat_scalars`` emits.
            if suffix not in ("fraction", "membership"):
                raise SteeringExprError(
                    f"unknown manifold gate channel "
                    f"'{probe}:{suffix}'; expected "
                    f"'<manifold>:fraction', '<manifold>:membership', "
                    f"'<manifold>@<label>', or '<manifold>~<label>'",
                    col=suffix_tok.col,
                )
            probe = f"{probe}:{suffix}"
        elif nxt == "AT":
            # Manifold label-similarity gate: ``<manifold>@<label>``.
            # Label existence is not validated at parse time — the
            # manifold artifact may not be loaded yet, and a stale
            # gate against a removed label should report inactive (no
            # matching key in ``flat_scalars``), not raise.
            self._consume()
            label_tok = self._expect("IDENT")
            probe = f"{probe}@{label_tok.value}"
        elif nxt == "TILDE":
            # Fuzzy-manifold soft-assignment gate: ``<manifold>~<label>`` —
            # fires on the node's assignment *probability* ∈ [0,1] (the
            # normalized counterpart to the ``@<label>`` negated distance).
            # Same verbatim-storage / ``flat_scalars`` correspondence; label
            # existence unvalidated at parse time (stale → inactive, not raise).
            self._consume()
            label_tok = self._expect("IDENT")
            probe = f"{probe}~{label_tok.value}"
        else:
            if nxt == "DOT":
                self._consume()
                rhs = self._expect("IDENT")
                probe = f"{probe}.{rhs.value}"
            # Optional coordinate-axis index for a multi-axis probe:
            # ``personas[3]`` / ``angry.calm[0]``.  Selects one axis of the
            # probe's coordinate readout.  The session writes a matching
            # ``<probe>[<i>]`` scalar channel into ``probe_scores`` for
            # every coord axis, and aliases the bare ``<probe>`` to axis 0,
            # so an un-indexed gate on a 2-node concept reads its single
            # coordinate exactly as before.  Indexing is not accepted after
            # the ``:fraction`` / ``@label`` manifold channels — those are
            # already scalar.
            if self._peek().kind == "LBRACK":
                self._consume()
                idx_tok = self._expect("NUM")
                idx_val = float(idx_tok.value)
                idx = int(idx_val)
                if idx_val != idx or idx < 0:
                    raise SteeringExprError(
                        f"coordinate index must be a non-negative integer, "
                        f"got '{idx_tok.value}'",
                        col=idx_tok.col,
                    )
                self._expect("RBRACK")
                probe = f"{probe}[{idx}]"
        op_tok = self._consume()
        op_kind = op_tok.kind
        if op_kind not in ("GT", "GTE", "LT", "LTE"):
            raise SteeringExprError(
                f"expected comparison op (>, >=, <, <=) in '@when:' "
                f"clause, got {op_kind} ({op_tok.value!r})",
                col=op_tok.col,
            )
        op_str = {"GT": ">", "GTE": ">=", "LT": "<", "LTE": "<="}[op_kind]
        # Allow a leading ``-`` on the threshold so ``@when:x>-0.5``
        # parses as ``threshold = -0.5``.
        threshold = self._signed_num()
        return Trigger.when(probe, op_str, threshold)  # pyright: ignore[reportArgumentType]  # dict[str,str] widens the literal op value; runtime is always a valid ProbeGateOp

    def _signed_num(self) -> float:
        """Parse an optionally signed numeric literal: ``[+|-] NUM``."""
        sign = +1.0
        if self._peek().kind == "MINUS":
            self._consume()
            sign = -1.0
        elif self._peek().kind == "PLUS":
            self._consume()
        num_tok = self._expect("NUM")
        return sign * float(num_tok.value)

    def _selector(self) -> _Selector:
        base = self._atom()
        if base.namespace in UNIT_NAMESPACES and self._peek().kind in (
            "PERCENT", "TILDE", "ORTHO",
        ):
            raise SteeringExprError(_UNIT_COMPOSE_MSG, col=self._peek().col)
        if self._peek().kind == "PERCENT":
            # Manifold position term: ``base % NUM(,NUM)*`` (coord form)
            # OR ``base % IDENT`` (label form, sugar for "the coords of
            # the node labeled <s>"; resolved at scope entry).  The
            # parser only collects the position payload — arity (against
            # the domain's intrinsic dimension) for the coord form, and
            # label existence + role-paired lookup for the label form,
            # are checked at manifold-load time when both the domain
            # and the node labels are known.
            self._consume()
            head = self._peek()
            position: tuple[float, ...] | str
            if head.kind == "IDENT":
                # Label form: a single bare identifier.  No comma list
                # — a manifold node label is one slug, not a tuple.
                # Variant suffixes (``%pirate:role-pirate``) are not
                # meaningful on a position and so not part of the
                # grammar; if the user typed one, it lexed as a
                # separate ``:`` token and will fall out as a parse
                # error on the trailing arm below.
                tok = self._consume()
                position = str(tok.value)
            else:
                coords: list[float] = [self._signed_num()]
                while self._peek().kind == "COMMA":
                    self._consume()
                    coords.append(self._signed_num())
                position = tuple(coords)
            if self._peek().kind in ("TILDE", "ORTHO", "PERCENT"):
                raise SteeringExprError(
                    _MANIFOLD_COMPOSE_MSG,
                    col=self._peek().col,
                )
            return _Selector(
                base=base, operator=None, onto=None,
                manifold_position=position,
            )
        if self._peek().kind in ("TILDE", "ORTHO"):
            op_tok = self._consume()
            op = "~" if op_tok.kind == "TILDE" else "|"
            onto = self._atom()
            if onto.namespace in UNIT_NAMESPACES:
                raise SteeringExprError(_UNIT_COMPOSE_MSG, col=onto.col)
            if self._peek().kind in ("TILDE", "ORTHO"):
                nxt = self._peek()
                raise SteeringExprError(
                    "chained projection is not allowed; "
                    "use one '~' or '|' per term",
                    col=nxt.col,
                )
            return _Selector(base=base, operator=op, onto=onto)
        return _Selector(base=base, operator=None, onto=None)

    def _namespaced_name(self, namespace: str) -> str:
        """Consume the name after ``<namespace>/`` (the ``/`` already consumed).

        ``sae/<int>`` takes a feature id and ``jlens``/``slens`` also take an
        exact vocab id ``#<int>``; ``neuron``/``head`` take a unit
        (``L8n344``), returned canonical; everything else is an identifier.
        """
        if namespace in UNIT_NAMESPACES:
            return self._unit_name(namespace)
        if namespace == "sae" and self._peek().kind == "NUM":
            return str(self._non_negative_int("SAE feature id"))
        if namespace in _TOKEN_ID_NAMESPACES and self._peek().kind == "HASH":
            hash_tok = self._consume()
            if self._peek().col != hash_tok.col + 1:
                raise SteeringExprError(
                    "token id must follow '#' with no space", col=self._peek().col,
                )
            return f"#{self._non_negative_int('token id')}"
        return str(self._expect("IDENT").value)

    def _unit_name(self, namespace: str) -> str:
        """Consume a unit name (``L8n344``) and return its canonical text."""
        tok = self._peek()
        if tok.kind != "IDENT":
            example = "L8n344" if namespace == "neuron" else "L20h17"
            raise SteeringExprError(
                f"'{namespace}/' takes a unit, e.g. '{namespace}/{example}'",
                col=tok.col,
            )
        self._consume()
        try:
            unit = str(parse_unit_ref(namespace, str(tok.value)))
        except UnitRefError as exc:
            raise SteeringExprError(str(exc), col=tok.col) from None
        if self._peek().kind == "DOT":
            raise SteeringExprError(
                f"'{namespace}/{unit}' is a whole unit and takes no '.'",
                col=self._peek().col,
            )
        return unit

    def _non_negative_int(self, what: str) -> int:
        tok = self._expect("NUM")
        value = float(tok.value)
        if value != int(value) or value < 0:
            raise SteeringExprError(
                f"{what} must be a non-negative integer", col=tok.col,
            )
        return int(value)

    def _atom(self) -> _Atom:
        first = self._expect("IDENT")
        col = first.col
        namespace: str | None = None
        concept = str(first.value)
        if self._peek().kind == "SLASH":
            self._consume()
            namespace = concept
            concept = self._namespaced_name(namespace)
        if self._peek().kind == "DOT":
            self._consume()
            rhs = self._expect("IDENT")
            concept = f"{concept}.{rhs.value}"
        variant = "raw"
        if self._peek().kind == "COLON":
            self._consume()
            v = self._expect("IDENT")
            variant = str(v.value)
            if namespace in RESERVED_NAMESPACES:
                raise SteeringExprError(
                    f"'{namespace}/' atoms name instrument units and take no "
                    f"':{variant}' variant",
                    col=v.col,
                )
        return _Atom(
            namespace=namespace, concept=concept, variant=variant, col=col,
        )


# ------------------------------------------------------------ resolve/fold ---

def _with_variant(canonical: str, variant: str) -> str:
    return canonical if variant == "raw" else f"{canonical}:{variant}"


def format_layer_clause(layers: LayerScope) -> str:
    """The canonical ``{…}`` clause of a :data:`LayerScope` (``""`` for ``None``).

    Indices render sorted with each run of consecutive layers as ``a-b``
    (:func:`~drowse.model.layers.format_layer_indices`); ``workspace``
    renders lower-case.
    """
    if layers is None:
        return ""
    if layers == "workspace":
        return "{workspace}"
    return "{" + format_layer_indices(layers) + "}"


def split_layer_clause(key: str) -> tuple[str, LayerScope]:
    """Split a ``Steering.alphas`` key into its bare key and :data:`LayerScope`.

    ``"honest{12-20}"`` → ``("honest", (12, …, 20))``; a key without a clause
    returns ``(key, None)``.  A plain vector carries its clause only in its
    key, so the composer reads it back through here.
    """
    if not key.endswith("}"):
        return key, None
    start = key.rfind("{")
    if start < 0:
        raise SteeringExprError(f"steering key {key!r} has an unopened layer clause")
    body = key[start + 1:-1]
    try:
        parsed = parse_layer_selector(body, allow_sample=False)
    except LayerSelectorError as exc:
        raise SteeringExprError(f"steering key {key!r}: layer clause: {exc}") from None
    if parsed is None:
        raise SteeringExprError(f"steering key {key!r} has an empty layer clause")
    scope: LayerScope
    if parsed == "all":
        scope = None
    elif isinstance(parsed, list):
        scope = tuple(sorted(parsed))
    else:
        scope = "workspace"
    return key[:start], scope


def _term_scope(term: _Term) -> LayerScope:
    """A parsed term's canonical scope: ``{all}`` is no clause."""
    return None if term.layers in (None, "all") else cast(LayerScope, term.layers)


def _resolve_atom(atom: _Atom) -> str:
    """Return the ``alphas_key`` for a **pole** atom.

    The key is the canonical concept name from ``canonicalize_atom``,
    prefixed with ``<namespace>/`` when the user explicitly typed a
    namespace (so two installed packs sharing a concept name —
    ``alice/foo`` vs ``bob/foo`` — stay distinct through the
    registry-key path), and suffixed with ``:<variant>`` when the
    variant is anything other than ``raw``.

    This is the **pole canonicalization tier in isolation** — the base
    of a projection / ablation term and a projection ``onto`` direction
    are always vector concepts and must not route through the manifold
    tiers; the plain-term path in :func:`_fold` runs the full
    :func:`~drowse.store.selectors.resolve_bare_atom` ladder instead.

    A reserved instrument namespace keeps its name verbatim.
    """
    if atom.namespace in RESERVED_NAMESPACES:
        return f"{atom.namespace}/{atom.concept}"
    from drowse.store.selectors import canonicalize_atom

    raw = atom.concept
    if atom.variant != "raw":
        raw = f"{raw}:{atom.variant}"
    canonical, variant = canonicalize_atom(raw)
    if atom.namespace is not None:
        canonical = f"{atom.namespace}/{canonical}"
    return _with_variant(canonical, variant)


def _merge_plain(
    alphas: "dict[str, AlphaEntry]",
    key: str,
    coeff: float,
    trig: Optional[Trigger],
) -> None:
    if key not in alphas:
        alphas[key] = coeff if trig is None else (coeff, trig)
        return
    existing = alphas[key]
    if isinstance(existing, ProjectedTerm):
        raise SteeringExprError(
            f"concept '{key}' appears both as a plain term and as a "
            f"projection target; use distinct references"
        )
    prev_coeff: float
    prev_trig: Optional[Trigger]
    if isinstance(existing, tuple):
        prev_coeff = float(existing[0])
        prev_trig = existing[1]
    else:
        prev_coeff = float(cast(float, existing))
        prev_trig = None
    if prev_trig is None and trig is None:
        alphas[key] = prev_coeff + coeff
        return
    if prev_trig is not None and trig is not None and prev_trig == trig:
        alphas[key] = (prev_coeff + coeff, trig)
        return
    raise SteeringExprError(
        f"concept '{key}' appears with conflicting triggers; "
        f"merge triggers explicitly or split into separate Steering entries"
    )


def _merge_projected(
    alphas: "dict[str, AlphaEntry]",
    key: str,
    op: Literal["~", "|"],
    base: str,
    onto: str,
    coeff: float,
    trig: Trigger,
    layers: LayerScope,
) -> None:
    if key not in alphas:
        alphas[key] = ProjectedTerm(
            coeff=coeff, trigger=trig, operator=op, base=base, onto=onto,
            layers=layers,
        )
        return
    existing = alphas[key]
    if not isinstance(existing, ProjectedTerm):
        raise SteeringExprError(
            f"projection '{key}' conflicts with a plain entry of the same name"
        )
    if existing.trigger != trig:
        raise SteeringExprError(
            f"projection '{key}' appears with conflicting triggers"
        )
    alphas[key] = ProjectedTerm(
        coeff=existing.coeff + coeff,
        trigger=trig, operator=op, base=base, onto=onto, layers=layers,
    )


def _merge_ablation(
    alphas: "dict[str, AlphaEntry]",
    target_key: str,
    coeff: float,
    trig: Trigger,
    layers: LayerScope,
) -> None:
    key = f"!{target_key}{format_layer_clause(layers)}"
    if key not in alphas:
        alphas[key] = AblationTerm(
            coeff=coeff, trigger=trig, target=target_key, layers=layers,
        )
        return
    existing = alphas[key]
    if not isinstance(existing, AblationTerm):
        raise SteeringExprError(  # pragma: no cover — key namespace is disjoint
            f"ablation '{key}' conflicts with a non-ablation entry of the same name"
        )
    if existing.trigger != trig:
        raise SteeringExprError(
            f"ablation '{key}' appears with conflicting triggers; "
            f"merge triggers explicitly or split into separate Steering entries"
        )
    alphas[key] = AblationTerm(
        coeff=existing.coeff + coeff, trigger=trig, target=target_key,
        layers=layers,
    )


def _merge_unit_ablation(
    alphas: "dict[str, AlphaEntry]",
    unit: UnitRef,
    coeff: float,
    trig: Trigger,
) -> None:
    key = f"!{unit.qualified}"
    existing = alphas.get(key)
    if existing is None:
        alphas[key] = UnitAblationTerm(unit=unit, coeff=coeff, trigger=trig)
        return
    if not isinstance(existing, UnitAblationTerm):
        raise SteeringExprError(  # pragma: no cover — key namespace is disjoint
            f"unit ablation '{key}' conflicts with another entry of the same name"
        )
    if existing.trigger != trig:
        raise SteeringExprError(
            f"unit ablation '{key}' appears with conflicting triggers; "
            f"merge triggers explicitly or split into separate Steering entries"
        )
    alphas[key] = UnitAblationTerm(
        unit=unit, coeff=existing.coeff + coeff, trigger=trig,
    )


def _resolve_manifold_atom(atom: _Atom) -> str:
    """Return the registry key for a manifold atom.

    Unlike :func:`_resolve_atom`, manifold atoms do *not* go through
    pole canonicalization — a manifold name is not a concept and must not
    alias to one.  The key is the bare name, namespace-prefixed only when
    the user typed a namespace, variant-suffixed when not ``raw``.
    """
    key = atom.concept
    if atom.namespace is not None:
        key = f"{atom.namespace}/{key}"
    return _with_variant(key, atom.variant)


def _fmt_position(position: tuple[float, ...] | str) -> str:
    """Render a manifold position payload back to grammar form.

    Coord tuple → comma-joined ``%g`` list (``0.3,0.8``).  Label string
    → the slug verbatim (``pirate``).  The two forms are unambiguous
    by shape — a node label is a slug-shaped identifier, a coord list
    starts with a digit / sign.
    """
    if isinstance(position, str):
        return position
    return ",".join(f"{c:g}" for c in position)


def _exact_position(position: tuple[float, ...] | str) -> str:
    """A position at full precision (``repr``), for error text."""
    if isinstance(position, str):
        return position
    return ",".join(repr(c) for c in position)


def _expand_along_onto_coeffs(
    coeffs: tuple[float, ...],
) -> tuple[float, float]:
    """Map a 1/2-length coefficient run to ``(along, onto)``.

    The manifold ``%`` coefficient slot expands a comma-run with the
    collapse op (``onto``) defaulting **off** — it is opted into:

    - ``c`` → ``along = c, onto = 0``  (pure directional slide)
    - ``a, o`` → ``along = a, onto = o``

    ``along`` is *the* steering knob: it slides the projected foot toward
    the target on the manifold and keeps everything else.  ``onto`` (flatten
    the off-manifold in-subspace residual) is aggressive and so off by
    default.  The off-*subspace* residual is always kept verbatim — that is
    the composition invariant, so it has no coefficient.

    A run of length 0 or > 2 is a programming error here — the parser
    rejects the 3-coeff case at parse time and a term always carries at
    least one coeff — but raise rather than index blindly.
    """
    n = len(coeffs)
    if n == 1:
        return (coeffs[0], 0.0)
    if n == 2:
        a, o = coeffs
        return (a, o)
    raise SteeringExprError(
        "a manifold coefficient slot takes 1 or 2 comma-separated "
        f"values (along, onto); got {n}"
    )


def _merge_manifold(
    alphas: "dict[str, AlphaEntry]",
    manifold: str,
    coeffs: tuple[float, float],
    position: tuple[float, ...] | str,
    trig: Trigger,
    layers: LayerScope,
) -> None:
    along, onto = coeffs
    key = f"{manifold}%{_fmt_position(position)}{format_layer_clause(layers)}"
    if key not in alphas:
        alphas[key] = ManifoldTerm(
            along=along, onto=onto,
            trigger=trig, manifold=manifold, position=position, layers=layers,
        )
        return
    existing = alphas[key]
    if not isinstance(existing, ManifoldTerm):
        raise SteeringExprError(  # pragma: no cover — key namespace is disjoint
            f"manifold '{key}' conflicts with a non-manifold entry"
        )
    if existing.trigger != trig:
        raise SteeringExprError(
            f"manifold '{key}' appears with conflicting "
            f"triggers; merge triggers explicitly or split into separate "
            f"Steering entries"
        )
    if existing.position != position:
        # The key renders coordinates with ``%g`` (six significant digits),
        # so two distinct positions can share it; summing them would move
        # one term silently onto the other's point.
        raise SteeringExprError(
            f"manifold '{key}' names two positions that agree to six "
            f"significant digits ({_exact_position(existing.position)} and "
            f"{_exact_position(position)}); round them to one point or move "
            f"them further apart"
        )
    alphas[key] = ManifoldTerm(
        along=existing.along + along,
        onto=existing.onto + onto,
        trigger=trig, manifold=manifold, position=position, layers=layers,
    )


def _fold(
    terms: list[_Term],
    *,
    namespace: Optional[str],
    profile_names: Optional[set[str]] = None,
) -> "Steering":
    from drowse.steering.terms import Steering

    alphas: "dict[str, AlphaEntry]" = {}
    registered_profiles = profile_names or set()
    for term in terms:
        sel = term.selector
        scope = _term_scope(term)
        clause = format_layer_clause(scope)
        # Manifold position terms resolve before ``_resolve_atom`` — a
        # manifold name must not run through concept pole-resolution.
        if sel.manifold_position is not None:
            if term.ablation:
                raise SteeringExprError(_MANIFOLD_COMPOSE_MSG)
            mfld_key = _resolve_manifold_atom(sel.base)
            mfld_trig = (
                term.trigger if term.trigger is not None else Trigger.BOTH
            )
            _merge_manifold(
                alphas, mfld_key,
                _expand_along_onto_coeffs(term.coeffs),
                sel.manifold_position, mfld_trig, scope,
            )
            continue
        # Bare-atom tier ladder.  A plain term (no projection operator, no
        # ablation) runs the unified ladder that
        # :func:`~drowse.store.selectors.resolve_bare_atom` owns end to end —
        # manifold-label tier first (a bare slug matching a node label →
        # label-form ``%`` push), then the composite-name tier (a 2-node
        # ``pca`` manifold name like ``happy.sad`` → its node-0 (+) pole),
        # then the pole canonicalization that falls through to the plain
        # vector path below.  Every bipolar pole is itself a node label, so a
        # bare pole resolves through the label tier as an affine ``%`` push.
        # Ordering + cross-tier ``AmbiguousSelectorError`` arbitration live in
        # the resolver, not here.
        if (
            sel.operator is None
            and not term.ablation
            and sel.base.namespace not in RESERVED_NAMESPACES
        ):
            profile_key = _resolve_atom(sel.base)
            if profile_key in registered_profiles:
                if len(term.coeffs) > 1:
                    raise SteeringExprError(
                        "comma-separated coefficients are only valid for "
                        "`manifold % position` terms"
                    )
                _merge_plain(
                    alphas, profile_key + clause, term.coeff, term.trigger,
                )
                continue

            from drowse.store.selectors import resolve_bare_atom

            atom = resolve_bare_atom(
                sel.base.concept,
                namespace=namespace,
                typed_namespace=sel.base.namespace,
                variant=sel.base.variant,
            )
            if atom.kind == "label":
                assert atom.manifold is not None
                mfld_trig = (
                    term.trigger if term.trigger is not None else Trigger.BOTH
                )
                _merge_manifold(
                    alphas, atom.manifold.manifold_key,
                    _expand_along_onto_coeffs(term.coeffs),
                    atom.manifold.label, mfld_trig, scope,
                )
                continue
            if atom.kind == "name":
                assert atom.manifold_name is not None
                mfld_trig = (
                    term.trigger if term.trigger is not None else Trigger.BOTH
                )
                _merge_manifold(
                    alphas, atom.manifold_name.manifold_key,
                    _expand_along_onto_coeffs(term.coeffs),
                    atom.manifold_name.pole_label, mfld_trig, scope,
                )
                continue
            # kind == "pole": fall through to the plain vector path below
            # (re-resolved via ``_resolve_atom`` for the namespace-prefixed
            # registry key).
        # Past this point the term is unambiguously a vector (plain /
        # projection / ablation) — the comma-separated coefficient run is a
        # manifold-``%``-only construct, so reject it here rather than
        # silently dropping the extra values.
        if len(term.coeffs) > 1:
            raise SteeringExprError(
                "comma-separated coefficients are only valid for "
                "`manifold % position` terms"
            )
        base_key = _resolve_atom(sel.base)
        coeff = term.coeff
        # ``_Term.trigger`` already carries a resolved Trigger object
        # (built by :class:`_Parser._term`) — preset names map through
        # _TRIGGER_PRESETS at parse time, gates produce a fresh
        # Trigger directly.  Pass through verbatim.
        trig: Trigger | None = term.trigger
        if term.ablation:
            if sel.operator is not None:
                raise SteeringExprError(
                    "ablation does not compose with projection operators; "
                    "ablate the base concept and project separately"
                )
            effective_trig = trig if trig is not None else Trigger.BOTH
            if sel.base.namespace in UNIT_NAMESPACES:
                unit = parse_unit_ref(sel.base.namespace, sel.base.concept)
                _merge_unit_ablation(alphas, unit, coeff, effective_trig)
                continue
            _merge_ablation(alphas, base_key, coeff, effective_trig, scope)
            continue
        if sel.operator is None:
            _merge_plain(alphas, base_key + clause, coeff, trig)
            continue
        # Projection terms.  Projection math is insensitive to the sign of
        # the onto direction (``a|b`` yields the same result as
        # ``a|(-b)``); the base sign is already folded into ``coeff``.
        assert sel.onto is not None
        onto_key = _resolve_atom(sel.onto)
        effective_trig = trig if trig is not None else Trigger.BOTH
        op: Literal["~", "|"] = cast(Literal["~", "|"], sel.operator)
        syn_key = f"{base_key}{op}{onto_key}{clause}"
        _merge_projected(
            alphas, syn_key, op, base_key, onto_key,
            coeff, effective_trig, scope,
        )
    return Steering(alphas=alphas)


# ---------------------------------------------------------------- public ---

def parse_expr(
    text: str,
    *,
    namespace: Optional[str] = None,
    profile_names: Optional[set[str]] = None,
) -> "Steering":
    """Parse a steering expression string into a :class:`Steering` IR.

    ``namespace`` scopes bare-atom resolution to a single namespace; when
    ``None``, :func:`drowse.store.selectors.resolve_bare_atom` raises
    :class:`~drowse.store.selectors.AmbiguousSelectorError` if a bare atom
    matches manifolds across multiple namespaces.

    ``profile_names`` is a session-local override set: when a plain term's
    canonical registry key exactly matches an in-memory profile, that profile
    wins over the installed-manifold tiers for that parse only.
    """
    if not text or not text.strip():
        raise SteeringExprError("empty steering expression")
    toks = _lex(text)
    terms = _Parser(toks).parse()
    return _fold(terms, namespace=namespace, profile_names=profile_names)


def format_expr(steering: "Steering") -> str:
    """Render a :class:`Steering` back into canonical expression form.

    Round-trips with :func:`parse_expr` for any IR produced by the parser.
    Entries whose trigger equals :data:`Trigger.BOTH` omit the ``@`` tag;
    entries whose coefficient is negative are emitted via ``-`` separators
    (first term carries the sign verbatim).  A layer clause renders in
    canonical form (:func:`format_layer_clause`); a plain term's rides in its
    key.
    """
    parts = [part.text for part in term_parts(steering)]
    if not parts:
        return ""
    out = parts[0]
    for p in parts[1:]:
        if p.startswith("-"):
            out += " - " + p[1:].lstrip()
        else:
            out += " + " + p
    return out


TermOp = Literal["push", "ablate", "project", "position"]


@dataclass(frozen=True)
class TermPart:
    """One term of a :class:`Steering`, in order: its text and its fields.

    ``text`` is the term alone as :func:`format_expr` writes it (its leading
    sign included: ``-!jlens/storm``); :func:`format_expr` is these texts
    joined.  The fields are what a client needs to show the term as a chip
    without a parser of its own (the dashboard's recipe row,
    ``POST .../steering/validate``):

    - ``op``: ``push`` (a plain term: a vector, a pole, an instrument atom, a
      neuron), ``ablate`` (``!x``, units included), ``project`` (``a~b`` /
      ``a|b``) or ``position`` (``m%pos``);
    - ``target``: the selector without coefficient, clause or trigger
      (``default/honest:sae``, ``jlens/#12``, ``neuron/L8n344``; a
      projection's ``a~b``; a position's manifold);
    - ``coeff`` (a position's ``along``) and ``onto`` (a position's collapse,
      0 otherwise);
    - ``position``: a position's node label or coordinates, else ``None``;
    - ``layers``: the ``{…}`` clause as a :data:`LayerScope`;
    - ``trigger``: the ``@`` clause without the ``@`` (``response``,
      ``when:slens/owl>3``), ``None`` for :data:`Trigger.BOTH`.
    """

    text: str
    op: TermOp
    target: str
    coeff: float
    onto: float
    position: str | tuple[float, ...] | None
    layers: LayerScope
    trigger: str | None


def term_parts(steering: "Steering") -> list[TermPart]:
    """Every term of ``steering`` in order, each as :class:`TermPart`."""
    default_trig = steering.trigger
    out: list[TermPart] = []
    for name, val in steering.alphas.items():
        if isinstance(val, AblationTerm):
            out.append(TermPart(
                _fmt_ablation(val), "ablate", val.target, val.coeff, 0.0, None,
                val.layers, _trigger_text(val.trigger),
            ))
        elif isinstance(val, UnitAblationTerm):
            out.append(TermPart(
                _fmt_bang(val.unit.qualified, val.coeff, val.trigger), "ablate",
                val.unit.qualified, val.coeff, 0.0, None, None,
                _trigger_text(val.trigger),
            ))
        elif isinstance(val, ProjectedTerm):
            out.append(TermPart(
                _fmt_projected(val), "project",
                f"{val.base}{val.operator}{val.onto}", val.coeff, 0.0, None,
                val.layers, _trigger_text(val.trigger),
            ))
        elif isinstance(val, ManifoldTerm):
            out.append(TermPart(
                _fmt_manifold(val), "position", val.manifold, val.along,
                val.onto, val.position, val.layers, _trigger_text(val.trigger),
            ))
        else:
            if isinstance(val, tuple):
                coeff, trig = float(val[0]), val[1]
            else:
                coeff, trig = float(val), default_trig
            target, layers = split_layer_clause(name)
            out.append(TermPart(
                _fmt_plain(name, coeff, trig), "push", target, coeff, 0.0,
                None, layers, _trigger_text(trig),
            ))
    return out


def _trigger_text(trig: Trigger) -> str | None:
    """A term's ``@`` clause without the ``@``; ``None`` for both phases."""
    return None if trig == Trigger.BOTH else _trigger_name(trig)


def _fmt_plain(name: str, coeff: float, trig: Trigger) -> str:
    body = f"{coeff:g} {name}"
    if trig != Trigger.BOTH:
        body += "@" + _trigger_name(trig)
    return body


def _fmt_projected(p: ProjectedTerm) -> str:
    body = f"{p.coeff:g} {p.base}{p.operator}{p.onto}{format_layer_clause(p.layers)}"
    if p.trigger != Trigger.BOTH:
        body += "@" + _trigger_name(p.trigger)
    return body


def _fmt_ablation(a: AblationTerm) -> str:
    return _fmt_bang(f"{a.target}{format_layer_clause(a.layers)}", a.coeff, a.trigger)


def _fmt_bang(target: str, coeff: float, trigger: Trigger) -> str:
    if coeff == 1.0:
        body = f"!{target}"
    elif coeff == -1.0:
        body = f"-!{target}"
    else:
        body = f"{coeff:g} !{target}"
    if trigger != Trigger.BOTH:
        body += "@" + _trigger_name(trigger)
    return body


def _fmt_manifold(m: ManifoldTerm) -> str:
    # Render the shortest coefficient form the (along, onto) pair collapses
    # to — the inverse of ``_expand_along_onto_coeffs`` — so the round-trip is
    # byte-for-byte.  The collapse op defaults off, so: onto zero → one coeff
    # (along); else two (along, onto).
    #
    # The parser applies the term's *leading* sign (carried by ``along``) to
    # every value in the comma-run (``-0.6,0.3`` parses as ``(-0.6, -0.3)``),
    # so to make a programmatically-built mixed-sign term round-trip we render
    # ``onto`` *relative to* ``along``'s sign: literal = onto·sign(along), which
    # the parser then re-multiplies by sign(along) back to ``onto``.  For the
    # parser's own output (where the whole run shares one sign) this is a no-op.
    # The sign is the sign bit, not ``< 0``: ``%g`` renders a negative zero as
    # ``-0``, whose leading sign the parser applies to ``onto`` too.
    pos = _fmt_position(m.position)
    if m.onto == 0.0:
        coeff_str = f"{m.along:g}"
    else:
        onto_lit = -m.onto if math.copysign(1.0, m.along) < 0 else m.onto
        coeff_str = f"{m.along:g},{onto_lit:g}"
    body = f"{coeff_str} {m.manifold}%{pos}{format_layer_clause(m.layers)}"
    if m.trigger != Trigger.BOTH:
        body += "@" + _trigger_name(m.trigger)
    return body


def _phase_name(base: Trigger) -> str | None:
    """Render a gate-free Trigger's window, or ``None`` if it has no form.

    Covers the five presets and the two counted decode windows
    (:meth:`Trigger.first` / :meth:`Trigger.after`) — i.e. every window a
    public factory can produce.  A ``Trigger`` hand-built with a field
    combination outside that set (say ``first_n`` *and* ``after_n``, or a
    narrowed phase on top of a count) has no grammar form and returns
    ``None``.
    """
    if base in _TRIGGER_CANONICAL:
        return _TRIGGER_CANONICAL[base]
    if base.first_n is not None and base == Trigger.first(base.first_n):
        return f"first:{base.first_n}"
    if base.after_n is not None and base == Trigger.after(base.after_n):
        return f"after:{base.after_n}"
    return None


def _trigger_name(trig: Trigger) -> str:
    """Render a :class:`Trigger` into its ``@`` clause (no leading ``@``).

    Raises :class:`SteeringExprError` for a Trigger with no grammar form,
    rather than emitting a sentinel that would not parse back.  Every public
    factory (:meth:`Trigger.first` / :meth:`Trigger.after` /
    :meth:`Trigger.when`, the five presets) and their gate-narrowed compounds
    round-trip, so this only fires on a hand-built dataclass.
    """
    gate = trig.gate
    if gate is None:
        name = _phase_name(trig)
        if name is None:
            raise SteeringExprError(
                f"trigger {trig!r} has no steering-expression form; it is "
                f"programmatic-only and cannot be rendered to a string"
            )
        return name
    gate_str = f"when:{gate.probe}{gate.op}{_fmt_number(gate.threshold)}"
    base = replace(trig, gate=None)
    # ``Trigger.when``'s own window is GENERATED_ONLY, so a bare gate renders
    # without a phase clause; any other window renders the compound form.
    if base == Trigger.GENERATED_ONLY:
        return gate_str
    phase = _phase_name(base)
    if phase is None:
        raise SteeringExprError(
            f"trigger {trig!r} has no steering-expression form; it is "
            f"programmatic-only and cannot be rendered to a string"
        )
    return f"{phase}&{gate_str}"


def _fmt_number(x: float) -> str:
    """Render a probe threshold compactly while staying parseable.

    The grammar accepts a leading ``-`` before NUM, so negative
    thresholds render with the minus sign in-line.  Uses ``%g`` so
    integer-valued thresholds drop the trailing ``.0`` and small
    fractions render at full precision without scientific notation.
    """
    if x < 0:
        return f"-{abs(x):g}"
    return f"{x:g}"


def referenced_selectors(
    text: str,
) -> list[tuple[Optional[str], str, str]]:
    """Return every **concept atom** ``(namespace, concept, variant)`` in ``text``.

    Walks the AST before pole resolution so namespace prefixes survive —
    useful at install time, when the CLI needs to know which pack to fetch
    for each atom.  Projection terms contribute two entries (base + onto);
    a ``{…}`` layer clause is not part of an atom and never appears.

    Concept atoms *only*: a ``%`` manifold term names an artifact, not a
    concept, and is skipped.  A caller preflighting artifacts (rather than
    concepts) — deciding what to fetch or register before a generation — has
    to handle manifold terms itself by walking ``parse_expr(text).alphas``
    for :class:`ManifoldTerm` values.
    """
    if not text or not text.strip():
        return []
    toks = _lex(text)
    terms = _Parser(toks).parse()
    out: list[tuple[Optional[str], str, str]] = []
    for term in terms:
        sel = term.selector
        if sel.manifold_position is not None:
            continue
        out.append((sel.base.namespace, sel.base.concept, sel.base.variant))
        if sel.onto is not None:
            out.append((sel.onto.namespace, sel.onto.concept, sel.onto.variant))
    return out


__all__ = [
    "DEFAULT_COEFF",
    "DEFAULT_ABLATION_COEFF",
    "RESERVED_NAMESPACES",
    "AblationTerm",
    "LayerScope",
    "ManifoldTerm",
    "ProjectedTerm",
    "SteeringExprError",
    "TermOp",
    "TermPart",
    "UnitAblationTerm",
    "format_layer_clause",
    "parse_expr",
    "format_expr",
    "term_parts",
    "referenced_selectors",
    "split_layer_clause",
]
