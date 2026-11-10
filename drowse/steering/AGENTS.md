# drowse/steering

The steering language and its lowering: the expression grammar and its IR
(`Steering`), triggers and probe gates, the composer that turns a steering
stack into a `SteeringPlan`, and the forward hooks that run it. The kernel
math (`subspace_inject`, `synthesize_subspace`, fold and projection) lives in
`drowse/geometry/`; this package decides what is injected, where, when and how
strongly.

| Module | Owns |
|---|---|
| `grammar.py` | Lexer, parser, fold; `parse_expr`, `format_expr`, `referenced_selectors`; `TermPart` (one term with its trigger as clause text and as `phase`/`first_n`/`after_n`/`gate`, and a projection's `(base, operator, onto)`, for `POST .../steering/validate`); term types `ProjectedTerm`, `AblationTerm`, `UnitAblationTerm`, `ManifoldTerm`; `LayerScope`, `format_layer_clause`, `split_layer_clause`; `DEFAULT_COEFF = 0.5`, `DEFAULT_ABLATION_COEFF = 1.0` |
| `terms.py` | `Steering` (frozen IR), `AlphaEntry`, `SteeringStackEntry`, `term_entry` / `entry_trigger` / `entry_coeff` |
| `triggers.py` | `Trigger` + presets, `ProbeGate`, `TriggerContext` |
| `composer.py` | `SteeringComposer` (stack, scopes, resolution, composition, plan cache, stack queries, `term_layers`), `SteeringScope`, `TermLayers`, `resolve_scope_role`, `select_term_layers`, `affine_manifold_push`, the `ArtifactSource` / `NeutralSource` / `UnitSource` protocols |
| `hooks.py` | `SteeringPlan`, `_lower_affine_subspaces`, `AffineOp`, `SteeringHook`, `UnitTerm` / `UnitOp` / `UnitHook` (unit ablation), `SteeringManager`, `install_persistent_offset_hooks`, the gain constants |

Layering: `grammar`/`terms`/`triggers` sit above `store` (parsing resolves bare
atoms against installed manifolds via `store.selectors`) and below `readout`;
`composer`/`hooks` sit above `lens`/`sae`/`units` and below `generation`.
Lower layers that must refuse work under steering take
`SteeringComposer.assert_unsteered` as an injected callable.

Flow: text → `parse_expr` → `Steering` → `SteeringComposer.open_scope`
(`resolve` + role policy) → `SteeringScope.__enter__` → `push` →
`rebuild_hooks` → `plan_for` (cache) → `compose` → `SteeringPlan` →
`SteeringManager.install` → `SteeringPlan.lower` → `AffineOp` / curved groups
on `SteeringHook`s, or the compiled offset buffers.

## Expression grammar (authoritative)

Every surface (Python, YAML `vectors:`, HTTP, CLI, loom recipes) parses through
`parse_expr`. `fit/bake.py::_parse_merge_expr` walks the parser's private AST
(`_lex`, `_Parser`, `_Term`/`_Selector`/`_Atom`) to accept only
namespace-qualified additive scalar terms. drowse.ai reimplements the grammar.

```text
expr       := [sign] term (sign term)*
sign       := "+" | "-"
term       := [coeffs ["*"]] ["!"] selector ["{" layers "}"] ["@" trigger]
coeffs     := NUM ["," signed_num]           # 2 values only on a % term: along,onto
layers     := <model/layers.py selector>     # entries INT | INT "-" INT, comma-joined; "all" | "workspace"
selector   := atom [("~" | "|") atom | "%" position]
position   := signed_num ("," signed_num)* | IDENT     # coord list | node label
atom       := ("jlens" | "slens") "/" "#" INT          # exact vocab id
            | "sae" "/" INT                            # SAE feature
            | ("neuron" | "head") "/" UNIT             # L8n344 / L20h17
            | [IDENT "/"] IDENT ["." IDENT] [":" IDENT]  # ns / name . name : variant
trigger    := phase ["&" gate] | gate
phase      := preset | ("first" | "after") ":" INT
preset     := "both" | "before" | "prompt" | "after" | "thinking" | "response" | "generated"
gate       := "when" ":" probe op signed_num
probe      := [IDENT "/"] pname ( ":" ("fraction" | "membership")
                                | ("@" | "~") IDENT
                                | ["." IDENT] ["[" INT "]"] )
pname      := IDENT | "#" INT (jlens/slens) | INT (sae) | UNIT (neuron/head)
op         := ">" | ">=" | "<" | "<="
signed_num := ["+" | "-"] NUM
UNIT       := "L" INT ("n" | "h") INT                  # any letter case; n under neuron/, h under head/
```

Lexical rules: `NUM` is unsigned (`1`, `.5`, `2.`, `1e-3`); every digit is
ASCII `0-9`. A term's sign comes from the leading/separating `+`/`-` and
applies to every value of a comma run (`-0.6,0.3 m%x` is `along=-0.6,
onto=-0.3`). `IDENT` is `[A-Za-z][A-Za-z0-9_]*`, and a `-` between two
identifier characters joins segments, so `a-b` is one name (slugged to `a_b`)
and `0.5 a-0.3 b` is a parse error; write `a - b`. Multi-word names use `_`;
quotes, `==` and `a.b.c` are errors. `SteeringExprError` (400) carries the
failing `col`.

The lexer takes a `{…}` clause whole and hands its body to
`model/layers.py::parse_layer_selector(body, allow_sample=False)`, the one
layer-selector parser. The clause follows the whole selector (after `%pos`,
`~b`, `|b`) and precedes `@`. Errors, each with a `col`: an empty clause,
`sample`, an empty entry, a negative index, a reversed range (`8-4`), a layer
named twice (`4-8,6`), an unterminated `{`, a second clause, a `~`/`|`/`%`
after the clause, a clause after `@` (gate probes take none), and a clause on
a selector with a unit atom (units carry their own layer).

**Unit atoms.** `neuron/L8n344` and `head/L20h17` name one MLP neuron or one
attention head. The name lexes as one `IDENT` and goes through
`naming.parse_unit_ref` (any letter case, leading zeros dropped), so keys,
`format_expr` and `referenced_selectors` carry the canonical `L8n344`; the
letter must match the namespace. Errors: a malformed unit (`neuron/L8h344`,
`neuron/8`), a `.` after a unit, a `:variant`, a unit as an operand of `~`,
`|` or `%`, a layer clause, and a head without `!` (a head has no single write
direction). Gate probes take units too (`@when:neuron/L8n344 > 2`); a unit
probe is one scalar, so a channel suffix after it is a parse error.

| Form | IR value (`Steering.alphas` key) | Lowered as |
|---|---|---|
| `α x` | `ManifoldTerm` via the bare-atom ladder, else `α` / `(α, Trigger)` under `x` | push |
| `α a~b` / `α a\|b` | `ProjectedTerm` under `a~b` / `a\|b` | Mahalanobis LEACE `project_profile`: keep / remove the component of `a` along `b`, then push |
| `α !x` | `AblationTerm` under `!x` | LEACE toward neutral, `h' = h − α((h − μ)·Σ⁻¹d / d·Σ⁻¹d)·d`: reads with the covector, writes along `d`; bare `!x` is α = 1 |
| `α neuron/L8n344` | `α` / `(α, Trigger)` under `neuron/L8n344` | push along the neuron's `+`-pole write at its layer (a one-layer profile); a negative α pushes along the `−` pole's |
| `α !neuron/L8n344`, `α !head/L20h17` | `UnitAblationTerm(unit, coeff, trigger)` under `!neuron/L8n344` | a `UnitOp` pre-hook on the down / output projection: the unit's activation (a neuron's `act`, a head's `z_h`) becomes `(1 − α)·x + α·mean`, its neutral mean; not a residual direction ablation |
| `along[,onto] m%pos` | `ManifoldTerm` under `m%pos` | affine fit: push fragment; curved fit: own term |
| any of these but a unit atom, `+ {L}` | the same value with `layers` set (`LayerScope`), under the key plus the canonical clause (`deceptive{12-20}`, `!jlens/Paris{24-30}`, `m%x{workspace}`); a plain term carries it only in its key | the same, restricted to the selected layers |

`LayerScope` is `None` (every fitted layer), `"workspace"`, or a sorted index
tuple; `{all}` is `None`. The clause is part of the term's identity: one
selector under two clauses is two keys, equal clauses merge, and written order
is irrelevant (`a{5,4}` is `a{4-5}`). `split_layer_clause(key)` reads a key
back into `(bare key, LayerScope)`.

Omitted coefficient: 0.5 (`DEFAULT_COEFF`), or 1.0 after `!`. Parse-time
rejections: `!` with `~`/`|`/`%`; `%` with `~`/`|` or a second `%`; chained
projections; a two-value comma run on anything that does not resolve to a `%`
term (a bare node label does: `0.6,0.3 pirate` is legal); more than two
values; more than one `@` clause or gate per term. Repeated keys merge: equal
triggers sum coefficients (`along` and `onto` separately); differing triggers,
or one key used both as a plain term and as a projection, raise. A coordinate
`%` key renders its position with `%g`, so two positions equal to six
significant digits share a key and raise rather than merge. Projection,
ablation and `%` terms default to `Trigger.BOTH`; a plain term without `@`, or
with `@both`, stores a bare float that inherits `Steering.trigger`, so `x@both`
and `x` are one IR.

### Triggers

| `@` form | `Trigger` | prefill | thinking tokens | response tokens |
|---|---|---|---|---|
| none, `@both` | `Trigger.BOTH` | yes | yes | yes |
| `@before` = `@prompt` | `Trigger.PROMPT_ONLY` | yes | no | no |
| `@response` = `@generated` | `Trigger.GENERATED_ONLY` | no | yes | yes |
| `@after` | `Trigger.AFTER_THINKING` | no | no | yes |
| `@thinking` | `Trigger.THINKING_ONLY` | no | yes | no |
| `@first:N` / `@after:N` | `Trigger.first(N)` / `Trigger.after(N)` | no | `gen_step < N` / `gen_step >= N` | same |
| `@when:G` | `Trigger.when(probe, op, t)` | no | while G holds | while G holds |
| `@<phase>&when:G` | `replace(<phase>, gate=…)` | no | phase window while G holds | same |

`@response` includes thinking tokens; only bare `@after` excludes them, and
`@after:N` is a token window, not a phase. `gen_step` counts raw generated
positions from 0, thinking included. A gate cannot fire during prefill, so
`@before&when:…` parses but never fires.

### Probe gates

The gate's probe string is stored verbatim in `ProbeGate.probe` and equals a
scalar key the read side writes into `TriggerContext.probe_scores`: the
attached probe's exact name plus a channel suffix (a probe attached as
`default/emotions` is gated as `default/emotions@happy`).

| Shape | Channel (`readout.types`) | Produced by |
|---|---|---|
| `p`, `a.b`, `p[i]` | `Axis(i)`; bare = axis 0 | geometry (one key per coord axis, domain-frame); `jlens/<w>` / `jlens/#id` (mean fitted-layer probability); `sae/<id>` (normalized strength); `neuron/L8n344` (activation over its pole's neutral `act_top`, signed), `head/L20h17` (write norm over its `act_top`). Lens and SAE emit axis 0 only; a higher index validates but never fires. A unit gate needs no attached probe. |
| `p:fraction` | `Fraction` | geometry: share of the centered activation in the subspace, [0, 1] |
| `p:membership` | `Membership` | geometry: tube-fit density, [0, 1] |
| `p@label` | `Distance` | geometry: negated distance in typical label spacings (larger = closer); only for nodes in the reading's `nearest` list; `neutral` names the frame anchor |
| `p~label` | `Assignment` | geometry: soft-assignment probability, [0, 1] |

Channel semantics and units are in `drowse/readout/AGENTS.md`. A dotted probe
name takes only the axis forms. `Trigger.active` treats a missing key as
inactive, never an error. The decode loop replaces `probe_scores` after every
forward, so a gate on forward t reads the reading scored after forward t−1
(the prefill for the first decode step). Validation is at generation preflight
in `ReadPlane.resolve_gates` (`generation/read_plane.py`): a gate naming a
lens/SAE probe on a channel its family cannot produce raises
`UnsupportedProbeChannelError` (400); geometry accepts every channel; a unit
outside the model is a `UnitIndexError` (400); any other gate naming no
attached probe is silently inactive.

### Round trip

`format_expr` renders the canonical form: resolved registry keys (`0.5 pirate`
→ `0.5 default/personas%pirate`), `%g` numbers (six significant digits, so
lossy beyond that; the same rendering spells a coordinate `%` position in its
key), no tag for `Trigger.BOTH`, canonical preset spellings (`before`,
`response`), `!x` / `-!x` for ±1 ablations, a `%` term's `onto` relative to
`along`'s sign (so the loom `inverted` regen, which negates only `along`,
re-parses exactly) and the canonical layer clause (`format_layer_clause`:
sorted, runs as `a-b`, no spaces; `{all}` renders nothing).
`parse_expr(format_expr(s)) == s` holds for every IR the parser produces and
every trigger built from presets, `first`/`after`/`when` or
`replace(preset, gate=…)`; any other hand-built `Trigger` has no grammar form
and `format_expr` raises `SteeringExprError`, which results, loom recipes and
scoring hit when they stamp `str(steering)`. `referenced_selectors` returns the
pre-resolution concept atoms `(namespace, concept, variant)`, skipping `%`
terms and layer clauses (pack-install checks in `cli/config_file.py`).

## Names, atoms and resolution

- **Canonical names** (`drowse/naming.py`): `canonical_concept_name` lowercases
  and slugs `[^a-z0-9]+ → _`; bipolar poles join with `BIPOLAR_SEP = "."`
  (`formal.casual`), and a pre-joined `a.b` slugs each side. `.` is the
  separator because `~` is the projection operator and HF repo names reject
  it. `NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$` (`base/integrity.py`) governs
  names and namespaces; node labels use `LABEL_REGEX` (no `.`) so `%label` and
  bare labels stay unambiguous; `@` is forbidden (HF revision separator).
- **Reserved namespaces** (`RESERVED_NAMESPACES`: `jlens`, `slens`, `sae`,
  `neuron`, `head`) name instrument units: the name is kept verbatim and
  case-preserving (`jlens/Paris` ≠ `jlens/paris`; unit names are canonicalized
  as above), takes no `:variant`, and skips every manifold tier. The session
  wires resolvers for `jlens`, `slens` (the token's S-lens footprint direction,
  `register_input_direction`), `sae` and `neuron` (`ArtifactRegistry.
  set_atom_resolver`; `neuron` is `UnitsInstrument.register_direction`, which
  checks the unit against the model: a 400 at `open_scope`).
- **Variants**: `raw` (default), `sae` / `sae-<release>`, `role` /
  `role-<id>` (role-augmented fit; its role becomes the scope's assistant
  label), `from` / `from-<src>` (cross-model transfer). Non-`%` atoms are
  checked against `VARIANT_SUFFIX_RE` at parse time; a `%` operand's variant is
  checked when the manifold loads.

Plain-term resolution in `_fold`, in order:

1. Reserved namespace → key `ns/name`.
2. `profile_names` override: a canonical key exactly matching an in-memory
   profile (`open_scope` passes the registry's names: extracts, bakes,
   instrument atoms, and any 2-node manifold already folded as an
   `!`/`~`/`|` operand) is a plain vector. So `0.5 formal.casual` is a `%`
   push in a fresh session but a folded-vector push after
   `session.extract("formal.casual")` or `!formal.casual`, and the two weight
   layers differently (folded `mahalanobis_share` versus the pole's
   neutral-anchored coordinate).
3. `store.selectors.resolve_bare_atom`: **label tier** (bare slug only: no
   typed namespace, `raw`, no `.`) → a unique node label across installed
   manifolds becomes `<ns>/<manifold>%<label>`; **name tier** (`raw`) → a
   2-node `pca` manifold name becomes `%` toward node 0, the + pole;
   **pole tier** → `canonicalize_atom` (slug + variant), a plain vector key.
   A collision in either manifold tier raises `AmbiguousSelectorError`. The
   ladder reads a memoized walk of installed manifold folders, so labels
   resolve whether or not the manifold is fitted for the loaded model.

Projection operands and ablation targets skip both manifold tiers and must
fold to a single direction (a 2-node `pca` manifold name, an instrument atom or
an in-memory profile): `!formal.casual`, not `!formal`. A `%` left operand is a
manifold name verbatim. Keys carry `<ns>/` only when typed and `:<variant>`
only when not `raw`.

## Composition (`composer.py`)

`SteeringComposer` reaches the session only through constructor arguments: the
`ArtifactSource`, `NeutralSource` and `UnitSource` protocols (the session's
`ArtifactRegistry`, `NeutralBaseline` and `units.runtime.UnitsRuntime`), the
manager, the exclusive generation lock, the event bus and two callbacks.

**Resolution.** `open_scope(value)` parses (with `profile_names`), then
`resolve()` concretizes and loads every artifact: `concretize_key` pins a bare
`:sae`/`:role`/`:from` to the one fitted variant (`UnknownVariantError` 404 /
`AmbiguousVariantError` 400); `ensure_profile_registered` returns an in-memory
profile, an instrument atom, or a fitted 2-node manifold folded to its
direction; `ensure_manifold_loaded` loads `%` targets. Unknown names and labels
raise here, before the scope is entered. `resolve_scope_role` picks the scope's
assistant-role label: explicit `:role-<id>` keys must agree (else
`SteeringExprError`), plain keys mixed in warn; each `%` term implies its
nearest node's role (disagreements resolve to the highest `|along|` with a
warning), and an explicit role beats an implied one.

**Scopes and the stack.** Entering a `SteeringScope` pushes its frame (and
role) and reinstalls the flattened head: later frames overwrite equal keys, and
the innermost frame naming a role sets `active_role`. `push` takes the
generation lock (another thread's push waits for an in-flight generation) and
raises `ConcurrentGenerationError` (409) from inside a generation callback. It
is transactional: if composition or install raises, the frame pops and the
outer plan stays installed, so composition errors surface on `__enter__`.
`pop` is fail-closed: if the outer head no longer composes, all steering is
cleared before the error propagates. Each change emits `SteeringApplied` (or
`SteeringCleared`); a head that steers the prefill calls
`on_prefill_steering_change` (the session drops its unsteered prefix cache).

**`compose(entries)`** builds a fresh plan without touching installed state. It
calls `neutral.ensure()` (the first steer on a fresh model may build the
neutral baseline) and needs the means plus a whitener covering every steered
layer (`WhitenerError` otherwise).

| Entry | Lowering |
|---|---|
| `(α, trigger)` vector | profile → `fold_directions_to_subspace` (neutral-anchored rank-1 ray, + pole at `‖d_L‖`) → `affine_manifold_push(folded, "+")`, coefficient α |
| `ProjectedTerm` | `project_profile(base, onto, op, whitener=…)`, memoized on the identities of base, onto and whitener under the clause-free key (one projection serves every clause), never registered → folded and pushed like a vector |
| `AblationTerm` | target profile's per-layer direction(s) → ablate fragment with its coefficient |
| `UnitAblationTerm` | `resolve` checks the unit against the model (`UnitIndexError` 400); `compose` takes its `UnitSite` (module, coordinate span, neutral mean) from `UnitSource.site` and registers `SteeringPlan.add_unit_ablation(…, coeff, trigger)`; it joins no affine group. The plan cache records the neutral unit statistics it read. Without unit statistics: `UnitsUnavailableError` (422); on a model `BlockSpec` refuses: `UnsupportedBlockError` (422) |
| affine `ManifoldTerm` | `affine_manifold_push(manifold, position)`, coefficient `along` (`onto` is vacuous and dropped) |
| curved `ManifoldTerm` | `SteeringPlan.add_manifold(…, layers=…)` (its own term) |

**Layer scoping.** Before grouping, `select_term_layers(key, fitted, scope,
n_layers=…)` turns each term's `LayerScope` into layers: `fitted` is a
vector's or ablation target's profile layers, a projection's projected layers,
or a manifold's fitted layers. An index scope must lie inside `fitted`;
`workspace` resolves against the model depth and intersects `fitted`. A
selected layer outside the fit, or an empty intersection, raises
`SteeringCompositionError` (422) naming the term and its fitted layers. The
term's directions or push fragments are restricted to the selection (a curved
term passes `layers` to `add_manifold`, which restricts its shares and lowered
layers), so synthesis, the mean-1 share normalization and the compiled offsets
see only selected layers. A plain vector's scope is read from its key; the
other kinds carry `layers`. Ablation coefficients stay exact under any
selection (`eff_along · κ` is the user's coefficient).

`SteeringComposer.term_layers(entries)` reports where each resolved term acts,
for the dashboard's term editor: its fitted layers, its share at each (mean 1)
and the layers its clause selects. Each term is weighed alone and unscoped: a
push by its whitened displacement `‖Δ_L‖_M`, a curved `%` by its baked share,
an ablation evenly, a unit at its own layer.

Push and ablate fragments group by trigger equality; each group becomes one
`synthesize_subspace` call (one merged affine subspace per layer, push wins a
direction shared with an ablation) registered as `__affine__<i>`. Distinct
triggers make distinct groups, each normalizing shares over the union of its
terms' layers, so a scoped term sharing a trigger with an unscoped one is
normalized over the union, and a scoped term alone in its group over its
selection. `affine_manifold_push` with a label uses that node's per-layer
`node_coords`; with coordinates it blends the per-layer coords by cardinal RBF
weights over the shared layout (exact at nodes). Wrong coordinate arity raises
`ManifoldArityError`; a non-poised layout raises `SteeringExprError` advising
the label form.

**Plan cache.** `plan_for` keys an LRU (`_PLAN_CACHE_SIZE = 16`, enough for the
push/pop pair of each `generate(steering=…)`) on `tuple(entries.items())`. A hit
is reused only while every registry object composition read is still the
identical object (`_CachedPlan.deps`; a re-extract, eviction, whitener rebuild
or neutral refresh misses). Plans memoize lowering per `(device, dtype)`.

**Stack queries.** `gate_refs()` gives the read plane its `(scalar key,
GateRef)` pairs; `steering_needs_probe_gating()` decides whether the engine
wires the gate callback; `steering_active_in_prefill()` (a term steers prefill
iff `trigger.prompt and trigger.gate is None`) gates prefix-cache reuse;
`snapshot_steering_alphas()` feeds result receipts. `assert_unsteered()` raises
`ConcurrentExtractionError` for artifact-producing work inside a scope, whose
hooks would bake into the artifact.

## Lowering and hooks (`hooks.py`)

`SteeringPlan.lower(device, dtype)` places curved groups first at each layer (a
curved term lowers only at the layers of its `shares`). Two curved spans
sharing a layer with max Mahalanobis |cosine| (`B_new · P_prevᵀ`) above
`_CURVED_ORTHO_TOL = 1e-3` raise `OverlappingManifoldError` (422); disjoint
layer clauses let one curved manifold appear twice (`m%a{0-3} + m%b{4-7}`).
Each affine term then goes through `_lower_affine_subspaces`, the single
affine lowering: mean-1 share normalization, orthogonalization against the
curved span at that layer (Mahalanobis-orthogonally through the curved frames'
`read` rows; the curved term owns shared directions; κ is carried through the
re-orthonormalization; a fully contained span drops the layer),
`eff_along = share_L · _SUBSPACE_GAIN`, and `kappa = requested_κ / eff_along`,
so `eff_along · κ` is exactly the user's ablation coefficient and ablation
strength is independent of share and gain. The transient `AffineOp` and the
compiled offsets (`LoweredAffine.offset`) both consume this result.

### Gains

`_normalize_shares_mean1` makes `Σ_L share_L = n_layers`, so a 1-layer or
30-layer fit, 4-dim or 16-dim, lands in the same α band; `n_layers` counts the
layers a group or curved term acts on. The raw share is
`SynthesizedSubspace.share` (the whitened push displacement `‖Δ_L‖_M`; a
positive proxy on pure-ablation layers) or a curved fit's baked
`mahalanobis_share`, which covers exactly its fitted layers and is normalized
over the term's selected layers. `synthesize_subspace` makes each push target
whitened-unit, so every target gets the same whitened budget (`Σ_L eff_along_L
= gain · n_layers`; rationale in `ARCHITECTURE.md` §6.4). Affine `along` has no
clamp or water-fill: a high-share layer overshoots by design, bounded by the
whitened-unit target.

| Constant | Value | Applies as | Calibration |
|---|---|---|---|
| `_SUBSPACE_GAIN` | 8.0 | affine: `eff_along_L = share_L · 8`, α already folded into the target | gemma-4-12b α sweeps (`formal.casual%formal`, `personas%caveman`, `personas%hacker`): α ≈ 0.5 expressive, α ≈ 1.0 strong / over-steer. Re-fit when frames began writing along `δ` rather than `Σ⁻¹δ`, matching neutral-text NLL cost at α ∈ {0.25, 0.5, 1} (six bipolar concepts on SmolLM2-360M, gemma-3-4b, Llama-3.1-8B, gemma-4-12b). |
| `_MANIFOLD_ALONG_GAIN` | 2.0 | curved: `eff_along_L = along · share_L · 2`; periodic `BoxDomain`: `min(1, along · 2)` on every layer | `months_loop%january` on gemma-4-12b (`along = 1.0` vivid and coherent), and equal-cost `along` on spectral `emotions` (SmolLM2-360M, gemma-4-12b). `eff_along = 1` lands on the node. |
| `_MANIFOLD_ONTO_GAIN` | 0.5 | curved: `eff_onto_L = min(1, onto · share_L · 0.5)` | gemma-4-12b `emotions%dominant` at along 0.3: onto ≈ 0.5 sweet spot, 1.0 strong but coherent. Too much `onto` erases per-token variation and loops. |

Curved `along` and `onto` are fractions, capped at 1 before gain; a negative
value raises `SteeringCompositionError` (422) in `add_manifold` because a
curved position has no opposite. Periodic fits drop share weighting (share ×
gain would send some layers past the node, each wrapping to a different one),
so every `along ≥ 0.5` lands all layers on the node. `fit_mode=auto` can
resolve flat on one model and curved on another, so the same negative `%`
coefficient pushes away on one and raises on the other. Recommended affine α
is about 0.5, toward 1.0 for stronger expression.

### Hooks and installation

`AffineOp` applies `h += c − ((h·Pₐᵀ − μPₐᵀ) ⊙ s) Bₐ`, with the fixed push
`c = eff_along · target @ B` and the activation-dependent ablation restricted
to the nonzero-κ axes: read rows `Pₐ` (`read`) measure, write rows `Bₐ`
(`basis`) move (`s = eff_along · κ`). A pure push is one `add_`; a rank-1
ablation is a dot plus `addcmul_` (cheaper than a tiny GEMM under MPS
dispatch). Constants are built in fp32 and cast once to the residual dtype.
There is no norm cap on this path (a flat fit has no RBF to extrapolate).

`SteeringHook.recompose` arms a layer. One always-active `AffineGroup` makes
the hook static: `hook_fn` is a single `AffineOp.apply` with no `TriggerContext`
read. Otherwise it walks the groups, skipping inactive triggers
(`_trigger_active` short-circuits on `is Trigger.BOTH`, and
`_canonical_trigger` interns equal-to-BOTH triggers at registration so that
identity holds). Curved groups run `subspace_inject` in fp32 with a
warm-started foot: the previous token's foot seeds one Gauss-Newton step; a
cold foot (first fire, or a shape change from a re-prefill or batch change)
seeds at the origin with `_MANIFOLD_COLD_GN_STEPS = 4`. The last position's
foot carries from prefill into decode; `reset_manifold_feet()` cold-starts
every generation.

**Unit ablation.** `SteeringPlan.add_unit_ablation` records a `UnitTerm`
(module, input coordinate span, neutral mean, coefficient, trigger);
`lower_units` groups terms by module and trigger into one `UnitOp`: fp32
`keep` (1, and `1 − α` on the units' coordinates) and `bias` (0, and `α·mean`)
constants, composed in registration order when spans overlap. `UnitOp.apply` is
`x.mul_(keep).add_(bias)` on the projection's input at every position the
forward carries except the attention sink: the unit means count positions from
1, so position 0 keeps its own activation. `TriggerContext.sink_column` names
the column holding absolute position 0 (0 for a forward from the start, `None`
on incremental decode steps), and the `UnitHook` holds that column aside and
writes it back. A left-padded batch has per-row sink columns, so
`session/batch.py` sends a batch with a unit ablation to the serial loop. A
`UnitHook` is a forward pre-hook on the down or output projection; unit terms
make `static_offsets()` `None`, since the compiled graph was not traced with a
module pre-hook.

`SteeringManager.install(plan, …)`: when the session adopted compiled offset
buffers (`install_persistent_offset_hooks` runs before `torch.compile` on
CUDA/MPS) and `plan.static_offsets()` returns a map (no curved or unit term,
every group `Trigger.BOTH`, every layer a pure push), it writes the buffers in
place (`copy_`; reassigning would force a retrace; untouched layers are zeroed)
and attaches no transient hook. Otherwise it zeroes the buffers and attaches
or re-arms one transient `SteeringHook` per steered layer and one `UnitHook`
per module carrying unit ablations. Lowering and validation run before any
state changes, so a raising install keeps the previous plan; `clear_all()` is
the inverse. Routing predicates for `session/engine.py`: `all_fast_path()` (no
transient or unit hook, so the compiled graph is valid) and `static_steerable()`
(every layer and unit hook static, so StaticCache stays eligible). On a compiled
session any transient hook, a unit hook or the units read tap routes to the
eager original module.

`TriggerContext` is `SteeringManager.ctx`. `generation/decode.py::generate_steered`
sets `is_prefill`, `thinking` and `gen_step` before each forward and replaces
`probe_scores` after it; hooks only read it. Hooks write the layer output in
place and do no host sync (the throughput gate is in the root `AGENTS.md`);
validation, overlap checks and constant preparation belong in compose or lower.

## Tests

`tests/steering/` is CPU-only. `tests/support/composer.py::SteeringRig` drives
the composer without a session; `tests/units/test_unit_steering.py` checks unit
ablation and neuron pushes against hand-patched forwards.
