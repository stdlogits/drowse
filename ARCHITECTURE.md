# Drowse architecture

How the system fits together: how artifacts are fitted, how a steering
expression reaches a model forward, how hidden states become measurements, and
how the library, server, dashboard and browser edition share those paths.

This file stays at the cross-cutting level: the data flow, the reasons behind
the strict boundaries, and the invariants a change in one package can break in
another. Contracts, constants and module maps live in each package's
`AGENTS.md`; this file points to them instead of restating them. The layer
order and the repo-wide rules are in the root `AGENTS.md`.

## 1. System in one page

One `DrowseSession` owns one loaded model. Three front doors reach it:

- the Python API (`DrowseSession`);
- `drowse serve`, which exposes OpenAI `/v1/*`, Ollama `/api/*` and the native
  `/drowse/v1/*` protocol on one port;
- the Svelte dashboard mounted by that server, which drowse.ai also runs in
  the browser against a WebGPU runtime.

All of them converge on the same generation, steering, capture and measurement
paths.

```mermaid
flowchart LR
    Clients["Python / OpenAI / Ollama / dashboard"] --> Session["DrowseSession"]
    Session --> Engine["GenerationEngine"]
    Engine --> Loom["LoomTree + Recipe"]
    Engine --> Composer["SteeringComposer"]
    Composer --> Hooks["SteeringManager + layer hooks"]
    Engine --> ReadPlane["ReadPlane"]
    ReadPlane --> Runs["geometry / lens / SAE / units runs"]
    Hooks --> Model["Hugging Face causal LM"]
    Model --> Capture["HiddenCapture + token logits"]
    Capture --> Runs
    Runs --> Envelope["measurement envelope"]
    Envelope --> Results["TokenEvent / GenerationResult / loom token rows"]
    Store["$DROWSE_HOME artifacts"] --> Session
    Session --> Store
```

The engine has two deliberately separate planes:

- The **steering plane** resolves an expression, composes every compatible
  term into one update per affected layer, and injects it.
- The **read plane** plans hidden-state retention, binds frozen per-generation
  instrument state, and produces geometry, lens, SAE and unit readings
  without touching the steering representation.

They meet at **gates**: a steering trigger may depend on a read-side scalar, so
the capture plan must retain and score exactly the channels needed before the
next steered forward.

## 2. Ownership

| Concern | Owner |
|---|---|
| Model loading, fingerprints, layer discovery, chat/role/scene rendering | `model/` (`load.py`, `fingerprint.py`, `arch.py`, `scene.py`) |
| Session facade, exclusive-GPU lock, artifact registry, neutral baseline | `session/session.py`, `registry.py`, `neutral.py` |
| The generation transaction and decode routing | `session/engine.py` (`GenerationEngine`), `session/batch.py` |
| Decode loop, sampler, rendering, prefix cache, compile routing | `generation/` |
| Expression grammar and IR | `steering/grammar.py`, `terms.py`, `triggers.py` |
| Resolution, composition, gains, hooks | `steering/composer.py`, `steering/hooks.py` |
| Steering math (synthesis, the kernel, fold, projection) | `geometry/inject.py`, `geometry/fold.py` |
| Read contract, Monitor, capture planning, envelope | `readout/` |
| Applying the read plan to one generation | `generation/read_plane.py::ReadPlane` |
| Lens, SAE and units families | `lens/`, `sae/`, `units/` (block description and unit directions in `model/blocks.py`, `model/units.py`) |
| Extraction and fitting | `fit/` (`pipeline.py`, `corpus.py`, `capture.py`, `node_stats.py`, `publish.py`, `preflight.py`) |
| Artifact formats, selectors, distribution | `store/` |
| Conversation tree, recipes, replay, transcripts | `loom/` |
| Protocols, dashboard mount, CLI | `server/`, `web/`, `cli/` |

Dependencies point downward only (`tests/invariants/test_layering.py`). A
collaborator gets its dependencies through its constructor; nothing outside
`drowse/session/` reads a session private. That rule is what lets each
collaborator be tested without a session and lets the browser runtime replace
whole subsystems behind the same contracts.

## 3. Artifacts and runtime objects

### 3.1 The manifold is the steering artifact

A manifold is a set of labeled nodes, a domain or discovered layout, and a
per-model fit at selected residual-stream layers. A fitted layer is a
`LayerSubspace` (`geometry/subspace.py`) in one of two geometries:

- **Flat**: a neutral-anchored affine frame (`mean`, orthonormal `basis`, real
  per-layer node coordinates). A rank-1, 2-node flat fit is the ordinary
  bipolar steering vector; a 1-node flat fit is a monopolar ray from the
  neutral mean.
- **Curved**: the same enclosing frame plus an `r³` polyharmonic RBF surface
  through the node centroids, a neutral-origin foot, and optionally a learned
  tube thickness σ(z).

`Manifold` (`geometry/manifold.py`) groups the layers with the authoring
metadata. `SynthesizedSubspace` is not durable: it is the per-dispatch result
of composing an expression's affine terms.

`Profile` (`geometry/profile.py`) is the in-memory `{layer: direction}` view
that vector-shaped consumers need: comparison, projection operands, lens and
SAE atoms, GGUF export. Extracted concepts are not stored twice. A fitted
2-node manifold is folded to a `Profile` on demand
(`geometry/fold.py::folded_directions`), and the folder stays the only stored
artifact.

### 3.2 Fit modes

`manifold.json::fit_mode` is one of five:

| Mode | Coordinates | Result |
|---|---|---|
| `authored` | supplied with a domain | curved, exact at nodes |
| `pca` | discovered from the corpora | flat |
| `spectral` | discovered from neighborhood geometry | curved, GCV-smoothed |
| `auto` | `select_topology` picks per model | flat, curved, or periodic |
| `baked` | none (a precomputed direction from `bake`) | flat, not refittable |

Extraction is discover-`pca` capped at rank 1. There is no separate vector
method and no `pca` selector variant. Because `auto` resolves per model, the
same folder can be flat on one model and curved on another; the resolved
geometry and the ranked candidates are recorded in the fitted sidecar.

### 3.3 Templates

Some categories are referenced rather than embodied: days of the week, months,
directions. "Someone who is Tuesday" is nonsense, so persona-framed elicitation
does not apply. A template (`store/templates.py`) is a slot token, candidate
values, and one or more multi-turn contexts whose final assistant turn carries
the slot exactly once (never in the shared history). It has two consumers:

- `ChoiceScorer` (`generation/scoring.py`, via `session.score_choices` /
  `score_template`) reads the restricted-choice distribution over the values
  from the raw model distribution, optionally under steering. This answers the
  project's core question directly: did steering shift the distribution, not
  just the argmax?
- `manifold from-template` materializes one node corpus per value and keeps a
  `template_ref`, whose content hash folds into the fit identity so a context
  or value edit refits.

### 3.4 Per-model read sources

The Jacobian lens and SAEs are model-scoped sources, not concept manifolds.
Both use `base/source_registry.py::SourceFamily`: an `active.json` selection,
Drowse-owned `local/<name>/` artifacts, and commit-pinned provider bindings
whose payloads stay in the Hugging Face cache.

- A lens is one kind (`LensKey`) with fp32 per-layer matrices at one or both
  residual sites (`out`, `mid`). Two families, two rules each: the R family
  (J standard, R RelP) holds output Jacobians `J_l = E[∂h_final/∂h_l]`, read
  through `W_U` (what a vector will make the model say); the S family (K, S)
  holds input Jacobians `K_l = E[∂h_l/∂e]`, read against the input embedding
  (what it has read). Format v7 stores every kind and site (`local:j`,
  `local:r`, `local:s`, `local:k`), and one estimator fits them all.
  Providers: `rs-r` and `rs-s` (stdlogits/rs-lenses, both sites),
  `workspace-r`, `neuronpedia`, `workspace-j`. One source per family is
  selected, so an R and an S lens are resident together. Which family reads
  a place is a depth rule (`lens/depth.py`): S early, R late, both weighted
  in a crossover band; the site is the place's own (`out` for a residual, a
  neuron's write or a head's input; `mid` for a neuron's read or a head's
  write).
- An S read ranks only tokens the model saw often enough: the token mask
  takes the bound neuron-lens catalog's counts, else the lens corpus's counts
  (`lens counts`, `lens/token_counts.py`), else the tokenizer's rules.
- An SAE is a local ReLU dictionary trained here or a SAELens release (the
  `[sae]` extra), held resident at one hook layer per session.
- The neuron-lens catalog (`units/catalog.py`, `unit fetch`) is a
  commit-pinned binding of unit labels, firing tables and token counts for
  Qwen3-8B and Gemma 4 12B; nothing else needs it.

A source switch becomes visible at the next generation's bind, never halfway
through a bound run. Details: `drowse/lens/AGENTS.md`, `drowse/sae/AGENTS.md`.

### 3.5 Loom and results

`LoomTree` (`loom/tree.py`) is the session's authoritative conversation graph.
A node records role and cast label, recipe, raw token ids, token rows with
their measurements (slim for a generated or a captured prompt token, section
7.4), the reply's aggregate readings and settled aggregate envelope, committed
thinking, notes and stars. An authored node's prompt rows arrive once, from the
first prefill that re-reads them, with their replay anchor (`prompt_capture`,
section 7.5). **Provenance is recipe presence**: a node with a `Recipe` was generated
(in either seat, since the model can speak the user seat), and `recipe=None`
means authored. Rendering, transcripts, authored capture and joint logprobs all
key on this; never infer provenance from `role`.

Generation returns a list-like `RunSet` of `GenerationResult`; streaming yields
`TokenEvent`s and exposes the result at close. The loom and the results share
data but are not aliases: `stateless=True` calls, and every OpenAI and Ollama
request, produce results without touching the tree.

## 4. State and artifact identity

All state resolves through `base.paths.drowse_home()` (`$DROWSE_HOME`, default
`~/.drowse`). The tree is documented in `drowse/store/AGENTS.md`; lens and SAE
subtrees in their packages. The server never saves a conversation: the
dashboard keeps its own library in the browser (IndexedDB, written between
generations) and backup downloads, and Python saves only to an explicit
`LoomTree.save(path)`.

Every model-bound artifact keys on one of two identities from
`model/fingerprint.py`:

- `model_source_fingerprint`: metadata only, computable before load. The
  weight-free preflights (`fit/preflight.py`, `lens/lifecycle.py`) use it to
  prove a repeat is a no-op without loading weights.
- `loaded_model_fingerprint`: the live weights. Fitted tensors bind it.

A fitted manifold sidecar carries the loaded fingerprint, a `nodes_sha256`
over everything authored (labels, corpora, roles, kinds, coords or
hyperparameters, the resolved template), a `capture_sha256` over the exact
rendered token ids and pool positions, the selected layers and
`MANIFOLD_FIT_POLICY_VERSION`. A cached fit is reused only when all of them
match; the registry recomputes `capture_sha256` on load, so rendering must stay
deterministic. Provider bindings pin commits. Declared files and fitted
payloads are sha256-verified before adoption.

Writers use stable artifact locks outside removable folders, temporary files,
fsync, and atomic pointer or manifest swaps. A fit publishes by
compare-and-swap against an authoring snapshot (`fit/publish.py`), so a paused
fit can never overwrite a folder that was cleared or replaced meanwhile. Long
fits (lenses) publish immutable shard generations and leave the previous
complete generation authoritative until the new pointer is durable. Readers
never treat an incomplete or mismatched artifact as a partial success.

## 5. Extraction and fitting

There is one pipeline, `fit/pipeline.py::ManifoldExtractionPipeline`. A
steering vector is the 2-node flat case, so `session.extract` and `manifold
fit` run the same stages: inputs (snapshot, identity, cache check), capture,
geometry (pure tensors), publish. Details: `drowse/fit/AGENTS.md`.

### 5.1 Conversational corpora

`fit/corpus.py::CorpusGenerator` has the model answer a shared set of baseline
prompts (`drowse/data/baseline_prompts.json`, 48 prompts) in character. The
concept rides a system prompt chosen by `kind` (abstract "someone {c}",
concrete "{art} {c}", or a custom template) and a swapped assistant-role label.
The corpus is a response list aligned `response[i] ↔ prompt[i % k]`.

Capture pools the swapped-back `[user: prompt, assistant: response]` pairs in
standard-assistant space: the persona is generation-only and its signal rides
the text. Only an explicit role (`--role`, the `:role-<name>` variant) is also
substituted at capture, and the same substitution is applied at steering time,
so the extract baseline equals the steer baseline.

The neutral corpus (`drowse/data/neutral_statements.json`) answers the same
prompts with no persona. One length directive leads the system prompt at all
four sites (node and neutral generation, node and neutral capture). As
common-mode it cancels in `node − neutral`, it matches capture framing to
generation framing, and it keeps answers short. The shared prompts hold topic
common-mode across nodes, so centroids stay comparable without a per-manifold
scenario set.

### 5.2 Capture and the baseline-match contract

Rows are rendered and tokenized once and pooled at
`model/tokens.py::last_content_index`: the last non-special token before the
chat-template trailer. The monitor, the read plane, the naturalness eval and
the drowse.ai capture pool at the same position. Capture batches across node
boundaries in right-padded batches that halve on OOM, aborts after the last
requested block (the final norm and LM head never run), and reduces to
centroids on device unless a curved fit needs rows, which spool to a
layer-major store. A content-addressed capture cache
(`models/<safe>/manifold_capture/`) lets geometry-only refits skip the
forwards.

Fit capture renders through `apply_chat_template` (or
`model/role_templates.py::apply_with_role`), while generation renders through
the scene stitcher (`model/scene.py`). The stitcher's byte-exact round-trip
validation against the template is what keeps the two renders identical; a
family that fails it falls back to the template path.

### 5.3 Neutral anchoring and the Mahalanobis metric

Neutral activations are cached fp32 per model (`store/neutral.py`, format 5).
This is a range invariant: late-layer residual channels overflow fp16 and
would poison every covariance-derived read. The same forward measures the
mid means (each block's residual after attention at the same pooled
positions: the centring mean of an S read at the lens `mid` site, as the
activations' `X.mean(0)` is at `out`) and the
unit statistics the units family reads in and ablates toward (one hook pair
per block, streamed; `drowse/units/AGENTS.md`), stored in the same shards; a
model whose blocks `BlockSpec` refuses gets the cache with units unavailable.

`geometry/whitener.py::LayerWhitener` keeps the centered neutral rows and a
Woodbury form of the regularized inverse covariance. It applies `Σ⁻¹`, computes
Mahalanobis norms and cosines, forms reduced-space Grams and the LEACE
projection, all without a dense hidden-size-squared matrix.

Every activation-space surface requires complete finite whitener coverage of
its layers (the root `AGENTS.md` "Mahalanobis only" rule), because a few
massive-activation ("rogue") channels dominate the Euclidean metric on real
LMs.

Every fitted frame is neutral-anchored: the stored mean is the projection of
the neutral mean into the span, node coordinates are measured from it, and
coordinate 0 means neutral. The basis comes from the node-centroid scatter
around the nodes' own mean, never around neutral: anchor-centering would inject
the neutral offset as a spurious axis and break PCA@2 ≡ difference-of-means.
With a whitener the basis maximizes `vᵀS_b v / vᵀΣv` (the Fisher/LDA
objective), which divides each direction by its background variance so the
rogue channels cancel the way they cancel in a difference of means.

### 5.4 Discovering and fitting geometry

For discover modes each layer contributes its whitened, node-mean-centered Gram
matrix, and coordinates come from their mean, the consensus Gram. Whitening
puts every layer in common units, so the average is signal-weighted and needs
no reference layer: a layer where the nodes do not separate drops out on its
own. PCA coordinates are its leading eigenvectors (up to `var_threshold`,
capped at `max_dim`); spectral coordinates come from Laplacian eigenmaps on a
k-NN graph; `auto` chooses flat versus curved by GCV and detects periodic axes
by Vietoris–Rips H1 persistence (`geometry/topology.py`).

Flat fits store the frame and real per-layer node coordinates; their display
layout is translated onto neutral so `% 0,…,0` reads as neutral. Curved fits
add the RBF surface, smoothed by a GCV-selected λ against centroid noise
(authored fits stay exact so a node is an exact steering target), and raw
curved fits learn the σ tube from the retained rows.

Each fit bakes a per-layer Mahalanobis `share`, the cross-layer budget the
hooks normalize at apply time. Discriminative layer selection (Selective
Steering, Dang & Ngo 2026) keeps a flat axis at a layer only when the node
projections relative to neutral straddle zero: same-side layers encode
intensity, not polarity. `--no-dls` opts out. Curved fits have no polarity and
skip DLS.

A monopolar concept (`extract(concept)` with no baseline) is a genuine 1-node
fold of `concept − ν` into a neutral-anchored ray: neutral is the implicit
negative pole, and the difference already cancels common-mode like DiM does.

## 6. Steering language and lowering

### 6.1 One grammar

Python, YAML `vectors:`, HTTP, CLI and loom recipes all parse through
`steering/grammar.py::parse_expr`, and `format_expr` round-trips the result.
Results, recipes and scoring stamp `str(steering)`, so every stored steering is
replayable. The authoritative grammar, resolution ladder and trigger table are
in `drowse/steering/AGENTS.md`. In brief:

| Form | Meaning |
|---|---|
| `0.3 warm` | push toward a pole or label |
| `a~b`, `a\|b` | keep / remove the Mahalanobis component of `a` shared with `b` |
| `!a` | mean-ablate `a` (`h' = h − α(h·d̂ − μ·d̂)d̂`) |
| `along[,onto] M%label`, `M%x,y` | move toward a node or coordinates of a manifold |
| `x{12-20}`, `x{4-8,20}`, `x{workspace}` | act only at these layers (any term kind) |
| `@response`, `@after`, `@first:N`, … | restrict the token phase or window |
| `@when:probe>t` | fire only while a read-side scalar holds |
| `jlens/<word>`, `sae/<id>` | instrument atoms in reserved namespaces |
| `neuron/L8n344`, `!head/L20h17` | model units: a neuron push along its write (a one-layer profile through the one kernel), a unit mean-ablation (`UnitAblationTerm` → a `UnitOp` pre-hook replacing the unit's activation by its neutral mean) |

### 6.2 Resolution and composition

`SteeringComposer` turns a parsed `Steering` into a `SteeringPlan`.
`open_scope` resolves and loads every artifact before the scope is entered, so
an unknown name fails before any state changes. A bare slug resolves through
the ladder in `store/selectors.py::resolve_bare_atom` (node label, then 2-node
name, then plain key); collisions raise `AmbiguousSelectorError` instead of
choosing silently. The scope's assistant-role label is resolved from the terms
(explicit `:role-<id>` keys must agree; implied roles from `%` terms resolve to
the strongest with a warning).

A term acts on every layer it is fitted at unless a `{…}` layer clause scopes
it. The clause is the layer-selector language of `model/layers.py` (indices,
`a-b` ranges, `workspace`, `all`) and part of the term's identity, so the same
selector under two clauses is two terms. Composition restricts the term's
per-layer fragments (a curved term's layers) to the selection before
grouping; `workspace` is the model's depth band intersected with the fitted
layers, and a selected layer outside the fit, or an empty intersection, is a
422.

Composition groups terms by trigger. Vectors are folded into neutral-anchored
rank-1 frames, `~`/`|` results are folded the same way, ablations contribute
collapse axes, and affine `%` terms contribute their fitted basis and target.
All affine terms of one trigger become one `SynthesizedSubspace` per layer
(`geometry/inject.py::synthesize_subspace`), so terms sharing a layer
superpose instead of colliding; partial, repeated, negative and non-orthogonal
ablations get exact per-axis coefficients. Each curved `%` term stays its own
term. Two curved spans may share a layer only when they are orthogonal
(`OverlappingManifoldError` otherwise), and the merged affine span is
orthogonalized against them, the curved term owning the overlap.

One resolution subtlety is easy to miss: once the registry holds a profile
under a 2-node name (after `session.extract`, or after the name was used as a
`!`/`~`/`|` operand), a bare `formal.casual` parses as a folded-vector push
instead of a `%` push toward its positive node. The two weight layers
differently (folded share versus the pole's neutral-anchored coordinate).

### 6.3 The kernel

`geometry/inject.py::subspace_inject` is the semantic definition of every
injection. It splits `h = mean + H_m + H_n + H_o` (surface foot, off-surface
in-subspace residual, off-subspace residual) and applies two operations:

- **along** translates the foot by a fixed `along·(target − origin)` offset.
  A fixed offset preserves each token's in-subspace spread; lerping each foot
  onto the absolute target erases it and loop-collapses at strong push. On a
  curved surface the off-surface residual is transported to the new tangent
  frame by a norm-preserving rotation that is exactly the identity when the
  foot does not move.
- **onto** shrinks the off-surface residual toward the surface: by `(1 − onto)`
  with no σ-field, or toward the local tube thickness σ(z) with one, so
  `onto = 1` lands on the surface's typical set rather than its zero-thickness
  wire. It is vacuous for flat terms.

`H_o`, the off-subspace residual, is always kept verbatim. It contains every
composing neighbor's span, which is why a vector and several orthogonal
manifolds compose with zero cross-talk.

On a flat subspace the foot is the projected coordinate and the kernel reduces
to an analytic slide, with no foot solve, RBF or norm cap (a flat fit cannot
extrapolate off-domain). At runtime affine terms run
`steering/hooks.py::AffineOp`, a compose-time split of that branch, pinned
equal to the kernel by `tests/steering/test_hooks_triggers.py`; only curved
terms call `subspace_inject` per token, with a warm-started foot solve and a
`3·‖h‖` norm cap against off-domain RBF extrapolation.

### 6.4 Gains and calibration

The kernel applies no gain; `steering/hooks.py` owns the constants and their
calibration notes (`drowse/steering/AGENTS.md`). Two design decisions make one
coefficient scale meaningful across very different fits:

- **Whitened-unit targets.** `synthesize_subspace` makes each affine push
  target a whitened unit direction and sets the share to the whitened
  displacement. Raw node distance cannot be the knob: a tight pole sits about
  0.3 from neutral and a far persona centroid about 17, a ~100× span no single
  gain calibrates. Every target gets the same whitened budget, linear in α.
- **Mean-1 share normalization.** The per-layer share is normalized so its
  layer sum equals the layer count, so a 1-layer or 30-layer fit, 2-node or
  107-node, lands in the same α band. The count is of the layers the trigger
  group acts on, so a layer clause selecting one layer of a 30-layer fit lands
  there too. There is no lever, no N correction, and no clamp on affine
  `along`: a high-share layer is meant to overshoot, bounded by the
  whitened-unit target.

The recommended affine α is about 0.5, toward 1.0 for stronger expression.
Curved `along` and `onto` are fractions in `[0, 1]`; the hooks scale them into
a per-layer `eff_along`, where `eff_along = 1` lands on the node and larger
values overshoot it, bounded by the curved norm cap. Periodic fits drop share
weighting and clamp `eff_along` to 1 so no layer wraps past the target node to
a different one.

### 6.5 Triggers and gates

`TriggerContext` carries the prefill/decode phase, the thinking state, the raw
generation step and the latest gate scalars. Hooks consult it only for dynamic
terms; the static always-active path never reads it.

Phases are easy to misread: `@response` (`@generated`) includes thinking
tokens, and only bare `@after` is response-only. `@after:N` is a token window,
not a phase.

Gate keys name an attached probe plus a channel: an axis (`p`, `p[i]`), a
subspace fraction (`p:fraction`), tube membership (`p:membership`), a
normalized label distance (`p@label`) or a soft assignment (`p~label`).
Geometry produces every channel; lens and SAE probes produce one strength
channel. At generation preflight `ReadPlane.resolve_gates`
(`generation/read_plane.py`) rejects a gate on a lens or SAE probe for a
channel that family cannot produce (`UnsupportedProbeChannelError`); a gate
naming no attached probe is silently inactive. Gates lag one forward: scalars
scored after forward `t` steer forward `t + 1`, and the prefill sees none.
Channel units and semantics: `drowse/readout/AGENTS.md`.

## 7. The read plane

### 7.1 Four instrument families

`session.instruments` registers four families behind one contract
(`readout/protocol.py::Instrument`):

| Family | Facade | Reading | Unit |
|---|---|---|---|
| `geometry` | `session.geometry` (`GeometryInstrument` over `Monitor`) | `ProbeReading` | domain coordinate; Mahalanobis fraction and label distance; curved membership and assignment |
| `lens` | `session.lens` (`LensInstrument`) | `ScalarReading` | `jlens/`: one token's mean fitted-layer probability through the R lens; `slens/`: its footprint z through the S lens, averaged over the layers with the depth rule's `w_S` |
| `sae` | `session.sae` (`SaeInstrument`) | `ScalarReading` | feature activation over Neuronpedia's `maxActApprox` when known, else raw activation |
| `units` | `session.units` (`UnitsInstrument`) | `ScalarReading` | a neuron's activation over its pole's reading scale (signed), a head's write norm over its own: the bound neuron-lens catalog's `act_top`, else the neutral statistics', soft-floored (`units/stats.py::UnitScales`); `meta.scale` names the source |

The units family reads the model's own parts rather than the residual: its
bound run owns a tap (one forward hook per touched down or output projection,
present only while a probe or gate consumes it) and its statistics come from
the neutral pass (neutral cache v5: a neuron's mean activation and per-pole
`act_top`, a head's mean `z_h` and write-norm `act_top`). A gate on a unit
needs no attached probe. `drowse/units/AGENTS.md` owns the definitions.

Units are also read statically, from their weights
(`units/reads.py::UnitReader`, `session.units.read|top|find`): each unit
vector goes through `LensInstrument.read` at the place the lens rule assigns
it (a neuron's write at `out`, its read direction at `mid`; a head's OV write
at `mid`, its query/key/value inputs at `L − 1` out), so every place returns
the depth rule's S and R halves. Hub tokens are hidden by neuron-lens's
layer-contrastive z, whose per-layer population statistics are a derived
cache under `models/<id>/units/contrast/`. The neuron-lens catalog
(`units/catalog.py`, Qwen3-8B and Gemma 4 12B) is a commit-pinned binding of
labels and firing tables under `models/<id>/units/catalog/`; its labels are
data shown with their credit and link, never inputs to a computation. The
`neurons` dictionary of `lens decompose` (each layer's signed neuron writes,
Mahalanobis like the token dictionaries) is `unit find`.

Geometry readings carry coordinates, fraction, nearest labels, residual,
membership/assignment, per-layer traces and depth summaries. Every geometry
probe is a fitted `Manifold`, flat or curved, read in the whitened metric and
reported in its domain frame (a rank-1 probe reads `1.0` at its positive node).
Live token reads, aggregates and replays come from the same scorer, so they are
directly comparable. Lens, SAE and unit readings
carry one value with an explicit unit, a per-layer trace and a depth summary.
The `ProbeReading` projection of a scalar reading exists only at the vendor
extension boundary (`TokenEvent.probe_readings` and the OpenAI/Ollama
`x-drowse-probe-readings` body key).

The R-family readout is `softmax(W_U · norm(J_l h))` at each fitted layer. One
unit, the per-layer probability `p_l`, backs every R surface: strength is
`mean_l p_l`, depth center and spread are weighted by `p_l`, and `jlens/`
probes, gates and the live wire report it. The S family reads a residual's
pullback `K_lᵀ (h_l − μ_l)` (centred on the neutral mean at that site) against
the input embedding as a footprint z over the vocabulary, ranking only the
token mask's kept tokens (`lens/AGENTS.md`). `lens top`, `lens.read` and the
token anatomy pick the family by depth (`lens/depth.py`) and say which. A
`jlens/<word>` steering atom is a different object: the activation direction
`W_U[v] @ J_l`; `slens/<word>` is the token's S footprint at each layer. An
`sae/<id>` atom is the decoder row as a one-layer profile; nothing clamps and
decodes a feature inside the forward.

### 7.2 The per-generation transaction

Each family runs `close_run → prepare → plan → bind` per generation and closes
in a `finally`. `prepare` snapshots probe specs, live intent and source
identity (the lens may adopt a new disk artifact here); `plan` declares
demanded layers, tail depth and gate keys; `bind` opens a frozen
`InstrumentRun`. A run measures a frozen binding, which is why source switches
and lens/SAE detaches are safe mid-generation.

`generation/read_plane.py::ReadPlane` executes the transaction for one
generation: it binds the runs, applies the capture plan, arms the step sinks,
builds the gate callback, and runs the finalize reads. The decode loop computes
one `step_id = len(generated_ids)` per forward and hands the same value to the
capture sink, the gate callback and the token tap. Within a forward, gate and
display reads share work through the geometry sink row or the lens/SAE/units
step-keyed stash, so one forward is never scored twice.

### 7.3 Capture planning

`readout/planner.py` is pure: `resolve_read_demand` folds the generation's
consumers into one demand, and `plan_capture` picks one physical
`HiddenCapture` mode: gating subset, lean incremental, incremental,
aggregate-only, or full. The planner, not an instrument, decides upgrades when
two families need different retention from the same forward. The mode table,
with each mode's finalize read, is in `drowse/readout/AGENTS.md`.

The rule behind the table: scoring is conditional on need. A generation with
probes attached but no per-token consumer keeps a bounded tail ring and scores
once at finalize. Capture hooks copy or accumulate tensors only; host-side
scoring runs after the forward, so a deep capture hook never drains the device
queue before the remaining layers and the LM head are enqueued.

### 7.4 One measurement envelope

`readout/measurements.py` builds the one versioned record (`version: 2`) used
by native WebSocket token frames, final aggregates, persisted loom rows and
replay endpoints:

```text
{
  version,
  scope: token | aggregate | replay,
  provenance: captured | replayed,
  scores,
  instruments: {
    geometry: {binding?, readings},
    lens:     {binding, readings?, readout?},
    sae:      {binding, readings?, readout?},
    units:    {binding, readings}
  },
  anatomy?          # replay only: one token, every layer
}
```

`readings` are attached probes in each family's native shape; `readout` is the
family's discovery surface (the R lens's layer-by-vocabulary matrix and aggregate
chips, the SAE's active features). The flat `scores` are derived joins for
cross-family consumers such as token tinting, not separate measurements; a
reading's per-layer trace lives in the reading alone. A generated loom token
row keeps `token_row_measurements(envelope)`: what the transcript's tint, the
reply trace and the tree filter read per token without a model, the flat
`scores` and each geometry reading's `fraction` and `coords`, at
`READING_DIGITS` (6) significant digits. Everything else about the token (a
geometry reading's nearest nodes, assignment, residual, membership and depth;
the lens and SAE readings, readouts and bindings; the unit readings) is the
token view's, which replays the token (section 7.5) while a model is loaded;
the lens and SAE replays and the anatomy return the attached probes' readings
too. An authored prompt row is the same projection, replayed through its
node's anchor (section 7.5). The node keeps its reply's settled aggregate once
(`aggregate_measurements`: each reading's summary, no traces, no scores), which
the rack and the probe inspector draw after a reload. A row of a call with
`persist_per_layer_scores` keeps `summary_measurements(envelope,
per_layer=True)`, every reading whole with its traces but never the probe
inspector's subspace coordinates. A SmolLM2 generated row's measurements with
the 17 default probes are about 1.6 kB (about 2 kB a row with its
alternatives; a summary row was about 9.7 kB), where the whole envelope is
about 118 kB. A prompt row with those probes, a curved one and a lens pin
under a live R lens is about 1.8 kB, where it was about 42 kB kept whole: a
2,513-token paste is a 4.6 MB node and autosave record, where it was a 106 MB
node past the dashboard's 64 MiB autosave limit.
Bindings record source and steering context, so a historical row stays
interpretable after the active lens or SAE changes; the lens binding names a
source per lens family (`sources.R`, `sources.S`) that produced the row. A
replay envelope may carry a token `anatomy` (section 7.5) beside its
instruments. The shape is pinned by
`tests/readout/test_measurements_envelope.py` and the browser golden
`measurement-envelope-v2.json`.

### 7.5 Replay and token drilldown

Each generated token row stores a `raw_index` into its node's raw token ids.
An authored turn's rows have none: their node carries a `PromptCapture`
(`loom/tree.py`) written with them, and `raw_index` addresses its captured
rows (thinking rows, then visible rows). It records what a replay of the
capture needs and nothing that a later deletion could take away: the
capturing call's steering, the render's resolved system prompt, thinking flag
and flat mode, where each channel starts in that render, and the digest of
the render before the later channel. The replay renders the path to the node
with those parameters (no generation prompt), checks the head against the
digest (an earlier turn edited since refuses, 400; an edit of the node's own
text keeps reading the original, since its tokens come from the rows), and
runs the head plus the channel's tokens before the anchored one as one
prefill under the capture's steering. A prefill fires a term exactly when it
has a prompt slot and no probe gate, so the expression's prompt-phase
steering replays exactly and no gate or phase window refuses. The capture
reads each prompt token cold (`Monitor.cold_reads`), as a replay does, so a
curved probe's foot never depends on the token before. Unit activity, a whole
decode record's read, refuses an authored turn.
`loom/replay.py::TokenReplayer` backs every family's `token_readout`: it
rebuilds the node's render and raw decode prefix to the producing position,
runs the capture forward under the recorded recipe steering (or the unsteered
counterfactual), and hands the pooled hidden states to the family. The forward
runs as generation ran it (`replay_segments`: one forward when every term
fires everywhere, else the prefill and the decode runs where the triggers
fire alike), so one token reads the same number in every family and in the
node replay. A probe-gated term fired on the generation's live probe scores,
which the node does not record: the replay steps one decode position at a time
under a gate-only read-plane capture (`ReadPlane.gate_run`) and scores the
gates after each forward as the decode loop does, so it fires where generation
did (slower: one forward a position). A thinking-phase window on a family that
thinks follows the decode loop's delimiter machine, which the node does not
record either, so a replay under one refuses (422).

The node replay (`loom/node_replay.py::NodeReplayer`) is the same rebuild
(`TokenReplayer.forward`) running a forward of its own, tapped where the
model's parts meet the residual (each block's input, `mid`, output before and
after the steering hooks, the down and output projections) and stopped after
the last block. **Token anatomy** reads one token at every layer: the
residual at `out` through both lens families, the attention write at `mid`
and the MLP and steering writes at `out` as directions; the neurons and heads
with the largest direct contribution to the token's logit through the R lens
(the token's unembedding row, its final norm linearized, pulled back by
`J_Lᵀ` at `out` for neurons and `mid` for heads); and, under eager attention
only when a head can be ranked, the top heads' attention rows. It rides the
replay envelope's `anatomy`. **Unit activity** reads every neuron and head at
every token of a turn in the units family's unit and reduces it on the device
(peaks, traces, one token's float16 population). Each leaves the device in one
transfer. Unlike a token readout, the node replay drives the trigger context:
with every open term always active it is one forward, otherwise the prefill
and then KV-cached runs of decode positions where every trigger fires alike,
as generation ran them, and one position a forward under a probe gate (its
scores replayed as above); a thinking-phase window on a thinking family, which
the node does not record, refuses (422).

**Residual or direction.** Every lens read says what it reads
(`lens/read.py`, `of`, no default). The S read of a residual is centred on the
neutral layer mean at its site (the stream's large common component would
otherwise swamp the position's own tokens); a direction (a steering vector, a
unit's write, a write delta) and every R read are read as given. A residual
read with no mean refuses (`NeutralMeansError`, 422): `read_prompt`, `slens/`
probes and gates, and the anatomy's `reads_as` all read residuals.

A token fork (`loom/ops.py::plan_fork`) replays the raw prefix, forces the
replacement token or text, then samples under the node's recipe. The recorded
seed re-applies by default and the sampler still draws on every forced step, so
the RNG stays aligned through the fork; a new seed asks for a different ending.

## 8. The generation transaction

### 8.1 Preamble

`generate`, `generate_stream`, fan, batch, sweep and every server request reach
`session/engine.py::GenerationEngine.generate_one`, inside `transaction()`, the
single-flight guard:

1. resolve the call into a frozen `CallSpec`: `loom/recipe.py::compose_call`
   merges the cast member, the call kwargs, the regen anchor and its modifier
   (rules in `drowse/loom/AGENTS.md`), and the steering expression is parsed and
   resolved;
2. plan the token consumers;
3. create the loom nodes (unless stateless), then enter the steering scope, so
   a bad expression leaves the tree untouched;
4. render through `PromptRenderer` (scene grammar, cast labels, committed
   thinking);
5. plan the reads and `ReadPlane.begin`;
6. decode with a `TokenTap` when something consumes tokens;
7. pop the steering scope, so finalize reads run unsteered, then finalize.

No sampling or steering argument mutates the session defaults. Same-thread
re-entry from a token callback is refused by the engine's `GenState` phase,
not by the lock.

### 8.2 Decode loop

`generation/decode.py::generate_steered` runs under `torch.inference_mode()`:
stop check, `step_id`, trigger context, forward, capture step, gate scoring,
sample, thinking/stop bookkeeping, token callback. The sampler applies
penalties, logit bias, temperature, a top-k cap (1024 by default) and top-p
over a fixed-width `torch.topk` pool. Per-token logprobs, alternatives and
perplexity describe that sampler distribution; completion scoring reads the
raw distribution. Each decoded token costs exactly one device-to-host transfer:
the token id, entropy, chosen logprob and top-K ride one packed copy.

The thinking state machine keeps thinking and response tokens distinct across
delimiter- and channel-based templates, and unfinished thinking never leaks
into the response text.

### 8.3 Routing

- **Prefix cache.** One opt-in slot (`generation/prefix_cache.py`), prefilled
  unsteered with capture suspended. A generation reuses it only when no active
  term fires in prefill (`SteeringComposer.steering_active_in_prefill`) and it
  needs no hidden states or thinking. A mismatch misses; it never mis-hits.
- **StaticCache** when the runtime has it, graph capture is thread-safe, and
  steering is absent or entirely the static single-affine fast path; the
  compiled-module rule below can still drop it.
- **The compiled module** only when the live hook topology matches the traced
  one: no transient capture hooks and steering absent, or lowered to the
  persistent offset buffers (a static pure-push plan). Anything else runs the
  eager module on a DynamicCache instead of retracing per topology.

### 8.4 Finalize

Finalize decodes the response slice, pools the last content position (the same
`last_content_index` the fit used), produces the full aggregate readings from
the retained slice with a cold curved foot solve, returns optional hidden
states, builds the `GenerationResult` and its aggregate envelope, emits events,
and stamps raw token ids and aggregates onto the loom node. Teardown is
exception-hard: runs close, capture and transient hooks detach, the scope pops,
and the lock releases even if the model, a stream consumer or finalize raises.

### 8.5 Multi-row runs

`generate(n > 1)` fans out siblings with derived seeds, and `generate_sweep`
runs the Cartesian product of `{concept: [alphas]}`; both are always serial.
`generate_batch` takes one batched Hugging Face `generate` when the call passes
one eligibility predicate (stateless, always-active triggers, no per-token
observation, no seeded sampling, penalties or stop strings); otherwise rows run
serially, sharing a prefilled common prefix when steering leaves the prefill
alone. Details: `drowse/session/AGENTS.md`.

## 9. Server

`server/app.py::create_app` serves one session:

| Surface | Purpose |
|---|---|
| `/v1/*` | OpenAI-compatible models, chat completions and completions; `steering` rides the request body |
| `/api/*` | Ollama-compatible metadata, chat and generate; `steer` rides `options` |
| `/drowse/v1/*` | sessions, tree, profiles, probes, manifolds, templates, instruments, SSE operations, polled preparations, one WebSocket per session |
| `/` | the committed dashboard bundle, mounted last (`web=True`) |

Handlers validate (`NativeRequest`, `extra="forbid"`), lock, call public
session APIs, and serialize. OpenAI and Ollama generations are always
stateless; only the native WebSocket grows the loom. One FIFO asyncio lock
serializes protocol callers before they reach the synchronous engine (300 s,
then 503); lens fits and SAE training hold the exclusive-GPU section outside
that lock, so a generation arriving mid-fit gets an immediate 409 instead of
queueing behind a long job.

Generation never waits on a WebSocket reader. Each connection writes one
ordered outbound buffer; a reader that falls 256 droppable frames (`token`,
`tree_mutated`) behind lags, misses frames, then gets one `resync` and
rebuilds what it missed from `GET /tree`, while the reply keeps generating
into the loom. Only Stop, a real disconnect or a send the peer accepts nothing
of for 30 s ends a reply `cancelled`; keepalive pings carry no pong timeout,
since a slow reader answers them only after its backlog.

`server/errors.py::describe_error` is the one error classifier for every HTTP,
SSE, polled and WebSocket surface. Typed `DrowseError`s keep their status and
text, a bare `ValueError` is a logged and scrubbed 500, and the wire `code` is
the exception class name.

Authentication is middleware on the private prefixes only, before body
parsing. With a key, every request needs `Authorization: Bearer`; without one,
the peer and `Host` must be loopback (a DNS-rebinding defense). Browsers
cannot set WebSocket headers, so the dashboard sends the key as a
`drowse.auth.<base64url>` subprotocol; a `?token=` query is scrubbed and never
accepted. TLS, quotas, rate limits and multi-tenant isolation are deployment
concerns. Details: `drowse/server/AGENTS.md`.

## 10. Dashboard and the browser edition

The Svelte 5 dashboard lives in `webui/` and builds into the committed
`drowse/web/dist/`, which is exactly what wheel users run; source and bundle
change together and CI fails on drift. It is drawn in Ledger (the
`@stdlogits/ledger` package, pinned at v0.1.1). Its home is the workbench: the thread (the loom as
the path you are on, its map and weave one step away), the page (the
conversation, with the steering recipe as chips over the composer) and the
inspector (whatever is selected: a token's drilldown and anatomy, a probe, a
unit, a steering term; with nothing selected, the instruments: probes, live
lens readout, SAE features, sampling). Manifolds, Units, Lens and Model are
rooms beside it, as routes. The dashboard writes steering only as chips over
the structured terms the server returns (`parts`); it never parses or takes
expression text, while agents (WebMCP), HTTP, the CLI and Python keep the
grammar (`drowse/web/AGENTS.md`). REST
response types are generated from the server's TypedDicts
(`scripts/generate_webui_types.py` → `webui/src/lib/types.gen.ts`); request
bodies and WS/SSE frames are hand-written in `webui/src/lib/types.ts`.

[drowse.ai](https://github.com/stdlogits/drowse.ai) builds the same `webui/src`
with a WebGPU runtime in place of `HttpRuntimeClient`. Shared code reaches the
runtime only through the `RuntimeClient` contract
(`webui/src/lib/runtime/contracts.ts`) and the five `@runtime-*` aliases;
hosted-only code (the WebGPU runtime, workers, catalog, its own TypeScript
steering parser and archive mirror) lives in drowse.ai.

Python is the reference implementation the browser replays. The parity
references are:

| Semantics | Python reference | Golden |
|---|---|---|
| expression grammar | `steering/grammar.py` | `tests/fixtures/browser_steering_expression.json` |
| sampler | `generation/decode.py` | `sampling-parity-v1.json` |
| affine lowering and the kernel | `steering/hooks.py`, `geometry/inject.py` | `affine-*-lowering-v1.json`, `post-block-rank-one-v2.json` |
| curved kernel, SAE and lens token directions, gates | `geometry/inject.py`, `sae/backend.py`, `lens/jlens.py`, `steering/triggers.py` | `structured-program-parity-v1.json` |
| Fisher fit, DLS, topology and RBF | `geometry/subspace.py`, `fit/capture.py::compute_dls_axes`, `geometry/topology.py`, `geometry/rbf.py` | `affine-fisher-fit-v1.json`, `dls-axes-v1.json`, `topology-rbf-parity-v1.json`, `topology-adversarial-v1.csv` |
| measurement envelope | `readout/measurements.py` | `measurement-envelope-v2.json` |
| loom ops and events | `loom/tree.py`, `loom/ops.py` | `loom-operation-transcript-v1.json` |
| generation-settings composition | `loom/recipe.py::compose_call` | `recipe-compose-v1.json` |
| tree filter | `loom/filter.py` | (ported, no golden) |
| `.drowse` archive and install | `store/archive.py`, `store/install.py` | `tests/browser_drowse_archive_bridge.py` |

The browser also mirrors `RESERVED_NAMESPACES`, `METHOD_FIT_MODES`, the
manifold sidecar format and lens format v7 (a v7 reader accepts the optional
`split_half`), and its hosted R-lens fit calls `lens.store.save_lens_at`; its
estimator call moves from the removed `fit_jacobian_lens` to
`lens.fit.fit_lens(kind=, sites=("out",))` after 6.0.0 (`drowse/lens/AGENTS.md`). Goldens are
byte-frozen and recomputed by `tests/browser/test_browser_parity.py`; a
semantic change regenerates the golden in the same change, and drowse.ai must
match before its submodule pin moves past it.

## 11. Lens, SAE and catalog lifecycle

`lens/fit.py::fit_lens` is the one lens estimator, transport
(`lens/transport.py`, ported from neuron-lens and generalized over
`BlockSpec`). It fits any kind at either or both sites over a
pretraining-like corpus (FineWeb-Edu by default, or any pinned Hub dataset or
file): each document is linearized once from the block description, then
basis seeds sweep through hand-written per-block maps, cotangents down from
the last block for J/R, tangents up from the embedding for K/S, so one sweep
covers every layer and both sites. No autograd runs; the autograd estimator
survives only as the test oracle. Standard rules give J/K, RelP rules
(`lens/relp.py`: LN, identity, half) give R/S; forwards are unchanged. A
fit records its split-half difference per site and layer, the check on
whether more documents would help, unless it is a corpus extension of a
durable artifact (whose halves are gone) or runs with `split_half=False`
(`--no-split-half`). The fit verifies the model's description
first and refuses undescribable blocks (`UnsupportedBlockError`), adapts to
OOM, and on MPS drains the queue and guards against zero rows. It checkpoints
every `--checkpoint-every` documents and every ten minutes, and SIGINT or
SIGTERM finishes the document in flight and checkpoints before exiting. `lens/lifecycle.py::plan_lens_fit` chooses noop, top-up, resume or
fresh from metadata alone, and a no-op is proved without loading the model.
Corpus resolution and streaming run in an isolated, cancellable worker process.

`sae train` fits a local residual-post ReLU SAE; `sae fetch` binds a SAELens
release at a pinned commit. Both adapt to one resident runtime
(`sae/runtime.py::SaeRuntime`). The last selected SAE restores at session
construction. Live discovery, pinned probes and gates share one encode per
step.

`drowse serve` adopts a compatible prepared source for each lens family in a
fixed preference order (never rewriting a selection the user made) and turns
the R family's live readout on; the live SAE is discovered only when the
dashboard is mounted (`drowse/cli/AGENTS.md`).

`unit fetch` binds the neuron-lens catalog at a pinned commit (labels, firing
tables and token counts; no weights, no model load). A bound catalog sets the
unit reading scale (its `act_top`) and the S token mask; without one, units
divide by the neutral statistics' scale and the mask takes `lens counts`, and
every surface names the source it used.

## 12. Distribution and transfer

`pack` owns manifold lifecycle: list, show, install, search, push, remove,
clear, refresh and export. A Hub pack is a `drowse-manifold`-tagged model repo
holding exactly one root `.drowse`, a summary `manifold.json`, and a card
whose `drowse:` block carries the search fields; drowse.ai shares the
convention. `.drowse` v1 (`store/archive.py`) is a narrow ZIP transport for
one manifold closure and its one referenced template. It never carries model
weights, lenses, SAEs or caches. Every entry's path, size, hash and tensor
header is validated before a staged install; checksums prove integrity, not
publisher identity.

Cross-model transfer (`geometry/procrustes.py`, `geometry/transfer.py`) fits a
per-layer affine alignment from shared neutral activations, maps means as
points and bases as directions, re-bakes the share in the target model's
Mahalanobis metric, and writes the `from-<source>` variant. Curved transfer
rejects maps that would make a scalar tube anisotropic. Discover layouts are
model-specific, so transfer maps fitted activation geometry, not a universal
coordinate system.

A fitted 2-node flat manifold exports as a llama.cpp control-vector GGUF
(`folded_directions = δ̂_L · share_L`). llama.cpp's uniform scale reproduces
that share profile, which approximates but does not equal drowse's live affine
injection. Multi-rank and curved fits are rejected.

## 13. Concurrency and consistency

The synchronization boundaries are deliberately separate:

- **The exclusive-GPU section.** One reentrant `_gen_lock` serializes all
  model work: MPS's single global command buffer is not thread-safe, and two
  threads committing it abort the process. Collaborators receive
  `base/exclusive.py::ExclusiveGuard`; a busy section fails fast (409).
  Steering push and pop wait out an in-flight generation instead. A steering
  scope opens outside the exclusive section, never inside it, or the push
  would deadlock.
- **Unsteered artifact work.** Extraction, fits, corpus generation, lens fits,
  SAE training and neutral compute refuse inside a steering scope
  (`SteeringComposer.assert_unsteered`); the hooks would bake into the
  artifact.
- **The server's request lock** serializes protocol callers before they enter
  the engine.
- **Frozen instrument bindings.** Registry and source changes become visible
  at the next bind.
- **Leaf state locks** (the lens `state_lock`, the SAE runtime's) cover source
  adoption, probe specs and live configuration. Take the exclusive guard first,
  never while holding one.
- **The loom reservation** (`LoomTree.reserve`) refuses conflicting edits to an
  in-flight generation's branch (409).
- **Artifact locks** are keyed to stable logical identities outside removable
  folders; global order is manifest lock, then template lock, then fitted-pair
  locks. Publication is compare-and-swap.

Collaborators read the tree through a getter (`tree=lambda: self.tree`),
because `restore_tree` replaces it. Do not collapse these boundaries into one
global lock: the separation lets unrelated readers and targets proceed while
each owner keeps exactly the state it claims.

## 14. Performance

The rules are architectural, not micro-optimizations; the root `AGENTS.md`
lists them and the throughput gates in `tests/gpu/test_smoke.py` protect the
ordinary steered path (≥ 85% of vanilla tok/s, ablation ≥ 80%) and the
served dashboard's instrumented decode (≥ 50%).

Beyond those rules:

- Capture is layer- and retention-planned; ordinary generation never retains a
  `[tokens, layers, hidden]` history.
- Curved foot following is paid only by genuinely curved terms: per token a
  Gauss-Newton step and frame transport, and an `n ≥ 2` SVD that hops to CPU
  on MPS.
- The whole flat geometry roster shares one Woodbury `Σ⁻¹h` apply and batched
  reduced-space products per layer. A curved probe reads all its layers in
  one stacked pass (one batched foot solve, per-layer reductions, means folded
  in layer order), bit-identical to reading them one at a time.
- The R family's live lens readout and SAE reads reuse the generation's capture and stashes; no
  second forward.
- The alternative-token width is resolved once per generation and shared by
  logits, lens, SAE, hover and replay.

Changes to hooks, capture, monitoring or generation must be checked on a real
model as well as by unit tests.

## 15. Change map

Change the narrow owner first:

- **Steering syntax**: parser, formatter and round-trip tests in
  `steering/grammar.py`, then composer lowering, then the grammar golden (a
  drowse.ai interface change), then docs.
- **Manifold math**: pure implementation and tests in `geometry/`, the codec
  and schema in `store/`, orchestration in `fit/`. A change that alters fitted
  results bumps `MANIFOLD_FIT_POLICY_VERSION` and may regenerate goldens.
- **A read channel or family**: `readout/types.py` and the protocol, the
  family, the planner union, the envelope (and its browser golden), the server
  schema and the webui types. A new family also needs a
  `session.instruments` entry, `generation/token_payloads.py::_FAMILY_SLOTS`
  and `server/instrument_routes.py::CAPABILITIES` (both pinned to the registry
  by test), and edits where `ProbeRoster`, `ReadPlane`, `FinalReads` and the
  engine's `TokenTap` name the families (the units family is the worked
  example: `drowse/units/`).
- **A model architecture**: `model/arch.py` discovers blocks structurally, so
  a new block location is one `_LAYER_PATHS` entry; "tested" means a
  `TESTED_ARCHS` entry plus dispatch and numerics cases; role and scene
  support need a `SEAT_HEADERS` entry and a drift-test tokenizer; logits quirks
  go in `model/readout.py::LogitReadout`. Keep `model_type` branches out of
  capture and hooks (`drowse/model/AGENTS.md`).
- **A native endpoint**: reuse a session or store owner, annotate the return
  type with a `server/response_models.py` TypedDict, regenerate
  `types.gen.ts` (CI checks it), use kebab-case paths. Request bodies and
  WS/SSE frames are hand-written in `webui/src/lib/types.ts`.
- **A `SamplingConfig` field**: it is persisted in every recipe and required on
  read, so it breaks saved trees and transcripts and changes the browser seam.
- **The dashboard**: edit `webui/src`, run `npm run check`, rebuild and commit
  `drowse/web/dist` in the same change.

The test suite is part of the architecture. Exact-schema, round-trip,
concurrency, failure-injection and cache-identity tests are often the only
executable statement of why a boundary is strict.

## 16. Known boundaries

- Lens atoms and probes (`jlens/`, `slens/`) name one tokenizer token;
  multi-token words raise `MultiTokenWordError`. An `slens/<word>` atom steers along the word's S-lens
  footprint at the S lens's `out` layers; a token the S read's mask drops is
  refused.
- Unit ablation rewrites a projection's input through a module pre-hook, so a
  steered generation with `!neuron/…`/`!head/…` (or a unit probe's tap) runs
  the eager module on a compiled session. Unit statistics measure the neutral
  corpus only (96 documents, so `act_top` is the median of the top quarter of
  document maxima, a median 12x below neuron-lens's on Qwen3-8B), taken from
  position 1 on, and a mean-ablation replaces every position but the
  attention sink, which keeps its own activation. A catalogued model (Qwen3-8B,
  Gemma 4 12B) with its catalog bound reads units over the catalog's
  `act_top` instead; any other reads relative to the neutral text, and says so.
- `BlockSpec` (`model/blocks.py`) describes dense decoders only: llama,
  mistral, qwen2/3 and gemma2/3/4 without per-layer inputs, KV sharing or MoE.
  MoE, fused gate-up or qkv projections, hybrid linear-attention stacks,
  attention sinks and unknown norms raise `UnsupportedBlockError` (422), so
  transport fits and unit reads refuse those models. Discovery is structural;
  `verify_model_spec` is what catches a forward term the structure hides.
- Lens format v7 records no embedding width, so stored K/S lenses have
  `d_emb = d_model`. drowse.ai's browser reader accepts only J/R packs with
  one `out` site and exact site keys; packs with a `mid` site, an input kind
  or a `split_half` record need its loader.
- Provider lenses and SAEs exist only for covered models and revisions; local
  fits are explicit, expensive workflows.
- Curved manifolds are slower than flat steering, and overlapping curved spans
  are rejected.
- Replay reproduces always-active affine steering exactly, a probe-gated
  history one forward a position and a prompt token's capture under any
  steering, but cannot recreate a thinking-phase history. A generated token's
  curved-probe reading came from a foot solve warm-started on the token
  before; its replay solves cold, so the foot-dependent parts of the token
  view (per-layer coordinates, depth, residual) can sit in another local
  minimum than the row's own coordinates.
- Discover coordinates are model-specific, and `fit_mode=auto` can resolve flat
  on one model and curved on another, so a negative `%` coefficient pushes away
  on one and raises on the other.
- The fit identity omits `dls` and the neutral cache, so `--no-dls` and a
  regenerated neutral corpus leave existing fits in place unless forced
  (`drowse/fit/AGENTS.md`).
- Recipe receipts stamp geometry probes only; transcript headers hash all
  four families.
- CPU works but is not the intended interactive path for large models.
- `drowse serve` is a trusted-local or lab server, not a hardened multi-tenant
  inference platform.

## 17. Open frontiers

- **Persona access.** The fitted `personas` subspace is a near-1-D
  "persona-ness" fan, so distinct personas can express the same generic intense
  register. That is a steering-access problem, not the rogue-channel problem:
  whitening verifiably fixed the latter.
- **Per-target coherence.** Whitened-unit targets unify push *strength* across
  targets, but not *coherence*: a hard persona such as `hacker` shatters at
  roughly half the effective gain a robust concept tolerates (about 2×).
  `_SUBSPACE_GAIN = 16.0` is the coherence-first compromise and is tagged a
  prototype; tune α down per target.
- **Low-α drift.** A steering trajectory can pass through neighboring attractor
  basins at low displacement (`personas%hacker` surfacing a cyber-security
  cluster before locking into the persona). That is signal about the model's
  internal structure, not a drowse bug, and it is unexplored.
- **Blending.** Off-node midpoints between distinct persona nodes produce
  coherent blended content at the sweet-spot α. Whether interpolation between
  basins holds generally, and on curved fits, is open.
- **Lenses beyond dense blocks.** Transport is written per block from
  `BlockSpec`; mixture-of-experts and hybrid linear-attention stacks are
  refused, and extending it to them is open.
- **The depth band.** The S/R crossover band is a prior (15–35% of depth,
  from neuron-lens's Qwen3-8B measurements); measuring it per model is open.
- **Axes whose neutral sits at a pole.** The neutral corpus is the model's
  default answers, which is one pole of some axes: DLS keeps few or no layers
  there, and a `%pole` push can point the wrong way (`drowse/fit/AGENTS.md`).
  6.0 recast the three where it was so (honest.deceptive,
  refusing.compliant, sincere.manipulative) as the one-node `deceptive`,
  `refusing` and `manipulative`. `religious.secular` sits at its secular pole
  on most layers of SmolLM2-360M and Qwen3-8B; what to do with it is open.
