# Drowse architecture

How artifacts are fitted, how a steering expression reaches a forward, how
hidden states become measurements, and the math and typing that keep those
paths consistent. Repo-wide rules are in the root `AGENTS.md`; contracts,
constants and file formats in each package's `AGENTS.md`.

## 1. System in one page

One `DrowseSession` owns one loaded model. The Python API, `drowse serve`
(OpenAI, Ollama and native protocols) and the dashboard all reach
`GenerationEngine`, which drives two planes over one forward:

- the **steering plane** (`SteeringComposer` → `SteeringManager` hooks)
  composes every compatible term into one update per layer and injects it;
- the **read plane** (`ReadPlane` → geometry, lens, SAE and units runs) plans
  hidden-state retention, binds frozen per-generation instrument state, and
  produces one measurement envelope without touching the steering
  representation.

They meet at **gates**: a trigger may depend on a read-side scalar, so the
capture plan must score exactly the channels the next steered forward needs.

## 2. Layering

Dependencies point downward through the layers in the root `AGENTS.md`. A
lower layer that needs a higher object types it as a narrow `Protocol` (the
composer's `ArtifactSource`, the read contract's `TokenReplay`) and takes it
through its constructor; nothing outside `drowse/session/` reads a session
private. That lets each collaborator be tested without a session and lets
drowse.ai replace whole subsystems behind the same contracts.

## 3. Artifacts

### 3.1 The manifold is the one steering artifact

A manifold is labeled nodes, a domain or discovered layout, and a per-model
fit at selected residual layers. Each fitted layer is a `LayerSubspace`:
**flat** (a neutral-anchored affine frame with per-layer node coordinates) or
**curved** (the same frame plus an `r³` polyharmonic RBF surface through the
node centroids and, for raw fits, a tube thickness σ(z)). A 2-node flat fit is
the bipolar steering vector and a 1-node flat fit a monopolar ray from the
neutral mean. `fit_mode` is `authored` (curved, exact at nodes), a discover
mode (`pca` flat, `spectral` curved, `auto` choosing flat, curved or periodic
per model), or `baked` (a precomputed direction). `Profile`
(`{layer: direction}`) is an in-memory view folded from an R = 1 fit, and
`SynthesizedSubspace` an expression's per-dispatch composition; neither is
stored.

### 3.2 Templates

Some categories are referenced rather than embodied (weekdays, months), so
persona elicitation does not apply. A template is a slot token, candidate
values and multi-turn contexts whose final assistant turn carries the slot.
`ChoiceScorer` (`session.score_choices` / `score_template`) reads the
restricted-choice distribution over the values from the raw model
distribution, optionally under steering: did steering shift the distribution,
not just the argmax? `manifold from-template` materializes one node corpus per
value, and the template's content folds into the fit identity.

### 3.3 Per-model read sources

Lenses and SAEs are model-scoped `SourceFamily`s (`base/source_registry.py`):
an active selection, Drowse-owned `local/` artifacts, and commit-pinned
provider bindings. The lens has two families. The R family (J standard, R
RelP) holds output Jacobians `J_l = E[∂h_final/∂h_l]`, read through the
unembedding: what a vector will make the model say. The S family (K, S) holds
input Jacobians `K_l = E[∂h_l/∂e]`, read against the input embedding: what the
stream has read. Each has an `out` and a `mid` (post-attention) site; one
source per family is resident, and `lens/depth.py` picks the family for a
place (S early, R late, both weighted in a crossover band). The units catalog
binds neuron-lens labels and firing tables, display data that never enters a
computation. A source switch becomes visible at the next bind.

### 3.4 Loom and results

`LoomTree` is the conversation graph: per node the recipe, raw token ids,
token rows with their measurements, the reply's settled aggregate and the
annotations. **Provenance is recipe presence**: a node with a `Recipe` was
generated (in either seat, since the model can speak the user seat) and
`recipe=None` means authored; rendering, transcripts and prompt capture key on
that, not on `role`. `stateless=True` calls and every OpenAI/Ollama request
return results without touching the tree.

## 4. Identity and persistence

Model-bound artifacts key on `model_source_fingerprint` (metadata only, so
weight-free preflights prove a repeat is a no-op without loading weights) or
`loaded_model_fingerprint` (the live weights, which fitted tensors bind).
Hashes decide **staleness and cache hits**, never integrity. A fitted manifold
is reused only when its `nodes_sha256` (everything authored: labels, corpora,
roles, kinds, coordinates, hyperparameters, the resolved template),
`capture_sha256` (the rendered token ids and pool positions), model
fingerprint and `MANIFOLD_FIT_POLICY_VERSION` all match; the registry
recomputes `capture_sha256` on every load, so rendering must stay
deterministic. Lenses key on corpus digests and `lens_identity`; provider
bindings pin commits. Readers require the fields they use and ignore the rest,
so dropping a field never strands a file, and derived caches recompute rather
than migrate.

Writes are atomic (same-directory staging, fsync, replace) under stable locks
outside removable folders, in one order: manifest, template, fitted pairs. A
fitted pair writes its tensor before its sidecar, so a sidecar always names a
complete tensor. A fit computes for minutes against an authoring snapshot and
publishes by compare-and-swap (`fit/publish.py`), raising
`ManifoldAuthoringChangedError` (409) if the folder moved. Lens fits write
immutable shard generations behind one atomic pointer, so the previous
complete generation stays authoritative until the new pointer is durable.

## 5. Extraction and fitting

One pipeline, `fit/pipeline.py::ManifoldExtractionPipeline`, serves
`session.extract` and `manifold fit`: inputs (snapshot, identity, cache
check), capture, geometry (pure tensors), publish (`drowse/fit/AGENTS.md`).

### 5.1 Conversational corpora

`fit/corpus.py::CorpusGenerator` has the model answer 48 shared baseline
prompts in character, the concept riding a system prompt and a swapped
assistant-role label. Capture pools the swapped-back pairs in
standard-assistant space, so the persona exists only at generation and its
signal rides the text (an explicit `:role-<name>` is substituted at capture
and again at steering time, so the two baselines match). The neutral corpus
answers the same prompts with no persona, and one length directive leads the
system prompt at all four sites: as common mode, prompt topic and directive
both cancel in `node − neutral`.

### 5.2 Capture and the baseline-match contract

Rows pool at `model/tokens.py::last_content_index`, the last non-special token
before the chat-template trailer, where generation's aggregate reads also
pool. Fit capture renders through the chat template and generation through
the scene stitcher (`model/scene.py`), whose round-trip validation against the
template keeps the two renders identical; a family that fails it falls back to
the template. Capture stops after the last requested block and goes through a
content-addressed cache, so geometry-only refits skip the forwards.

### 5.3 Typed frames in the Mahalanobis metric

**The metric.** On real LMs a few massive-activation channels dominate the
Euclidean norm, so every activation-space quantity uses the neutral covariance
Σ. Neutral activations are cached fp32 (late layers overflow fp16), with each
block's `mid` mean and the unit statistics from the same forward.
`geometry/whitener.py::LayerWhitener` keeps the centered neutral rows
`Z (N, D)` and a ridge λ, and applies `Σ⁻¹v = (v − Zᵀ(NλI + ZZᵀ)⁻¹Zv)/λ` in
O(ND), never forming a D×D matrix.

**Vectors and covectors.** A write is a vector `w` added to the residual; a
read is a covector `u` that scores a residual by `u·(h − mean)`. In the
Mahalanobis metric they differ: the unit-cost write (`wᵀΣ⁻¹w = 1`) that most
raises a reading `u` is `w ∝ Σu`. For a two-class split with mean displacement
δ the best linear read is the LDA covector `Σ⁻¹δ`, and its raised write is
`Σ·Σ⁻¹δ = δ`, the difference of means.

**The fit.** Let `X` be the `(K, D)` node centroids centered on their own mean
(centering on neutral would add the neutral offset as a spurious axis and
break PCA@2 ≡ DiM). Eigenvectors `a_r` of the whitened Gram `G = XΣ⁻¹Xᵀ`
(eigenvalues `μ_r`) rank axes by whitened between-node variance, the
Fisher/LDA objective, so rogue channels do not choose them. Then

- `basis_r = Xᵀa_r/√μ_r` are the write rows: combinations of node
  displacements, Mahalanobis-orthonormal (`basis Σ⁻¹ basisᵀ = I`);
- `read_r = Σ⁻¹basis_r` are their dual covectors (`read·basisᵀ = I`);
- `mean = (ν·readᵀ)·basis`, the neutral mean ν projected into the span, so
  coordinate 0 is neutral.

`P = basisᵀ·read` is the Mahalanobis-orthogonal projector onto the span, and:

- **a 2-node push is DiM**: the frame writes along δ;
- **a full `!` ablation is LEACE**: `h − P(h − mean)`, at rank 1
  `h − ((h − ν)·Σ⁻¹d / dᵀΣ⁻¹d)·d` (`mean − ν` lies in the reads' kernel);
- **the monitor reads the LDA score**: a coordinate is `(h − mean)·readᵀ`;
- reduced coordinates are whitened, so every reduced norm (the foot solve,
  the σ tube, the share) is a Mahalanobis norm.

Every frame obeys this contract: fitted, folded (`basis = d/‖d‖_M`,
`read = Σ⁻¹d/‖d‖_M`), synthesized (§6.2) and transferred (§12, where writes
map as directions and reads are recomputed in the target metric). Objects that
arrive as reads stay reads. A lens token atom `u` (`W_U[v]·J_l`, or an S
footprint) steers along its raised vector `Σu`, whose fold reads with
`Σ⁻¹Σu = u`, so `!jlens/x` zeroes exactly that token's reading. The lens
decompositions whiten each atom by its type (`Σ^{-1/2}` for vectors such as
neuron writes, `Σ^{1/2}` for covectors such as token atoms), so every score is
a plain pairing. Hence the rule: never write along a read row or read with a
write row. `MANIFOLD_FIT_POLICY_VERSION` (2) stamps fits made under it.

### 5.4 Discovering and fitting geometry

Discover modes average every fit layer's whitened, node-centered `(K, K)` Gram
into a consensus Gram. Whitening puts layers in common units, so the average
is signal-weighted with no reference layer, and a layer where the nodes do not
separate drops out on its own. PCA coordinates are its leading eigenvectors,
spectral ones Laplacian eigenmaps; `auto` decides flat versus curved by GCV
and proposes (never certifies) periodic axes from H1 persistence
(`geometry/topology.py`). Discovered curved surfaces are GCV-smoothed against
centroid noise; authored ones interpolate exactly, so a node is an exact
steering target.

Each fit bakes a per-layer `share`, the node spread in the Mahalanobis metric
(`‖δ_L‖_M/√2` at K = 2), which the hooks use as the cross-layer budget.
Discriminative layer selection (Dang & Ngo 2026) keeps a flat axis at a layer
only when the nodes' readings relative to neutral straddle zero: same-side
layers encode intensity, not polarity. Curved fits skip it. A monopolar
concept (`extract(concept)` with no baseline) is a 1-node fold of
`concept − ν`, neutral being the implicit negative pole.

## 6. Steering language and lowering

### 6.1 One grammar

Python, YAML `vectors:`, HTTP, CLI and recipes parse through
`steering/grammar.py::parse_expr`, and `format_expr` round-trips it, so every
stamped `str(steering)` replays. Terms are pushes, Mahalanobis projections
(`a~b`, `a|b`), LEACE ablations (`!a`), manifold positions (`M%label`),
instrument atoms (`jlens/`, `slens/`, `sae/`) and units (`neuron/`, `!head/`),
each with an optional layer clause and trigger. The authoritative grammar is
in `drowse/steering/AGENTS.md`.

### 6.2 Resolution and composition

`SteeringComposer.open_scope` resolves and loads every artifact before the
scope is entered, so an unknown name fails before any state changes. A bare
slug resolves through one ladder (`store/selectors.py::resolve_bare_atom`:
node label, then 2-node name, then plain key); a collision raises
`AmbiguousSelectorError` instead of choosing. A layer clause is part of a
term's identity and restricts its fragments before grouping.

Composition groups terms by trigger. Within a group, vectors and `~`/`|`
results fold to rank-1 frames, ablations contribute collapse axes and affine
`%` terms their basis and target, and all of them become one
`SynthesizedSubspace` per layer (`geometry/inject.py::synthesize_subspace`),
so terms sharing a layer superpose rather than collide. Orthonormalization
runs on (write, read) pairs in the Mahalanobis metric, push first, and the
leftover ablation operator is diagonalized, so partial, repeated, negative and
non-orthogonal ablations get exact per-axis coefficients. Each curved `%` term
stays its own term; two curved spans may share a layer only when
Mahalanobis-orthogonal (`OverlappingManifoldError` otherwise), and the affine
span is orthogonalized against them. The one term that is not a residual
injection is unit ablation: a `UnitOp` pre-hook blends a unit's activation
toward its neutral mean.

### 6.3 The kernel

`geometry/inject.py::subspace_inject` defines every injection. It splits
`h = mean + H_m + H_n + H_o` (surface foot, off-surface in-subspace residual,
off-subspace residual), reading with `read` and writing with `basis`:

- **along** translates the foot by a fixed `along·(target − origin)`. A fixed
  offset keeps each token's in-subspace spread; lerping each foot onto the
  target would erase it and loop-collapse at strong push. On a curved surface
  `H_n` is carried to the new tangent frame by a norm-preserving rotation that
  is the identity when the foot does not move.
- **onto** shrinks `H_n` toward the surface: by `(1 − onto)`, or toward the
  local tube σ(z), so `onto = 1` lands on the typical set rather than the
  zero-thickness wire. It is vacuous on flat terms.
- **`H_o = (I − P)(h − mean)` is kept verbatim.** It holds every
  Mahalanobis-orthogonal neighbor's span, which is why a vector and several
  orthogonal manifolds compose with zero cross-talk.

On a flat frame the kernel is an analytic slide (no foot solve, RBF or norm
cap); affine terms run `steering/hooks.py::AffineOp`, a compose-time split of
that branch tested equal to the kernel. Only curved terms call the kernel per
token, with a warm-started foot and a `3·‖h‖` cap against off-domain RBF
extrapolation.

### 6.4 Gains and calibration

The kernel applies no gain. Two decisions make one coefficient scale mean the
same across very different fits:

- **Whitened-unit targets.** `synthesize_subspace` makes each push target a
  whitened unit direction and sets the share to the whitened displacement. Raw
  node distance cannot be the knob: a tight pole and a far persona centroid
  sit orders of magnitude apart in distance from neutral.
- **Mean-1 shares.** Per-layer shares are normalized to sum to the number of
  layers the group acts on, so a 1-layer or 30-layer fit, 2-node or 107-node,
  lands in one α band. Affine `along` is unclamped: a high-share layer
  overshoots by design, bounded by the whitened-unit target.

The constants live in `steering/hooks.py` with their calibration notes: affine
`eff_along_L = share_L · _SUBSPACE_GAIN` (8), with α ≈ 0.5 expressive and 1.0
strong; curved `eff_along_L = along · share_L · _MANIFOLD_ALONG_GAIN` (2),
where `eff_along = 1` lands on the node (periodic fits clamp it there, so no
layer wraps to another node). An ablation's κ is divided by `eff_along`, so
its strength is exactly the user's coefficient.

### 6.5 Triggers and gates

`TriggerContext` carries the phase, thinking state, generation step and latest
gate scalars; only dynamic terms read it. A gate key names a probe plus a
channel: geometry produces every channel (axis, `:fraction`, `:membership`,
`@label`, `~label`), lens, SAE and unit probes one strength axis. A unit gate
needs no attached probe; any other gate naming no attached probe is inactive.
Gates lag one forward: scalars scored after forward `t` steer forward `t + 1`,
and the prefill sees none. `@response` includes thinking tokens; only bare
`@after` is response-only.

## 7. The read plane

### 7.1 Four instrument families

`session.instruments` registers four families behind one contract
(`readout/protocol.py::Instrument`). **Geometry** (`Monitor`) reads every
probe, a fitted `Manifold`, in the whitened metric and reports a
`ProbeReading` in its domain frame; **lens**, **SAE** and **units** return
`ScalarReading`s with an explicit unit, the units family reading the model's
parts through hooks present only while something consumes them. Live reads,
aggregates and replays come from one scorer, so they compare directly.

The R readout is `softmax(W_U · norm(J_l h))` per layer; the S read pulls a
residual back, `K_lᵀ(h − μ_l)`, and scores it against the input embedding. A
read always says what it reads (`lens/read.py`): an S read of a **residual**
centers on the neutral mean at its site, since the stream's common component
would swamp the position's own tokens, while a **direction** (a steering
vector, a unit write, a write delta) is read as given. Units are also read
statically, each weight vector through the lens at the place and site the
depth rule assigns (`units/reads.py`).

### 7.2 The per-generation transaction

`ReadPlane` runs each family's `close_run → prepare → plan → bind` per
generation and closes in a `finally`: `prepare` snapshots probe specs and
source identity, `plan` declares demand (layers, tail depth, gate keys), and
`bind` opens a frozen `InstrumentRun`, which is why source switches and
detaches are safe mid-generation. One `step_id` per forward reaches the
capture sink, gate callback and token tap, so no forward is scored twice.

### 7.3 Capture planning

`readout/planner.py` is pure: it folds a generation's consumers into one
demand and picks one capture mode (table in `drowse/readout/AGENTS.md`),
upgrading retention when two families need different things from one forward.
Scoring is conditional on need: with probes attached and no per-token
consumer, capture keeps a bounded tail ring and scores once at finalize. Hooks
only copy tensors and scoring runs after the forward, so a deep hook never
drains the device queue mid-forward.

### 7.4 One measurement envelope

`readout/measurements.py` builds one versioned record for token frames, final
aggregates, persisted rows and replays: per-family `instruments` (native
`readings`, a discovery `readout`, a source-and-steering `binding` that keeps
a historical row interpretable after the active source changes) and a derived
axis-0 `scores` join for cross-family consumers. Persistence keeps
projections: a loom token row keeps `scores` and each geometry reading's
`fraction` and `coords` (what tint, trace and the tree filter need without a
model), a node its settled aggregate; the rest is replayed on demand.

### 7.5 Replay and drilldown

A token's full view is recomputed, not stored. `loom/replay.py::TokenReplayer`
rebuilds the node's render and decode prefix and runs the forward as
generation ran it (`replay_segments`), under the recorded steering or the
unsteered counterfactual, so one token reads the same in every family. A
probe-gated term replays one position a forward, scoring gates as decode did;
a thinking-phase window, which the node does not record, refuses (422).
Authored turns are captured by the first prefill that reads them, and the node
keeps a `PromptCapture` anchor (steering, render parameters, a digest of the
render head) so their replay never depends on the capturing reply.
`loom/node_replay.py::NodeReplayer` adds taps where the model's parts meet the
residual, for **token anatomy** (one token at every layer, through both lens
families, with the units that contributed most to its logit) and **unit
activity** (every neuron and head across a turn).

## 8. The generation transaction

Every generation surface reaches `GenerationEngine.generate_one` inside
`transaction()`, the single-flight guard. It resolves the call into a frozen
`CallSpec` and the steering before any loom node exists; creates the nodes,
enters the scope, renders, plans the reads and decodes; then pops the scope,
so finalize reads run unsteered. Teardown is unconditional, and removes a
reply node a failure left empty.

Per-token logprobs describe the sampler's distribution; completion scoring
reads the raw one. Routing follows what the hooks can promise. The unsteered
**prefix cache** is reused only when no active term steers the prefill and
the call needs neither hidden states nor thinking. **StaticCache** runs when
every steered layer is the static single-affine fast path. The **compiled
module** runs only when the live hook topology matches the traced one
(steering absent, or a static pure push written into the persistent offset
buffers); anything else runs eager rather than retracing.

## 9. Server

`server/app.py::create_app` serves one session; OpenAI and Ollama generations
are stateless, and only the native protocol grows the loom. A FIFO asyncio
lock serializes protocol callers (503 after 300 s); lens fits and SAE training
hold the exclusive-GPU section outside it, so a generation arriving mid-fit
gets an immediate 409 instead of queueing behind a long job. Generation never
waits on a WebSocket reader: a lagging connection misses frames, then gets one
`resync` and rebuilds from `GET /tree`. One classifier
(`server/errors.py::describe_error`) serves every surface (details:
`drowse/server/AGENTS.md`).

## 10. Dashboard and the browser edition

The Svelte 5 dashboard (`webui/`, drawn in Ledger) builds into the committed
`drowse/web/dist/`, exactly what wheel users run. It holds steering only as
chips over the structured terms the server returns (`parts` from
`.../steering/validate`) and never parses expression text; agents, HTTP, the
CLI and Python keep the grammar. REST response types are generated from the
server's TypedDicts.

drowse.ai builds the same `webui/src` with a WebGPU runtime, reached only
through the `RuntimeClient` contract and the `@runtime-*` aliases, and
reimplements the grammar, sampler, lowering, kernel and fit. Python is the
reference and there are no goldens here: drowse.ai pins this repo as a
submodule and computes parity from the pinned Python, so a semantic change
here is an interface change it catches up to when it moves the pin.

## 11. Lens, SAE and catalog lifecycle

`lens/fit.py::fit_lens` is the one lens estimator, a transport over the
`BlockSpec` description (`model/blocks.py`): each document is linearized once,
then basis seeds sweep hand-written per-block maps (cotangents down from the
last block for J/R, tangents up from the embedding for K/S), so one sweep
covers every layer and both sites with no autograd. A fit records its
split-half difference (whether more documents would help) and checkpoints,
and `lens/lifecycle.py` chooses noop, top-up, resume or fresh from metadata
alone. SAEs trained here and SAELens releases adapt to one resident
`SaeRuntime`. A bound units catalog sets the unit reading scale and the S
token mask, and every surface names the source it used.

## 12. Distribution and transfer

A Hub pack is a `drowse-manifold`-tagged model repo holding one root
`.drowse` (`store/archive.py`): a narrow ZIP transport for one manifold
closure and its template, never weights, lenses, SAEs or caches; drowse.ai
shares the convention.

Cross-model transfer (`geometry/procrustes.py`, `geometry/transfer.py`) fits a
per-layer affine alignment from shared neutral activations, maps means as
points and write rows as directions, re-orthonormalizes in the target's
Mahalanobis metric (recomputing reads there), re-bakes the share, and writes a
`from-<source>` variant. A curved tube becomes the volume-preserving isotropic
approximation when the map is not an isometry between the metrics. Discover
layouts are model-specific, so transfer maps fitted geometry, not a universal
coordinate system. A 2-node flat fit exports as a llama.cpp control-vector
GGUF, each layer's write row scaled to Mahalanobis norm `share_L`, which
approximates drowse's injection.

## 13. Concurrency and consistency

The boundaries are deliberately separate, so unrelated readers and targets
never serialize behind one global lock. The exclusive-GPU section fails fast
(409) while steering push and pop wait out a generation; artifact work refuses
inside a steering scope; instrument bindings are frozen per run, and leaf
`state_lock`s are taken after the exclusive section; the loom reservation
refuses edits to an in-flight branch (409); artifact locks follow §4.

## 14. Performance

The rules and throughput gates are in the root `AGENTS.md`. Capture is layer-
and retention-planned, curved foot following is paid only by curved terms,
the flat geometry roster shares one Woodbury `Σ⁻¹h` apply per layer, and live
lens and SAE reads reuse the generation's capture.

## 15. Change map

Change the narrowest owner first:

- **Steering syntax**: parser, formatter and round-trip tests in
  `steering/grammar.py`, then composer lowering, then docs; a semantic change
  is a drowse.ai interface change.
- **Manifold math**: pure code and tests in `geometry/`, the codec in `store/`,
  orchestration in `fit/`; a change to fitted results bumps
  `MANIFOLD_FIT_POLICY_VERSION`.
- **A read channel or family**: `readout/types.py` and the protocol, the
  family, the planner, the envelope, the server schema and the webui types. A
  new family also needs a `session.instruments` entry,
  `generation/token_payloads.py::_FAMILY_SLOTS` and
  `server/instrument_routes.py::CAPABILITIES` (pinned to the registry by test),
  and the places `ProbeRoster`, `ReadPlane`, `FinalReads` and `TokenTap` name
  the families; `drowse/units/` is the worked example.
- **A model architecture**: structural discovery in `model/arch.py`; keep
  `model_type` branches out of capture and hooks.
- **A `SamplingConfig` field**: recipes persist and require every field, so a
  new one changes saved trees, transcripts and the browser seam.

Tests are CPU-first, against a real session on a tiny generated checkpoint;
the GPU suite is one smoke test plus the throughput gates (`tests/AGENTS.md`).

## 16. Known boundaries

- `BlockSpec` describes dense decoders only (no MoE, fused projections,
  linear attention, attention sinks or KV sharing); lens transport and unit
  reads refuse other models (`UnsupportedBlockError`, 422).
- Unit ablations and taps are hooks the compiled graph was not traced with,
  so they run eager. Without a bound catalog, unit scales are relative to the
  neutral corpus.
- A generated token's curved reading came from a warm foot and its replay
  solves cold, so foot-dependent details can land in another local minimum.
- `fit_mode=auto` can resolve flat on one model and curved on another, where
  a negative `%` coefficient raises.
- The fit identity omits `dls` and the neutral cache, so `--no-dls` or a
  regenerated neutral corpus leaves existing fits in place unless forced.
- `drowse serve` is a trusted-local or lab server, not a hardened multi-tenant
  platform.

## 17. Open frontiers

- **Persona access.** `personas` fits as a near-1-D "persona-ness" fan, so
  distinct personas can express one generic intense register.
- **Per-target coherence.** Whitened-unit targets unify push strength, not
  coherence: `hacker` breaks at about half the gain a robust concept takes.
  `_SUBSPACE_GAIN` is the coherence-first compromise.
- **Drift and blending.** Low-α trajectories pass through neighboring basins
  (`personas%hacker` surfaces a cyber-security cluster first), and off-node
  midpoints blend coherently; how general either is, on curved fits too, is
  open.
- **Lenses beyond dense blocks** (MoE, hybrid linear attention), and a
  per-model S/R depth band in place of the 15–35% prior.
- **Axes whose neutral sits at a pole.** The neutral corpus is the model's
  default answers, one pole of some axes, where DLS keeps few layers and a
  `%pole` push can point the wrong way. Three such axes became one-node
  concepts (`deceptive`, `refusing`, `manipulative`); `religious.secular` is
  open (`drowse/fit/AGENTS.md`).
