# drowse/geometry

Activation-space geometry: the manifold/subspace representation behind every
steering term and geometry probe, the steering kernel, the RBF surface, coordinate
discovery and topology selection, the Mahalanobis whitener, the direction view
(fold, `Profile`, GGUF) and cross-model transfer. Pure tensor math; it imports
only `errors`/`naming`/`events` and `base`. The only file I/O is the `Profile`
safetensors/GGUF codec. The manifold tensor codec is
`store/manifold_tensors.py`, fit orchestration is `fit/`, and the gains that
scale the kernel are in `steering/hooks.py`.

## Module map

Internal order: `domains`, `rbf` → `subspace` → `invert`, `manifold`, `tube` →
`inject`, `topology`, `fold`, `transfer`; `packing` sits under `manifold`;
`whitener`, `procrustes`, `gguf`, `profile` stand alone.

| Module | Owns |
|---|---|
| `domains.py` | `ManifoldDomain` and its five domains; the `manifold.json` domain spec |
| `rbf.py` | `RbfFitPlan`, `RbfFit`, `eval_rbf`/`_jacobian`/`_jet` (one layer or an `(L, X, m)` stack), `rbf_cardinal_weights` |
| `subspace.py` | `LayerSubspace`, `StackedRbf`, `pca_basis`, `dual_read`, `fit_affine_subspace`, `fit_curved_subspace`, `subspace_share`, `DEFAULT_N_COMPONENTS = 64` |
| `manifold.py`, `packing.py` | `Manifold`, `MANIFOLD_FIT_POLICY_VERSION`; `pack_to_device` (one buffer per dtype) |
| `invert.py`, `tube.py` | foot solve (`_gn_step`, `invert_parameterization`, `invert_stacked`, `neutral_origin`); the σ-field |
| `inject.py` | `synthesize_subspace`, `_ortho_frame`, `subspace_inject` (the kernel) |
| `topology.py` | `derive_pca_coords`, `derive_spectral_coords`, `neutral_layout_coord`, `select_topology` |
| `whitener.py` | `LayerWhitener` |
| `fold.py` | `fold_directions_to_subspace`, `folded_directions`, `project_profile` |
| `profile.py`, `gguf.py` | `Profile` and its safetensors codec, `linear_sum`; the llama.cpp control-vector codec |
| `procrustes.py`, `transfer.py` | `fit_alignment`, `alignment_quality`; `transfer_manifold_subspaces` |

## Invariants

- **Mahalanobis metric.** Fit basis, share, synthesis, `~`/`|`,
  `Profile.cosine_similarity`, transfer and (in `readout/`) monitor reads all
  use the whitener's metric; consumers check `whitener.covers_all(layers)` and
  raise `WhitenerError` (reason: root `AGENTS.md`). The deliberate Euclidean
  code: `pca_basis` without a whitener (the behavior-space fit in
  `fit/naturalness.py`, and tests), `Profile.projected_away`, a layer whose
  neutrals are exactly zero (λ = 1, Σ_reg = I), and a head's write norm
  `‖W_O^h z_h‖` (`units/`), a scalar read against that head's own neutral
  `act_top` and never compared geometrically.
- **Typed frames.** `basis` rows are write vectors, `read` rows their dual
  covectors, `read @ basisᵀ = I`. Fitted, folded, synthesized and transferred
  frames are Mahalanobis-orthonormal with `read = basis Σ⁻¹`, so `basisᵀ read`
  is the Mahalanobis-orthogonal projector and reduced norms are Mahalanobis
  norms. Fits, the kernel, `AffineOp`, synthesis, fold and transfer read with
  `read` and write with `basis`. Without a whitener the frame is
  Euclidean-orthonormal and `read == basis`.
- **μ-centered scatter, neutral origin.** `pca_basis` centers centroids on
  their own mean (centering on neutral injects the neutral offset as an axis
  and breaks PCA@2 ≡ difference-of-means); the frame then anchors at
  `mean = P_basis(ν_L)` so reduced coordinate 0 is neutral.
- **Fit-time math is CPU fp32**; norms are fp32 everywhere (fp16 sum-of-squares
  overflows at hidden sizes ≥ 2048).
- **No host sync on the hot path.** `subspace_inject`, `eval_rbf*`, `_gn_step`
  and `LayerSubspace.sigma_at` run in forward hooks. Two known exceptions,
  curved n ≥ 3 only: `_gn_step` uses `torch.linalg.solve` and
  `_principal_pairs_svd` hops to CPU on MPS.
- **`MANIFOLD_FIT_POLICY_VERSION`** stamps every fitted sidecar. Bump it when
  PCA/Fisher selection, topology choice, RBF or σ fitting, DLS or share
  allocation changes incompatibly: fitted tensors then refuse to load and the
  fit caches miss.
- drowse.ai reimplements the kernel, `synthesize_subspace`, the Fisher fit,
  `subspace_share`, `RbfFitPlan` and the topology functions in its WebGPU
  runtime.

## Frames: `LayerSubspace` and `Manifold`

`LayerSubspace` is one layer's frame: `mean (D,)`, `basis (R, D)`, `read (R, D)`
(`validate_structure` checks `read @ basisᵀ ≈ I`), unit-box normalization
`coord_offset`/`coord_scale (m,)`, and one of two shapes:

- **Affine (flat)**, `is_affine ⇔ node_params is None`: `node_coords (K, R)` are
  the neutral-anchored per-layer node positions (the steering target);
  `affine_map (m, R)` is `None` (identity) except after a rectangular transfer.
  `select_axes(kept)` bakes a DLS prune into the basis (affine only).
- **Curved**: `node_params (K, m)`, `rbf_weights (K, R)`, `poly_coeffs (m+1, R)`;
  raw-space fits add the σ pair (`sigma_rbf_weights (K, 1)`,
  `sigma_poly_coeffs (m+1, 1)`), SAE-space curved fits have none.

`Manifold` adds the domain, `node_labels`, `node_coords (K, n)` (authoring and
display layout), `node_roles`/`node_kinds`, `feature_space` (`raw` or
`sae-<release>`, decoded to model space), `mahalanobis_share` (one finite
positive scalar per fitted layer) and `origin` (the neutral foot of each
curved layer). A fit is all-affine or all-curved. `nearest_node_index` uses
chordal distance between embedded coordinates, so periodic axes wrap.

## Domains

A domain embeds n authoring coordinates into R^m (`embed`, `embed_jacobian`,
`clamp_position`), supplies the foot-solve frame (`tangent_jacobian`,
`retract_position`) and implements `translate_foot`, the kernel's `along` step
(every token's foot moves by the same neutral→target offset, parallel-
transported to the foot). `BoxDomain` is intervals × circles (periodic axes
embed as `(cos, sin)` and translate along the short arc; open axes clamp to
`[lo, hi]`, default `[0, 1]`); `SphereDomain` uses hyperspherical angles with a
chordal metric (keeps r³ conditionally positive definite); `KleinBottleDomain`
(`(u + 2π, v) ~ (u, −v)` in R⁴) and `ProjectivePlaneDomain` (Veronese embedding
in R⁵) are quotient-aware; `CustomDomain` is the identity carrier for discover
coordinates and synthesized affine subspaces. Sphere, Klein and projective are
authored-only; `readout/monitor.py` uses their `project_mean` to average a
coordinate across layers.

## Fitting a frame

`pca_basis(X, whitener=, layer=)` is PCA in the whitened space mapped back with
the right types (the Fisher/LDA problem `(S_b, Σ)`). Eigenvectors `a` of
`G = X Σ⁻¹ Xᵀ` (eigenvalues `μ`) rank axes by whitened between-node variance,
so the rogue channels raw PCA chases do not choose them. Write rows are
`Xᵀa/√μ` (Mahalanobis-orthonormal since `aᵀGa = μ`); read rows are
`aᵀ(Σ⁻¹X)/√μ`, the LDA covectors, made exactly dual by `dual_read` (one
`(R, R)` solve against fp32 Woodbury drift). `R = min(n_components, K − 1,
rank)`. A flat fit's node coordinates are the discover layout's whitened PCA
coordinates.

- `fit_affine_subspace` (flat: `fit_mode=pca`, every steering vector as K = 2):
  `node_coords = (centroids − ν)·readᵀ`; `orient_to=0` makes node 0's
  coordinate non-negative, so a 2-node axis writes `+δ` and reads `Σ⁻¹δ`.
- `fit_curved_subspace` (authored/spectral, K ≥ 3): the same frame plus an RBF
  through the neutral-anchored reduced coordinates. `smoothing=None`
  interpolates exactly (a node is an exact steering target); `"auto"`/float
  fits the penalized surface and records λ/edf/GCV on `LayerFit.rbf`.
- `subspace_share = sqrt(Σ_k c_kᵀ (B Σ⁻¹ Bᵀ) c_k)` over μ-centered
  coordinates: the per-layer budget is signal spread, independent of where
  neutral sits. At K = 2 it is `‖δ_L‖_M / √2`, so the mean-1 profile equals
  DiM's. The monitor weights its cross-layer read by the same share.

## RBF surface and σ-field

One `r³` polyharmonic RBF plus an affine polynomial per layer: conditionally
positive definite of order 2 in every dimension, C², the natural cubic spline at
n = 1. `RbfFitPlan.build(node_params, smoothing=)` depends only on the layout,
so a multi-layer fit factorizes once and each layer (and the σ-field) is
right-hand-side work. It normalizes to the unit box (the kernel amplifies
scale), raises `PoisednessError` unless K ≥ m + 1 and the nodes affinely span
R^m, and caches the saddle LU (symmetric indefinite).

The penalized fit is the thin-plate/Duchon spline `[[E + λI, Q], [Qᵀ, 0]]`, λ
scaled by the mean off-diagonal kernel magnitude. `"auto"` minimizes the
multi-output GCV `K·‖(I − S_λ)Y‖² / tr(I − S_λ)²` over 40 log-spaced points in
`[1e-6, 1e3]` through the Demmler-Reinsch eigensystem (λ = 0 excluded, ties to
the smallest). Smoothing changes coefficients only, never the evaluators.
`rbf_cardinal_weights` gives exact interpolation weights (partition of unity);
the composer uses them for coordinate `%` on a flat manifold.

`fit_sigma_field` turns each node's reduced within-node covariance
(`fit/node_stats.py`) into one off-surface σ: `off_surface_vars` projects onto
the normal complement of the actual local tangent range (one economy SVD, so
folds with rank < n stay honest) and divides by the normal degrees of freedom;
σ² is floored at `1e-3 × median`, and `log σ` gets its own GCV-smoothed RBF
(a second moment from about 48 samples is noisy). The field is isotropic.
`onto` steers into it; the monitor reads it as `:membership`.

## Foot solve

`_gn_step` is one Levenberg-Marquardt step of `argmin_p ‖s(p) − q‖` with the
analytic RBF Jacobian chained through `tangent_jacobian` and a diagonal floor
for folds (n ≤ 2 closed form). `invert_parameterization` runs cold with 12
iterations from the 3 nearest nodes (best residual wins, which guards periodic
and fold minima) or warm with 4 from the carried foot plus the nearest node;
`invert_stacked` is the same solve over a `StackedRbf`, one query per layer.
`neutral_origin` projects ν onto a curved layer's surface. Origins are per
layer, and the steering follower takes one step with no restarts, so its seed
must already be in the right basin.

## Composition and the kernel

`synthesize_subspace(push, ablate, neutral_means, *, whitener)` composes every
affine term of one trigger group (vectors, poles, `~`/`|` results, `!`, affine
`%`) into one `SynthesizedSubspace` per layer, on CPU fp32:

- Push rows `(basis_rows, coord_target, coeff)` and ablation rows `(dirs,
  coeff)` are write vectors; their reads come from one batched Woodbury `Σ⁻¹`,
  and they are orthonormalized push-first in the Mahalanobis metric (push wins
  a shared direction). An ablation axis reads with `Σ⁻¹d` and writes along `d`:
  a full `!` is LEACE against the neutral mean.
- The leftover ablation operator `A = Σ coeffᵢ uᵢuᵢᵀ` is diagonalized, so
  partial, repeated, negative and non-orthogonal ablations get exact per-axis
  `kappa` (0 on push axes).
- **Whitened along-normalization.** `target = P @ Σ coeffᵢ·wdᵢ/‖wdᵢ‖_M` (`P`
  the merged read rows) and `share = ‖Δ‖_M` of the raw displacement. Every
  target gets the same whitened budget (`Σ_L eff_along_L = gain·n_layers`,
  linear in α), spread by where its signal lives (`ARCHITECTURE.md` §6.4). A
  pure-ablation layer gets a positive proxy share that the hook's κ
  compensation cancels.
- Missing neutral means or partial whitener coverage raise `WhitenerError`.

`subspace_inject(h, subspace, domain, target, foot_seed, along, onto, …)` splits
`h = mean + H_m + H_n + H_o` (surface foot, off-surface in-subspace residual,
off-subspace residual) and works in reduced coordinates (`read` rows read,
`basis` rows write; reduced norms are Mahalanobis norms; O(R)):

- **along** translates the foot by `along·(target − origin)` via
  `domain.translate_foot`. A fixed offset preserves per-token in-subspace
  spread; lerping each foot onto the target would erase it and loop-collapse at
  strong push. On the curved path `H_n` moves to the new tangent frame by the
  minimal principal-angle rotation (`_frame_rotation_transport`): norm
  preserving and exactly the identity when the foot does not move, so
  `along = 0` returns `h` whatever the foot-solve accuracy. Frames use modified
  Gram-Schmidt (`linalg.qr` is unimplemented on MPS); principal pairs are
  closed-form on device for n ≤ 2.
- **onto** then shrinks `H_n`: by `(1 − onto)` without a σ-field (onto = 1 lands
  on the wire), or toward σ(z) with one (onto = 1 lands one σ off the wire, on
  the typical set; direction kept; a residual inside the tube is never
  expanded).
- **`H_o` is kept verbatim.** It is the Mahalanobis-orthogonal complement
  `(I − basisᵀ read)(h − mean)`, every composing neighbor's span, so a vector
  and N orthogonal manifolds compose with zero cross-talk.
- **Affine shortcut.** On a flat subspace `q = (h − mean)·Pᵀ`, `H_n ≡ 0`,
  `onto` is vacuous: `p_new = q + along·(target − κ·q)`,
  `h_new = h + (p_new − q)·B`. No foot solve, RBF or norm cap.
- **Curved norm cap** `‖h_new‖ ≤ norm_cap·‖h‖` (3) bounds off-domain RBF
  extrapolation; it is a Euclidean norm (through the frame's `B Bᵀ` Gram) and
  scales the whole output, `H_o` included.

The kernel returns fp32 plus this token's refined pre-slide foot, which seeds
the next token. It applies no gain and no clamp: per-layer `eff_along` and
`eff_onto`, share normalization, `_SUBSPACE_GAIN` (8.0), `_MANIFOLD_ALONG_GAIN`
(2.0), `_MANIFOLD_ONTO_GAIN` (0.5), the periodic clamp and the cold-start step
count are in `steering/hooks.py`. At runtime only curved terms call
`subspace_inject`; affine terms run `steering/hooks.py::AffineOp`, a split of
the affine branch that `tests/steering/test_hooks_triggers.py` pins equal to
the kernel.

## Discover coordinates and topology

Discover coordinates are layer-agnostic: the pipeline averages every fit
layer's whitened, node-mean-centered `(K, K)` Gram into a consensus Gram.
`derive_pca_coords` keeps the smallest eigen-prefix crossing `var_threshold`
(0.70), capped at `max_dim` (8). `derive_spectral_coords` builds a union k-NN
heat-kernel graph (`k_nn = max(5, ⌈log K⌉)`, bandwidth = median neighbor
distance) and picks k at the eigenvalue-ratio cliff (absolute gaps grow like k²
on S¹ and over-pick); `min_dim` floors it. Spectral needs K ≥ 4 and a
connected graph and is noisy below about 50 nodes. `neutral_layout_coord`
locates ν in a flat layout by landmark MDS, so `% 0,…,0` reads as neutral.

`select_topology` (`fit_mode=auto`) makes two decoupled decisions, since a
single reconstruction score always crowns the highest-dimensional candidate:

1. **Flat vs curved.** Summed per-layer GCV (`K·RSS/(K − edf)²`) of the flat
   OLS fit and the penalized curved fit on the same Fisher-reduced targets;
   the curved candidate is floored to the flat dimension.
2. **Periodic axes.** Vietoris-Rips H1 over a window of twice the longest MST
   edge counts persistent bars (`persistence_frac`, 0.5). Charts come from a
   distance-MDS span or Laplacian eigenpairs that are near-circular and not
   harmonics, and must pass ambient-rank, phase-coverage and false-neighbor
   checks. When H1 counts nothing, `_faint_cycle_coords` tries a tour for a
   faint ring or clustered ring (months, weekdays); its `_CYCLE_*` guards
   accept two false negatives (a very loose cluster heap, an ellipse beyond
   6:1) for near-zero false positives on grids, fans, arcs, lines and blobs.

A viable periodic candidate (finite GCV no worse than the mean-only baseline)
becomes a `BoxDomain` of 2π axes and wins outright (a plane approximates a
circle but loses its closure); otherwise curved wins only on strictly lower
GCV. `TopologyChoice` carries the ranked candidates plus the winner's
diagnostics. This proposes periodic charts without certifying topology: H1
cannot tell a circle from a cylinder or Möbius strip. Detection abstains above
K = 128, and an exceeded 500,000-triangle budget is recorded as a non-viable
`periodic-unresolved` candidate; abstention is not evidence of trivial topology.

## Whitener

`LayerWhitener` keeps per layer the centered neutral rows `X (N, D)`, the ridge
`λ_L = ‖X‖²_F/(N·D) · ridge_scale` (`DEFAULT_RIDGE_SCALE = 1.0`) and
`K = (NλI + XXᵀ)⁻¹` by Cholesky, so `Σ⁻¹v = (1/λ)(v − Xᵀ K X v)` is O(ND) with no
D×D matrix (CPU fp32, batched). N is a multiple of the 48 baseline prompts, so
Σ is rank-deficient and the ridge is required. A layer with non-finite factors
is left out, so `covers_all` means finite factors everywhere. It is built by
`store/neutral.py::load_neutral_whitener` or the session's lazy neutral state.

Primitives: `apply_inv` (vector → covector), `apply` (`Σv`, covector → vector;
the lens atoms), `whiten`/`whiten_dual` (`Σ^{-1/2}` on vectors, `Σ^{1/2}` on
covectors, exact inverses; the lens decompositions fold `W` into their
dictionaries), `mahalanobis_norm`/`mahalanobis_cosine`, `leace_project`
(`~` keeps `⟨b, d⟩_M/⟨d, d⟩_M · d`, `|` erases it), `subspace_gram(layer, B) =
B Σ⁻¹ Bᵀ`, `woodbury_factors` (device-resident factors for the monitor).

## Direction view: fold, `Profile`, GGUF

A steering vector is the R = 1 affine manifold. `fold_directions_to_subspace`
folds a per-layer direction (a merge, a `~`/`|` result, a registered profile, a
monopolar `concept − ν`) into a neutral-anchored one-pole ray: `basis =
d/‖d‖_M`, `read = Σ⁻¹d/‖d‖_M`, pole at coordinate `‖d_L‖_M` (so it sits at
`ν + d`), share `‖d_L‖_M` (not the √2 form; one direction has no node cloud),
no DLS, no origin. `folded_directions` reads an R = 1 fit back as
`{L: basis_L · share_L}`: the `Profile` behind `extract()`, `manifold
compare`/`why`, merges and GGUF export. `project_profile` is the per-layer LEACE
`~`/`|` (`|` passes layers missing from `onto` through, `~` drops them, an empty
result raises `SteeringCompositionError`).

`Profile` wraps `dict[int, Tensor]` and saves as fp32 `layer_<i>` safetensors
plus a JSON sidecar (`format_version`, `drowse_version`, `method` ∈
{`profile`, `manifold_pca`, `merge`}, free-form `provenance`), written
atomically under `artifact_lock`. `linear_sum` merges over the union of layers
(`strict=True` requires equal coverage).

`gguf.py` reads and writes llama.cpp control vectors (`direction.<L>` fp32,
optional lazy `gguf` import). `store/gguf.py::export_gguf_manifold` exports
2-node flat fits as `folded_directions`; llama.cpp applies one uniform scale,
so the per-layer weighting is the Mahalanobis share, an approximation of
drowse's injection.

## Cross-model transfer

`fit_alignment` fits `A_L(x) = M_L x + b_L` per shared layer from matched
neutral activations (`fit/transfer.py` orchestrates; `store/alignment.py`
caches): orthogonal Procrustes factored through the two row spaces for equal
hidden sizes (no unobserved D − N complement is invented), minimum-norm least
squares otherwise. Fewer than 10 shared layers raises `AlignmentError`; the
median of the per-layer centered R² is the sidecar's `transfer_quality_estimate`.

`transfer_manifold_subspaces` maps means as points and write rows as directions
(source reads are source-metric covectors and do not transfer),
re-orthonormalizes in the target whitener's metric (`B_tgt = L⁻¹B_raw`,
`L Lᵀ = B_raw Σ_tgt⁻¹ B_rawᵀ`, reads recomputed there) and folds `L` into
`affine_map` or the RBF coefficients, so world points are preserved.
Rank-collapsing maps are rejected. A curved layer's scalar σ becomes the
volume-preserving isotropic approximation `σ·(Π sᵢ)^(1/R)` (`sᵢ` the singular
values of `L`) unless `L` is orthogonal. Shares are re-baked with the target
whitener, curved origins re-solved against the target neutral means, and
uncovered layers drop.

Tests: `tests/geometry/` (CPU-only; `test_manifold_math.py` holds the kernel identities).
