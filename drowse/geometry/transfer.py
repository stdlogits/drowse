"""Cross-model transfer of a fitted manifold's per-layer subspaces.

The pure-tensor core of ``manifold transfer``: map each layer's frame through
a per-layer affine alignment (source → target activation space), keep the
runtime frame orthonormal in the *target* Mahalanobis metric by a Cholesky
reparameterization (with its dual read rows recomputed there), re-bake the
Mahalanobis share in the target metric, and re-solve curved origins against
the target neutral mean.  The alignment fit is :mod:`drowse.geometry.procrustes`;
folder I/O and orchestration live in :mod:`drowse.fit.transfer`.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

import torch

from drowse.errors import DrowseError, WhitenerError
from drowse.geometry.invert import neutral_origin
from drowse.geometry.manifold import Manifold
from drowse.geometry.rbf import eval_rbf
from drowse.geometry.subspace import LayerSubspace, dual_read, subspace_share

if TYPE_CHECKING:
    from drowse.geometry.whitener import LayerWhitener


class LayerMap(Protocol):
    """One layer's source → target affine map (e.g. a Procrustes alignment)."""

    def apply_vectors(self, vectors: torch.Tensor, /) -> torch.Tensor:
        """Map direction rows ``(.., D_src)`` by the linear factor only."""
        ...

    def apply_points(self, points: torch.Tensor, /) -> torch.Tensor:
        """Map points ``(.., D_src)`` by the full affine map."""
        ...


def transfer_manifold_subspaces(
    src: Manifold,
    alignment: Mapping[int, LayerMap],
    *,
    whitener: "LayerWhitener | None",
    target_layer_means: Mapping[int, torch.Tensor],
    from_model: str,
    to_model: str,
) -> Manifold:
    """Map a fitted manifold's per-layer subspaces into a target model's space.

    The pure-tensor core of the cross-model Procrustes transfer (the folder
    read/write orchestration stays in :func:`drowse.store.manifold_lifecycle.
    transfer_manifold`).  Takes the already-loaded **source** ``Manifold``, a
    per-layer affine alignment map (the compact factorized
    :class:`drowse.geometry.procrustes.LayerAlignment` returned by
    :func:`drowse.geometry.procrustes.fit_alignment`), and the **target** model's
    whitener, and returns a new
    ``Manifold`` whose subspaces live in target space.

    Means and manifold points use the affine map (linear factor plus fitted
    translation); basis rows (write vectors) and direction profiles use only
    its linear factor.  The source read rows are source-metric covectors and
    do not transfer: the mapped write rows ``B_raw`` are reparameterized to a
    target-Mahalanobis-orthonormal frame ``B_tgt = L⁻¹ B_raw`` (``L Lᵀ =
    B_raw Σ_tgt⁻¹ B_rawᵀ``) with reads ``L⁻¹ B_raw Σ_tgt⁻¹``, and every
    affine/RBF reduced-coordinate coefficient is transformed by the exact
    companion matrix ``L``, preserving world points. A collapsed source span
    is rejected.  ``L`` is orthogonal only where the alignment carries the
    source metric exactly onto the target's; otherwise a curved fit's scalar
    tube thickness turns anisotropic, which the σ-field cannot encode, so σ is
    rescaled by ``(Π sᵢ)^(1/R)`` (``sᵢ`` the singular values of ``L``): the
    volume-preserving isotropic approximation.  Layers the alignment doesn't
    cover drop.

    **Target-metric share re-bake (mandatory).**  The source fit's per-layer
    Mahalanobis ``share`` is a per-model quantity (``Σ`` belongs to
    ``from_model``), so it can't carry across.  The target ``whitener`` is
    **required** and must cover every transferred layer (all-or-nothing,
    mirroring the fit gate); the share is recomputed in target space via
    :func:`subspace_share` (``sqrt(Σ_k coordsᵀ (B_tgt Σ_tgt⁻¹ B_tgtᵀ) coords)``
    — the same formula the fit pipeline bakes).  A missing or non-covering
    whitener raises :class:`~drowse.errors.WhitenerError`; there is no
    Euclidean rebake. For curved layers, the target neutral mean is projected
    into the transferred frame and inverted on the transferred surface,
    yielding the target-model origin directly.

    Folder-level format guards (empty alignment, source fit missing) are the
    caller's concern; this function raises only
    :class:`~drowse.errors.WhitenerError` (missing / partial target whitener) and
    ``ValueError`` for unrepresentable geometry or when ``alignment`` covers
    none of the source's fitted layers.
    """
    src.validate_runtime_geometry()

    covered = [layer for layer in src.layers if layer in alignment]
    if not covered:
        raise ValueError(
            f"alignment for {from_model!r} → {to_model!r} covered none of the "
            f"source manifold's fitted layers ({sorted(src.layers)})"
        )
    # The frame is orthonormal in the target metric and the source model's
    # Mahalanobis share is per-model (Σ and the neutral activations are both
    # ``from_model`` quantities), so the **target** whitener is mandatory and
    # must cover every transferred layer (all-or-nothing, mirroring the fit
    # gate).  No Euclidean path — a missing / partial whitener is an error.
    if whitener is None or not whitener.covers_all(covered):
        raise WhitenerError(
            "manifold transfer requires a Mahalanobis whitener covering every "
            f"transferred layer {sorted(covered)}; generate neutral "
            "activations for the TARGET model first (the Euclidean path is gone)"
        )

    # Map each covered layer's subspace into target space.  ``M_L`` is
    # ``(D_tgt, D_src)`` so ``mean_tgt = M_L @ mean_src`` and each basis row
    # transforms the same way → ``basis_tgt = basis_src @ M_L^T``.
    new_layers: dict[int, LayerSubspace] = {}
    for layer in covered:
        sub = src.layers[layer]
        M_L = alignment[layer]
        mean_f = sub.mean.to(torch.float32)
        basis_f = sub.basis.to(torch.float32)
        raw_basis = M_L.apply_vectors(basis_f).to(device="cpu", dtype=torch.float32)
        mean_tgt_f = M_L.apply_points(mean_f)

        rank = int(raw_basis.shape[0])
        if raw_basis.ndim != 2 or raw_basis.shape[1] < rank:
            raise ValueError(
                f"alignment for layer {layer} cannot carry rank-{rank} source "
                f"subspace into target shape {tuple(raw_basis.shape)}"
            )
        # Reparameterize B_raw = L @ B_tgt with B_tgt orthonormal in the target
        # metric.  Runtime projection/injection requires an orthonormal frame,
        # while multiplying every reduced output by L preserves each mapped
        # world-space manifold point exactly.
        raw_read = whitener.apply_inv(layer, raw_basis)            # Σ_tgt⁻¹ B_rawᵀ rows
        raw_gram = raw_basis @ raw_read.transpose(0, 1)
        raw_gram = 0.5 * (raw_gram + raw_gram.transpose(0, 1))
        identity = torch.eye(rank, dtype=raw_gram.dtype, device=raw_gram.device)
        if torch.allclose(raw_gram, identity, atol=1e-5, rtol=1e-5):
            # Keep an already-orthonormal frame as is (an identity map within
            # one model's metric needs no coordinate change).
            basis_tgt_f = raw_basis.contiguous()
            read_tgt_f = raw_read.contiguous()
            reduced_map = identity
        else:
            eigenvalues = torch.linalg.eigvalsh(raw_gram)
            tol = (
                torch.finfo(raw_gram.dtype).eps * max(raw_basis.shape)
                * float(eigenvalues.abs().max())
            )
            if bool((eigenvalues <= tol).any()):
                raise ValueError(
                    f"alignment for layer {layer} collapses the rank-{rank} manifold "
                    "subspace; refusing a rank-deficient transfer"
                )
            chol = torch.linalg.cholesky(raw_gram)                 # L, lower
            basis_tgt_f = torch.linalg.solve_triangular(
                chol, raw_basis, upper=False,
            ).contiguous()
            read_tgt_f = dual_read(basis_tgt_f, torch.linalg.solve_triangular(
                chol, raw_read, upper=False,
            ))
            reduced_map = chol.contiguous()
        kwargs: dict[str, Any] = {
            "mean": mean_tgt_f.to(dtype=sub.mean.dtype),
            "basis": basis_tgt_f.to(device=sub.basis.device, dtype=sub.basis.dtype),
            "read": read_tgt_f.to(device=sub.read.device, dtype=sub.read.dtype),
        }
        if sub.is_affine:
            source_map = (
                torch.eye(
                    sub.rank, dtype=torch.float32, device=reduced_map.device,
                )
                if sub.affine_map is None
                else sub.affine_map.to(
                    device=reduced_map.device, dtype=torch.float32,
                )
            )
            kwargs["affine_map"] = (source_map @ reduced_map).to(
                dtype=sub.basis.dtype,
            )
            kwargs["node_coords"] = (
                None if sub.node_coords is None
                else (sub.node_coords.to(torch.float32) @ reduced_map).to(
                    dtype=sub.node_coords.dtype,
                )
            )
        else:
            assert sub.rbf_weights is not None and sub.poly_coeffs is not None
            kwargs["rbf_weights"] = (
                sub.rbf_weights.to(torch.float32) @ reduced_map
            ).to(dtype=sub.rbf_weights.dtype)
            kwargs["poly_coeffs"] = (
                sub.poly_coeffs.to(torch.float32) @ reduced_map
            ).to(dtype=sub.poly_coeffs.dtype)
            if sub.sigma_poly_coeffs is not None:
                # The scalar tube thickness is exact only under an isometry of
                # the reduced frame; the companion map is one only where the
                # alignment carries the source metric onto the target's.  The
                # ball of radius σ maps to an ellipsoid with semi-axes σ·sᵢ
                # (``sᵢ`` the companion's singular values); keep its volume:
                # σ ← σ·(Π sᵢ)^(1/R), a shift of the log-σ field's constant term.
                log_scale = float(
                    torch.linalg.svdvals(reduced_map).clamp_min(1e-30).log().mean()
                )
                sigma_poly = sub.sigma_poly_coeffs.to(torch.float32).clone()
                sigma_poly[0] += log_scale
                kwargs["sigma_poly_coeffs"] = sigma_poly.to(
                    dtype=sub.sigma_poly_coeffs.dtype,
                )
        new_layers[layer] = replace(sub, **kwargs)

    curved_layers = {
        layer for layer, subspace in new_layers.items() if not subspace.is_affine
    }
    missing_means = sorted(curved_layers - set(target_layer_means))
    if missing_means:
        raise WhitenerError(
            "manifold transfer requires target neutral means covering every "
            f"transferred layer; missing {missing_means}"
        )

    new_share: dict[int, float] = {}
    for layer, sub_tgt in new_layers.items():
        sub_f = sub_tgt.to(device=torch.device("cpu"), dtype=torch.float32)
        # ``coords`` are the reduced node values in subspace-coordinate space.
        # For a K=1 affine folded ray there is no node cloud to μ-center: its
        # share is the target-metric norm of the actual neutral→pole world
        # direction. For K>=2 affine and every curved fit, ``subspace_share``
        # computes the μ-centered whitened spread
        # ``sqrt(Σ_k c_kᵀ M_R c_k)`` (``M_R = B_tgt Σ_tgt⁻¹ B_tgtᵀ`` via
        # ``subspace_gram``, the *target* Σ⁻¹ restricted to the transferred
        # basis) — the same formula the fit pipeline bakes, now in target space.
        # It μ-centers internally only if fed μ-centered coords, so do the
        # centering here: flat fits carry neutral-anchored real coords in
        # ``node_coords``; curved fits read μ-centered node values off the RBF.
        if sub_f.is_affine:
            coords = sub_f.node_coords  # (K, R) neutral-anchored
            if coords is None:  # affine ⇒ node_coords set; sidecar corruption guard
                raise DrowseError(
                    "transfer_manifold_subspaces: affine LayerSubspace has"
                    " node_coords=None — the saved manifold sidecar may be corrupt"
                )
            if coords.shape[0] == 1:
                world_direction = coords[0] @ sub_f.basis
                new_share[layer] = float(
                    whitener.mahalanobis_norm(layer, world_direction)
                )
                continue
        else:
            _np, _rw, _pc = sub_f.rbf_params()
            coords = eval_rbf(_np, _rw, _pc, _np)  # (K, R)
        mu_coords = coords - coords.mean(dim=0, keepdim=True)  # μ-center
        new_share[layer] = subspace_share(
            mu_coords, sub_f.basis, whitener=whitener, layer=layer,
        )

    new_origin = {
        layer: neutral_origin(
            sub_tgt, src.domain, target_layer_means[layer], src.node_coords,
        )
        for layer, sub_tgt in new_layers.items()
        if not sub_tgt.is_affine
    }

    transferred = replace(
        src, layers=new_layers,
        mahalanobis_share=new_share,
        origin=new_origin,
    )
    transferred.validate_runtime_geometry()
    return transferred


__all__ = ["LayerMap", "transfer_manifold_subspaces"]
