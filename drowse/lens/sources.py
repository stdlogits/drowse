"""The lens source family: local fits, provider bindings, per-family selection.

Drowse owns lenses it fits and stores those payloads below ``DROWSE_HOME``
(:mod:`drowse.lens.store`).  External publishers retain ownership of theirs: a
Hugging Face lens stays in the Hub cache and Drowse writes only a small,
commit-pinned :class:`LensBinding` that maps the provider's files to
:class:`~drowse.lens.key.LensKey` s, one file per site.

Sources are addressed ``local:<kind>[-<tag>]`` (``local:r``, ``local:s``) or by
a provider binding's bare name (``rs-r``, ``rs-s``, ``workspace-r``,
``neuronpedia``, ``workspace-j``).  The selection holds one active R-family
(J/R) and one S-family (K/S) lens, so the two are resident together; on disk
the two slots keep their side names (``output``, ``input``).  A source carries
one or both sites (``out``, ``mid``), and a read at a site the selected source
lacks says so rather than falling back to the other site.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, TypeGuard

import torch
import yaml

from drowse.base.atomic import artifact_lock, write_json_atomic
# The three Hub indirections are imported (not redefined) so ``base.hf`` stays
# the single monkeypatchable HF seam; tests patch them on this module.
from drowse.base.hf import HFError, _hf_api, _hf_hub_download, _hf_snapshot_download
from drowse.base.integrity import hash_file
from drowse.base.source_registry import LOCAL, SourceFamily, SourceRef
from drowse.lens.jlens import JacobianLens
from drowse.lens.key import LENS_SIDES, LENS_SITES, LOCAL_LENS_NAME, LensKey, local_lens_kind
from drowse.errors import RequestError, SourceSpecError

LENS_SOURCE_FORMAT_VERSION = 2
LENS_BINDING_FORMAT_VERSION = 2
HUGGINGFACE = "huggingface"
NEURONPEDIA_REPO = "neuronpedia/jacobian-lens"
WORKSPACE_REPO = "camilablank/workspace-lenses"
RS_LENSES_REPO = "stdlogits/rs-lenses"
#: The text corpus every rs-lenses file was fitted on (its model card).  A
#: file that carries no embedded fit record (Gemma 4's) is bound to it.
RS_LENSES_CORPUS = (
    "HuggingFaceFW/dclm_100BT-shuffled@2fa015e4044ec442a0734e89658cdcc538d10dd4"
)
RS_LENSES_SEQ_LEN = 128
RS_LENSES_SKIP_FIRST = 16


@dataclass(frozen=True)
class _Provider:
    """One fetchable source name: who publishes it and which lens it is.

    ``files`` maps each site the provider publishes to its path below the
    model's folder (``<model stem>/<path>``); a fetch binds every one the
    repository holds, and at least one.  Neuronpedia has no layout (its file
    is found through a config match) and publishes ``out`` only.
    """

    provider: str
    repo: str
    kind: str
    files: Mapping[str, str] = field(default_factory=dict)
    estimator: str | None = None  # the declared provenance estimator

    @property
    def keys(self) -> tuple[LensKey, ...]:
        """Every key this provider can bind, in site order."""
        sites = tuple(self.files) or ("out",)
        return tuple(LensKey(self.kind, site) for site in LENS_SITES if site in sites)

    @property
    def side(self) -> str:
        return LensKey(self.kind).side


#: Every fetchable source name.  The binding name is the public source token
#: (``lens fetch``/``use``/``show``/``rm`` all speak it), so the R and S lenses
#: of one release, or both arms of a matched workspace J/R pair, coexist per
#: model and switch with ``lens use``.
LENS_PROVIDERS: dict[str, _Provider] = {
    "rs-r": _Provider(
        "rs-lenses", RS_LENSES_REPO, "R",
        files={"out": "r-lens/lens.pt", "mid": "r-lens/mid.pt"}, estimator="relp",
    ),
    "rs-s": _Provider(
        "rs-lenses", RS_LENSES_REPO, "S",
        files={"out": "s-lens/out.pt", "mid": "s-lens/mid.pt"}, estimator="relp",
    ),
    "neuronpedia": _Provider("neuronpedia", NEURONPEDIA_REPO, "J"),
    "workspace-r": _Provider(
        "workspace-lenses", WORKSPACE_REPO, "R",
        files={"out": "r-lens/lens.pt"}, estimator="relp",
    ),
    "workspace-j": _Provider(
        "workspace-lenses", WORKSPACE_REPO, "J",
        files={"out": "j-lens/lens.pt"}, estimator="standard",
    ),
}
_PROVENANCE_FIELDS = {
    "neuronpedia": {"config_file", "config_sha256", "dim_batch"},
    "workspace-lenses": {"estimator", "rules_json", "target_layer"},
    "rs-lenses": {"estimator", "rules_json", "embedded"},
}
#: Automatic attachment and source-picker order, per family.  The provider
#: tier first: rs-lenses' R-lens (both sites), the workspace R-lens, the
#: official paper lens, the matched workspace J-lens; on the S side rs-lenses'
#: S-lens.  Drowse-fitted local artifacts are the fallback tier (R before J,
#: S before K).
DEFAULT_LENS_SOURCE_ORDER: dict[str, tuple[str, ...]] = {
    "output": ("rs-r", "workspace-r", "neuronpedia", "workspace-j"),
    "input": ("rs-s",),
}
_LOCAL_KIND_ORDER = {"output": ("r", "j"), "input": ("s", "k")}


def lens_source_side(ref: SourceRef) -> str:
    """The selection slot of a source: ``output`` (J/R) or ``input`` (K/S)."""
    if ref.local:
        return LensKey(local_lens_kind(ref.name)).side
    return LENS_PROVIDERS[ref.name].side


def lens_family(kind: str) -> str:
    """A kind's family as users see it: ``R`` (J/R, what a vector is
    disposed to say) or ``S`` (K/S, what it has read).  ``output``/``input``
    is the internal slot name (:func:`lens_source_side`)."""
    return "R" if LensKey(kind).side == "output" else "S"


def _lens_side(model_id: str, ref: SourceRef) -> str:
    del model_id
    return lens_source_side(ref)


#: The lens source family: ``local:<kind>[-<tag>]`` artifacts and bare
#: provider bindings, one active source per family slot.
LENS_SOURCES = SourceFamily(
    label="lens",
    dirname="lens",
    format_version=LENS_SOURCE_FORMAT_VERSION,
    provider_kind=HUGGINGFACE,
    provider_prefix=None,
    local_name=LOCAL_LENS_NAME,
    provider_name_ok=lambda name: name in LENS_PROVIDERS,
    slots=LENS_SIDES,
    slot_of=_lens_side,
)


def lens_source_preference_key(source: str) -> tuple[int, int, int, str]:
    """Stable default ordering for prepared lens sources.

    R-family (J/R) sources sort before S-family (K/S) ones.  Within a family
    the provider tier comes first (:data:`DEFAULT_LENS_SOURCE_ORDER`), then
    the local fits (``local:r``, ``local:j``; ``local:s``, ``local:k``), then
    any tagged fit; unknown names sort last.
    """
    unknown = (len(LENS_SIDES), 0, 0, source)
    try:
        ref = LENS_SOURCES.parse(source)
    except ValueError:
        return unknown
    side = lens_source_side(ref)
    side_rank = LENS_SIDES.index(side)
    if not ref.local:
        return (side_rank, 0, DEFAULT_LENS_SOURCE_ORDER[side].index(ref.name), source)
    order = _LOCAL_KIND_ORDER[side]
    rank = order.index(ref.name) if ref.name in order else len(order)
    return (side_rank, 1, rank, ref.name)


# -- bindings ---------------------------------------------------------------

@dataclass(frozen=True)
class BoundLensFile:
    """One provider file holding one lens key's matrices."""

    path: str
    sha256: str
    layers: tuple[int, ...]


@dataclass(frozen=True)
class LensBinding:
    """A commit-pinned binding of provider files to lens keys.

    One kind, one file per bound site.  Common identity (publisher and
    base-model commits, dimensions, corpus recipe) is typed; ``provenance``
    carries the provider's own fit record (a Neuronpedia config digest, a
    workspace arm's declared estimator and rules, an rs-lenses estimator and
    whether its files embedded their record), validated per provider.
    """

    name: str
    model_id: str
    model_revision: str
    repo_id: str
    repo_revision: str
    files: dict[LensKey, BoundLensFile]
    d_model: int
    n_prompts: int
    corpus: str
    seq_len: int
    skip_first: int
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def provider(self) -> str:
        return LENS_PROVIDERS[self.name].provider

    @property
    def kind(self) -> str:
        """The bound lens kind (J/R/K/S); every file of a binding shares it."""
        return LENS_PROVIDERS[self.name].kind

    @property
    def sites(self) -> tuple[str, ...]:
        """The bound sites, in ``LENS_SITES`` order."""
        held = {key.site for key in self.files}
        return tuple(site for site in LENS_SITES if site in held)

    def file(self, site: str) -> BoundLensFile:
        """The file bound at ``site`` (``KeyError`` when the site is unbound)."""
        return self.files[LensKey(self.kind, site)]

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": LENS_BINDING_FORMAT_VERSION,
            "kind": HUGGINGFACE,
            "provider": self.provider,
            "name": self.name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "repo_id": self.repo_id,
            "repo_revision": self.repo_revision,
            "files": {
                key.label: {
                    "path": bound.path,
                    "sha256": bound.sha256,
                    "layers": list(bound.layers),
                }
                for key, bound in self.files.items()
            },
            "d_model": self.d_model,
            "n_prompts": self.n_prompts,
            "corpus": self.corpus,
            "seq_len": self.seq_len,
            "skip_first": self.skip_first,
            "provenance": dict(self.provenance),
        }


_BINDING_FIELDS = {
    "format_version", "kind", "provider", "name", "model_id",
    "model_revision", "repo_id", "repo_revision", "files", "d_model",
    "n_prompts", "corpus", "seq_len", "skip_first", "provenance",
}


def _int_at_least(value: Any, minimum: int) -> TypeGuard[int]:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _layer_tuple(value: Any) -> tuple[int, ...]:
    if (
        not isinstance(value, list) or not value
        or any(not _int_at_least(layer, 0) for layer in value)
        or value != sorted(set(value))
    ):
        raise ValueError("lens binding has invalid layers")
    return tuple(value)


def _parse_binding(payload: Any, model_id: str, name: str) -> LensBinding:
    """Strictly validate one current-format binding payload."""
    provider = LENS_PROVIDERS.get(name)
    if (
        provider is None
        or not isinstance(payload, dict)
        or set(payload) != _BINDING_FIELDS
        or payload["format_version"] != LENS_BINDING_FORMAT_VERSION
        or payload["kind"] != HUGGINGFACE
        or payload["provider"] != provider.provider
        or payload["name"] != name
        or payload["model_id"] != model_id
    ):
        raise ValueError("lens binding has an invalid schema or identity")
    for key in ("model_revision", "repo_id", "repo_revision", "corpus"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError(f"lens binding has invalid {key}")
    for key, minimum in (
        ("d_model", 1), ("n_prompts", 1), ("seq_len", 1), ("skip_first", 0),
    ):
        if not _int_at_least(payload[key], minimum):
            raise ValueError(f"lens binding has invalid {key}")
    raw_files = payload["files"]
    allowed = {key.label for key in provider.keys}
    if not isinstance(raw_files, dict) or not raw_files or not set(raw_files) <= allowed:
        raise ValueError(f"{name} must bind a nonempty subset of {sorted(allowed)}")
    files: dict[LensKey, BoundLensFile] = {}
    for label, entry in raw_files.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "layers"}
            or not isinstance(entry["path"], str) or not entry["path"]
            or not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64
        ):
            raise ValueError(f"lens binding file {label} is invalid")
        files[LensKey.parse(label)] = BoundLensFile(
            entry["path"], entry["sha256"], _layer_tuple(entry["layers"]),
        )
    provenance = payload["provenance"]
    if (
        not isinstance(provenance, dict)
        or set(provenance) != _PROVENANCE_FIELDS[provider.provider]
    ):
        raise ValueError("lens binding has invalid provenance")
    if provider.provider == "neuronpedia":
        if (
            not isinstance(provenance["config_file"], str)
            or not provenance["config_file"]
            or not isinstance(provenance["config_sha256"], str)
            or len(provenance["config_sha256"]) != 64
            or not _int_at_least(provenance["dim_batch"], 1)
        ):
            raise ValueError("Neuronpedia lens binding has invalid provenance")
    elif (
        provenance["estimator"] != provider.estimator
        or not isinstance(provenance["rules_json"], str)
        or (
            provider.provider == "workspace-lenses"
            and not _int_at_least(provenance["target_layer"], 0)
        )
        or (
            provider.provider == "rs-lenses"
            and not isinstance(provenance["embedded"], bool)
        )
    ):
        raise ValueError(f"{provider.provider} lens binding has invalid provenance")
    return LensBinding(
        name=name,
        model_id=model_id,
        model_revision=payload["model_revision"],
        repo_id=payload["repo_id"],
        repo_revision=payload["repo_revision"],
        files=files,
        d_model=payload["d_model"],
        n_prompts=payload["n_prompts"],
        corpus=payload["corpus"],
        seq_len=payload["seq_len"],
        skip_first=payload["skip_first"],
        provenance=dict(provenance),
    )


def load_external_lens_binding(model_id: str, name: str) -> LensBinding | None:
    if name not in LENS_PROVIDERS:
        return None
    path = LENS_SOURCES.binding_path(model_id, name)
    if not path.exists():
        return None
    try:
        return _parse_binding(json.loads(path.read_text()), model_id, name)
    except (OSError, ValueError, TypeError, KeyError):
        return None


# -- provider checkpoints ---------------------------------------------------

#: A provider file's dictionary, by the side its kind reads.  Output kinds
#: follow the workspace layout (``J`` by ``source_layers``), input kinds
#: rs-lenses' S files (``K`` by ``layers``).  ``provenance`` is optional.
_CHECKPOINT_KEYS = {
    "output": ("J", "source_layers"),
    "input": ("K", "layers"),
}


def _load_provider_checkpoint(path: str, kind: str) -> dict[str, Any]:
    """One provider ``.pt`` (memory-mapped: a multi-GB fp32 lens is paged on
    demand rather than read whole), with its exact key set checked."""
    matrices, layers = _CHECKPOINT_KEYS[LensKey(kind).side]
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = {matrices, "n_prompts", layers, "d_model"}
    if (
        not isinstance(payload, dict)
        or not required <= set(payload)
        or set(payload) - required - {"provenance"}
    ):
        raise ValueError(
            f"external {kind}-lens checkpoint has an invalid schema "
            f"(expected {sorted(required)} and an optional provenance)"
        )
    return payload


def _validated_site_matrices(
    payload: dict[str, Any],
    kind: str,
    *,
    expected_d_model: int | None = None,
    expected_layers: int | None = None,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """``(matrices by layer, n_prompts, d_model)`` of one checked provider file."""
    matrices_key, layers_key = _CHECKPOINT_KEYS[LensKey(kind).side]
    matrices = payload.get(matrices_key)
    layers = payload.get(layers_key)
    d_model = payload.get("d_model")
    n_prompts = payload.get("n_prompts")
    if (
        not isinstance(matrices, dict)
        or not isinstance(layers, list)
        or not layers
        or layers != sorted(set(layers))
        or set(matrices) != set(layers)
        or not _int_at_least(d_model, 1)
        or not _int_at_least(n_prompts, 1)
        or (expected_d_model is not None and d_model != expected_d_model)
        or (
            expected_layers is not None
            and any(
                not isinstance(layer, int) or not 0 <= layer < expected_layers
                for layer in layers
            )
        )
    ):
        raise ValueError(f"external {kind}-lens checkpoint metadata is incompatible")
    clean: dict[int, torch.Tensor] = {}
    for layer in layers:
        value = matrices[layer]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (d_model, d_model)
            or not value.dtype.is_floating_point
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"external {kind}-lens layer {layer} is invalid")
        clean[int(layer)] = value
    return clean, int(n_prompts), int(d_model)


def _external_lens(
    kind: str, sites: Mapping[str, dict[int, torch.Tensor]], *, n_prompts: int, d_model: int,
) -> JacobianLens:
    return JacobianLens(
        sites.get("out"), mid=sites.get("mid"),
        n_prompts=n_prompts, d_model=d_model, key=LensKey(kind),
    )


def _config_dimensions(model_id: str, revision: str) -> tuple[int, int]:
    """``(d_model, n_layers)`` from the config pinned to ``revision``."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=False, revision=revision,
    )
    text = getattr(config, "text_config", config)
    hidden = getattr(text, "hidden_size", getattr(text, "n_embd", None))
    layers = getattr(text, "num_hidden_layers", getattr(text, "n_layer", None))
    if not _int_at_least(hidden, 1) or not _int_at_least(layers, 1):
        raise ValueError(f"{model_id} config declares no hidden size / layer count")
    return int(hidden), int(layers)


def _resolve_model_for_fetch(model_id: str) -> tuple[str, int, int]:
    """The immutable model revision and dimensions a fetch validates against.

    The commit is resolved first and the config read pinned to it, so the
    dimensions provably belong to the revision the binding records.
    """
    from drowse.model.fingerprint import resolve_model_revision

    model_revision = resolve_model_revision(model_id)
    if model_revision is None:
        raise RequestError(
            f"could not resolve immutable Hugging Face model revision for {model_id}"
        )
    try:
        d_model, n_layers = _config_dimensions(model_id, model_revision)
    except ValueError as exc:
        raise RequestError(f"could not resolve model dimensions for {model_id}") from exc
    return model_revision, d_model, n_layers


def _repo_commit(repo_id: str, revision: str) -> str:
    sha = getattr(_hf_api().model_info(repo_id, revision=revision), "sha", None)
    if not isinstance(sha, str) or not sha:
        raise HFError(f"could not resolve immutable revision for {repo_id}")
    return sha


def _match_official_config(
    root: Path, model_id: str, dataset: str,
) -> tuple[Path, dict[str, Any], bytes]:
    matches: list[tuple[Path, dict[str, Any], bytes]] = []
    for path in sorted(root.glob(f"*/jlens/{dataset}/config.yaml")):
        raw = path.read_bytes()
        payload = yaml.safe_load(raw)
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("hf_model_name"), str)
            and payload["hf_model_name"].casefold() == model_id.casefold()
        ):
            matches.append((path, payload, raw))
    if not matches:
        raise FileNotFoundError(f"Neuronpedia has no official Jacobian lens for {model_id!r}")
    if len(matches) != 1:
        raise ValueError(f"multiple Neuronpedia lens configs match {model_id!r}")
    return matches[0]


def _select_checkpoint(files: list[str], config_file: str) -> str:
    parent = str(Path(config_file).parent)
    candidates = sorted(
        filename
        for filename in files
        if str(Path(filename).parent) == parent
        and Path(filename).suffix == ".pt"
        and "jacobian_lens" in Path(filename).name
    )
    if not candidates:
        raise FileNotFoundError(f"no Jacobian-lens checkpoint beside {config_file}")
    n1000 = [path for path in candidates if "_n1000.pt" in path]
    plain = [path for path in candidates if path.endswith("_jacobian_lens.pt")]
    return (n1000 or plain or candidates)[0]


def _fetch_neuronpedia(
    model_id: str, *, repo_id: str, revision: str, dataset: str = "Salesforce-wikitext",
) -> LensBinding:
    """Bind the official paper lens for ``model_id``."""
    provider = LENS_PROVIDERS["neuronpedia"]
    repo_revision = _repo_commit(repo_id, revision)
    model_revision, d_model, n_layers = _resolve_model_for_fetch(model_id)
    root = Path(_hf_snapshot_download(
        repo_id, revision=repo_revision,
        allow_patterns=[f"*/jlens/{dataset}/config.yaml"],
    ))
    config_path, fit_config, config_raw = _match_official_config(root, model_id, dataset)
    config_file = config_path.relative_to(root).as_posix()
    files = list(_hf_api().list_repo_files(repo_id, revision=repo_revision))
    checkpoint = _select_checkpoint(files, config_file)
    checkpoint_path = _hf_hub_download(repo_id, checkpoint, revision=repo_revision)
    matrices, n_prompts, lens_d_model = _validated_site_matrices(
        _load_provider_checkpoint(checkpoint_path, provider.kind), provider.kind,
        expected_d_model=d_model, expected_layers=n_layers,
    )
    fit_raw = fit_config.get("fit")
    fit: Mapping[str, Any] = fit_raw if isinstance(fit_raw, Mapping) else {}
    dataset_raw = fit_config.get("dataset")
    dataset_cfg: Mapping[str, Any] = dataset_raw if isinstance(dataset_raw, Mapping) else {}
    return LensBinding(
        name="neuronpedia",
        model_id=model_id,
        model_revision=model_revision,
        repo_id=repo_id,
        repo_revision=repo_revision,
        files={LensKey(provider.kind): BoundLensFile(
            checkpoint, hash_file(Path(checkpoint_path)), tuple(sorted(matrices)),
        )},
        d_model=lens_d_model,
        n_prompts=n_prompts,
        corpus=":".join(
            str(dataset_cfg.get(key, "")) for key in ("name", "config", "split")
        ),
        seq_len=int(fit.get("max_seq_len", 128)),
        skip_first=0,
        provenance={
            "config_file": config_file,
            "config_sha256": hashlib.sha256(config_raw).hexdigest(),
            "dim_batch": int(fit.get("dim_batch", 1)),
        },
    )


def _embedded_record(
    payload: dict[str, Any], model_id: str, provider: _Provider, key: LensKey,
    *, required: bool,
) -> dict[str, Any] | None:
    """Validate a provider file's embedded fit record against what it is bound as.

    Returns ``None`` for a file with no record when ``required`` is false
    (rs-lenses' Gemma 4 files carry none).  A record must name this model and
    the provider's estimator; an rs-lenses record must also name the kind and
    site it is bound as, so a misplaced file cannot bind.
    """
    prov = payload.get("provenance")
    if prov is None and not required:
        return None
    if not isinstance(prov, dict):
        raise ValueError(f"{provider.provider} lens checkpoint carries no provenance dict")
    prov_model = prov.get("model_id")
    if not isinstance(prov_model, str) or prov_model.casefold() != model_id.casefold():
        raise ValueError(
            f"{provider.provider} lens provenance names "
            f"{prov_model!r}, not the requested model {model_id!r}"
        )
    try:
        config = json.loads(prov.get("config_json", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{provider.provider} lens provenance config is unreadable") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{provider.provider} lens provenance config is not a dict")
    if config.get("estimator") != provider.estimator:
        raise ValueError(
            f"{provider.provider} {key.label} file declares estimator "
            f"{config.get('estimator')!r}; expected {provider.estimator!r}"
        )
    if provider.provider == "rs-lenses" and (
        config.get("lens") != key.kind or config.get("target") != key.site
    ):
        raise ValueError(
            f"rs-lenses file bound as {key.label} declares lens "
            f"{config.get('lens')!r} at {config.get('target')!r}"
        )
    corpus = prov.get("dataset_id")
    if not isinstance(corpus, str) or not corpus:
        raise ValueError(f"{provider.provider} lens provenance has invalid dataset_id")
    fields: dict[str, int] = {}
    for name, minimum in (("t_max", 1), ("skip_first", 0)):
        value = prov.get(name)
        if not _int_at_least(value, minimum):
            raise ValueError(f"{provider.provider} lens provenance has invalid {name}")
        fields[name] = int(value)
    if provider.provider == "workspace-lenses":
        target = prov.get("target_layer")
        if not _int_at_least(target, 0):
            raise ValueError("workspace-lenses lens provenance has invalid target_layer")
        fields["target_layer"] = int(target)
    rules = config.get("rules", {})
    return {
        "corpus": corpus,
        "seq_len": fields["t_max"],
        "skip_first": fields["skip_first"],
        "rules_json": json.dumps(rules if isinstance(rules, dict) else {}, sort_keys=True),
        **({"target_layer": fields["target_layer"]} if "target_layer" in fields else {}),
    }


def _fetch_layout(
    model_id: str, name: str, *, repo_id: str, revision: str,
) -> LensBinding:
    """Bind a provider that publishes ``<model stem>/<path>`` per site.

    The workspace arms bind their one ``out`` file; rs-lenses binds every site
    the repository holds for the model (``r-lens/{lens,mid}.pt``,
    ``s-lens/{out,mid}.pt``), at least one, all of one kind and one fit.
    """
    provider = LENS_PROVIDERS[name]
    repo_revision = _repo_commit(repo_id, revision)
    model_revision, d_model, n_layers = _resolve_model_for_fetch(model_id)
    stem = model_id.rsplit("/", 1)[-1].casefold()
    published = set(_hf_api().list_repo_files(repo_id, revision=repo_revision))
    wanted = {site: f"{stem}/{path}" for site, path in provider.files.items()}
    present = {site: path for site, path in wanted.items() if path in published}
    if not present:
        models = sorted({
            str(Path(f).parts[0]) for f in published
            if any(f.endswith(f"/{path}") for path in provider.files.values())
        })
        raise FileNotFoundError(
            f"{repo_id} has no {name} lens for {model_id!r} "
            f"(expected {' or '.join(sorted(wanted.values()))}; "
            f"published models: {', '.join(models) or 'none'})"
        )
    files: dict[LensKey, BoundLensFile] = {}
    sites: dict[str, dict[int, torch.Tensor]] = {}
    records: list[dict[str, Any] | None] = []
    n_prompts: set[int] = set()
    for site in LENS_SITES:
        if site not in present:
            continue
        key = LensKey(provider.kind, site)
        local_path = _hf_hub_download(repo_id, present[site], revision=repo_revision)
        payload = _load_provider_checkpoint(local_path, provider.kind)
        matrices, count, _ = _validated_site_matrices(
            payload, provider.kind, expected_d_model=d_model, expected_layers=n_layers,
        )
        records.append(_embedded_record(
            payload, model_id, provider, key,
            required=provider.provider == "workspace-lenses",
        ))
        n_prompts.add(count)
        sites[site] = matrices
        files[key] = BoundLensFile(
            present[site], hash_file(Path(local_path)), tuple(sorted(matrices)),
        )
    if len(n_prompts) != 1:
        raise ValueError(f"{name} files for {model_id!r} disagree on n_prompts")
    embedded = [record for record in records if record is not None]
    if embedded and len(embedded) != len(records):
        raise ValueError(f"{name} files for {model_id!r} disagree on carrying provenance")
    recipes = {
        (record["corpus"], record["seq_len"], record["skip_first"]) for record in embedded
    }
    if len(recipes) > 1:
        raise ValueError(f"{name} files for {model_id!r} were fitted on different corpora")
    if embedded:
        corpus, seq_len, skip_first = recipes.pop()
    else:
        corpus, seq_len, skip_first = RS_LENSES_CORPUS, RS_LENSES_SEQ_LEN, RS_LENSES_SKIP_FIRST
    if provider.provider == "workspace-lenses":
        record = embedded[0]
        provenance: dict[str, Any] = {
            "estimator": provider.estimator,
            "rules_json": record["rules_json"],
            "target_layer": record["target_layer"],
        }
    else:
        provenance = {
            "estimator": provider.estimator,
            "rules_json": embedded[0]["rules_json"] if embedded else "{}",
            "embedded": bool(embedded),
        }
    return LensBinding(
        name=name,
        model_id=model_id,
        model_revision=model_revision,
        repo_id=repo_id,
        repo_revision=repo_revision,
        files=files,
        d_model=d_model,
        n_prompts=n_prompts.pop(),
        corpus=corpus,
        seq_len=seq_len,
        skip_first=skip_first,
        provenance=provenance,
    )


def fetch_lens_source(
    model_id: str,
    source: str,
    *,
    repo_id: str | None = None,
    revision: str = "main",
    force: bool = False,
    activate: bool = True,
) -> LensBinding:
    """Fetch a named external lens into the HF cache and bind it.

    The CLI/server dispatch point.  ``rs-r`` and ``rs-s`` are rs-lenses' R-
    and S-lens (both sites, Qwen3-8B and Gemma 4 12B text), ``workspace-r``
    the workspace RelP lens, ``workspace-j`` the standard lens fitted on the
    same forward passes, ``neuronpedia`` the official paper lens.  No payload
    is copied into ``DROWSE_HOME``; the binding pins the publisher and
    base-model repositories to commits.  ``activate`` selects the binding in
    its family's slot.
    """
    provider = LENS_PROVIDERS.get(source)
    if provider is None:
        raise SourceSpecError(
            f"unknown lens fetch source {source!r}; expected "
            + ", ".join(sorted(LENS_PROVIDERS))
        )
    repo = repo_id or provider.repo
    binding = (
        _fetch_neuronpedia(model_id, repo_id=repo, revision=revision)
        if not provider.files
        else _fetch_layout(model_id, source, repo_id=repo, revision=revision)
    )
    path = LENS_SOURCES.binding_path(model_id, binding.name)
    with artifact_lock(path):
        current = load_external_lens_binding(model_id, binding.name)
        if force or current != binding:
            write_json_atomic(path, binding.to_json())
    if activate:
        LENS_SOURCES.select(model_id, SourceRef(HUGGINGFACE, binding.name))
    return binding


def load_external_lens(
    model_id: str, name: str,
) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load one pinned external lens (every bound site) from the provider
    cache, offline-only."""
    binding = load_external_lens_binding(model_id, name)
    if binding is None:
        return None
    try:
        config_file = binding.provenance.get("config_file")
        if isinstance(config_file, str):
            config_path = _hf_hub_download(
                binding.repo_id, config_file,
                revision=binding.repo_revision, local_files_only=True,
            )
            raw = Path(config_path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != binding.provenance["config_sha256"]:
                raise ValueError("cached Neuronpedia config digest changed")
        sites: dict[str, dict[int, torch.Tensor]] = {}
        for site in binding.sites:
            bound = binding.file(site)
            checkpoint_path = _hf_hub_download(
                binding.repo_id, bound.path,
                revision=binding.repo_revision, local_files_only=True,
            )
            matrices, n_prompts, _ = _validated_site_matrices(
                _load_provider_checkpoint(checkpoint_path, binding.kind), binding.kind,
                expected_d_model=binding.d_model,
            )
            if n_prompts != binding.n_prompts or tuple(sorted(matrices)) != bound.layers:
                raise ValueError("cached external lens does not match its binding")
            sites[site] = matrices
        lens = _external_lens(
            binding.kind, sites, n_prompts=binding.n_prompts, d_model=binding.d_model,
        )
    except (OSError, ValueError, RuntimeError, TypeError):
        return None
    return lens, external_lens_sidecar(binding)


def external_lens_sidecar(binding: LensBinding) -> dict[str, Any]:
    """Runtime metadata for a binding, without loading provider weights.

    Shaped like a local manifest where the two agree (``kind``, ``sites``,
    dimensions, corpus, model identity) so one identity/compatibility check
    serves both; ``_source`` carries the provider record.
    """
    return {
        "kind": binding.kind,
        "sites": {
            site: {"layers": list(binding.file(site).layers)} for site in binding.sites
        },
        "n_prompts": binding.n_prompts,
        "d_model": binding.d_model,
        "dtype": "external",
        "corpus_spec": binding.corpus,
        "seq_len": binding.seq_len,
        "skip_first_positions": binding.skip_first,
        "model_fingerprint": None,
        "model_source_fingerprint": binding.model_revision,
        "_source": {
            "kind": HUGGINGFACE,
            "name": binding.name,
            "provider": binding.provider,
            "repo_id": binding.repo_id,
            "repo_revision": binding.repo_revision,
            "model_id": binding.model_id,
            "model_revision": binding.model_revision,
            "files": {
                bound_key.label: {"path": bound.path, "sha256": bound.sha256}
                for bound_key, bound in binding.files.items()
            },
            **binding.provenance,
        },
    }


def load_external_lens_sidecar(model_id: str, name: str) -> dict[str, Any] | None:
    binding = load_external_lens_binding(model_id, name)
    return None if binding is None else external_lens_sidecar(binding)


# -- lifecycle --------------------------------------------------------------

def load_lens_source_sidecar(model_id: str, ref: SourceRef) -> dict[str, Any] | None:
    """A source's metadata: a local manifest or a binding's descriptor."""
    if ref.local:
        from drowse.lens.store import load_local_lens_sidecar

        return load_local_lens_sidecar(model_id, ref.name)
    return load_external_lens_sidecar(model_id, ref.name)


def _sidecar_sites(sidecar: Mapping[str, Any] | None) -> list[str]:
    sites = sidecar.get("sites") if sidecar is not None else None
    if not isinstance(sites, Mapping):
        return []
    return [site for site in LENS_SITES if site in sites]


def list_lens_sources(model_id: str) -> list[dict[str, Any]]:
    """Every prepared source, in default preference order.

    Each row names its kind (``lens``), its family (``family``: ``R`` or
    ``S``, the slot it selects into) and the sites it carries (``sites``;
    empty for a local artifact whose manifest no longer validates).
    """
    actives = {LENS_SOURCES.active(model_id, side) for side in LENS_SIDES}
    rows: list[dict[str, Any]] = []
    for name in LENS_SOURCES.local_names(model_id):
        from drowse.lens.store import load_local_lens_sidecar

        ref = SourceRef(LOCAL, name)
        kind = local_lens_kind(name)
        rows.append({
            "source": LENS_SOURCES.format(ref),
            "kind": LOCAL,
            "name": name,
            "lens": kind,
            "family": lens_family(kind),
            "sites": _sidecar_sites(load_local_lens_sidecar(model_id, name)),
            "active": ref in actives,
            "path": str(LENS_SOURCES.path_of(model_id, ref)),
        })
    for name in sorted(LENS_PROVIDERS):
        binding = load_external_lens_binding(model_id, name)
        if binding is None:
            continue
        ref = SourceRef(HUGGINGFACE, name)
        rows.append({
            "source": name,
            "kind": HUGGINGFACE,
            "name": name,
            "lens": binding.kind,
            "family": lens_family(binding.kind),
            "sites": list(binding.sites),
            "provider": binding.provider,
            "repo_id": binding.repo_id,
            "repo_revision": binding.repo_revision,
            "checkpoints": {site: binding.file(site).path for site in binding.sites},
            "active": ref in actives,
            "path": str(LENS_SOURCES.path_of(model_id, ref)),
        })
    rows.sort(key=lambda row: lens_source_preference_key(str(row["source"])))
    return rows


def use_lens_source(model_id: str, source: str) -> SourceRef:
    """Select a prepared source from the public grammar (into its family's slot)."""
    ref = LENS_SOURCES.parse(source)
    if not LENS_SOURCES.exists(model_id, ref):
        raise FileNotFoundError(
            f"lens source {LENS_SOURCES.format(ref)} is not prepared for "
            f"{model_id} — fit it (local:…) or fetch it "
            f"({', '.join(sorted(LENS_PROVIDERS))})"
        )
    LENS_SOURCES.select(model_id, ref)
    return ref


def remove_lens_source(model_id: str, ref: SourceRef) -> bool:
    """Delete a local lens or forget a binding (provider cache untouched).

    Either way the selection is unpublished only when it named exactly this
    source.
    """
    if ref.local:
        from drowse.lens.store import remove_lens

        return remove_lens(model_id, ref.name)
    path = LENS_SOURCES.binding_path(model_id, ref.name)
    with artifact_lock(path):
        removed = path.exists()
        path.unlink(missing_ok=True)
    LENS_SOURCES.clear_if_active(model_id, ref)
    return removed
