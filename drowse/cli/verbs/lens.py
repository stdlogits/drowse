"""``drowse lens <verb>`` — the per-model lenses: fit, the local/external
source lifecycle (one R-family and one S-family source selected per model),
and the two readouts (``top``, ``decompose``)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drowse.cli.context import (
    CliError, emit_json, note, open_session, parse_layers, progress, user_errors,
)
from drowse.cli.spec import (
    FORCE, JSON, MODEL, MODEL_LOAD, MODEL_REQ, YES, OneOf, Verb, arg,
    layers_arg, positive_int,
)
from drowse.lens.key import (  # torch-free
    DEFAULT_FIT_DOCUMENTS, FITTABLE_KINDS, LENS_SITES, LensKey,
)

if TYPE_CHECKING:
    from drowse.lens.corpus import LensCorpus


#: ``--kind`` choices: the lower-case letter of each fittable
#: :class:`~drowse.lens.key.LensKey` kind, which is also the local artifact
#: name (``local:r``, ``local:j``, ``local:s``, ``local:k``).
LENS_KINDS: tuple[str, ...] = tuple(kind.lower() for kind in FITTABLE_KINDS)

_KIND_TITLES = {
    "R": "R-lens (RelP, R family)",
    "J": "J-lens (standard, R family)",
    "S": "S-lens (RelP, S family)",
    "K": "K-lens (standard, S family)",
}


def _sites(text: str) -> tuple[str, ...]:
    """``--sites``: a comma list of ``out``/``mid``, in canonical order."""
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names or len(set(names)) != len(names) or set(names) - set(LENS_SITES):
        raise argparse.ArgumentTypeError(
            f"sites must be out, mid or out,mid (got {text!r})"
        )
    return tuple(site for site in LENS_SITES if site in names)

_SOURCES_HELP = (
    "local:r, local:j, local:s, local:k (or local:<kind>-<tag>), or a fetched "
    "binding (rs-r, rs-s, workspace-r, neuronpedia, workspace-j)"
)


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

def _pinned_hub_corpus(args: argparse.Namespace) -> "LensCorpus | None":
    """The Hub corpus ``lens fit`` streams (``--hf-corpus``, repeated = a
    weighted mix; unset = the default web-text sample), pinned once for both
    the no-op proof and the stream.  ``None`` for ``--corpus FILE``."""
    from drowse.lens.corpus import parse_lens_corpus, pin_lens_corpus

    if args.corpus is not None:
        return None
    with user_errors(ValueError, exit_code=2):
        corpus = parse_lens_corpus(args.hf_corpus)
    return pin_lens_corpus(corpus)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def lens_fit_is_noop(
    args: argparse.Namespace,
    requested: "list[int] | str | None",
    *,
    docs: list[str] | None = None,
    hf_corpus: "LensCorpus | None" = None,
) -> bool:
    """Prove the requested fit already exists, without loading weights
    (:func:`~drowse.lens.lifecycle.prove_lens_fit_noop`), and report it.

    The corpus identity is ``docs`` for a file corpus, or the pinned
    ``hf_corpus`` spec plus prompt count before streaming.
    """
    from drowse.lens.lifecycle import prove_lens_fit_noop
    from drowse.lens.store import lens_artifact_size, lens_manifest_path
    from drowse.model.fingerprint import model_source_fingerprint

    if args.force or (docs is None and hf_corpus is None):
        return False
    name = args.kind
    sidecar = prove_lens_fit_noop(
        args.model, name,
        requested=requested,
        sites=args.sites,
        seq_len=args.seq_len,
        source_fingerprint=lambda: model_source_fingerprint(
            args.model, quantize=args.quantize, device=args.device,
        ),
        docs=docs,
        corpus_spec=None if hf_corpus is None else hf_corpus.spec,
        prompt_count=int(args.prompts),
    )
    if sidecar is None:
        return False
    size_mb = lens_artifact_size(args.model, name=name) / 1024**2
    note("Already fitted for this corpus and exact model source — nothing to do.")
    _print_fitted(sidecar)
    print(f"Artifact: {lens_manifest_path(args.model, name)} ({size_mb:.0f} MB across layer shards)")
    return True


def _print_fitted(sidecar: dict[str, Any]) -> None:
    """One line per site: layers, documents and the split-half spread."""
    from drowse.lens.store import lens_layers, lens_split_half

    kind = sidecar.get("kind", "?")
    print(
        f"Fitted {_KIND_TITLES.get(kind, kind)}: {sidecar.get('n_prompts', '?')} "
        f"prompts, d_model={sidecar.get('d_model', '?')}"
    )
    for site in sidecar.get("sites", {}):
        layers = lens_layers(sidecar, site)
        line = f"  {kind}:{site}  {len(layers)} layers (L{layers[0]}..L{layers[-1]})"
        split = sorted(lens_split_half(sidecar, site).values())
        if split:
            line += (
                f"  split-half min {split[0]:.3f} · median "
                f"{split[len(split) // 2]:.3f} · max {split[-1]:.3f}"
            )
        print(line)


def run_fit(args: argparse.Namespace) -> None:
    from drowse.lens.corpus import read_corpus_file, stream_lens_corpus
    from drowse.lens.store import (
        lens_artifact_size, lens_manifest_path, load_local_lens_sidecar,
    )

    requested = parse_layers(args.layers)
    if requested == "sample":
        raise CliError(
            "--layers sample is not a wall-time optimization; use --layers "
            "workspace or an explicit comma-separated band"
        )
    kind = args.kind.upper()
    hf_corpus = _pinned_hub_corpus(args)
    if hf_corpus is None:
        path = Path(args.corpus)
        docs = read_corpus_file(path, args.prompts)
        spec = f"file:{path.name}"
        if lens_fit_is_noop(args, requested, docs=docs):
            return
    else:
        if lens_fit_is_noop(args, requested, hf_corpus=hf_corpus):
            return
        note(f"Streaming {args.prompts} documents from {hf_corpus}...")
        docs, spec = stream_lens_corpus(args.prompts, hf_corpus)

    with open_session(args) as session:
        note(f"Fitting the {_KIND_TITLES[LensKey(kind).kind]} at "
             f"{' and '.join(args.sites)}{' from zero (-f)' if args.force else ''}.")
        session.lens.fit(
            docs, kind=kind, sites=args.sites, corpus_spec=spec, layers=requested,
            dim_batch=args.dim_batch, seq_len=args.seq_len, force=args.force,
            checkpoint_every=args.checkpoint_every, split_half=not args.no_split_half,
            on_progress=progress,
        )
    sidecar = load_local_lens_sidecar(args.model, args.kind)
    if sidecar is not None:
        _print_fitted(sidecar)
    size_mb = lens_artifact_size(args.model, name=args.kind) / 1024**2
    print(f"Artifact: {lens_manifest_path(args.model, args.kind)} ({size_mb:.0f} MB across layer shards)")


# ---------------------------------------------------------------------------
# source lifecycle
# ---------------------------------------------------------------------------

def _lens_source(model_id: str, source: str | None) -> Any:
    """``local:<kind>[-<tag>]`` / a binding name / ``None`` (the active
    R-family source, else the active S-family one) → a
    :class:`~drowse.base.source_registry.SourceRef`."""
    from drowse.lens.sources import LENS_SOURCES

    if source is None:
        ref = LENS_SOURCES.active(model_id, "output") or LENS_SOURCES.active(model_id, "input")
        if ref is None:
            raise CliError(f"no lens source for {model_id}", exit_code=1)
        return ref
    with user_errors(ValueError, exit_code=2):
        return LENS_SOURCES.parse(source)


def run_fetch(args: argparse.Namespace) -> None:
    from drowse.lens.sources import LENS_PROVIDERS, fetch_lens_source

    known = sorted(LENS_PROVIDERS)
    if args.source not in known:
        raise CliError(f"source must be one of {', '.join(known)}")
    note(f"Fetching the {args.source} lens for {args.model} into the Hugging Face cache...")
    binding = fetch_lens_source(
        args.model, args.source, repo_id=args.repo, revision=args.revision, force=args.force,
    )
    if args.json_output:
        emit_json(binding.to_json())
        return
    print(
        f"Fetched {binding.name} for {args.model}: {binding.kind}-lens at "
        f"{', '.join(binding.sites)}, {binding.n_prompts} prompts"
    )
    print(f"  provider: {binding.repo_id}@{binding.repo_revision}")
    for site in binding.sites:
        bound = binding.file(site)
        print(f"  {site + ':':<5} {bound.path} ({len(bound.layers)} layers)")
    note("Provider payload remains in the Hugging Face cache; binding is active.")


def run_ls(args: argparse.Namespace) -> None:
    from drowse.lens.sources import list_lens_sources

    rows = list_lens_sources(args.model)
    if args.json_output:
        emit_json(rows)
        return
    if not rows:
        print(f"No lens sources for {args.model}.")
    width = max((len(str(row["source"])) for row in rows), default=0)
    for row in rows:
        sites = ",".join(row["sites"]) or "unreadable"
        print(
            f"{'*' if row['active'] else ' '} {row['source']:<{width}}  "
            f"{row['lens']}-lens  {sites}"
        )


def run_use(args: argparse.Namespace) -> None:
    from drowse.lens.sources import lens_source_side, use_lens_source

    with user_errors(ValueError, exit_code=2):
        ref = use_lens_source(args.model, args.source)
    family = "R" if lens_source_side(ref) == "output" else "S"
    print(f"Active {family}-family lens for {args.model}: {args.source}")


def run_show(args: argparse.Namespace) -> None:
    from drowse.lens.sources import LENS_SOURCES, lens_family, load_lens_source_sidecar
    from drowse.lens.store import lens_artifact_size, lens_layers

    ref = _lens_source(args.model, args.source)
    label = LENS_SOURCES.format(ref)
    sidecar = load_lens_source_sidecar(args.model, ref)
    if sidecar is None:
        raise CliError(f"no lens source {label} for {args.model}", exit_code=1)
    path = LENS_SOURCES.path_of(args.model, ref)
    size_mb = lens_artifact_size(args.model, name=ref.name) / 1024**2 if ref.local else None
    is_active = ref in {LENS_SOURCES.active(args.model, side) for side in ("output", "input")}
    if args.json_output:
        emit_json({
            "model": args.model, "source": label, "path": str(path),
            "size_mb": None if size_mb is None else round(size_mb, 1),
            "active": is_active, **sidecar,
        })
        return
    skip_first = (
        sidecar["estimator_policy"]["skip_first_positions"] if ref.local
        else sidecar.get("skip_first_positions", "?")
    )
    family = lens_family(sidecar["kind"])
    print(
        f"{sidecar['kind']}-lens {label} for {args.model} ({family} family)"
        f"{' (active)' if is_active else ''}"
    )
    for site in ("out", "mid"):
        layers = lens_layers(sidecar, site)
        if layers:
            print(f"  {site + ':':<9} layers {layers[0]}..{layers[-1]} ({len(layers)})")
    print(f"  d_model:  {sidecar['d_model']}")
    print(f"  prompts:  {sidecar.get('n_prompts', '?')}")
    corpus = f"  corpus:   {sidecar.get('corpus_spec', '?')}"
    if sidecar.get("corpus_sha256"):
        corpus += f" (sha256 {str(sidecar['corpus_sha256'])[:12]}…)"
    print(corpus)
    print(f"  seq_len:  {sidecar.get('seq_len', '?')}, skip_first: {skip_first}")
    if size_mb is None:
        meta = sidecar["_source"]
        print(f"  source:   {meta['provider']} ({meta['repo_id']}@{meta['repo_revision']})")
        print(f"  binding:  {path}")
        print("  payload:  Hugging Face cache (provider-owned)")
    else:
        print(f"  artifact: {path} ({size_mb:.0f} MB across layer shards)")


def run_rm(args: argparse.Namespace) -> None:
    from drowse.lens.sources import LENS_SOURCES, remove_lens_source

    ref = _lens_source(args.model, args.source)
    label = LENS_SOURCES.format(ref)
    if not args.yes:
        what = "local lens" if ref.local else "external binding"
        raise CliError(f"refusing to remove {what} {label}; pass -y/--yes to confirm")
    if not remove_lens_source(args.model, ref):
        raise CliError(f"no lens source {label} for {args.model}", exit_code=1)
    if ref.local:
        print(f"Removed Drowse-owned lens {label} for {args.model}.")
    else:
        print(f"Forgot {label} binding; the Hugging Face cache was not modified.")


# ---------------------------------------------------------------------------
# readouts
# ---------------------------------------------------------------------------

def _half_line(half: Any, width: int) -> str:
    """One family's half of a read as ``R:out 0.70  tok tok tok`` (or why it
    is absent)."""
    head = f"{half.lens or half.family}"
    if not half.present:
        return f"{head:<6} {half.weight:.2f}  (absent: {half.absent})"
    tokens = "  ".join(t.token.strip() or repr(t.token) for t in half.tokens[:width])
    return f"{head:<6} {half.weight:.2f}  {tokens}"


def run_top(args: argparse.Namespace) -> None:
    layers = parse_layers(args.layers)
    with open_session(args) as session:
        reads = session.lens.read_prompt(
            args.prompt, site=args.site, layers=layers,
            positions=args.position, k=args.top_k,
        )
    if args.json_output:
        emit_json({
            "model": args.model,
            "prompt": args.prompt,
            "site": args.site,
            "positions": sorted(reads),
            # Per position, one read per layer: the R half (per-layer readout
            # probability p_l) and the S half (footprint z), each with its
            # depth weight and the lens it read through.
            "reads": {
                str(position): [read.to_dict() for read in rows]
                for position, rows in sorted(reads.items())
            },
        })
        return
    for position, rows in sorted(reads.items()):
        if len(reads) > 1:
            print(f"\nposition {position}:")
        for read in rows:
            # Shallow-faithful first: S, then R, as the depth rule orders them.
            print(f"  L{read.layer:>3}  {_half_line(read.s, args.top_k)}")
            print(f"        {_half_line(read.r, args.top_k)}")


def run_decompose(args: argparse.Namespace) -> None:
    layers = parse_layers(args.layers)
    with open_session(args) as session:
        out = session.lens.decompose(
            args.selector, k=args.top_k, layers=layers, dictionary=args.dictionary,
        )
    mean_share = sum(share for share, _ in out.values()) / len(out)
    if args.json_output:
        emit_json({
            "selector": args.selector, "model": args.model, "k": args.top_k,
            "dictionary": args.dictionary,
            "mean_share": round(mean_share, 4),
            "layers": {
                str(layer): {
                    "share": round(share, 4),
                    "tokens": [{"token": t, "coeff": round(c, 4)} for t, c in tokens],
                }
                for layer, (share, tokens) in out.items()
            },
        })
        return
    print(f"{args.dictionary} share of '{args.selector}' (k={args.top_k}):")
    print(f"  mean over {len(out)} layers: {mean_share:.1%}")
    for layer in sorted(out):
        share, tokens = out[layer]
        print(f"  L{layer:>3}  {share:>6.1%}  {'  '.join(t.strip() or repr(t) for t, _ in tokens[:6])}")


_MODEL_POS = arg("model", help="Model ID")
_SOURCE_OPT = arg("source", nargs="?", default=None, help=f"{_SOURCES_HELP} (default: active)")

GROUP = Verb(
    "lens",
    "Jacobian lenses (fit/fetch/ls/show/use/top/decompose/rm)",
    description="Per-model Jacobian lenses: R/J read what a residual is disposed "
                "to say, S/K what it has read. Fit or fetch them, read a prompt "
                "through the depth rule (S early, R late), and split directions "
                "against token dictionaries.",
    verbs=(
        Verb(
            "fit", "Fit one of the model's lenses over a text corpus",
            run=run_fit,
            description=(
                "Fit a lens per layer over a text corpus by transport: one "
                "forward plus ceil(d_model/dim_batch) passes through every "
                "block per document. --kind r (default) and j read what an "
                "activation makes the model say (J_l = E[dh_final/dh_l], RelP "
                "or standard); s and k what put it there (K_l = E[dh_l/de]). "
                "--sites picks the block output (out), the MLP input (mid) or "
                "both. A fit records its split-half difference per layer "
                "(unless it extends a finished fit's corpus, or --no-split-half); "
                "interrupted fits checkpoint and resume; an exact repeat is "
                "proven without loading the model."
            ),
            args=(
                MODEL,
                OneOf((
                    arg("--corpus", default=None, metavar="FILE",
                        help="One document per line (JSONL 'text' lines work too). "
                             "Default: stream FineWeb-Edu sample-10BT (needs "
                             "`pip install 'drowse.ai[hf]'`)"),
                    arg("--hf-corpus", action="append", default=None,
                        metavar="REPO[:CONFIG][@REV][,KEY=VALUE...]",
                        help="Stream a Hub dataset's train split instead, pinned to "
                             "a commit. Options: text_field=F, files=K (round-robin "
                             "over K data files), seed=S; repeat with weight=W to mix"),
                )),
                arg("--kind", choices=sorted(LENS_KINDS), default="r",
                    help="Lens: r = RelP R-lens (default), j = standard J-lens, "
                         "s = RelP S-lens, k = standard K-lens"),
                arg("--sites", type=_sites, default=LENS_SITES, metavar="out,mid",
                    help="Residual sites to fit: out (block output), mid (MLP "
                         "input), or both (default)"),
                arg("--prompts", type=positive_int, default=DEFAULT_FIT_DOCUMENTS, metavar="N",
                    help=f"Corpus documents to average over (default {DEFAULT_FIT_DOCUMENTS}; "
                         "the research lenses used 1000)"),
                arg("--seq-len", type=positive_int, default=None, metavar="T",
                    help="Truncate each document to T tokens (default 128)"),
                arg("--dim-batch", type=positive_int, default=None, metavar="K",
                    help="Basis seeds per transport pass (default 256; halves on OOM)"),
                arg("--checkpoint-every", type=positive_int, default=None, metavar="N",
                    help="Resumable checkpoint every N usable prompts (default 25)"),
                arg("--no-split-half", action="store_true",
                    help="Skip the split-half check (saves one lens of RAM)"),
                layers_arg("Fit only these layers, per site (default: every layer; "
                           "the out site of r/j stops below the last block); "
                           "workspace = the 40-90%% band. An output fit stops at "
                           "its lowest layer, an input fit at its highest"),
                FORCE.but(help="Restart from zero, replacing the local lens of this kind "
                               "(default: resume a matching partial fit; without -f a fit "
                               "never replaces one over more documents or layers)"),
                MODEL_LOAD,
            ),
        ),
        Verb(
            "fetch", "Fetch an external lens into its provider cache",
            run=run_fetch,
            description="Pin an external lens's repository and base-model revisions "
                        "and write only a small binding (one file per site it "
                        "publishes); the payload stays in the Hugging Face cache "
                        "and the binding becomes its family's active source. No "
                        "model load.",
            args=(
                _MODEL_POS,
                arg("source", nargs="?", default="neuronpedia",
                    help="neuronpedia (default), rs-r / rs-s (stdlogits/rs-lenses' "
                         "R- and S-lens, out and mid, Qwen3-8B and Gemma 4 12B), "
                         "workspace-r (RelP), or workspace-j"),
                arg("--revision", default="main", help="Provider revision to resolve and pin"),
                arg("--repo", default=None, metavar="REPO",
                    help="Override the provider repository"),
                FORCE.but(help="Refresh the binding"),
                JSON,
            ),
        ),
        Verb("ls", "List local and external lens sources for a model", run=run_ls,
             args=(_MODEL_POS, JSON)),
        Verb("show", "Show the active or selected lens source", run=run_show,
             args=(_MODEL_POS, _SOURCE_OPT, JSON)),
        Verb("use", "Select an already available lens source", run=run_use,
             args=(_MODEL_POS, arg("source", help=_SOURCES_HELP))),
        Verb(
            "top", "Read a prompt's residuals per layer through the depth rule",
            run=run_top,
            description="One forward pass on a raw prompt (no chat template); each "
                        "layer's residual at the site is read through both lens "
                        "families: the S read (footprint z, faithful early) and "
                        "the R read (softmax(W_U · norm(J_l h)), faithful late), "
                        "each with its depth weight and the lens it used.",
            args=(
                MODEL,
                arg("prompt", help="Raw prompt to read out"),
                arg("-k", "--top-k", type=positive_int, default=8, metavar="K",
                    help="Tokens per (layer, position) read and family (default 8)"),
                layers_arg("Layer indices or a mode (default: every layer a lens "
                           "holds at the site)",
                           metavar="L1,L2-L3,...|sample|workspace|all"),
                arg("--position", type=int, action="append", default=None, metavar="P",
                    help="Token position to read (repeatable; negative from the end; "
                         "default: final)"),
                arg("--site", choices=("out", "mid"), default="out",
                    help="out = each block's output (default); mid = the residual "
                         "after its attention, the MLP's input"),
                MODEL_LOAD, JSON,
            ),
        ),
        Verb(
            "decompose", "Split a steerable direction against a token dictionary",
            run=run_decompose,
            description="Best sparse combination of k atoms per layer: the "
                        "variance share and the contributing atoms. tokens-out "
                        "(default) is W_U J_l through the R-family lens (how "
                        "verbalizable the direction is); tokens-in is the S-family "
                        "footprint K_l e_v over the token mask (which tokens it "
                        "reads as having seen); neurons is the layer's neuron "
                        "writes, signed (the sign is the pole; `unit find` adds "
                        "labels).",
            args=(
                arg("selector", help="Steerable direction (concept, ns/name, "
                                     "jlens/<word> or slens/<word>)"),
                MODEL_REQ,
                arg("-k", "--top-k", type=positive_int, default=16, metavar="K",
                    help="Sparsity budget (default 16)"),
                arg("--dictionary", choices=("tokens-out", "tokens-in", "neurons"),
                    default="tokens-out", help="Atom dictionary (default tokens-out)"),
                layers_arg("Layer indices or a mode (default: every fitted layer "
                           "the direction covers)"),
                MODEL_LOAD, JSON,
            ),
        ),
        Verb(
            "rm", "Remove a local lens or forget an external binding",
            run=run_rm,
            description="Delete a local lens, or forget an external binding without "
                        "touching its provider cache.",
            args=(_MODEL_POS, _SOURCE_OPT, YES),
        ),
    ),
)
