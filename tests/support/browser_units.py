"""The ``units-tiny-models-v1`` fixture and the hook ABI v5 goldens.

drowse.ai runs units, token anatomy, the ``mid`` site and the S lens on
WebGPU (hook ABI ``block-sites-v5``).  These goldens are what its compiled
libraries and host code are built and checked against, computed here by the
engine itself on two tiny real architectures (``gemma3_text`` and ``qwen3``)
whose weights come from a portable integer generator, so drowse.ai rebuilds
the same tensors in TypeScript or NumPy without reading a checkpoint.

One builder per golden (``build_<name>(arch)``) returns the file's JSON;
:func:`write_goldens` runs every builder, enforces the validator (finite
values, margins at every top-k boundary, size caps) and writes the files.
``tests/browser/test_browser_parity.py`` rebuilds each golden and compares
within the file's own ``tolerance`` block.  Regenerate after a deliberate
semantic change with ``PYTHONPATH=. python -m tests.support.browser_units
--write`` (a seam change: drowse.ai's runtime must match before its pin moves).
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from tests.support.tiny_models import byte_tokenizer

GOLDENS = Path(__file__).resolve().parents[1] / "fixtures" / "browser"
MODEL_FILE = "units-tiny-models-v1.json"
FIXTURE_MODELS = "units-tiny-models-v1"
HOOK_ABI = "block-sites-v5"
SCHEMA_VERSION = 1
ARCHES = ("gemma3", "qwen3")
FAMILY = {"gemma3": "gemma3_text", "qwen3": "qwen3"}
#: The generator's seed (B1-goldens §1.2).
SEED = 0x5EED0005
#: Size caps: every file, and the whole set.
FILE_CAP = 256 * 1024
TOTAL_CAP = 2 * 1024 * 1024
#: Arrays up to this many numbers are JSON lists; longer ones base64.
INLINE_MAX = 256
#: The validator's margin: ``|a − b| ≥ MARGIN · max(1, |a|, |b|)``.
MARGIN = 1e-3

#: Every golden carries its comparison policy (B1-goldens §3).
TOLERANCE: dict[str, Any] = {
    "python": {"atol": 1e-6, "rtol": 1e-5},
    "library": {"atol": 1e-5, "rtol": 1e-4, "probability_atol": 1e-5, "z_atol": 1e-4},
    "rounded": {"atol": 2e-6},
    "ids": "exact",
}

_MASK = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# The portable generator (32-bit integer arithmetic only)
# ---------------------------------------------------------------------------


def _h(x: np.ndarray) -> np.ndarray:
    """The lowbias32 mix, elementwise over uint64 arrays holding uint32s."""
    x = x & _MASK
    x ^= x >> 16
    x = (x * 0x7FEB352D) & _MASK
    x ^= x >> 15
    x = (x * 0x846CA68B) & _MASK
    x ^= x >> 16
    return x


def _seed(ordinal: int) -> int:
    return int(_h(np.array([(SEED + 0x9E3779B9 * ordinal) & _MASK], dtype=np.uint64))[0])


def _u(ordinal: int, count: int) -> np.ndarray:
    seed = _seed(ordinal)
    return _h(np.arange(count, dtype=np.uint64) ^ np.uint64(seed))


def quantized_tensor(ordinal: int, shape: Sequence[int], exponent: int) -> torch.Tensor:
    """A q4-exact fp32 tensor: groups of 32 along the last axis, a ±7 in
    every group, power-of-two group scales ``2^−(E + bit)``."""
    count = int(np.prod(shape))
    last = int(shape[-1])
    if last % 32:
        raise ValueError(f"a quantized tensor's last axis must be a multiple of 32, got {last}")
    u = _u(ordinal, count)
    q = (u % 15).astype(np.int64) - 7
    first = np.arange(count) % 32 == 0
    q[first] = np.where(u[first] >= 0x80000000, 7, -7)
    groups = np.arange(count // 32, dtype=np.uint64)
    bits = _h(groups ^ np.uint64(_seed(ordinal) ^ 0x80000000)) % 2
    scale = np.ldexp(1.0, -(exponent + bits.astype(np.int64)))
    w = q.astype(np.float64) * np.repeat(scale, 32)
    return torch.from_numpy(w.astype(np.float32).reshape(tuple(shape)))


def norm_tensor(ordinal: int, shape: Sequence[int], arch: str) -> torch.Tensor:
    """An exact fp32 norm weight: Gemma's ``1 + w`` offsets in ``[−¼, ¼)``,
    Qwen3's plain gains in ``[¾, 1¼)``."""
    u = _u(ordinal, int(np.prod(shape)))
    frac = (u >> 9).astype(np.float64) * 2.0**-24
    w = frac - 0.25 if arch == "gemma3" else 0.75 + frac
    return torch.from_numpy(w.astype(np.float32).reshape(tuple(shape)))


def lens_matrix(ordinal: int, d: int) -> torch.Tensor:
    """``0.75·δ_ij + ((u >> 16) mod 256 − 128)·2⁻¹⁰``, exact in fp32."""
    u = _u(ordinal, d * d)
    noise = (((u >> 16) % 256).astype(np.float64) - 128.0) * 2.0**-10
    m = 0.75 * np.eye(d) + noise.reshape(d, d)
    return torch.from_numpy(m.astype(np.float32))


def q4_round_trip(w: torch.Tensor) -> torch.Tensor:
    """A symmetric 4-bit quantizer's round trip over groups of 32 along the
    last axis (``scale = max|w| / 7``, ``q = round(w / scale) + 7``)."""
    groups = w.double().reshape(-1, 32)
    scale = groups.abs().amax(-1, keepdim=True) / 7.0
    q = torch.round(groups / scale) + 7.0
    return ((q - 7.0) * scale).float().reshape(w.shape)


# ---------------------------------------------------------------------------
# The fixture models
# ---------------------------------------------------------------------------

_D, _FF, _LAYERS, _HEADS, _KV, _HD = 64, 96, 4, 4, 2, 16
#: Per tensor role, its scale exponent ``E_t`` (tuned until the validator
#: passes, then frozen; recorded per tensor in the model file).
_EXPONENTS = {
    "embed_tokens": 5, "q_proj": 5, "k_proj": 5, "v_proj": 5, "o_proj": 5,
    "gate_proj": 5, "up_proj": 5, "down_proj": 5,
}


def fixture_config(arch: str) -> Any:
    tokenizer = byte_tokenizer(FAMILY[arch])
    common: dict[str, Any] = dict(
        vocab_size=len(tokenizer), hidden_size=_D, intermediate_size=_FF,
        num_hidden_layers=_LAYERS, num_attention_heads=_HEADS, num_key_value_heads=_KV,
        head_dim=_HD, max_position_embeddings=4096, tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if arch == "gemma3":
        config = transformers.AutoConfig.for_model(
            "gemma3_text", **common, query_pre_attn_scalar=16, sliding_window=4,
            layer_types=["sliding_attention", "full_attention"] * 2,
        )
    else:
        config = transformers.AutoConfig.for_model("qwen3", **common)
    config._attn_implementation = "eager"
    return config


def tensor_table(model: Any) -> list[tuple[str, tuple[int, ...]]]:
    """The model's own tensors in state-dict order (the tied head left out)."""
    seen: set[int] = set()
    out = []
    for name, tensor in model.state_dict().items():
        if name == "lm_head.weight" or id(tensor) in seen or not name.endswith("weight"):
            continue
        out.append((name, tuple(int(d) for d in tensor.shape)))
    return out


def _role(name: str) -> str:
    return name.rsplit(".", 2)[-2]


def build_model(arch: str) -> tuple[Any, Any, list[dict[str, Any]]]:
    """``(model, tokenizer, tensors)``: the fixture model with every weight
    from the generator, and its tensor list (name, shape, kind, exponent)."""
    tokenizer = byte_tokenizer(FAMILY[arch])
    config = fixture_config(arch)
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.eval().requires_grad_(False)
    params = dict(model.named_parameters())
    rows: list[dict[str, Any]] = []
    for ordinal, (name, shape) in enumerate(tensor_table(model)):
        role = _role(name)
        if role in _EXPONENTS:
            value = quantized_tensor(ordinal, shape, _EXPONENTS[role])
            if not torch.equal(q4_round_trip(value), value):
                raise AssertionError(f"{name} is not q4-exact")
            kind, exponent = "q4", _EXPONENTS[role]
        else:
            value, kind, exponent = norm_tensor(ordinal, shape, arch), "norm", None
        params[name].copy_(value)
        rows.append({
            "name": name, "shape": list(shape), "kind": kind, "exponent": exponent,
            "sha256": hashlib.sha256(value.numpy().astype("<f4").tobytes()).hexdigest(),
        })
    model.tie_weights()
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    return model, tokenizer, rows


def write_checkpoint(arch: str, path: Path) -> Path:
    model, tokenizer, _rows = build_model(arch)
    model.config._name_or_path = str(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


# ---------------------------------------------------------------------------
# Scripts and corpora (B1-goldens §1.3)
# ---------------------------------------------------------------------------

#: The raw text of the short prompt (after the family's BOS, when it has one).
SHORT_TEXT = {"gemma3": "Kp=q;Wx7&zR", "qwen3": "Kp=q;Wx7&zR!"}
#: The two-turn chat prompt.
CHAT = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi there"},
]
#: The generated node's decode budget (greedy).
DECODE_STEPS = 6
#: How many neutral statements (and baseline prompts) the fixture uses.
NEUTRAL_COUNT = 24
#: The token-count mask drops these byte symbols (counted 0 times).
DROPPED_SYMBOLS = "QXZqxz#$%^`|"


def neutral_corpus() -> tuple[list[str], list[str]]:
    """``(statements, baseline prompts)``: the first :data:`NEUTRAL_COUNT` of
    drowse's bundled ones, frozen into the model file."""
    from importlib import resources

    def bundled(name: str) -> list[str]:
        with resources.files("drowse.data").joinpath(name).open() as f:
            return list(json.load(f))[:NEUTRAL_COUNT]

    return bundled("neutral_statements.json"), bundled("baseline_prompts.json")


def short_prompt_ids(tokenizer: Any, arch: str) -> list[int]:
    """The 12-position raw prompt: BOS where the family has one, then bytes."""
    ids = list(tokenizer.encode(SHORT_TEXT[arch], add_special_tokens=False))
    if tokenizer.bos_token_id is not None:
        ids = [int(tokenizer.bos_token_id), *ids]
    if len(ids) != 12:
        raise AssertionError(f"the short prompt is {len(ids)} positions, not 12")
    return ids


# ---------------------------------------------------------------------------
# Encoding and the validator
# ---------------------------------------------------------------------------


def _f32(x: float) -> float:
    """``x`` as the shortest decimal that reads back as the same float32."""
    return float(str(np.float32(x)))


def encode(value: Any) -> Any:
    """A tensor or array as JSON: a list up to :data:`INLINE_MAX` numbers,
    else a little-endian base64 record."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    integral = np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_
    if array.size <= INLINE_MAX:
        flat = [int(x) for x in array.reshape(-1)] if integral else [_f32(x) for x in array.reshape(-1)]
        if array.ndim <= 1:
            return flat
        return {"shape": list(array.shape), "values": flat}
    if integral:
        data, dtype, encoding = array.astype("<i4").tobytes(), "int32", "int32-le-base64"
    else:
        data, dtype, encoding = array.astype("<f4").tobytes(), "float32", "float32-le-base64"
    return {
        "dtype": dtype, "encoding": encoding, "shape": list(array.shape),
        "data": base64.b64encode(data).decode("ascii"),
    }


def decode(value: Any) -> np.ndarray:
    """The inverse of :func:`encode`."""
    if isinstance(value, dict) and "encoding" in value:
        dtype = "<i4" if value["dtype"] == "int32" else "<f4"
        raw = np.frombuffer(base64.b64decode(value["data"]), dtype=dtype)
        return raw.reshape(value["shape"])
    if isinstance(value, dict) and "values" in value:
        return np.asarray(value["values"]).reshape(value["shape"])
    return np.asarray(value)


class FixtureError(AssertionError):
    """The validator refused a golden (it would not discriminate)."""


def _margin(a: float, b: float) -> bool:
    return abs(a - b) >= MARGIN * max(1.0, abs(a), abs(b))


def require_top_margin(values: Sequence[float], k: int, what: str) -> None:
    """``values`` sorted descending: consecutive returned values and the
    k-th against the best excluded keep the margin."""
    take = min(int(k), len(values))
    for i in range(take - 1 + (1 if take < len(values) else 0)):
        if not _margin(float(values[i]), float(values[i + 1])):
            raise FixtureError(
                f"{what}: entries {i} and {i + 1} tie within the margin "
                f"({values[i]!r} vs {values[i + 1]!r})"
            )


def require_finite(tree: Any, what: str, *, limit: float | None = None) -> None:
    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict) and "encoding" in node and node.get("dtype") == "float32":
            array = decode(node)
            if not np.isfinite(array).all():
                raise FixtureError(f"{what}{path}: non-finite value")
            if limit is not None and np.abs(array).max(initial=0) > limit:
                raise FixtureError(f"{what}{path}: |x| > {limit}")
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(child, f"{path}.{key}")
        elif isinstance(node, list):
            for i, child in enumerate(node):
                walk(child, f"{path}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            raise FixtureError(f"{what}{path}: non-finite value")

    walk(tree, "")


def header(name: str, arch: str | None, model_sha256: str | None) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "golden": name,
        "hookAbi": HOOK_ABI,
        "arch": arch,
        "fixtureModels": {"name": FIXTURE_MODELS, "sha256": model_sha256},
        "tolerance": TOLERANCE,
    }


# ---------------------------------------------------------------------------
# Comparison (the parity tests)
# ---------------------------------------------------------------------------


def _close(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> bool:
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= atol + rtol * np.abs(a)))


def compare(expected: Any, actual: Any, *, path: str = "", atol: float = 1e-6, rtol: float = 1e-5) -> list[str]:
    """Every difference between two goldens, values within the tolerance,
    ints, strings, booleans and structure exactly."""
    problems: list[str] = []
    if isinstance(expected, dict) and ("encoding" in expected or "values" in expected):
        if not isinstance(actual, dict):
            return [f"{path}: expected an array"]
        a, b = decode(expected), decode(actual)
        if expected.get("dtype") == "int32" or (a.dtype.kind in "iu" and b.dtype.kind in "iu"):
            if a.shape != b.shape or not np.array_equal(a, b):
                problems.append(f"{path}: integer arrays differ")
        elif not _close(a.astype(np.float64), b.astype(np.float64), atol, rtol):
            worst = float(np.max(np.abs(a - b))) if a.shape == b.shape else math.inf
            problems.append(f"{path}: arrays differ (max |Δ| {worst:.3g})")
        return problems
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            return [f"{path}: keys {sorted(expected)} != {sorted(actual) if isinstance(actual, dict) else actual!r}"]
        for key in expected:
            problems += compare(expected[key], actual[key], path=f"{path}.{key}", atol=atol, rtol=rtol)
        return problems
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [f"{path}: list lengths differ"]
        for i, (x, y) in enumerate(zip(expected, actual, strict=True)):
            problems += compare(x, y, path=f"{path}[{i}]", atol=atol, rtol=rtol)
        return problems
    if isinstance(expected, float) or isinstance(actual, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not isinstance(expected, (int, float)):
            return [f"{path}: {expected!r} != {actual!r}"]
        if not abs(float(expected) - float(actual)) <= atol + rtol * abs(float(expected)):
            return [f"{path}: {expected!r} != {actual!r}"]
        return []
    if expected != actual:
        return [f"{path}: {expected!r} != {actual!r}"]
    return []


# ---------------------------------------------------------------------------
# The fixture session
# ---------------------------------------------------------------------------

#: The hand-built lenses (B1-goldens §1.3): every read meets a present half,
#: an absent half and the logit-lens identity (R at the last block's ``out``).
LENS_LAYERS: dict[tuple[str, str], tuple[int, ...]] = {
    ("R", "out"): (0, 1), ("R", "mid"): (1, 3), ("S", "out"): (0, 1), ("S", "mid"): (0, 2),
}
#: The crossover band set on the lens instrument.
BAND = (0.2, 0.8)
_LENS_ORDINAL = 1000


def fixture_lens(kind: str) -> Any:
    from drowse.lens.jlens import JacobianLens
    from drowse.lens.key import LensKey

    tables: dict[str, dict[int, torch.Tensor]] = {"out": {}, "mid": {}}
    for i, ((family, site), layers) in enumerate(sorted(LENS_LAYERS.items())):
        if family != kind:
            continue
        for layer in layers:
            tables[site][layer] = lens_matrix(_LENS_ORDINAL + 16 * i + layer, _D)
    return JacobianLens(tables["out"], mid=tables["mid"], n_prompts=3, d_model=_D, key=LensKey(kind))


@dataclass
class Fixture:
    """One arch's live fixture: the session on its checkpoint, inside a
    scratch ``$DROWSE_HOME`` with the fixture corpus, lenses, band and token
    counts in place."""

    arch: str
    session: Any
    home: Path

    @property
    def tokenizer(self) -> Any:
        return self.session.tokenizer

    @property
    def spec(self) -> Any:
        from drowse.model.blocks import model_spec

        return model_spec(self.session.model)


@contextmanager
def _deterministic() -> Iterator[None]:
    threads = torch.get_num_threads()
    previous = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.set_num_threads(threads)


@contextmanager
def fixture(arch: str, *, lenses: bool = True) -> Iterator[Fixture]:
    """The arch's fixture session (CPU, fp32, no probes) in a scratch home."""
    from drowse.lens.depth import LensBand
    from drowse.lens.store import LensProvenance, save_lens
    from drowse.lens.token_counts import MIN_COUNT_TOKENS, tokenizer_fingerprint, write_corpus_counts
    from drowse.model.fingerprint import loaded_model_fingerprint
    from drowse.session.session import DrowseSession
    from tests.support.sessions import drowse_home

    statements, prompts = neutral_corpus()
    with tempfile.TemporaryDirectory(prefix=f"drowse-units-{arch}-") as tmp, _deterministic():
        root = Path(tmp)
        home = root / "home"
        home.mkdir()
        (home / "neutral_statements.json").write_text(json.dumps(statements))
        (home / "baseline_prompts.json").write_text(json.dumps(prompts))
        checkpoint = write_checkpoint(arch, root / f"units-tiny-{arch}")
        with drowse_home(home):
            session = DrowseSession.from_pretrained(
                str(checkpoint), device="cpu", dtype="float32", probes=[],
            )
            try:
                if lenses:
                    rows = int(session.model.get_input_embeddings().weight.shape[0])
                    counts = torch.full((rows,), 1_000, dtype=torch.int64)
                    for symbol in DROPPED_SYMBOLS:
                        counts[session.tokenizer.convert_tokens_to_ids(symbol)] = 0
                    write_corpus_counts(
                        session.model_id, counts, corpus="units-tiny-fixture", documents=24,
                        tokens=MIN_COUNT_TOKENS, tokenizer=tokenizer_fingerprint(session.tokenizer),
                    )
                    for kind in ("R", "S"):
                        save_lens(fixture_lens(kind), session.model_id, LensProvenance(
                            corpus_spec="units-tiny-fixture", corpus_sha256="0" * 64, seq_len=8,
                            dim_batch=1,
                            model_fingerprint=loaded_model_fingerprint(session.model, session.model_id),
                        ))
                        session.lens.select_source(f"local:{kind.lower()}")
                    session.lens.band = LensBand(*BAND)
                yield Fixture(arch, session, home)
            finally:
                session.close()


# ---------------------------------------------------------------------------
# Helpers the builders share
# ---------------------------------------------------------------------------


def _trigger(trigger: Any) -> dict[str, Any]:
    gate = trigger.gate
    return {
        "prompt": trigger.prompt, "generated": trigger.generated,
        "thinking": trigger.thinking, "response": trigger.response,
        "first_n": trigger.first_n, "after_n": trigger.after_n,
        "gate": None if gate is None else str(gate),
    }


def _unit_tables(fx: Fixture) -> list[dict[str, Any]]:
    """The installed unit-ablation hooks, as the browser's tables: per
    module its groups' ``keep``/``bias`` (fp32) and trigger."""
    from drowse.steering.hooks import UnitHook

    tables = []
    for block in fx.spec.blocks:
        for site, module in (("act", block.mlp.down), ("z", block.attn.o)):
            for hook in module._forward_pre_hooks.values():
                owner = getattr(hook, "__self__", None)
                if not isinstance(owner, UnitHook):
                    continue
                for group in owner.groups:
                    keep = group.op.keep.float().cpu()
                    bias = group.op.bias.float().cpu()
                    touched = (keep != 1) | (bias != 0)
                    tables.append({
                        "layer": block.layer, "site": site, "width": int(keep.shape[0]),
                        "trigger": _trigger(group.trigger),
                        "entries": [int(i) for i in touched.nonzero().reshape(-1)],
                        "keep": encode(keep[touched]), "bias": encode(bias[touched]),
                    })
    return tables


@contextmanager
def _capture_blocks(fx: Fixture, *, positions: Sequence[int] | None) -> Iterator[dict[str, list[torch.Tensor]]]:
    """Every forward's block outputs (after the steering hooks) and its
    last-position logits: ``{"out": [[L, n, d] per forward], "logits": [[V]]}``.
    ``positions`` picks absolute rows (``None``: every row of the forward)."""
    store: dict[str, list[torch.Tensor]] = {"out": [], "logits": []}
    blocks = fx.spec.blocks
    pending: list[torch.Tensor] = []

    def block_hook(_m: Any, _args: Any, output: Any) -> None:
        h = output[0] if isinstance(output, tuple) else output
        if len(pending) == len(blocks):  # a forward that never reached the head
            pending.clear()
        pending.append(h[0].detach().float().clone())

    def head_hook(_m: Any, _args: Any, output: Any) -> None:
        store["logits"].append(output[0, -1].detach().float().clone())
        store["out"].append(torch.stack(pending))
        pending.clear()

    handles = [block.module.register_forward_hook(block_hook) for block in blocks]
    handles.append(fx.session.model.get_output_embeddings().register_forward_hook(head_hook))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


def _forward(fx: Fixture, ids: Sequence[int]) -> None:
    with torch.inference_mode():
        fx.session.model(input_ids=torch.tensor([list(ids)]))


# ---------------------------------------------------------------------------
# 1. The fixture models
# ---------------------------------------------------------------------------


def build_units_tiny_models() -> dict[str, Any]:
    """The weights' definition, digests and scripts; integers and hashes only,
    so the file is byte-identical on every platform."""
    statements, prompts = neutral_corpus()
    arches: dict[str, Any] = {}
    for arch in ARCHES:
        model, tokenizer, rows = build_model(arch)
        vocab = json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
        chat = tokenizer.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True)
        chat_ids = list(tokenizer.encode(chat, add_special_tokens=False))
        config = model.config.to_dict()
        keep = (
            "model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "rms_norm_eps",
            "tie_word_embeddings", "hidden_activation", "hidden_act", "query_pre_attn_scalar",
            "sliding_window", "layer_types", "rope_parameters", "attn_logit_softcapping",
            "final_logit_softcapping", "max_position_embeddings",
        )
        arches[arch] = {
            "family": FAMILY[arch],
            "config": {key: config[key] for key in keep if key in config},
            "tensors": rows,
            "tokenizer": {
                "size": len(tokenizer), "vocab_sha256": hashlib.sha256(vocab).hexdigest(),
                "specials": list(tokenizer.all_special_tokens),
                "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id,
                "pad": tokenizer.pad_token_id,
            },
            "short_prompt": short_prompt_ids(tokenizer, arch),
            "chat_prompt": {"messages": CHAT, "ids": chat_ids, "positions": len(chat_ids)},
            "lenses": {
                f"{family}:{site}": list(layers) for (family, site), layers in sorted(LENS_LAYERS.items())
            },
        }
    return {
        **header("units-tiny-models", None, None),
        "generator": {
            "seed": SEED, "hash": "lowbias32", "golden_ratio": 0x9E3779B9,
            "group": 32, "group_scale_tag": 0x80000000,
            "q4": "q = (u mod 15) − 7, group's first element ±7 by u ≥ 2³¹; scale 2^−(E + H(seed ^ tag ^ g) mod 2)",
            "norm": {"gemma3": "(u >> 9)·2⁻²⁴ − 0.25", "qwen3": "0.75 + (u >> 9)·2⁻²⁴"},
            "lens": "0.75·δ_ij + ((u >> 16) mod 256 − 128)·2⁻¹⁰, ordinal 1000 + 16·i + layer over the sorted (family, site) list",
        },
        "arches": arches,
        "neutral": {"statements": statements, "baseline_prompts": prompts},
        "token_counts": {"dropped": list(DROPPED_SYMBOLS), "count": 1000},
        "band": list(BAND),
        "decode_steps": DECODE_STEPS,
    }


def _flat(value: Any) -> bool:
    if isinstance(value, dict):
        return all(not isinstance(v, (dict, list)) or (isinstance(v, list) and _flat(v)) for v in value.values())
    if isinstance(value, list):
        return all(not isinstance(v, (dict, list)) for v in value)
    return True


def _dump(value: Any, depth: int) -> str:
    """Indented JSON whose scalar lists and flat records sit on one line."""
    if _flat(value):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    pad = " " * (depth + 1)
    if isinstance(value, dict):
        items = [f"{pad}{json.dumps(k)}: {_dump(v, depth + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * depth + "}"
    items = [f"{pad}{_dump(v, depth + 1)}" for v in value]
    return "[\n" + ",\n".join(items) + "\n" + " " * depth + "]"


def _serialize(obj: Any) -> bytes:
    return (_dump(obj, 0) + "\n").encode("utf-8")


def model_file_sha256() -> str:
    return hashlib.sha256(_serialize(build_units_tiny_models())).hexdigest()


# ---------------------------------------------------------------------------
# 2. block-sites
# ---------------------------------------------------------------------------

SITES = ("block_in", "z", "attn_o", "mid", "act", "mlp_o", "out")


def build_block_sites(fx: Fixture) -> dict[str, Any]:
    """Every in-block site at every position of the short raw prompt
    (``tap_token`` per position, one cacheless forward each)."""
    from drowse.loom.node_replay import tap_token

    spec = fx.spec
    ids = short_prompt_ids(fx.tokenizer, fx.arch)
    tensor = torch.tensor([ids])
    sites: dict[str, list[list[torch.Tensor]]] = {site: [] for site in SITES}
    with torch.inference_mode():
        for p in range(len(ids)):
            taps = tap_token(fx.session.model, spec, tensor, p, patterns=False)
            row = {
                "block_in": list(taps.block_in), "z": taps.z, "attn_o": taps.attn_out,
                "mid": list(taps.mid), "act": taps.acts, "mlp_o": taps.mlp_out, "out": list(taps.out),
            }
            for site in SITES:
                sites[site].append([t.float().cpu() for t in row[site]])
        logits = fx.session.model(input_ids=tensor).logits[0, -1].float()
    stacked = {
        site: torch.stack([torch.stack(per_pos) for per_pos in values]).transpose(0, 1)  # [L, T, ·]
        for site, values in sites.items()
    }
    checks = []
    for block in spec.blocks:
        L = block.layer
        attn = stacked["attn_o"][L]
        mlp = stacked["mlp_o"][L]
        if block.attn_out_norm is not None:
            attn = block.attn_out_norm.apply(attn)
            mlp = block.mlp_out_norm.apply(mlp)
        checks.append({
            "layer": L,
            "mid_residual": _f32(float((stacked["mid"][L] - stacked["block_in"][L] - attn).abs().max())),
            "out_residual": _f32(float((stacked["out"][L] - stacked["mid"][L] - mlp).abs().max())),
        })
    return {
        **header("block-sites", fx.arch, model_file_sha256()),
        "ids": ids,
        "sites": {site: encode(value) for site, value in stacked.items()},
        "logits_last": encode(logits),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# 3. unit-stats
# ---------------------------------------------------------------------------


def build_unit_stats(fx: Fixture) -> dict[str, Any]:
    """The neutral unit statistics, the reading scales and the mid/out means."""
    from drowse.units.stats import typical_scale

    runtime = fx.session.units.reads.runtime
    stats = runtime.stats()
    scales = runtime.scales()
    layers = sorted(stats.layers)
    lens = fx.session.lens

    def stack(get: Callable[[int], torch.Tensor]) -> Any:
        return encode(torch.stack([get(L).float() for L in layers]))

    neuron_tops = torch.cat([scales.layer(L).neuron_top.reshape(-1) for L in layers])
    head_tops = torch.cat([scales.layer(L).head_top.reshape(-1) for L in layers])
    return {
        **header("unit-stats", fx.arch, model_file_sha256()),
        "n_documents": stats.n_documents,
        "neuron_mean": stack(lambda L: stats.layer(L).neuron_mean),
        "neuron_scale": stack(lambda L: stats.layer(L).neuron_scale),
        "head_mean": stack(lambda L: stats.layer(L).head_mean),
        "head_scale": stack(lambda L: stats.layer(L).head_scale),
        "mid_mean": stack(lambda L: lens.neutral_mean("mid", L)),
        "out_mean": stack(lambda L: lens.neutral_mean("out", L)),
        "scales": {
            **scales.describe(),
            "neuron": stack(lambda L: scales.layer(L).neuron),
            "head": stack(lambda L: scales.layer(L).head),
            "typical": {
                "neuron": [_f32(float(typical_scale(scales.layer(L).neuron_top))) for L in layers],
                "head": [_f32(float(typical_scale(scales.layer(L).head_top))) for L in layers],
                "model_neuron": _f32(float(typical_scale(neuron_tops))),
                "model_head": _f32(float(typical_scale(head_tops))),
            },
        },
    }


# ---------------------------------------------------------------------------
# 10. attention-patterns
# ---------------------------------------------------------------------------

PATTERN_POSITIONS = (2, 5, 8)


def _patterns(fx: Fixture, ids: Sequence[int]) -> Any:
    from drowse.loom.node_replay import eager_attention, tap_token

    spec = fx.spec
    rows = []
    with torch.inference_mode(), eager_attention(spec, True):
        for p in PATTERN_POSITIONS:
            taps = tap_token(fx.session.model, spec, torch.tensor([list(ids)]), p, patterns=True)
            assert taps.patterns is not None
            rows.append({"position": p, "layers": [encode(row.float().cpu()) for row in taps.patterns]})
    return rows


def _min_entropy(fx: Fixture, ids: Sequence[int]) -> float:
    from drowse.loom.node_replay import eager_attention

    with torch.inference_mode(), eager_attention(fx.spec, True):
        out = fx.session.model(input_ids=torch.tensor([list(ids)]), output_attentions=True)
    worst = math.inf
    for probs in out.attentions:
        p = probs[0, :, 1:].float()  # positions ≥ 1 (position 0 has one key)
        entropy = -(p.clamp_min(1e-30).log() * p).sum(-1)
        worst = min(worst, float(entropy.min()))
    return worst


def build_attention_patterns(fx: Fixture) -> dict[str, Any]:
    """Every head's attention row at positions 2, 5 and 8 (past Gemma's local
    window of 4), every layer, unsteered and under ``!head/L1h3``."""
    ids = short_prompt_ids(fx.tokenizer, fx.arch)
    entropy = _min_entropy(fx, ids)
    if entropy < 0.1:
        raise FixtureError(f"attention-patterns: an attention row saturates (entropy {entropy:.3g})")
    base = _patterns(fx, ids)
    with fx.session.steering("!head/L1h3"):
        steered = _patterns(fx, ids)
    return {
        **header("attention-patterns", fx.arch, model_file_sha256()),
        "ids": ids,
        "min_entropy": _f32(entropy),
        "cases": [
            {"name": "unsteered", "steering": None, "rows": base},
            {"name": "ablated-head", "steering": "!head/L1h3", "rows": steered},
        ],
    }


# ---------------------------------------------------------------------------
# 4. unit-ablation
# ---------------------------------------------------------------------------

#: The fixture direction the mixed case pushes along.
FIXTURE_DIRECTION = "deceptive_fixture"

#: The gated case's condition per arch: open on some decode steps, shut on
#: others (``neuron/L1n5``'s readings over the greedy decode).
GATES = {"gemma3": "neuron/L1n5 > 0", "qwen3": "neuron/L1n5 < -0.25"}

ABLATION_CASES: tuple[tuple[str, str, int], ...] = (
    ("neurons-full", "!neuron/L1n5 + !neuron/L1n40", 0),
    ("head-local", "!head/L0h1", 0),
    ("head-global", "!head/L1h3", 0),
    ("coefficients", "0.5 !neuron/L2n7 + 2 !neuron/L2n8 - !head/L2h0", 0),
    ("triggers", "!neuron/L1n5@prompt + !head/L2h1@response + !neuron/L3n9@first:2", 4),
    ("gated", "!neuron/L2n7@when:{gate}", 4),
    ("mixed", f"0.4 {FIXTURE_DIRECTION} + !neuron/L1n5", 2),
    ("push-and-ablate", "0.3 neuron/L2n7 + !neuron/L2n7", 0),
)


def _register_fixture_direction(fx: Fixture) -> None:
    from tests.support.sessions import register_vectors

    if FIXTURE_DIRECTION not in fx.session.registry.profiles:
        register_vectors(fx.session, FIXTURE_DIRECTION)


@contextmanager
def _capture_inputs(fx: Fixture) -> Iterator[list[list[int]]]:
    """The input ids of every forward through the model."""
    seen: list[list[int]] = []
    current: list[list[int]] = []

    def hook(_m: Any, args: Any, kwargs: dict[str, Any]) -> None:
        ids = kwargs.get("input_ids", args[0] if args else None)
        current[:] = [] if ids is None else [[int(x) for x in ids[0].tolist()]]

    def head(_m: Any, _args: Any, _output: Any) -> None:
        seen.extend(current)

    handles = [
        fx.session.model.register_forward_pre_hook(hook, with_kwargs=True),
        fx.session.model.get_output_embeddings().register_forward_hook(head),
    ]
    try:
        yield seen
    finally:
        for handle in handles:
            handle.remove()


def _greedy(steps: int) -> Any:
    from drowse.generation.sampling import SamplingConfig

    return SamplingConfig(temperature=0.0, max_tokens=steps, seed=0)


def _run_case(fx: Fixture, ids: Sequence[int], steering: str | None, steps: int) -> dict[str, Any]:
    """One case's forwards: a prefill (plus ``steps`` greedy decode steps
    through the real decode loop, raw mode).  Per forward the last-position
    logits and ``out`` (every layer at positions 0, 1 and the last of a
    prefill; a decode step's one row); the last layer's ``out`` at every
    prefill position."""
    text = fx.tokenizer.decode(list(ids))  # raw mode reads the BOS piece as its special
    with _capture_blocks(fx, positions=None) as store, _capture_inputs(fx) as inputs:
        if steps == 0:
            if steering is None:
                _forward(fx, ids)
            else:
                with fx.session.steering(steering):
                    tables = _unit_tables(fx)
                    _forward(fx, ids)
        else:
            # ``steps`` decode forwards after the prefill: steps + 1 tokens.
            run = fx.session.generate(
                text, raw=True, stateless=True, steering=steering, sampling=_greedy(steps + 1),
            ).first
            if len(run.tokens) != steps + 1:
                raise FixtureError(f"decode stopped after {len(run.tokens)} of {steps + 1} tokens")
            tables = None
    if steps and steering is not None:
        with fx.session.steering(steering):
            tables = _unit_tables(fx)
    if inputs[0] != list(ids):
        raise FixtureError(f"the prefill saw {inputs[0]}, not the short prompt {list(ids)}")
    forwards = []
    for i, (out, logits) in enumerate(zip(store["out"], store["logits"], strict=True)):
        rows = [0, 1, out.shape[1] - 1] if i == 0 else [out.shape[1] - 1]
        forwards.append({
            "ids": inputs[i],
            "logits": encode(logits),
            "out_rows": rows,
            "out": encode(out[:, rows]),
        })
    return {
        "forwards": forwards,
        "out_last_layer": encode(store["out"][0][-1]),
        "tables": tables if steering is not None else None,
    }


def build_unit_ablation(fx: Fixture) -> dict[str, Any]:
    """Exact mean-ablation at the projection inputs: neurons at ``down``'s,
    heads at ``o``'s, the sink row untouched, coefficients, triggers, a unit
    gate, beside an affine push, and a push and an ablation of one neuron."""
    _register_fixture_direction(fx)
    ids = short_prompt_ids(fx.tokenizer, fx.arch)
    base = {steps: _run_case(fx, ids, None, steps) for steps in sorted({s for *_x, s in ABLATION_CASES})}
    cases = []
    for name, steering, steps in ABLATION_CASES:
        steering = steering.format(gate=GATES[fx.arch])
        case = _run_case(fx, ids, steering, steps)
        ref = base[steps]
        first, ref_first = decode(case["forwards"][0]["out"]), decode(ref["forwards"][0]["out"])
        pushes = name in ("mixed", "push-and-ablate")  # an additive push writes every row
        if not pushes and not np.array_equal(first[:, 0], ref_first[:, 0]):
            raise FixtureError(f"unit-ablation {name}: the sink row's out moved")
        moved = max(
            float(np.abs(decode(f["logits"]) - decode(r["logits"])).max())
            for f, r in zip(case["forwards"], ref["forwards"], strict=False)
        )
        if moved < 1e-3:
            raise FixtureError(f"unit-ablation {name}: the logits move by only {moved:.3g}")
        cases.append({"name": name, "steering": steering, "decode_steps": steps, **case})
    out = {
        **header("unit-ablation", fx.arch, model_file_sha256()),
        "ids": ids,
        "direction": FIXTURE_DIRECTION,
        "unsteered": {str(steps): value for steps, value in base.items()},
        "cases": cases,
        "naive_post_block": _naive_post_block(fx, ids, cases) if fx.arch == "gemma3" else None,
    }
    return out


def _naive_post_block(fx: Fixture, ids: Sequence[int], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Gemma's sandwich makes a head's effect nonlinear at the block output:
    the naive correction ``out + γ_ao ⊙ Δ · r_ao`` (Δ the head's write
    change, ``r_ao`` the unablated attention output's norm scale) is not the
    ablated ``out``."""
    from drowse.loom.node_replay import tap_token

    spec = fx.spec
    layer, head = 1, 3
    block = spec.blocks[layer]
    stats = fx.session.units.reads.runtime.stats()
    hd = block.attn.head_dim
    p = len(ids) - 1
    with torch.inference_mode():
        taps = tap_token(fx.session.model, spec, torch.tensor([list(ids)]), p, patterns=False)
    z = taps.z[layer].float()
    delta_z = torch.zeros_like(z)
    delta_z[head * hd:(head + 1) * hd] = stats.layer(layer).head_mean[head].float() - z[head * hd:(head + 1) * hd]
    delta = block.attn.o.weight.float() @ delta_z
    norm = block.attn_out_norm
    assert norm is not None
    attn = taps.attn_out[layer].float()
    r = torch.rsqrt(attn.pow(2).mean() + norm.eps)
    naive = taps.out[layer].float() + norm.gain() * delta * r
    true = torch.from_numpy(decode(
        next(c for c in cases if c["name"] == "head-global")["forwards"][0]["out"],
    )[layer, -1].copy())
    gap = float((naive - true).abs().max())
    if gap < 1e-3:
        raise FixtureError(f"unit-ablation: the naive post-block correction matches ({gap:.3g})")
    return {"layer": layer, "head": head, "position": p, "naive_out": encode(naive), "gap": _f32(gap)}


# ---------------------------------------------------------------------------
# Nodes the replay goldens read
# ---------------------------------------------------------------------------


def generated_node(fx: Fixture, steering: str | None = None) -> str:
    """The chat prompt plus :data:`DECODE_STEPS` greedy tokens, in the tree
    (a fresh conversation from the root)."""
    fx.session.tree.navigate(fx.session.tree.root_id)
    runs = fx.session.generate(list(CHAT), steering=steering, sampling=_greedy(DECODE_STEPS))
    if len(runs.first.tokens) != DECODE_STEPS:
        raise FixtureError(f"the generated node stopped after {len(runs.first.tokens)} tokens")
    node = runs.node_ids[0]
    assert node is not None
    return node


#: The raw first turn's one-byte prompt per arch (greedy from it repeats
#: least on the fixture weights).
RAW_FIRST = {"gemma3": "=", "qwen3": "K"}


def raw_first_turn(fx: Fixture, tokens: int) -> str:
    """A raw-mode node whose first row is produced at position 0 (the sink):
    a one-token prompt (BOS, or one byte without one)."""
    text = RAW_FIRST[fx.arch]
    fx.session.tree.navigate(fx.session.tree.root_id)
    runs = fx.session.generate(text, raw=True, sampling=_greedy(tokens))
    if len(runs.first.tokens) != tokens:
        raise FixtureError(f"the raw first turn stopped after {len(runs.first.tokens)} tokens")
    node = runs.node_ids[0]
    assert node is not None
    return node


def _all_units(fx: Fixture) -> list[str]:
    spec = fx.spec
    return [
        f"neuron/L{b.layer}n{n}" for b in spec.blocks for n in range(spec.blocks[b.layer].d_ff)
    ] + [f"head/L{b.layer}h{h}" for b in spec.blocks for h in range(b.n_heads)]


def _replayer(fx: Fixture) -> Any:
    """A :class:`NodeReplayer` over the session's public collaborators, for
    the hand-built forwards (``_activity`` directly)."""
    from drowse.loom.node_replay import NodeReplayer
    from drowse.steering.triggers import TriggerContext

    session = fx.session
    return NodeReplayer(
        replayer=cast_any(None), model=session.model, lens=session.lens,
        units=session.units.reads.runtime, tokenizer=session.tokenizer,
        tree=lambda: session.tree, traits=cast_any(None), trigger_context=TriggerContext(),
    )


def cast_any(value: Any) -> Any:
    return value


def _hand_activity(fx: Fixture, forward: Any, *, top: int, units: Sequence[str]) -> dict[str, Any]:
    from drowse.loom.replay import ReplaySegment

    replayer = _replayer(fx)
    runtime = fx.session.units.reads.runtime
    refs = [runtime.parse(unit) for unit in units]
    segments = (ReplaySegment(0, int(forward.ids.shape[1]), prefill=False, gen_step=0),)
    with torch.inference_mode():
        result = replayer._activity(
            runtime.spec(), runtime.scales(), forward, segments,
            top=top, refs=refs, population=None, pop_layers=[],
        )
    result["node_id"] = "hand-built"
    result["steering"] = None
    return dict(result)


def _activity_diagnostics(
    fx: Fixture, full: dict[str, Any], excluded: set[int], case: str, *, ties: bool = False,
) -> dict[str, Any]:
    """Every unit's peak value, peak row and peak-row margin (from a call
    tracing every unit), the arrays the browser checks its device peaks by."""
    traces = full["traces"]
    units = list(traces)
    values = np.array([traces[u] for u in units], dtype=np.float64)  # [units, rows]
    rows = values.shape[1]
    keep = np.array([r not in excluded for r in range(rows)])
    peak = np.zeros(len(units))
    row = -np.ones(len(units), dtype=np.int64)
    margin = np.zeros(len(units))
    if keep.any():
        masked = np.where(keep, np.abs(values), -1.0)
        row = masked.argmax(1)
        peak = values[np.arange(len(units)), row]
        ranked = np.sort(masked, axis=1)[:, ::-1]
        margin = ranked[:, 0] - (ranked[:, 1] if rows > 1 else -1.0)
    return {
        "units": units,
        "peak": encode(peak.astype(np.float32)),
        "row": encode(row.astype(np.int32)),
        "row_margin": encode(margin.astype(np.float32)),
    }


def _check_activity(result: dict[str, Any], diag: dict[str, Any], top: int, case: str, *, ties: bool) -> None:
    units = diag["units"]
    peak = decode(diag["peak"])
    margin = decode(diag["row_margin"])
    index = {unit: i for i, unit in enumerate(units)}
    for kind in ("neuron", "head"):
        mags = sorted((abs(float(peak[i])) for u, i in index.items() if u.startswith(kind)), reverse=True)
        listed = [p for p in result["top"] if p["kind"] == kind]
        if listed and not ties:
            require_top_margin(mags, len(listed), f"unit-activity {case} {kind} peaks")
    if ties:
        return
    rows = decode(diag["row"])
    for unit in [p["unit"] for p in result["top"]] + list(result["traces"]):
        i = index[unit]
        if int(rows[i]) < 0:  # every row excluded: the unit never ranks
            continue
        scale = max(1.0, abs(float(peak[i])))
        if float(margin[i]) < MARGIN * scale:
            raise FixtureError(f"unit-activity {case}: {unit}'s peak row has no margin")


# ---------------------------------------------------------------------------
# 6. unit-activity
# ---------------------------------------------------------------------------

ACTIVITY_UNITS = ("neuron/L1n5", "head/L2h1", "neuron/L3n9")


def build_unit_activity(fx: Fixture) -> dict[str, Any]:
    """``UnitActivityJSON`` over generated, sink, hand-built and segmented
    forwards, with every unit's peak, row and row margin."""
    from drowse.errors import DrowseError
    from drowse.loom.replay import ReplayForward
    from drowse.steering.hooks import UnitGroup, UnitHook, UnitOp
    from drowse.steering.triggers import Trigger, TriggerContext

    session = fx.session
    every = _all_units(fx)
    cases: list[dict[str, Any]] = []

    def record(name: str, result: dict[str, Any], full: dict[str, Any], excluded: set[int],
               top: int, *, args: dict[str, Any], ties: bool = False) -> None:
        result = {k: v for k, v in result.items() if k != "node_id"}
        diag = _activity_diagnostics(fx, full, excluded, name)
        _check_activity(result, diag, top, name, ties=ties)
        cases.append({"name": name, "arguments": args, "declared_ties": ties, "result": result,
                      "diagnostics": {k: v for k, v in diag.items() if k != "units"}})

    generated = generated_node(fx)
    args = {"top": 8, "units": list(ACTIVITY_UNITS), "raw_index": 2, "layers": "0,2"}
    result = dict(session.unit_activity(generated, **args))
    full = dict(session.unit_activity(generated, top=1024, units=every))
    sink = {result["sink_raw_index"]} - {None}
    record("generated", result, full, sink, 8, args=args)

    first = raw_first_turn(fx, 3)
    # A raw node replays raw (its prompt is the raw render, not the template's).
    result = dict(session.unit_activity(first, top=8, raw_index=1, raw=True))
    full = dict(session.unit_activity(first, top=1024, units=every, raw=True))
    if result["sink_raw_index"] != 0:
        raise FixtureError("unit-activity sink: row 0 is not at the sink")
    record("sink", result, full, {0}, 8, args={"top": 8, "raw_index": 1, "raw": True})
    # drowse 6.0 answers a population at the sink (every value 0) rather
    # than refusing it; a refusal would be recorded as one.
    try:
        at_sink = dict(session.unit_activity(first, top=8, raw_index=0, raw=True))
    except DrowseError as exc:
        status, text = exc.user_message()
        cases.append({"name": "sink-population", "arguments": {"raw_index": 0, "raw": True},
                      "refusal": {"status": status, "text": text, "code": type(exc).__name__}})
    else:
        record("sink-population", at_sink, full, {0}, 8, args={"top": 8, "raw_index": 0, "raw": True})

    ids = short_prompt_ids(fx.tokenizer, fx.arch)

    def hand(name: str, runs: tuple[tuple[int, int], ...], rows: list[int], stop: int,
             excluded: set[int], *, top: int, units: Sequence[str], ties: bool = False) -> None:
        forward = ReplayForward(
            ids=torch.tensor([ids[:stop]]), prompt_length=stop,
            raw_ids=tuple(ids[r] for r in rows), raw_index=len(rows) - 1, runs=runs,
        )
        result = _hand_activity(fx, forward, top=top, units=units)
        full = _hand_activity(fx, forward, top=1024, units=every)
        record(name, result, full, excluded, top, ties=ties, args={
            "top": top, "units": list(units), "ids": ids[:stop],
            "runs": [list(run) for run in runs], "raw_ids": [ids[r] for r in rows],
        })

    # Two spans with a gap; the first starts with an unproduced row (at −1),
    # then the sink (row 1 at position 0).
    hand("two-spans", ((0, 0), (3, 7)), [0, 1, 2, 7, 8, 9], 9, {0, 1}, top=4, units=())
    # Every row excluded: one unproduced row and the sink.
    hand("all-excluded", ((0, 0),), [0, 1], 1, {0, 1}, top=8, units=("neuron/L1n5",))
    # A unit forced to exactly 0 on every row (``keep = 0, bias = 0``, built
    # through UnitOp: the grammar's terms always carry ``c·mean``) peaks at
    # the first produced non-sink row; a declared tie case.
    block = fx.spec.blocks[1]
    keep = torch.ones(block.d_ff)
    keep[5] = 0.0
    hook = UnitHook(block.mlp.down)
    hook.recompose(
        (UnitGroup(Trigger.BOTH, UnitOp(keep, torch.zeros(block.d_ff), device=torch.device("cpu"),
                                        dtype=torch.float32)),),
        TriggerContext(),
    )
    hook.attach()
    try:
        hand("excluded-and-zero", ((0, 0),), [0, 1, 2, 3, 4, 5], 5, {0, 1}, top=1024,
             units=("neuron/L1n5",), ties=True)
    finally:
        hook.detach()
    zero = next(c for c in cases if c["name"] == "excluded-and-zero")["result"]
    pick = next(p for p in zero["top"] if p["unit"] == "neuron/L1n5")
    if pick["peak"] != 0.0 or pick["raw_index"] != 2:
        raise FixtureError(f"unit-activity excluded-and-zero: L1n5 peaks {pick}")

    segmented = generated_node(fx, steering="!neuron/L1n5@prompt")
    full = dict(session.unit_activity(segmented, top=1024, units=every))
    excluded = {full["sink_raw_index"]} - {None}
    for top in (8, 7, 6, 5, 4):  # the largest top whose boundaries keep the margin
        result = dict(session.unit_activity(segmented, top=top))
        try:
            _check_activity(result, _activity_diagnostics(fx, full, excluded, "segmented"), top,
                            "segmented", ties=False)
        except FixtureError:
            continue
        break
    record("segmented", result, full, excluded, top, args={"top": top})
    return {**header("unit-activity", fx.arch, model_file_sha256()), "cases": cases}
