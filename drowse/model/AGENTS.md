# model/

Everything drowse knows about a Hugging Face causal LM below steering:
loading and identity, where the residual blocks are and what they compute
(`BlockSpec`), unit directions, the logits path, layer selectors, token spans, chat-template facts, role substitution, the scene
stitcher, and compile/StaticCache plumbing. The package imports only
`errors`/`naming`/`events`, `base` and `geometry`
(`tests/invariants/test_layering.py`) and holds no session state: every
function takes a model, tokenizer or config.
Render *policy* (which path a conversation takes) lives in
`generation/decode.py::build_chat_input` and `generation/render.py`; this
package supplies the mechanisms.

## Module map

| File | Owns |
|---|---|
| `load.py` | `load_model`: `LoadPlan` → materialize → finalize |
| `checkpoint.py` | Hardened safetensors reader; multimodal text-model extraction |
| `device.py` | Device/dtype defaults, `memory_gb`, `patch_torch_for_mps` |
| `fingerprint.py` | Source and loaded-model identities, `resolve_model_revision` |
| `arch.py` | `get_layers`, `TESTED_ARCHS`, `text_extractable`, `config_model_shape`, `workspace_layer_indices` |
| `readout.py` | Final residual → logits (`LogitReadout`, `get_unembedding`, `get_final_norm`) |
| `blocks.py` | `BlockSpec`/`ModelSpec`: dense decoder blocks described for transport and unit reads; `model_spec`, `block_specs`, `verify_model_spec`, the reference `block_forward` |
| `units.py` | Neuron writes and pole reads, head write and read rows, from a `BlockSpec` (fp32, CPU) |
| `layers.py` | The one layer-selector parser, formatter and resolvers (`--layers`, `layers=`, steering `{…}`) |
| `tokens.py` | `special_token_ids`, `last_content_index` |
| `chat.py` | `ChatTraits`: per-session chat facts and render cache |
| `role_templates.py` | `SEAT_HEADERS`, `resolve_seat_headers`, `validate_role`, the splice renderers |
| `scene.py` | Template autopsy → `TurnGrammar`; `render_scene`, `validate_turn_grammar` |
| `compile.py` | `compile_with_probe`: `torch.compile` + warmup probes, eager fallback |
| `static_cache.py` | StaticCache support probes and the one factory |

`DrowseSession.from_pretrained` calls `load_model`; then
`generation/compile_runtime.py::CompiledRuntime.prepare` calls
`compile_with_probe` and the StaticCache probes (CUDA/MPS only).
`DrowseSession.__init__` builds one `ChatTraits` and resolves `get_layers` once.

## Loading

`_resolve_load_plan` makes every decision from config reads alone (no weight
I/O, so `LoadPlan` is unit-testable); `_materialize_model` reads weights;
`_finalize_model` stamps identity and freezes the model. Decisions:

- **Revision pin.** The Hub commit is resolved once (`_pinned_revision`; `None`
  for a local directory; fails open to an unpinned read) and every config,
  tokenizer and weight read passes it, so a moving branch can't pair config A
  with weights B. Finalize writes it onto `config._commit_hash` (and the
  `text_config`'s): transformers no longer fills that slot, and J-lens
  bindings and capture fingerprints read it.
- **Remote code** is off. `trust_remote_code=None` reads
  `DROWSE_TRUST_REMOTE_CODE` (`1`/`true`/`yes`/`on`; the CLI has no flag);
  explicit bools win; anything else raises. Even when trusted, the model load
  drops it for a `model_type` native to transformers (stale `auto_map` code
  breaks). Metadata-only reads never trust remote code.
- **dtype** defaults to bf16 on CUDA and MPS, fp32 on CPU; not fp16, because
  some residual streams leave its range (Gemma-3-4b reaches ~1e5).
  Quantization (`4bit` nf4, `8bit`) is CUDA-only and dropped with a warning
  elsewhere.
- **Attention** `sdpa`; `flash_attention_2` on CUDA when `flash_attn` imports;
  `eager` on MPS for MLA models (`deepseek_v2`/`v3`: mismatched q/v head dims
  break MPS SDPA). Mistral tokenizers (`"mistral"` in the id) get
  `fix_mistral_regex=True`; the shipped regex mis-splits ~1% of tokens.
- **Text extraction** applies when `text_config.model_type` is a transformers
  causal-LM type (`arch.text_extractable`), even if the composite loads.
  Quantized loads skip it (the extractor cannot quantize).

Fallback ladder: unsupported attention → `eager`; any other failure of an
unquantized load → fp16 (CUDA) or fp32; an MPS weight-`CONVERSION` failure →
load on CPU, then move. Retries narrow `plan.load_kwargs` in place so an outer
retry never repeats a known-bad attempt. Never retried:
`_IncompleteCheckpointError` (drowse never ships a freshly initialized
parameter) and a quantized composite that only text extraction can load.

`load_text_from_multimodal` builds the text model with `from_config` under
`torch.device(device)` and streams shards tensor by tensor into it: ~1× peak
memory on unified memory, where `from_pretrained`'s stage-then-copy costs ~2×
(>128 GB for gemma-4-31B). Reads are chunked `pread`s on an `open_uncached` fd
(`F_NOCACHE` on macOS, so the checkpoint never fills the page cache MPS
shares). It strips the first `language_model.` segment, keeps `lm_head.*`,
skips the rest and dequantizes FP8 by `<name>_scale_inv`; no match raises
`NoTextWeightsExtracted` (→ full multimodal load), a missing parameter
`ValueError`. This is the one hand-written checkpoint parser; keep it hardened
(duplicate keys, size caps, traversal-safe shard names, exact extents).

Process-global effects: loads serialize on `_MODEL_LOAD_LOCK`; MPS loads
scope `HF_DEACTIVATE_ASYNC_LOAD=1` (transformers' concurrent copies can
segfault) and install `patch_torch_for_mps` once (`histc` on integers via
float for MoE routing, `ldexp` via CPU).

## Identity (`fingerprint.py`)

- `model_source_fingerprint` is provable before any weight is read: sha256 of
  the canonical id, resolved commit, exact hashes of every regular file in a
  local snapshot (no whitelist: remote code can read anything), the config,
  CUDA-effective quantization and parameter dtype. `None` without a commit or
  local files, or on a read error; callers must then decline metadata-only
  shortcuts. Load stamps it as `model._drowse_source_fingerprint`; weight-free
  preflights (`fit/preflight.py`, `fit/transfer.py`, the lens-fit no-op)
  recompute it.
- `loaded_model_fingerprint` identifies the weights a live module holds; fitted
  artifacts bind to it. A clean drowse load hashes config, class, sorted
  parameter/buffer schema and the source stamp. Anything else (in-memory,
  wrapped, or changed since load per `_model_state_signature`) also streams a
  bit-exact hash of every tensor. Memoized per module; unwraps `_orig_mod`.

Rules: the text-extraction load mutates the plan config in place (fills
`text_config._name_or_path`; `from_config` writes the dtype), and
`model_source_fingerprint` applies the same writes to its payload so a
pre-load recompute equals the stamp. Change what the loader writes onto the
config and mirror it here, or every weight-free no-op proof silently misses
(`test_source_fingerprint_recompute_matches_stamp_*`). Integrations that write
through `Tensor.data` or swap adapters (bypassing `_version`) must call
`invalidate_loaded_model_fingerprint`. Bump `_SOURCE_FINGERPRINT_VERSION` when
the payload shape changes.

## Architectures (`arch.py`, `readout.py`)

`get_layers` finds the block list structurally: the first non-empty
`nn.ModuleList` among `_LAYER_PATHS` (`model.layers`,
`model.language_model.layers`, `model.blocks`, `model.decoder.layers`,
`transformer.h`, `transformer.blocks`, `gpt_neox.layers`). There is no
per-family accessor table; anything with blocks on those paths runs, and a
`model_type` outside `TESTED_ARCHS` warns once.

`TESTED_ARCHS`: `llama`, `mistral`, `ministral3`, `gemma2`, `gemma3`,
`gemma3_text`, `gemma4`, `gemma4_text`, `gemma4_unified`,
`gemma4_unified_text`, `qwen2`, `qwen3`, `qwen3_5`, `qwen3_5_text`,
`qwen3_5_moe`, `glm`, `gpt_oss`, `talkie`. Mistral-Small-3 loads as `mistral`
(its `mistral3` composite's text sub-model); `ministral3` loads under its own
type; the gemma VLM composites are listed for callers who hand the full model
to `DrowseSession`.

**Lookups are exact.** `TESTED_ARCHS` and `SEAT_HEADERS` match `model_type`
verbatim, so each sub-variant a checkpoint can load as needs its own entry (a
text-extracted Qwen3.5-MoE loads as `qwen3_5_moe_text`, in neither table).
Adding an architecture: a new block location is one `_LAYER_PATHS` entry;
"tested" means cases in `test_architecture_dispatch.py` and
`test_model_numerics.py` plus the `TESTED_ARCHS` entry; role and scene support
mean a `SEAT_HEADERS` entry plus a tokenizer in `test_role_template_drift.py`.
Keep `model_type` branches out of capture and hooks; the forward side's only
per-family branches are in `readout.py`.

`config_model_shape` reads `(n_layers, hidden)` from the published config for
weight-free preflights and raises rather than guess (a guessed depth would
widen or narrow a proof). `workspace_layer_indices` is the canonical workspace
band, `0.40 ≤ l/(n_layers−1) ≤ 0.90`, falling back to every candidate on
shallow models.

`LogitReadout` is the logits path the J-lens runs without another forward:
final norm (found as a sibling of the block list), OPT `project_out`, head,
Cohere `logit_scale` or Falcon-H1 `lm_head_multiplier`, Granite
`logits_scaling`, then the `final_logit_softcapping` tanh cap. A family with
another logits quirk needs a branch here and a numerics case. A model whose
forward ignores `past_key_values` is handled in `generation/decode.py`.

## Blocks and units (`blocks.py`, `units.py`)

A `BlockSpec` describes one dense decoder block in semantic terms, so the
lens transport estimator and the unit reads consume one description instead
of each family's attribute names. Every described block computes

```text
h_mid = h + attn_out_norm(o(attention(q(a), k(a), v(a)))),   a = attn_in_norm(h)
h_out = (h_mid + mlp_out_norm(down(act(gate(m)) * up(m)))) * scalar,   m = mlp_in_norm(h_mid)
```

with causal softmax attention over grouped KV heads, optional per-head q/k/v
RMSNorms before RoPE, scores `q·k · scaling` capped by
`logit_softcap·tanh(s/logit_softcap)` when set, and an optional sliding window
(position `i` sees `j` iff `i − window < j ≤ i`).

| Field | Holds |
|---|---|
| `attn_in_norm`, `mlp_in_norm` | the pre-norms (`NormSpec`: module, `eps`, `offset` 0 for `w` or 1 for `1 + w`, `weighted`) |
| `attn_out_norm`, `mlp_out_norm` | Gemma's sandwich norms; `None` elsewhere |
| `attn` (`AttentionSpec`) | `q`/`k`/`v`/`o` (`nn.Linear`; `v is None` means the value is `k`'s raw output, Gemma 4's full layers), `q_norm`/`k_norm`/`v_norm`, `n_heads`, `n_kv_heads`, `head_dim`, `scaling`, `sliding_window`, `logit_softcap`, `rope_table` (the layer type a per-type rotary table takes), `qkv_bias`, `o_bias`, `kv_head(h)` |
| `mlp` (`MLPSpec`) | `gate`/`up`/`down`, `act` (`silu` or `gelu_tanh`), `d_ff`, `bias` |
| `scalar` | Gemma 4's `layer_scalar` (`scale()` is 1.0 without one) |

`ModelSpec` adds the embedding and `embed_scale` (√d on Gemma), the final
norm, `final_logit_softcap`, `unembed`, the rotary module (and whether it takes
a layer type), `n_layers`, `d_model`, `d_ff` (the widest block's) and `vocab`.
`activation`, `activation_factor` (the RelP identity rule's `act(g)/g`) and
`activation_derivative` are the two activations' math.

- **Discovery is structural.** `model_spec(model)` (and `block_specs(model)`)
  finds parts by module presence and config fields, never `model_type`. A
  block with `pre_feedforward_layernorm` is a sandwich block: its
  `post_attention_layernorm` normalizes the attention output. Norm conventions
  and the activation are identified by running the module on a probe and
  matching its output, and each gated MLP is checked to compute exactly
  `down(act(gate(x)) * up(x))`. Specs hold live module references; weights are
  read when used.
- **Refusals** raise `UnsupportedBlockError` (422, in `errors.py`) naming the
  layer (`layer` is `None` for the model-level part) and what was found: a
  mixture-of-experts MLP or block, a fused gate-up or qkv projection, a
  linear-attention layer (hybrid stacks), attention sinks or any other
  attention tensor, KV sharing, non-causal or NoPE attention, a norm that is
  not an RMSNorm with a `w` or `1 + w` weight, an activation other than SiLU
  or tanh-GELU, any unrecognized submodule (Gemma 4's per-layer inputs), a
  config multiplier (`*_multiplier`, `logits_scaling`, `logit_scale`) other
  than 1, and `llama_4_scaling_beta` query scaling.
- **Covered:** `llama`, `mistral`, `qwen2`, `qwen3`, `gemma2`, `gemma3_text`
  (and the `gemma3` composite), `gemma4_text` and `gemma4_unified_text` without
  per-layer inputs, KV sharing or MoE. `glm`/`glm4` (fused gate-up),
  `qwen3_5` (hybrid), `qwen3_5_moe`, `gpt_oss` (MoE, sinks) and `ministral3`
  (query scaling) are refused.
- **Verification.** A forward term the structure does not show is invisible
  to discovery (helium is llama-shaped but rotates RoPE pairs interleaved).
  `verify_model_spec(model)` runs the model once on probe tokens and holds
  every block to `block_forward` (the formula above, fp32, from the spec
  alone): the attention update `mid − h` from the block's real input and the
  output from its real `mid`, within 5% of the update. It runs a forward, so
  the caller holds the model exclusively and unsteered. Transport fits call
  it; static unit reads need only `model_spec`.

Unit directions (`units.py`) are fp32 CPU tensors, with neuron-lens's
definitions. Neuron `n` is coordinate `n` of the down projection's input.
`neuron_write(spec, n)` is `s · γ_out ⊙ W_down[:, n]` (`γ_out` the MLP output
norm's gain where the block has one, `s` the per-layer scalar), the `+` pole's
write as it lands at `out`; the `−` pole writes its negation.
`neuron_read(spec, n, pole)` is `â + b̂` or `â − b̂` for the unit rows
`a = γ_in ⊙ W_gate[n]`, `b = γ_in ⊙ W_up[n]` (`γ_in` the MLP input norm's
gain), not renormalized. `neuron_writes`/`neuron_reads`/`neuron_read_rows`
are the whole-layer matrices. `head_write_matrix(spec, h)` is
`γ_attn_out ⊙ W_O[:, h·hd:(h+1)·hd]`, `[d, head_dim]`, landing at `mid`
before the scalar applies; `head_read_rows(spec, h)` gives the query rows and
its KV head's key and value rows (keys as values when `v` is absent), each
times the attention input norm's gain. An index outside the block raises
`UnitIndexError` (400); whether a `NeuronRef`/`HeadRef` names a block of the
model is the caller's check.

## Layer selectors and token spans

`layers.py` is the one parser for `--layers` / `layers=` (CLI, server, lens,
manifold fits) and the steering grammar's `{…}` layer clause; torch-free at
import because the CLI parses before a model loads. Selectors: `None`/`all`,
`workspace`, `sample` (`SAMPLE_COUNT = 9` evenly spaced, readout only) or an
index list, written as comma-separated indices and inclusive `a-b` ranges
(`4-8,20`; whitespace is free). `parse_layer_selector` keeps written order and
raises `LayerSelectorError` (a 400 `RequestError` whose `offset` points at the
failing entry) on an empty entry, a negative index, a reversed range, a layer
named twice, or `sample` under `allow_sample=False`.
`format_layer_indices` is its inverse on index sets (sorted, consecutive runs
as `a-b`), the canonical form the grammar renders. `resolve_layer_selector`
returns an explicit list as given; `resolve_model_layers` (fits) sorts,
bounds-checks and refuses `sample`. A manifold fit and its weight-free
preflight both resolve through `resolve_model_layers`, so a no-op proof covers
exactly the fit's layers.

`tokens.last_content_index` is the single definition of the "last content
token" every single-state read pools from (fit capture, monitor per-token and
aggregate reads, `generation/read_plane.py`, naturalness); never re-derive it.
It trims template whitespace only after a closing special (SmolLM2 renders
`…content<|im_end|>\n`), then walks back past `special_token_ids` and floors at
0. Whitespace *before* the closing marker stays content (its state predicts
the marker; the browser capture pools the same token). `special_token_ids`
includes every `added_tokens_encoder` id, because some tokenizers (talkie's
`<|user|>`/`<|end|>`) add turn markers without marking them special.

## Chat traits (`chat.py`)

`ChatTraits` reads a tokenizer's chat behavior once per session: `eos_ids`
(generation-config eos ∪ tokenizer eos ∪ added end-of-turn tokens like
`<end_of_turn>`, `<turn|>`, `<|im_end|>`, `<|eot_id|>`), `think`
(`ThinkDelimiters` token ids for the decode loop), `takes_enable_thinking`
(substring sniff of the template), `is_base`, `logits_keep_kwarg`, a
token-text table and a bounded render cache (128 entries, 8 MiB; key built
in `generation/decode.py`). `DrowseSession.close()` clears the caches.
`adopt_eos` writes `eos_ids` into `generation_config` so HF `generate` (batch
paths) stops where the decode loop does; an extracted text sub-config can
under-specify end-of-turn (Gemma-4 lists `<eos>`, turns end `<turn|>`).
Delimiter detection covers switchable
templates (round-trip marker thinking), gpt-oss channels and the Mistral-3
`[THINK]` pair. These token ids are not the scene grammar's string pair: a
family can stream thinking (gpt-oss) yet carry no committed thinking.

## Role substitution (`role_templates.py`)

Templates branch on hardcoded role names, so a custom role is dropped, raises
or works depending on family. Drowse renders with the standard roles, splices
the label into the rendered string at the family's header bytes, and
tokenizes with `add_special_tokens=False`, keeping token boundaries where
triggers expect them. `SEAT_HEADERS` maps `model_type` → both seats'
`RoleHeader(before, after, label)`; the user label is always `user`:

- `qwen2`, `qwen3`, `qwen3_text`, `qwen3_moe`, `qwen3_5`, `qwen3_5_text`,
  `qwen3_5_moe`: `<|im_start|>` · `assistant` · `\n`
- `gemma2`, `gemma3`, `gemma3_text`: `<start_of_turn>` · `model` · `\n`
- `gemma4`, `gemma4_text`, `gemma4_unified`, `gemma4_unified_text`: `<|turn>` ·
  `model` · `\n`
- `llama`: `<|start_header_id|>` · `assistant` · `<|end_header_id|>`
- `glm`, `talkie`: `<|` · `assistant` · `|>`
- `gpt_oss`: `<|start|>` · `assistant` · `<|channel|>`; its user seat ends
  `<|message|>` (the one family whose seats differ)
- `mistral`, `mistral3`, `ministral3`: both seats `None` (positional `[INST]`)

`resolve_seat_headers` is the only resolver (splice paths, scene autopsy,
server capability flags). It checks `_TEMPLATE_HEADER_OVERRIDES`, keyed by
`(model_type, sha256(chat_template))`, first: SmolLM2-360M-Instruct is a
`llama` checkpoint whose template renders ChatML. Overrides key on exact
template bytes, never on header-shaped text in rendered content. `None` = family
absent; `SeatHeaders(None, None)` = present but label-free.

`validate_role` (`^[a-z0-9._-]+$`) is the one slug gate for every label surface
(extraction `--role`, manifold node roles, loom cast labels, scene turns);
`render_role_label` de-slugs `_` → space.

- `apply_with_role` puts one label per side on every turn (both `None` is a
  plain `apply_chat_template` pass-through). The assistant splice is strict: a
  missing header raises `RoleTemplateDriftError` (500) instead of letting the
  extraction baseline silently diverge from the steering one; the user splice
  tolerates absence. Fit capture renders role-baselined rows through it.
- `apply_with_per_turn_roles` honors per-message `"label"` keys by occurrence
  index, best-effort (gpt-oss's generation prompt lacks `<|channel|>`, so its
  `gen_role` never lands). Both sides splice in one scan of the original
  string; sequential passes would let a user turn labelled `assistant` (GLM
  `<|assistant|>`) be claimed as an assistant occurrence. This is the fallback
  when the stitcher is unavailable.

Errors: `RoleSubstitutionUnsupportedError` (400), `RoleTemplateDriftError`
(500), `InvalidRoleError` (`RequestError`, 400).

## Scene grammar (`scene.py`)

The cast model needs renders no template produces: non-alternating seats, a
label per turn, a generation prompt on either seat (gemma and mistral templates
raise on non-alternation). **Template autopsy** (`extract_turn_grammar`)
renders sentinel probe conversations through the live template and slices a
`TurnGrammar`: `prelude`, per-seat `SeatWrapper`s (open around the label
site, close), the system shape (a real turn, a gemma-style fold into the first
turn whatever its seat, or unsupported as on gemma-2), `gen_extra` (bytes the
generation prompt adds after the assistant open, probed with
`enable_thinking=False` where the template takes it) and its `True` variant,
trim flags, `last_assistant_special` (qwen3's empty-think insert into a
closing assistant turn), and the think pair with its history policy.

`render_scene` is pure stitching, `open(label) + content + close` per
`SceneTurn` plus a trailing open for `gen_seat`. Labels go into headers drowse
constructs, so the label↔turn binding is positional and nothing is
occurrence-matched. Trim applies to the turn text before thinking and fold
composition (gemma-3 folds raw system content around trimmed text). A
user-seat generation prompt is the bare user open.

`validate_turn_grammar` is the load-bearing gate: it stitches canonical
alternating conversations (plain, generation prompt, padded content, closed on
assistant, system, padded system, thinking generation prompt, as applicable)
and byte-compares each with `apply_chat_template`. A passing family's
alternating renders are bit-identical through the stitcher. That is why
generation can render through the stitcher while fit capture renders through
the template or `apply_with_role` and extract baseline still equals steer
baseline, and why a template revision flips the family to the template paths
instead of corrupting renders.

Autopsy failure (`SceneGrammarError`) loses scene mode: a label-free or
unregistered family, a probe render that raises, a transformed or repeated
sentinel, first-turn-special headers, a generation prompt that doesn't extend
the plain render, misplaced system content, or a trim that is asymmetric or
not plain (GLM-4.7-Flash fails). Feature defeats only drop the feature: no
system support, `gen_extra_thinking=None`, or no committed thinking.

Thinking: the caller supplies a literal `(open, close)` pair bracketing
reasoning in the family's render; `PromptRenderer.scene_grammar` passes
`("<think>", "</think>")` when the template mentions `<think>`, else `None`
(channel formats like gpt-oss Harmony have no valid pair). If a probe shows the
template strips history thinking, a turn's thinking renders only while it is
the last before the generation header. Thinking without a pair raises
`SceneThinkingUnsupportedError`, at commit time via
`loom/ops.py::check_thinking_commit`; finalize stamps `thinking_text` only when
the pair exists.

Routing (`generation/decode.py::build_chat_input`): with a template and a
validated grammar, the stitcher serves an optional leading system turn plus
user/assistant turns. A mid-conversation system turn, no grammar, or an
assistant-seat shape the grammar declines goes to `apply_chat_template` or
`apply_with_per_turn_roles`; a non-assistant `gen_seat` has no such fallback
and raises `SceneRenderError`. Only a template-less tokenizer uses
`render_scene_raw` (`Label: text` blocks). The session resolves the grammar
lazily once (`DrowseSession.scene_grammar`; `None` = unavailable, logged).
Seats stay binary; everything else is a label. A user-seat generation on a
family whose seat closes differ (gpt-oss) adds the user close as a stop string
(`generation/render.py::seat_stop_strings`). Errors: `SceneRenderError` (400),
`SceneGrammarError` (422), `SceneThinkingUnsupportedError` (400).

## Compile and StaticCache

`compile_with_probe` compiles (`dynamic=None`) and immediately runs probe
forwards: DynamicCache prefills at lengths 2 and 16 (forcing dynamic-shape
promotion), and under `reduce-overhead` a StaticCache prefill plus a decode
with `attention_mask=None`, a separate dynamo call shape where inductor bugs
have surfaced. Failure warns and returns the eager model, so breakage shows at
load. Process-global knobs: `use_static_cuda_launcher = False` (the static
launcher segfaults uncatchably on codegen bugs; Triton's raises a catchable
`TypeError`) and, on CUDA, `float32_matmul_precision("high")` (TF32 around
bf16 inference), restored only on probe failure.

`is_static_cache_supported` probes a 1-token StaticCache, cached per
weakly-held module (through `_orig_mod`), device and dtype; StaticCache is the
compile enabler on every backend, MPS included (fixed K/V shapes let inductor
reuse one trace). `is_cuda_graphs_supported` is the CUDA gate on top and picks
`reduce-overhead` over `default`. `make_static_cache` is the one factory
(callers size `max_cache_len ≥ prompt_len + max_new_tokens +
cache_position_offset`): it early-initializes layer buffers outside the
compiled graph (in-graph init blocks CUDA-graph capture and recompiles per
cache) and flags sliding layers that cannot slide this generation, which the
class-level `_patch_static_sliding_mask` gives a constant `get_mask_sizes`
(byte-identical in that regime) to stop the per-token recompile storm on
hybrid caches. Whether steering may take the static path is decided in
`steering/hooks.py` (`all_fast_path` / `static_steerable`).

## Tests

`tests/model/` runs on CPU without downloads: `test_blocks.py` and
`test_units.py` (every covered family's spec reproduces the model's forward;
the refusals; units against hand computation, on `tests/support/blocks.py`),
loader plan/ladder/extraction
and checkpoint hardening on mocked transformers, real module trees and 14 tiny
real architectures for `get_layers` and `LogitReadout` numerics, and the splice
and stitcher against the Jinja fakes in `tests/support/chat_templates.py`
(`test_scene_wiring.py` pins `build_chat_input` byte parity).
`test_role_template_drift.py` checks `SEAT_HEADERS` against real tokenizers in
the local HF cache; `DROWSE_TEST_LIVE_TOKENIZERS=1` permits downloads (run it
before a release). `tests/support/tiny_models.py` supplies genuine 4-layer
models with real family templates for CPU end-to-end tests; real-weight checks
are `tests/gpu/test_model_numerics_gpu.py` (`DROWSE_TEST_MODEL`).
