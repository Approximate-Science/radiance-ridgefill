#!/usr/bin/env python3
"""mini_model.py -- a MINI Qwen4-Exp of the SAME architecture as Qwen3.8-Flash-Next, with random
weights, as an HF-format checkpoint rad-convert turns into a .rad container (notes/mini-model.md).

  mini_model.py --out DIR [--real-config DIR] [--tokenizer DIR] [dims...]

WHY. The plugin's correctness gates must run with no GPU, and the real model is 122 GB: this builds
the smallest checkpoint the ARCHITECTURE accepts (radiance arch/qwen4exp_fp8, which arch/qwen4exp_kva.cpp
shadows) so `radiance --debug-accept-reference-kernels` can serve it on the host backend.

WHAT IS KEPT (the architecture's identity -- see notes/mini-model.md for the full table):
  every per-head dimension (24 query heads and 2 KV heads of 256 with partial rotary 64, 16 key /
  48 value delta-net heads of 128, 4 indexer heads of 128), the layer pattern (3 x GatedDeltaNet
  -> 1 x gated attention, full_attention_interval 4), the gated residual (hc_count 4, hc_lowrank
  320), PLE (ngram 3, 16 hash heads of 160, 128 shards, layer_ids [2] -> layer 1), the MTP head,
  the REAL vocab (248320 -- the marker token ids exist) and the real tokenizer files.
WHAT IS CUT: hidden 2560 -> 256 (a multiple of the recipe's 128-column blocks; try --hidden to go
  lower), layers 48 -> 8 (two full 4-groups), experts 512 -> 16 at top_k 10 -> 4, and the PLE
  n-gram table's base 20e6 -> 1000 (51B rows cannot fit a 3 GiB budget; the per-head geometry is
  untouched). The vision tower is dropped (the plugin's paths never touch it).

The weights are seeded random, scaled per fan-in so activations stay finite, with the reference's
own initialisations where they are semantic: norm gains ZERO (x_hat * (1 + w)), the delta net's
gated output norm ONES, A_log = log(0.05) and dt_bias 0 (decay ~0.966/step).

Output: one model.safetensors (bf16, written directly -- numpy has no bf16), config.json (the real
config's keys, the scaled values), the real tokenizer files, generation_config.json. bf16 tensors
are IEEE half-precision floats widened by the reader; we write uint16 bit patterns directly.
"""
import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

I64_DTYPES = {"I64"}


def bf16_bits(x16: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns (uint16), round-to-nearest-even, like torch's .to(bfloat16)."""
    u = x16.astype(np.float32).view(np.uint32)
    # round the low 16 bits up when the remainder is past halfway (ties to even handled by +bit0)
    rounded = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounded) >> 16).astype(np.uint16)


def write_safetensors(tensors: dict, path: Path):
    """One .safetensors file; bf16 as raw uint16 bits, i64 as int64. Same layout as HF's."""
    blobs, header = [], {}
    at = 0
    for name, (dt, arr) in tensors.items():
        raw = arr.astype({"BF16": "<u2", "I64": "<i8", "F32": "<f4", "I32": "<i4"}[dt],
                        copy=False).tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    head = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - len(head) % 8) % 8
    head += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for b in blobs:
            f.write(b)


def is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    d = 3
    while d * d <= v:
        if v % d == 0:
            return False
        d += 2
    return True


def ngram_primes(base: int, heads: int):
    """The 1st..heads-th prime above base-1, exactly as radiance's ngram_table_rows counts them."""
    out, p = [], base - 1
    for _ in range(heads):
        p += 1
        while not is_prime(p):
            p += 1
        out.append(p)
    return out


class Mini:
    """Every dimension, derived once, so tensor shapes and config.json cannot drift apart."""

    def __init__(self, args):
        self.n = args.hidden                      # hidden_size
        self.layers = args.layers
        self.experts = args.experts
        self.top_k = args.topk
        self.ngram_base = args.ngram_base
        self.hc = 4                               # hc_count (kept)
        self.lowrank = 320                        # hc_lowrank (kept)
        self.head_dim = 256                       # per-head (kept)
        self.n_head, self.n_head_kv = 24, 2       # per-head counts (kept)
        self.nk, self.nv, self.kdim = self.nk_nv()  # 16, 48, 128 (kept)
        self.n_ff = 640                           # moe_intermediate_size (kept)
        self.ple_embed = 2560                     # ple_embed_dim = 16 heads x 160 (kept)
        self.ple_heads = 16                       # (ngram-1) * heads_per_ngram (kept)
        self.ple_head_dim = self.ple_embed // self.ple_heads   # 160 (kept)
        self.shards = 128                         # split_ngram_parts (kept)
        self.vocab = 248320                        # the REAL vocab (kept)
        self.q_dim = self.n_head * self.head_dim                # 6144
        self.kv_dim = self.n_head_kv * self.head_dim             # 512
        self.hn = self.hc * self.n                                # the wide stream
        self.conv_dim = 2 * self.nk * self.kdim + self.nv * self.kdim   # 10240
        self.v_dim = self.nv * self.kdim                          # 6144
        self.ple_layer = 1                                        # ple_layer_ids [2], 1-indexed
        self.rng = np.random.default_rng(args.seed)
        self._scale = args.scale

    def nk_nv(self):
        return 16, 48, 128

    def rand(self, *shape, std=None):
        """N(0, std) with std = scale/sqrt(fan-in) by default: activations stay finite."""
        fan = max(1, int(np.prod(shape[1:])) if len(shape) > 1 else 1)
        return self.rng.normal(0.0, std if std is not None else self._scale / np.sqrt(fan),
                               size=shape).astype(np.float32)


def tensors(m: Mini) -> dict:
    t = {}
    add = lambda name, dt, arr: t.setdefault(name, (dt, np.asarray(arr)))
    rand = m.rand
    idx_qk_rows = (4 + 1) * 128          # indexer_n_heads 4, indexer_head_dim 128 (both kept)

    def w(name, *shape, std=None):
        add(name, "BF16", bf16_bits(rand(*shape, std=std)))

    def gain(name, n, one=False):
        add(name, "BF16", bf16_bits(np.full(n, 1.0 if one else 0.0, np.float32)))

    # ---- the vocabulary edges: real vocab, random rows.
    w("lm_head.weight", m.vocab, m.n)
    w("model.language_model.embed_tokens.weight", m.vocab, m.n)

    # ---- the 97th connection (no write half).
    hc_conn = lambda pre: (
        gain(f"{pre}.hc_norm.weight", m.hn),
        w(f"{pre}.input_mix_weight_down.weight", m.lowrank, m.hn),
        w(f"{pre}.input_mix_weight_up.weight", m.hn, m.lowrank),
    )
    hc_conn("model.language_model.hyper_connection_mixer")

    for l in range(m.layers):
        pre = f"model.language_model.layers.{l}"
        full = (l + 1) % 4 == 0                     # full_attention_interval 4
        for kind in ("attn_hyper_connection", "mlp_hyper_connection"):
            hc_conn(f"{pre}.{kind}")
            w(f"{pre}.{kind}.block_inject_weight.weight", m.hc, m.hn)
        if full:
            sa = f"{pre}.self_attn"
            w(f"{sa}.q_proj.weight", 2 * m.q_dim, m.n)     # gated: half of it the sigmoid gate
            w(f"{sa}.k_proj.weight", m.kv_dim, m.n)
            w(f"{sa}.v_proj.weight", m.kv_dim, m.n)
            w(f"{sa}.o_proj.weight", m.n, m.q_dim)
            gain(f"{sa}.q_norm.weight", m.head_dim, one=True)
            gain(f"{sa}.k_norm.weight", m.head_dim, one=True)
            ix = f"{sa}.indexer"
            w(f"{ix}.index_qk_proj.weight", idx_qk_rows, m.n)
            gain(f"{ix}.q_layernorm.weight", 128, one=True)
            gain(f"{ix}.k_layernorm.weight", 128, one=True)
        else:
            la = f"{pre}.linear_attn"
            w(f"{la}.in_proj_qkv.weight", m.conv_dim + m.v_dim, m.n)
            w(f"{la}.in_proj_z.weight", m.v_dim, m.n)
            w(f"{la}.in_proj_a.weight", m.nv, m.n)
            w(f"{la}.in_proj_b.weight", m.nv, m.n)
            w(f"{la}.out_proj.weight", m.n, m.v_dim)
            w(f"{la}.conv1d.weight", m.conv_dim, 1, 4)
            add(f"{la}.A_log", "BF16", bf16_bits(np.full(m.nv, np.log(0.05), np.float32)))
            add(f"{la}.dt_bias", "BF16", bf16_bits(np.zeros(m.nv, np.float32)))
            gain(f"{la}.norm.weight", m.kdim, one=True)     # the gated output norm: ONES
        moe = f"{pre}.mlp"
        w(f"{moe}.gate.weight", m.experts, m.n)
        w(f"{moe}.experts.gate_up_proj", m.experts, 2 * m.n_ff, m.n)
        w(f"{moe}.experts.down_proj", m.experts, m.n, m.n_ff)
        w(f"{moe}.shared_expert.gate_proj.weight", m.n_ff, m.n)
        w(f"{moe}.shared_expert.up_proj.weight", m.n_ff, m.n)
        w(f"{moe}.shared_expert.down_proj.weight", m.n, m.n_ff)
        w(f"{moe}.shared_expert_gate.weight", 1, m.n)
        if l == m.ple_layer:
            ple = f"{pre}.ple"
            w(f"{ple}.key_proj.weight", m.hn, m.ple_embed)
            w(f"{ple}.value_proj.weight", m.n, m.ple_embed)
            gain(f"{ple}.norm_key.weight", m.hn, one=True)
            gain(f"{ple}.norm_query.weight", m.hn, one=True)
            gain(f"{ple}.norm_conv.weight", m.hn, one=True)
            w(f"{ple}.conv1d.weight", m.hn, 1, 4)
            # ---- the n-gram hash constants and the per-head vocab geometry radiance's hash reads.
            add(f"{ple}.ple_embedding.layer_multipliers", "I64",
                m.rng.integers(1 << 62, (1 << 63) - 1, 3, dtype=np.int64))
            primes = ngram_primes(m.ngram_base, m.ple_heads)
            total = -(-sum(primes) // 128) * 128        # padded: make_ngram_vocab_size_divisible_by
            offs = np.cumsum([0] + primes[:-1])
            add(f"{ple}.ple_embedding.ngram_heads_vocab_sizes", "I64", np.array(primes, np.int64))
            add(f"{ple}.ple_embedding.ngram_heads_offsets", "I64", offs)
            # ---- the table: head h's prime rows at offset offs[h], padding at the end; 128 shards.
            table = rand(total, m.ple_head_dim)
            for h, p in enumerate(primes):
                table[offs[h]:offs[h] + p] = rand(p, m.ple_head_dim)
            rows = total // m.shards                   # total is a multiple of 128 by the padding
            for i in range(m.shards):
                add(f"{ple}.ple_embedding.ngram_embedding.shard_{i}.weight", "BF16",
                    bf16_bits(table[i * rows:(i + 1) * rows]))
    # ---- the MTP head: one full-attention layer + 512->16 routed experts, as the checkpoint ships it.
    add("mtp.pre_fc_norm_hidden.weight", "BF16", bf16_bits(np.zeros(m.hn, np.float32)))
    gain("mtp.pre_fc_norm_embedding.weight", m.n)
    w("mtp.fc_hidden.weight", m.n, m.n)
    w("mtp.fc_embedding.weight", m.n, m.n)
    hc_conn("mtp.hyper_connection_mixer")
    pre = "mtp.layers.0"
    for kind in ("attn_hyper_connection", "mlp_hyper_connection"):
        hc_conn(f"{pre}.{kind}")
        w(f"{pre}.{kind}.block_inject_weight.weight", m.hc, m.hn)
    sa = f"{pre}.self_attn"
    w(f"{sa}.q_proj.weight", 2 * m.q_dim, m.n)
    w(f"{sa}.k_proj.weight", m.kv_dim, m.n)
    w(f"{sa}.v_proj.weight", m.kv_dim, m.n)
    w(f"{sa}.o_proj.weight", m.n, m.q_dim)
    gain(f"{sa}.q_norm.weight", m.head_dim, one=True)
    gain(f"{sa}.k_norm.weight", m.head_dim, one=True)
    w(f"{sa}.indexer.index_qk_proj.weight", idx_qk_rows, m.n)
    gain(f"{sa}.indexer.q_layernorm.weight", 128, one=True)
    gain(f"{sa}.indexer.k_layernorm.weight", 128, one=True)
    moe = f"{pre}.mlp"
    w(f"{moe}.gate.weight", m.experts, m.n)
    w(f"{moe}.experts.gate_up_proj", m.experts, 2 * m.n_ff, m.n)
    w(f"{moe}.experts.down_proj", m.experts, m.n, m.n_ff)
    w(f"{moe}.shared_expert.gate_proj.weight", m.n_ff, m.n)
    w(f"{moe}.shared_expert.up_proj.weight", m.n_ff, m.n)
    w(f"{moe}.shared_expert.down_proj.weight", m.n, m.n_ff)
    w(f"{moe}.shared_expert_gate.weight", 1, m.n)
    return t


def write_config(real: Path, m: Mini, out: Path):
    cfg = json.loads((real / "config.json").read_text(encoding="utf-8"))
    tc = cfg["text_config"]
    tc["hidden_size"] = m.n
    tc["num_hidden_layers"] = m.layers
    tc["num_experts"] = m.experts
    tc["num_experts_per_tok"] = m.top_k
    tc["ngram_vocab_size_base"] = m.ngram_base
    tc["layer_types"] = ["full_attention" if (l + 1) % 4 == 0 else "linear_attention"
                         for l in range(m.layers)]
    tc["mtp"]["layer_types"] = ["full_attention"]
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the checkpoint directory (created)")
    ap.add_argument("--real-config", default="/var/home/dylan/projects/inference/radiance-kva/data/stub",
                    help="dir with the real config.json (default: the published container's source stub)")
    ap.add_argument("--tokenizer", default="/var/home/dylan/models-boot/tcc-qwen38-flash-next-mxfp4-fp8-gptq",
                    help="dir with the real tokenizer files")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--ngram-base", type=int, default=1000)
    ap.add_argument("--scale", type=float, default=1.0, help="weight std = scale/sqrt(fan-in)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.layers % 4:
        sys.exit("--layers must be a multiple of 4 (the full_attention_interval-4 pattern)")
    m = Mini(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = write_config(Path(args.real_config), m, out)
    t = tensors(m)
    write_safetensors(t, out / "model.safetensors")
    tok = Path(args.tokenizer)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "generation_config.json", "chat_template.jinja"):
        src = tok / f
        if not src.exists():
            src = Path(args.real_config) / f
        if src.exists():
            shutil.copy2(src, out / f)
        else:
            print(f"warning: {f} not found in either source dir", file=sys.stderr)
    (out / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": (out / "model.safetensors").stat().st_size},
        "weight_map": {name: "model.safetensors" for name in t}}, indent=1) + "\n",
        encoding="utf-8")
    size = (out / "model.safetensors").stat().st_size
    primes = ngram_primes(m.ngram_base, m.ple_heads)
    print(f"mini checkpoint at {out}: hidden {m.n}, {m.layers} layers, {m.experts} experts "
          f"top-{m.top_k}, ngram table {sum(primes)} rows (padded {-(-sum(primes)//128)*128}, "
          f"{m.shards} shards of {-(-sum(primes)//128)*128//m.shards}), vocab {m.vocab}; "
          f"model.safetensors {size/2**30:.3f} GiB, {len(t)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())