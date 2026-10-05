#!/usr/bin/env python3
"""Build the KVA projector folder: what a user downloads into `<model dir>/projector/` (PACKAGING.md §2, §3).

  kva_projector.py build --proj P --st RANK0.pt RANK1.pt --freq F --tokenizer DIR --container MODEL.rad \\
                         --rad-info-v FILE --spec kva-marker-spec.json --out DIR

The model file is only READ (its metadata, tokenizer, chat template and a few KiB of base tensors); nothing is
written to it. Files written (L = the projector's layers, S the lowest):
  proj.L<L>.safetensors   proj.L.weight [n_embd, stream_width] bf16, proj.L.bias [n_embd] bf16 (one file a layer,
                          so a partial download fails by name)
  correction.safetensors  st.L [heads, V, K] f32 per delta-net layer (cat of the per-rank fits, rank 0's heads first)
  rowsel.safetensors      score / score_none / score_all [vocab] f32 (the class table and its two controls)
  chat_template.jinja     the model's own template with the kva marker block in front (tools/kva_template.py merge)
  README.md               the install flow
  kva.json                the manifest: layout, defaults, the model fingerprint, every file's sha256

THE FINGERPRINT (arch/kva_match.h reads it; Dylan's DD-K split):
  cannot run if different  -> arch_id, `meta` (the model's own metadata values, as the engine prints them),
                              `vocab_sha256` (each token's text, NUL, type byte in id order, then the merge table)
  warns if different       -> `encodings` (every non-expert late-layer weight, from `rad-info -v`), `anchors` (sha256
                              of the planes of the hyper-connection norms around the split), `name`
The tensors are the bytes the container append carried (data/sidecar/kva-sidecar.safetensors): same readers.
"""
import argparse
import hashlib
import json
import mmap
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "dev"))
import kva_sidecar as S  # noqa: E402  (the source readers; the append itself is the retired route)
import kva_template as T  # noqa: E402

FORMAT = 1
ADAPTER = "qwen4exp"
# The metadata the plugin compares as "cannot run": dimensions, the layer layout and the delta-net head geometry.
META_KEYS = ("model_type", "n_layers", "n_embd", "n_vocab", "n_expert", "hc_count", "layer_types",
             "full_attention_interval", "linear_num_value_heads", "linear_value_head_dim", "linear_key_head_dim")
RAD_MAGIC, RAD_FORMAT_VER = 0x31444152, 2
HEADER = struct.Struct("<IIQ5Q16Q8Q")          # RadFileHeader, 248 B
KV = struct.Struct("<QII8s")                    # RadFileKV, 24 B
ENTRY = struct.Struct("<QQQII6qiiiIQQQ2Q")      # RadFileEntry, 136 B
PLANE = struct.Struct("<QQ")                    # RadFilePlane, 16 B
VOCAB = struct.Struct("<IIII5Q6i2IQ4Q")         # RadVocabHeader, 128 B
P_INT, P_STR, KV_F64 = 1, 2, 100


class Rad:
    """A .rad container read through its public format (abi/rad_format.h), memory-mapped read-only."""

    def __init__(self, path):
        self.f = open(path, "rb")
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        h = HEADER.unpack_from(self.m, 0)
        if h[0] != RAD_MAGIC or h[1] != RAD_FORMAT_VER:
            raise SystemExit(f"{path}: not a format-{RAD_FORMAT_VER} .rad container")
        (self.str_off, self.str_bytes, self.meta_off, self.meta_count, self.dir_off, self.dir_count,
         self.plane_off, self.plane_count, _, _, self.vocab_off, self.vocab_bytes) = h[8:20]
        self.arch = self.str(h[3])
        self.name = self.str(h[4]) or self.arch   # the engine names an unnamed model by its arch

    def str(self, off):
        if off >= self.str_bytes:
            return ""
        at = self.str_off + off
        return self.m[at:self.m.find(b"\0", at)].decode("utf-8")

    def meta(self):
        """{key: value} as the engine hands them to a plugin: ints %lld, floats %.17g, strings verbatim."""
        out = {}
        for i in range(self.meta_count):
            key, kind, _, raw = KV.unpack_from(self.m, self.meta_off + i * KV.size)
            if kind == P_INT:
                out[self.str(key)] = str(struct.unpack("<q", raw)[0])
            elif kind == KV_F64:
                out[self.str(key)] = "%.17g" % struct.unpack("<d", raw)[0]
            else:
                out[self.str(key)] = self.str(struct.unpack("<Q", raw)[0])
        return out

    def entry_sha256(self, name):
        for i in range(self.dir_count):
            e = ENTRY.unpack_from(self.m, self.dir_off + i * ENTRY.size)
            if self.str(e[0]) != name:
                continue
            digest = hashlib.sha256()
            for k in range(e[15], e[15] + e[14]):
                off, n = PLANE.unpack_from(self.m, self.plane_off + k * PLANE.size)
                digest.update(self.m[off:off + n])
            return digest.hexdigest()
        raise SystemExit(f"anchor tensor {name} is not in the container")

    def _vocab(self):
        return VOCAB.unpack_from(self.m, self.vocab_off)

    def vocab_sha256(self):
        v = self._vocab()
        n_tok, n_merge, text_off, type_off, merge_off = v[1], v[2], v[4], v[6], v[7]
        digest = hashlib.sha256()
        for i in range(n_tok):
            (so,) = struct.unpack_from("<Q", self.m, text_off + 8 * i)
            digest.update(self.str(so).encode("utf-8") + b"\0" + self.m[type_off + i:type_off + i + 1])
        digest.update(self.m[merge_off:merge_off + 8 * n_merge])
        return digest.hexdigest()

    def chat_template(self):
        return self.str(self._vocab()[17])


def rad_info_encodings(path, layers):
    """{name: encoding} of every non-expert weight of `layers` in saved `rad-info -v` output."""
    out = {}
    prefixes = tuple(f"blk.{layer}." for layer in layers)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].startswith(prefixes) and "_exps." not in fields[0]:
            out[fields[0]] = fields[1]
    if not out:
        raise SystemExit(f"{path}: no weight of layers {layers[0]}..{layers[-1]} (is it `rad-info -v` output?)")
    return dict(sorted(out.items()))


def save(tensors, path):
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path), metadata={"format": "kva-projector-1"})


def write_tensors(args, out):
    """The three tensor groups into their files; (split, layers, correction layers, projector source metadata)."""
    split, proj = S.projector_tensors(Path(args.proj), "proj")
    layers = sorted({int(k.split(".")[1]) for k in proj})
    for layer in layers:
        save({k: v for k, v in proj.items() if k.startswith(f"proj.{layer}.")}, out / f"proj.L{layer}.safetensors")
    st = S.correction_tensors(Path(args.st[0]), Path(args.st[1]), split, "st", None)
    save(st, out / "correction.safetensors")
    rowsel, kept, n_tok = S.rowsel_tensors(args.tokenizer, Path(args.freq))
    save({k.replace("kva.rowsel.", ""): v for k, v in rowsel.items()}, out / "rowsel.safetensors")
    print(f"split {split}; {len(layers)} projector layers; {len(st)} correction layers; "
          f"rowsel {kept} of {len(rowsel['kva.rowsel.score'])} rows kept ({n_tok} tokenizer ids)")
    first = proj[f"proj.{split}.weight"]
    return split, layers, sorted(int(k.split(".")[1]) for k in st), first.shape, next(iter(st.values())).shape


def fingerprint(rad, args, split, layers):
    meta = rad.meta()
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise SystemExit(f"{args.container}: metadata lacks {missing}")
    anchors = [f"blk.{split - 1}.attn_hc_norm.weight", f"blk.{split - 1}.ffn_hc_norm.weight",
               f"blk.{split}.attn_hc_norm.weight"]
    return {"arch_id": rad.arch,
            "name": rad.name,
            "meta": {k: meta[k] for k in META_KEYS},
            "vocab_sha256": rad.vocab_sha256(),
            "encodings": rad_info_encodings(args.rad_info_v, [split - 1] + layers),
            "anchors": {a: rad.entry_sha256(a) for a in anchors}}


def dials(spec):
    """The marker's dial tables, by kwarg, as numbers."""
    return {d["kwarg"]: sorted(float(v) for v in d["table"]) for d in spec["dials"]}


def write_template(rad, args, out):
    base = out / ".template-source.jinja"
    base.write_text(rad.chat_template(), encoding="utf-8")
    try:
        if T.main(["merge", "--spec", args.spec, "--base", str(base), "--out", str(out / "chat_template.jinja")]):
            raise SystemExit("kva_template.py merge failed")
        return hashlib.sha256(base.read_bytes()).hexdigest()
    finally:
        base.unlink()


README = """# KVA projector for {name}

The fitted tensors the radiance KVA plugin reads. The model file stays exactly as published.

1. Serve the stock model as you do today.
2. Install the plugin: unpack it to e.g. /opt/kva (architectures/qwen4exp_fp8.so, kernels/kva.so).
3. Put this folder next to your model:  hf download <this repo> --local-dir <model dir>/projector
4. Start radiance with  RADIANCE_HOME=/opt/kva:/opt/radiance/share/radiance  and RADIANCE_KVA=quality (every
   request) -- or, for per-request KVA, --override-chat-template <model dir>/projector/chat_template.jinja.
5. Check the startup log: "KVA: projector <folder> ... matches <model>".

Docker: mount the model's DIRECTORY (not the single file), or set RADIANCE_KVA_PROJECTOR to this folder.
RADIANCE_KVA_PROJ_PLACE=host keeps the projector maps off the cards (slower approximated chunks).
"""


def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rad = Rad(args.container)
    spec = T.load_spec(args.spec)
    split, layers, st_layers, pshape, sshape = write_tensors(args, out)
    model = fingerprint(rad, args, split, layers)
    template_sha = write_template(rad, args, out)
    (out / "README.md").write_text(README.format(name=rad.name), encoding="utf-8")
    types = model["meta"]["layer_types"].split()
    table = dials(spec)
    manifest = {
        "format": FORMAT, "plugin_min_version": "0.3.0", "adapter": ADAPTER, "adapter_abi": 1,
        "model": model, "split": split, "stream": "hc", "stream_width": pshape[1], "block_in_width": pshape[0],
        "layers": {str(L): "full_attn" if types[L] == "full_attention" else "recurrent" for L in layers},
        "tail": {"default": 2048, "min": 512, "table": [int(t) for t in table["kva_tail"]]},
        "correction": {"kind": "gdn_terminal_state", "alpha_default": 1.0, "alpha_table": table["kva_alpha"],
                       "heads": sshape[0], "sd": list(sshape[1:]), "layers": st_layers},
        "rowsel": {"classes": "cap,mixed,piece", "share_default": 0.25, "share_table": table["kva_share"]},
        "projector": {"dtype": "bf16", "layout": "plain_nk",
                      "files": {str(L): f"proj.L{L}.safetensors" for L in layers}},
        "marker": {"spec": spec, "template_source_sha256": template_sha},
        "fit": {"engine": "radiance 1.0.8",
                **{f"{role}_sha256": S.sha256_file(p) for role, p in
                   (("proj", args.proj), ("st0", args.st[0]), ("st1", args.st[1]), ("freq", args.freq),
                    ("tokenizer_json", Path(args.tokenizer) / "tokenizer.json"))}},
    }
    files = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "kva.json")
    manifest["files"] = {name: S.sha256_file(out / name) for name in files}
    (out / "kva.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    total = sum((out / n).stat().st_size for n in files) + (out / "kva.json").stat().st_size
    print(f"wrote {out}: {len(files) + 1} files, {total} bytes ({total / 2**30:.3f} GiB); model {rad.name}, "
          f"vocab {model['vocab_sha256'][:12]}..., {len(model['encodings'])} encodings, {len(model['anchors'])} anchors")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--proj", required=True, help="projector safetensors (layer.S..layer.L-1 [H, W+1] bf16)")
    b.add_argument("--st", nargs=2, required=True, metavar=("RANK0", "RANK1"), help="correction .pt per TP rank")
    b.add_argument("--freq", required=True, help="unigram table safetensors (`logfreq` [vocab])")
    b.add_argument("--tokenizer", required=True, help="checkpoint dir with config.json + tokenizer.json")
    b.add_argument("--container", required=True, help="the served .rad (read only)")
    b.add_argument("--rad-info-v", required=True, help="saved `rad-info -v` output of that container")
    b.add_argument("--spec", required=True, help="kva-marker-spec.json (tools/kva_template.py)")
    b.add_argument("--out", required=True, help="the folder to write (created)")
    args = ap.parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
