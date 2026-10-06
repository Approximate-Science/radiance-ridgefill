#!/usr/bin/env python3
"""Build the KVA projector folder: what a user downloads into `<model dir>/projector/` (PACKAGING.md §2, §3).

  kva_projector.py build --proj P --st RANK0.pt RANK1.pt --freq F --tokenizer DIR --container MODEL.rad \\
                         --rad-info-v FILE --spec kva-marker-spec.json --out DIR
  kva_projector.py int8 --from BF16_FOLDER --out DIR
  kva_projector.py final --from FOLDER --proj P --out DIR

The model file is only READ (its metadata, tokenizer, chat template and a few KiB of base tensors); nothing is
written to it. Files written (L = the projector's layers, S the lowest):
  proj.L<L>.safetensors   proj.L.weight [n_embd, stream_width] bf16, proj.L.bias [n_embd] bf16 (one file a layer,
                          so a partial download fails by name)
  correction.safetensors  st.L [heads, V, K] f32 per delta-net layer (cat of the per-rank fits, rank 0's heads first)
  rowsel.safetensors      score / score_none / score_all [vocab] f32 (the class table and its two controls)
  chat_template.jinja     the model's own template with the kva marker block in front (tools/kva_template.py merge)
  README.md               the install flow (NOT in the manifest: documentation is never hashed, see is_doc)
  kva.json                the manifest: layout, defaults, the model fingerprint, every other file's sha256

THE FINGERPRINT (arch/kva_match.h reads it; Dylan's DD-K split):
  cannot run if different  -> arch_id, `meta` (the model's own metadata values, as the engine prints them),
                              `vocab_sha256` (each token's text, NUL, type byte in id order, then the merge table)
  warns if different       -> `encodings` (every non-expert late-layer weight, from `rad-info -v`), `anchors` (sha256
                              of the planes of the hyper-connection norms around the split), `name`
The tensors are the bytes the container append carried (data/sidecar/kva-sidecar.safetensors): same readers.

THE INT8 VARIANT (`int8`; Stage E, R79) is its own folder, selected with RADIANCE_KVA_PROJECTOR: every map quantised
offline to the encoding the container's own int8 trunk uses, i8*bf16[1x128] -- codes i8 [n_embd, stream_width]
row-major and a bf16 scale per 128 columns of a row, value = code * scale, the scale absmax/127 rounded to bf16 and
each code rounded half-to-even against the ROUNDED scale (libquant's rtn rule). proj8.L<L>.safetensors holds
proj.L.codes, proj.L.scale and the unchanged bf16 proj.L.bias. The file is canonical planes only: the plugin
relayouts them at load through the int8 GEMM's own layout hook, so no kernel library's arrangement is ever written
here. Every other file is copied byte for byte; the manifest says `"dtype": "i8"` and names its source folder.
"""
import argparse
import hashlib
import json
import mmap
import shutil
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

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
4. Start radiance with  RADIANCE_HOME=/opt/kva:/opt/radiance/share/radiance  and RADIANCE_KVA=quality
   (or speed); the mode is server-wide.
5. Check the startup log: "KVA: projector <folder> ... matches <model>".

Docker: mount the model's DIRECTORY (not the single file), or set RADIANCE_KVA_PROJECTOR to this folder.
The projector is streamed from host memory: it costs each card one one-layer staging slot, not the whole map.
"""


def is_doc(name):
    """Documentation is never listed in kva.json's files: a hub serves the repo's README.md as its model card, so the
    README a user downloads is not the one written here, and a hashed README would make the loader refuse a correct
    folder (arch/kva_folder.h verifies exactly the listed files and ignores the rest)."""
    return name.lower().endswith(".md")


def hashed(folder, names):
    """The manifest's `files`: every listed name that is not documentation, with its sha256."""
    return {name: S.sha256_file(Path(folder) / name) for name in sorted(names) if not is_doc(name)}


def copy_docs(src, out):
    for doc in sorted(p for p in Path(src).iterdir() if p.is_file() and is_doc(p.name)):
        shutil.copyfile(doc, Path(out) / doc.name)


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
    manifest["files"] = hashed(out, files)
    (out / "kva.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    total = sum((out / n).stat().st_size for n in files) + (out / "kva.json").stat().st_size
    print(f"wrote {out}: {len(files) + 1} files, {total} bytes ({total / 2**30:.3f} GiB); model {rad.name}, "
          f"vocab {model['vocab_sha256'][:12]}..., {len(model['encodings'])} encodings, {len(model['anchors'])} anchors")


def quantise_i8(w):
    """i8*bf16[1x128] of a [N, K] map: (codes i8 [N, K], scale bf16 [N, K/128], max |w - code*scale| / max |w|)."""
    n, k = w.shape
    if k % 128:
        raise SystemExit(f"a map of {k} columns is not a whole number of 128-column groups")
    x = w.float().reshape(n, k // 128, 128)
    amax = x.abs().amax(-1)
    scale = (amax / 127).to(torch.bfloat16)
    scale[amax == 0] = 1.0   # a zero group decodes to zero whatever its scale; 1 as quant_act_i8g writes it
    codes = torch.round(x / scale.float().unsqueeze(-1)).clamp(-127, 127)
    err = (codes * scale.float().unsqueeze(-1) - x).abs().max() / x.abs().max()
    return codes.to(torch.int8).reshape(n, k), scale, float(err)


def int8(args):
    src, out = Path(getattr(args, "from")), Path(args.out)
    manifest = json.loads((src / "kva.json").read_text(encoding="utf-8"))
    proj = manifest["projector"]
    if proj["dtype"] != "bf16":
        raise SystemExit(f"{src}: its projector is {proj['dtype']}; the int8 variant is made from a bf16 folder")
    out.mkdir(parents=True, exist_ok=True)
    files, worst = {}, 0.0
    for layer, name in sorted(proj["files"].items(), key=lambda kv: int(kv[0])):
        t = load_file(str(src / name))
        codes, scale, err = quantise_i8(t[f"proj.{layer}.weight"])
        worst = max(worst, err)
        files[layer] = f"proj8.L{layer}.safetensors"
        save({f"proj.{layer}.codes": codes, f"proj.{layer}.scale": scale, f"proj.{layer}.bias": t[f"proj.{layer}.bias"]},
             out / files[layer])
    kept = [n for n in manifest["files"] if n not in proj["files"].values()]
    for name in kept:
        shutil.copyfile(src / name, out / name)
    copy_docs(src, out)
    manifest["projector"] = {"dtype": "i8", "layout": "i8_row128", "encoding": "i8*bf16[1x128]",
                             "files": files, "source": {"folder": src.name,
                                                        "kva_json_sha256": S.sha256_file(src / "kva.json")}}
    names = sorted(kept + list(files.values()))
    manifest["files"] = hashed(out, names)
    (out / "kva.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    total = sum((out / n).stat().st_size for n in names) + (out / "kva.json").stat().st_size
    print(f"wrote {out}: {len(names) + 1} files, {total} bytes ({total / 2**30:.3f} GiB); {len(files)} maps i8*bf16[1x128], "
          f"worst |w - dequant| {worst:.4g} of a map's max |w|")


def final(args):
    """FOLDER + the MTP `final` map (Stage D, DD-D): final.safetensors with final.weight [w, w] and final.bias [w] bf16,
    w the stream width, from the source projector's `final` [w, w + 1] (bias last). The source must be the file the
    folder was fitted from (its sha256 against the manifest's fit.proj_sha256). Every other file is copied byte for byte."""
    src, out = Path(getattr(args, "from")), Path(args.out)
    manifest = json.loads((src / "kva.json").read_text(encoding="utf-8"))
    want = (manifest.get("fit") or {}).get("proj_sha256")
    got = S.sha256_file(args.proj)
    if want != got:
        raise SystemExit(f"{args.proj} hashes {got[:12]}..., and {src} was fitted from {str(want)[:12]}...: not its final map")
    with safe_open(str(args.proj), "pt") as f:
        if "final" not in f.keys():
            raise SystemExit(f"{args.proj} holds no `final` map")
        t = f.get_tensor("final")
    wide = manifest["stream_width"]
    if t.dtype != torch.bfloat16 or list(t.shape) != [wide, wide + 1]:
        raise SystemExit(f"final is {t.dtype} {list(t.shape)}; expected bf16 [{wide}, {wide + 1}]")
    out.mkdir(parents=True, exist_ok=True)
    for name in manifest["files"]:
        shutil.copyfile(src / name, out / name)
    copy_docs(src, out)
    save({"final.weight": t[:, :-1], "final.bias": t[:, -1]}, out / "final.safetensors")
    manifest["final"] = {"file": "final.safetensors", "dtype": "bf16", "source_sha256": got}
    manifest["files"] = hashed(out, list(manifest["files"]) + ["final.safetensors"])
    (out / "kva.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {src.name} + final.safetensors ({(out / 'final.safetensors').stat().st_size} bytes)")


def reseal(args):
    """Rewrite an existing folder's kva.json so it lists no documentation (is_doc), touching no other file: every listed
    file that stays listed is first checked against the manifest's own hash (a changed weight is refused, never
    re-blessed), the fingerprint, anchors and every other field are kept as they are."""
    folder = Path(args.folder)
    path = folder / "kva.json"
    old = path.read_bytes()
    manifest = json.loads(old)
    for name, digest in manifest["files"].items():
        if is_doc(name):
            continue
        got = S.sha256_file(folder / name)
        if got != digest:
            raise SystemExit(f"{folder / name} hashes {got[:12]}..., the manifest says {digest[:12]}...: not resealing a changed folder")
    dropped = sorted(n for n in manifest["files"] if is_doc(n))
    manifest["files"] = {n: d for n, d in manifest["files"].items() if not is_doc(n)}
    new = (json.dumps(manifest, indent=1) + "\n").encode()
    path.write_bytes(new)
    print(f"resealed {folder}: kva.json {hashlib.sha256(old).hexdigest()} -> {hashlib.sha256(new).hexdigest()}; "
          f"no longer listed: {', '.join(dropped) or 'nothing'}; {len(manifest['files'])} files listed, none touched")


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
    q = sub.add_parser("int8", help="the int8 variant of a bf16 folder (its own folder)")
    q.add_argument("--from", required=True, help="a bf16 projector folder")
    q.add_argument("--out", required=True, help="the folder to write (created)")
    fm = sub.add_parser("final", help="a folder plus the MTP final map (its own folder)")
    fm.add_argument("--from", required=True, help="a projector folder")
    fm.add_argument("--proj", required=True, help="the projector safetensors the folder was fitted from (holds `final`)")
    fm.add_argument("--out", required=True, help="the folder to write (created)")
    rs = sub.add_parser("reseal", help="rewrite an existing folder's kva.json without documentation; no file touched")
    rs.add_argument("--folder", required=True, help="a projector folder")
    args = ap.parse_args(argv)
    if args.command == "reseal":
        reseal(args)
    elif args.command == "final":
        final(args)
    elif args.command == "int8":
        int8(args)
    else:
        build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
