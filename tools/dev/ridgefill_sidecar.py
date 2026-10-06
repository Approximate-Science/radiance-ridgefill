#!/usr/bin/env python3
"""Build and verify the RidgeFill sidecar shard: the fitted projector, the GDN state correction and the row-selection
score tables, as one safetensors shard plus an index naming only it, so the directory is a checkpoint
namespace `rad-convert --reuse --in-place` can append from.

  ridgefill_sidecar.py build  --proj P --st RANK0.pt RANK1.pt --freq F --tokenizer DIR --out DIR
  ridgefill_sidecar.py verify TARGET --proj P --st RANK0.pt RANK1.pt --freq F --tokenizer DIR [--rad-info CMD]
  ridgefill_sidecar.py build  --names refit --proj P [--st RANK0.pt RANK1.pt] --out DIR    (Stage 6 refit set)

Tensors written (L = the layers present in the inputs; no model number is typed here):
  ridgefill.proj.L.weight [H, W] bf16     projector layer.L[:, :W]   (W = its columns - 1)
  ridgefill.proj.L.bias   [H] bf16        projector layer.L[:, W]    (the bias column); `final` is not converted
  ridgefill.st.L          [2h, V, K] f32  cat(rank0 sum/count, rank1 sum/count) along heads: rank 0's heads first
  ridgefill.stswap.L      [2h, V, K] f32  the same with the halves swapped: the head-order negative control (R24)
  ridgefill.rowsel.score       [vocab] f32  rarity (-logfreq) where the token's class is kept, else -inf
  ridgefill.rowsel.score_none  [vocab] f32  all -inf: no row selected, rho = 1 (R41)
  ridgefill.rowsel.score_all   [vocab] f32  all 0: every row a match, ties by position (the all-rows check, R35)
`--names refit` writes only ridgefill.projr.L.{weight,bias} and ridgefill.str.L (no swap control, no row tables) into
ridgefill-sidecar-refit.safetensors, with source-hash keys ridgefill.src.{projr,str0,str1}.sha256. Its --st is optional: the
correction refit needs the refit projector IN the container (its speed run fills with it), so Stage 6 appends twice:
first the projector alone (no --st: no ridgefill.str.*, so RADIANCE_RIDGEFILL_ST=refit serves with no correction), then the
full refit set (the projector tensors are byte-identical and reused by name, only ridgefill.str.* is new).
The controls share the container with the real tensors because an in-place append cannot replace a weight and
there is no disk for a second container (orchestrator decision, 2026-10-04).

Metadata (in the shard header AND in rad-convert-set.txt as `key=value` lines for `rad-convert --set`, so the
container carries it): ridgefill.split, ridgefill.rowsel.share, ridgefill.rowsel.classes, and ridgefill.src.<role>.sha256 for every
file read. `verify` recomputes those hashes and compares them with a sidecar's header, a container's
`rad-info --meta`, or a saved copy of that output (R13).
"""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # tools/: ridgefill_rules
import ridgefill_rules as R  # noqa: E402

SET_FILE = "rad-convert-set.txt"
# Which copy of the fitted tensors a build writes (arch/ridgefill_config.h selects among them at serve time).
# shipped: the published fits, with the head-swap control and the row-selection tables.
# refit:   the Stage 6 fits from radiance's own captures, under their own names and source-hash keys, so
#          the two copies sit in one container side by side (an in-place append cannot replace a weight).
NAMES = {
    "shipped": dict(shard="ridgefill-sidecar.safetensors", proj="ridgefill.proj", st="ridgefill.st", swap="ridgefill.stswap",
                    roles=("proj", "st0", "st1")),
    "refit": dict(shard="ridgefill-sidecar-refit.safetensors", proj="ridgefill.projr", st="ridgefill.str", swap=None,
                  roles=("projr", "str0", "str1")),
}
SHARD = NAMES["shipped"]["shard"]
FORMAT = "ridgefill-sidecar-1"
# The selection share the paper's quality row was measured at (HANDOVER §2.4.5: changing it is Dylan's call).
SHARE = 0.25
TOKENIZER_FILES = {"config": "config.json", "tokenizer_json": "tokenizer.json",
                   "tokenizer_config": "tokenizer_config.json"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(args):
    """{role: path} of every file the build reads; a missing one stops the tool before any work. The
    row-selection inputs belong to the shipped set only."""
    names = NAMES[args.names]
    if args.names == "shipped" and not args.st:
        raise SystemExit("--names shipped needs --st RANK0 RANK1: the correction is part of the shipped set")
    files = dict(zip(names["roles"], [Path(args.proj)] + [Path(f) for f in args.st or ()]))
    if args.names == "shipped":
        if not (args.freq and args.tokenizer):
            raise SystemExit("--names shipped builds the row-selection tables: --freq and --tokenizer are required")
        files["freq"] = Path(args.freq)
        files.update({role: Path(args.tokenizer) / name for role, name in TOKENIZER_FILES.items()})
    elif args.freq or args.tokenizer:
        raise SystemExit("--names refit writes no row-selection tables: drop --freq and --tokenizer")
    missing = [f"{role}: {path}" for role, path in files.items() if not path.is_file()]
    if missing:
        raise SystemExit("input file(s) not found:\n  " + "\n  ".join(missing))
    return files


def source_hashes(files):
    return {f"ridgefill.src.{role}.sha256": sha256_file(path) for role, path in files.items()}


def projector_tensors(path, prefix):
    """(split, tensors): layer.L [H, W+1] -> <prefix>.L.weight [H, W] + .bias [H]; split = the lowest layer."""
    out = {}
    with safe_open(str(path), "pt") as f:
        layers = sorted(int(k.split(".", 1)[1]) for k in f.keys() if k.startswith("layer."))
        if not layers or layers != list(range(layers[0], layers[-1] + 1)):
            raise SystemExit(f"{path}: projector layers {layers} are not one contiguous run layer.S..layer.L-1")
        for layer in layers:
            w = f.get_tensor(f"layer.{layer}")
            if w.dim() != 2 or w.dtype != torch.bfloat16:
                raise SystemExit(f"{path}: layer.{layer} is {w.dtype} {list(w.shape)}; expected a 2-D bf16 [H, W+1]")
            out[f"{prefix}.{layer}.weight"] = w[:, :-1].contiguous()
            out[f"{prefix}.{layer}.bias"] = w[:, -1].contiguous()
    return layers[0], out


def load_correction(path):
    """{layer: f32 [h, V, K]} = sum / count, as b0/st_hook.py load_correction computes it."""
    d = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(d, dict) or set(d) != {"sum", "count"}:
        raise SystemExit(f"{path}: expected a dict with keys sum, count (b0/st_hook.py fit format)")
    if set(d["sum"]) != set(d["count"]):
        raise SystemExit(f"{path}: sum and count name different layers")
    return {int(layer): total / d["count"][layer] for layer, total in d["sum"].items()}


def correction_tensors(rank0_path, rank1_path, split, prefix, swap_prefix):
    rank0, rank1 = load_correction(rank0_path), load_correction(rank1_path)
    if set(rank0) != set(rank1):
        raise SystemExit(f"rank files name different layers: {sorted(rank0)} vs {sorted(rank1)}")
    if min(rank0) < split:
        raise SystemExit(f"correction layer {min(rank0)} is below the projector split {split}")
    out = {}
    for layer in sorted(rank0):
        a, b = rank0[layer], rank1[layer]
        if a.dtype != torch.float32 or a.shape != b.shape or a.dim() != 3:
            raise SystemExit(f"layer {layer}: halves {a.dtype} {list(a.shape)} / {b.dtype} {list(b.shape)}; "
                             "expected two f32 [h, V, K] of one shape")
        out[f"{prefix}.{layer}"] = torch.cat([a, b]).contiguous()
        if swap_prefix:
            out[f"{swap_prefix}.{layer}"] = torch.cat([b, a]).contiguous()
    return out


def rowsel_tensors(tokenizer_dir, freq_path):
    """(tensors, kept): the score tables over the MODEL's vocab rows (config.json), not the tokenizer's."""
    vocab = R.model_vocab(tokenizer_dir)
    logfreq = R.load_logfreq(freq_path)
    if len(logfreq) != vocab:
        raise SystemExit(f"{freq_path}: {len(logfreq)} entries, but the model has {vocab} vocab rows")
    texts = R.token_texts(R.load_tokenizer(tokenizer_dir))
    if len(texts) > vocab:
        raise SystemExit(f"tokenizer has {len(texts)} ids, more than the model's {vocab} vocab rows")
    score = torch.from_numpy(R.score_table(texts, logfreq))
    kept = int(torch.isfinite(score).sum())
    return {"ridgefill.rowsel.score": score,
            "ridgefill.rowsel.score_none": torch.full((vocab,), float("-inf"), dtype=torch.float32),
            "ridgefill.rowsel.score_all": torch.zeros(vocab, dtype=torch.float32)}, kept, len(texts)


def sort_header_metadata(path):
    """Rewrite the shard's header with __metadata__ in key order, same length, in place. safetensors writes
    the metadata map in hash order, which differs run to run, so without this two builds of the same inputs
    differ in a few header bytes and the shard's sha256 is not reproducible."""
    with open(path, "r+b") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
        header["__metadata__"] = dict(sorted(header["__metadata__"].items()))
        text = json.dumps(header, separators=(",", ":")).encode()
        if len(text) > size:
            raise SystemExit(f"{path}: re-serialised header is {len(text)} bytes, more than the {size} written")
        f.seek(8)
        f.write(text.ljust(size))


def write_index(out_dir, shard_name, tensors):
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    index = {"metadata": {"total_size": total}, "weight_map": {name: shard_name for name in sorted(tensors)}}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")


def build(args):
    names = NAMES[args.names]
    files = source_files(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("hashing sources ...", flush=True)
    meta = {"ridgefill.format": FORMAT, **source_hashes(files)}
    print("projector ...", flush=True)
    split, tensors = projector_tensors(files[names["roles"][0]], names["proj"])
    if args.st:
        tensors.update(correction_tensors(files[names["roles"][1]], files[names["roles"][2]], split,
                                          names["st"], names["swap"]))
    if args.names == "shipped":
        print("row-selection tables ...", flush=True)
        rowsel, kept, n_tok = rowsel_tensors(args.tokenizer, files["freq"])
        tensors.update(rowsel)
        meta.update({"ridgefill.split": str(split), "ridgefill.rowsel.share": repr(SHARE),
                     "ridgefill.rowsel.classes": ",".join(R.CLASSES)})
        print(f"rowsel: {kept} of {len(rowsel['ridgefill.rowsel.score'])} vocab rows kept ({n_tok} tokenizer ids)")
    assert all(" " not in v and "\n" not in v for v in meta.values()), "metadata values go on a command line"
    shard = out_dir / names["shard"]
    print(f"writing {shard} ({len(tensors)} tensors) ...", flush=True)
    save_file(tensors, str(shard), metadata=meta)
    sort_header_metadata(shard)
    write_index(out_dir, shard.name, tensors)
    (out_dir / SET_FILE).write_text("".join(f"{k}={v}\n" for k, v in sorted(meta.items())), encoding="utf-8")
    print(f"split {split}; {sum(k.startswith(names['proj'] + '.') for k in tensors) // 2} projector layers; "
          f"{sum(k.startswith(names['st'] + '.') for k in tensors)} correction layers")
    print(f"wrote {shard}, model.safetensors.index.json, {SET_FILE}")


def parse_rad_info_meta(text):
    """{key: value} of the ridgefill.* rows of `rad-info --meta` output (rows are `  key  value`)."""
    meta = {}
    for line in text.splitlines():
        fields = line.split(None, 1)
        if len(fields) == 2 and fields[0].startswith("ridgefill."):
            meta[fields[0]] = fields[1].strip()
    return meta


def read_meta(target, rad_info):
    if target.suffix == ".safetensors":
        with safe_open(str(target), "pt") as f:
            return f.metadata() or {}
    if target.suffix == ".rad":
        command = shlex.split(rad_info) + ["--meta", str(target)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=True)
        except FileNotFoundError:
            raise SystemExit(f"rad-info not found ({command[0]}); pass --rad-info, or save its --meta output "
                             "to a file and verify that file")
        return parse_rad_info_meta(result.stdout)
    return parse_rad_info_meta(target.read_text(encoding="utf-8"))


def verify(args):
    target = Path(args.target)
    if not target.is_file():
        raise SystemExit(f"{target} not found")
    expected = source_hashes(source_files(args))
    meta = read_meta(target, args.rad_info)
    bad = 0
    for key, want in sorted(expected.items()):
        have = meta.get(key)
        if have == want:
            print(f"OK        {key} {want}")
        else:
            bad += 1
            print(f"{'MISSING' if have is None else 'MISMATCH':9s} {key}: recorded {have}, recomputed {want}")
    for key in ("ridgefill.format",) + (("ridgefill.split", "ridgefill.rowsel.share", "ridgefill.rowsel.classes")
                                  if args.names == "shipped" else ()):
        bad += key not in meta
        print(f"{'' if key in meta else 'MISSING':9s} {key} = {meta.get(key)}")
    print(f"verify {target}: {'PASS' if bad == 0 else f'FAIL ({bad} problem(s))'}")
    return 0 if bad == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("build", "verify"):
        p = sub.add_parser(name)
        if name == "verify":
            p.add_argument("target", help="ridgefill-sidecar.safetensors, a .rad container, or saved `rad-info --meta` output")
            p.add_argument("--rad-info", default="rad-info", help="rad-info command (a .rad target), e.g. a docker run prefix")
        p.add_argument("--proj", required=True, help="projector safetensors (layer.S..layer.L-1 [H, W+1] bf16)")
        p.add_argument("--st", nargs=2, metavar=("RANK0", "RANK1"),
                       help="correction .pt per TP rank; required for shipped, optional for refit (projector alone)")
        p.add_argument("--names", choices=sorted(NAMES), default="shipped",
                       help="tensor-name set: shipped (ridgefill.proj/ridgefill.st + controls + row tables) or refit "
                            "(ridgefill.projr/ridgefill.str, Stage 6)")
        p.add_argument("--freq", help="unigram table safetensors (`logfreq` [vocab]); shipped only")
        p.add_argument("--tokenizer", help="checkpoint dir with config.json + tokenizer.json; shipped only")
        if name == "build":
            p.add_argument("--out", required=True, help="output directory (created)")
    args = ap.parse_args(argv)
    if args.command == "build":
        build(args)
        return 0
    return verify(args)


if __name__ == "__main__":
    sys.exit(main())
