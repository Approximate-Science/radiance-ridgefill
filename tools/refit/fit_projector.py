#!/usr/bin/env python3
"""Stage 6 projector refit (R42): capture.py acc's running sums -> the research pipeline's own ridge solve (qfn.fit
solve over kva.ridge.RidgeFit: centred ridge, unpenalised bias, one eigendecomposition for every lambda) -> lambda
by held-out mean block-input cosine -> the projector in tcc's per-layer layout + report-radiance-s<S>.json.

  fit_projector.py --sums DIR --held DIR --ckpt MODEL_DIR --shipped P.safetensors --out DIR
                   [--share 0.5] [--lambdas 0.001,0.003,0.01,0.03,0.1,0.3] [--threads 12]

--held: capture.py run's filed held-out documents (<held>/<format>/<name>/), converted once to capture_NNNNN.pt
under <out>/held-pt so qfn.fit.Held reads them unchanged. --ckpt: a checkpoint dir for the held-out screen's
fixed weights (layer S's connection read for the HC identity check; the attention layers' K / V / indexer-key
weights for kdir / ikdir / vrel), the same ones the shipped fit's report used. --shipped: scored on the SAME
radiance held-out rows (qfn.fit.his_metrics), the comparison that says what the refit buys on this engine.
Chat rows are mixed in at --share of the weighted rows (w = share * n_raw / ((1 - share) * n_chat), qfn.fit's rule).

Output <out>/kva-radiance-s<S>.safetensors: layer.S .. layer.{L-1} [hidden, hc*hidden + 1] bf16, the bias in the last
column -- tcc's layer format WITHOUT `final` (not captured: it feeds only MTP), so tools/kva_sidecar.py reads it and
tcc's own loader would not. The held-out cosines in the report are recomputed from the written (bf16) file, as the
shipped file's are, so both sides of the comparison carry the same rounding.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert  # noqa: E402
import kva_research  # noqa: E402

LAMBDAS = "0.001,0.003,0.01,0.03,0.1,0.3"      # the shipped fit's grid (report-big-s24.json)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_sums(sums_dir):
    """(split, {fmt: state}, {fmt: docs}) from capture.py acc's checkpoints (qfn.bigcap format)."""
    import torch
    from qfn import fit
    states, docs, splits = {}, {}, set()
    for fmt in ("raw", "chat"):
        paths = sorted(Path(sums_dir).glob(f"acc-{fmt}-s*.pt"))
        if len(paths) != 1:
            raise SystemExit(f"{sums_dir}: expected one acc-{fmt}-s<S>.pt, found {[p.name for p in paths]}")
        splits.add(int(paths[0].stem.rsplit("-s", 1)[1]))
        (_, state), = fit.stats_parts(paths[0])          # mmapped: combine() below writes new tensors
        states[fmt] = state
        docs[fmt] = list(torch.load(paths[0], map_location="cpu", mmap=True, weights_only=True)["docs"])
    if len(splits) != 1:
        raise SystemExit(f"raw and chat sums are of different splits: {splits}")
    return splits.pop(), states, docs


def mixed(raw, chat, share):
    """raw rows + w x chat rows, chat = `share` of the weighted rows (qfn.fit.main's mixed(), same formula)."""
    from kva.ridge import combine
    w = share * raw["n"] / ((1 - share) * chat["n"])
    return combine(raw, chat, w), w


def held_dirs(held, out):
    """capture_NNNNN.pt dirs of every filed held-out document (converted once)."""
    dirs = []
    for meta in sorted(Path(held).glob("*/*/meta.json")):
        dest = out / "held-pt" / meta.parent.parent.name / meta.parent.name
        if not list(dest.glob("capture_*.pt")):
            convert.write_pt(meta.parent, dest)
        dirs.append(dest)
    if not dirs:
        raise SystemExit(f"{held}: no filed held-out documents")
    return dirs


def write_projector(path, pick, lam, layout, meta):
    """layer.L = [W | b] bf16 for every late layer at lambda `lam`."""
    import torch
    from safetensors.torch import save_file
    tensors = {}
    for target in layout.targets:
        w, b = pick(lam, target)
        tensors[f"layer.{target.split('.')[1]}"] = torch.cat([w, b[:, None]], 1).to(torch.bfloat16).contiguous()
    save_file(tensors, str(path), metadata={k: str(v) for k, v in meta.items()})


def per_layer_bi(scores):
    return {str(l): round(m["bi"], 6) for l, m in sorted(scores["layers"].items())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sums", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--share", type=float, default=0.5, help="chat share of the weighted training rows")
    ap.add_argument("--lambdas", default=LAMBDAS)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args(argv)
    kva_research.root()
    import torch
    import torch.nn.functional as F
    from qfn import fit
    from qfn.weights import Checkpoint
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lambdas = [float(v) for v in a.lambdas.split(",")]
    split, states, docs = load_sums(a.sums)
    ck = Checkpoint(a.ckpt)
    lay = fit.Layout("plain", split, ck)
    lay.final = False                    # not captured (it feeds only MTP)
    if states["raw"]["layers"] != lay.targets:
        raise SystemExit(f"sums targets {states['raw']['layers'][:2]}... != {lay.targets[:2]}...")
    late = list(range(split, ck.num_layers))
    hcw = {l: ck.hc_read(l) for l in late}
    attw = {l: ck.attention(l) for l in late if ck.is_attention(l)}
    held = fit.Held(held_dirs(a.held, out), split, ck, (), hcw, attw, "cpu")
    hc_cos = F.cosine_similarity(held.base[split], held.bi[split], dim=-1).mean().item()
    log(f"held rows {held.b.shape[0]}; HC identity at layer {split}: cos {hc_cos:.6f}")
    shipped = fit.his_metrics(a.shipped, held, split, ck)
    log(f"shipped projector on radiance held rows: {shipped['summary']}")
    state, weight = mixed(states["raw"], states["chat"], a.share)
    rows = dict(raw=states["raw"]["n"], chat=states["chat"]["n"], chat_weight=weight, weighted=state["n"])
    del states
    t0 = time.time()
    scores, pick = fit.solve(state, held, lay, lambdas, ck)
    del state
    for lam in lambdas:
        log(f"lambda {lam}: {scores[lam]['summary']}")
    lam = fit.best(scores, "bi")
    path = out / f"kva-radiance-s{split}.safetensors"
    write_projector(path, pick, lam, lay, {"split": split, "lambda": lam, "share": a.share,
                                           "train_rows": rows["weighted"], "rows_raw": rows["raw"],
                                           "rows_chat": rows["chat"], "engine": "radiance",
                                           "fit": "qfn.fit plain without final: centred ridge, unpenalised bias, "
                                                  "lambda by held-out block-input cosine"})
    refit = fit.his_metrics(path, held, split, ck)
    log(f"solved in {time.time() - t0:.0f}s; lambda {lam}; refit file on held rows: {refit['summary']}")
    report = dict(split=split, share=a.share, lambdas=lambdas, chosen_lambda=lam, rows=rows,
                  docs={f: len(d) for f, d in docs.items()}, doc_keys=docs, held_rows=int(held.b.shape[0]),
                  held=[str(d) for d in held_dirs(a.held, out)], checkpoint=ck.fingerprint(),
                  hc_identity_check={str(split): hc_cos}, identity=fit.identity_metrics(held, split, ck)["summary"],
                  heldout=dict(refit=refit["summary"], shipped=shipped["summary"],
                               refit_bi_per_layer=per_layer_bi(refit), shipped_bi_per_layer=per_layer_bi(shipped)),
                  shipped=dict(file=a.shipped, sha256=sha256_file(a.shipped)),
                  scores={str(l): s["summary"] for l, s in scores.items()},
                  output=dict(file=path.name, sha256=sha256_file(path)))
    (out / f"report-radiance-s{split}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    log(f"wrote {path} and report-radiance-s{split}.json: held-out bi refit {refit['summary']['bi']:.4f} vs shipped "
        f"{shipped['summary']['bi']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
