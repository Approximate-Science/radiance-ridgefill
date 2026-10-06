#!/usr/bin/env python3
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""media_corpus.py -- R75's image + long-text corpus, generated: five synthetic pictures, each paired with a document.

  media_corpus.py --docs QUICK_PPL.jsonl --out DIR [--seed 75] [--chars 56000]

WHY GENERATED: R75 asks how well the text-fitted projector approximates text that FOLLOWS an image (its rotary
position runs behind its index, abi/rad_runtime.h). That needs images the model can read, and no personal or
downloaded picture belongs in a corpus: these are rendered from a seed -- a bar chart, a line chart, a flow diagram,
a table of figures and a pie chart, each with seeded labels and numbers -- so anyone with the same matplotlib
rebuilds the same pixels (`image_sha256` in the manifest says whether they did).

WHAT IT WRITES: DIR/img<i>-<kind>.png (640 x 480, no timestamp metadata) and DIR/media.jsonl, one line a picture:
{id, kind, image, image_sha256, docs, doc_index, chars, seed}. Picture i is paired with quick-corpus document
3 + i, cut to `chars` characters (~12-14K tokens; tools/media_ident.py refuses a prompt under its --min-prompt),
so pictures and texts are distinct and every document is long enough.

Needs matplotlib and numpy (the ridgefill venv).
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WORDS = ("north", "delta", "copper", "harbor", "signal", "orchid", "basalt", "ledger", "meadow", "quartz", "falcon",
         "tundra", "cobalt", "summit", "lantern", "glacier", "prairie", "beacon", "canyon", "willow")


def names(rng, n):
    return [w.capitalize() for w in rng.choice(WORDS, size=n, replace=False)]


def bar(ax, rng):
    labels = names(rng, 6)
    ax.bar(labels, rng.integers(5, 95, 6), color="tab:blue")
    ax.set_title(f"Quarterly output by site ({rng.integers(2001, 2030)})")
    ax.set_ylabel("units (thousands)")


def line(ax, rng):
    x = np.arange(12)
    for lab in names(rng, 3):
        ax.plot(x, np.cumsum(rng.normal(1.0, 2.0, 12)), marker="o", label=lab)
    ax.set_xticks(x, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title("Monthly balance by account")
    ax.legend()


def flow(ax, rng):
    steps = names(rng, 5)
    for i, s in enumerate(steps):
        ax.text(0.1 + 0.2 * i, 0.5 + (0.2 if i % 2 else -0.2), s, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round", fc="lightyellow", ec="black"))
        if i:
            ax.annotate("", xy=(0.1 + 0.2 * i - 0.06, 0.5 + (0.2 if i % 2 else -0.2)),
                        xytext=(0.1 + 0.2 * (i - 1) + 0.06, 0.5 + (0.2 if (i - 1) % 2 else -0.2)),
                        arrowprops=dict(arrowstyle="->"))
    ax.set_axis_off()
    ax.set_title("Process: " + " -> ".join(steps[:2]) + " ...")


def table(ax, rng):
    rows = names(rng, 6)
    cells = [[f"{v:,}" for v in rng.integers(100, 99999, 4)] for _ in rows]
    t = ax.table(cellText=cells, rowLabels=rows, colLabels=["Q1", "Q2", "Q3", "Q4"], loc="center")
    t.scale(1, 1.6)
    ax.set_axis_off()
    ax.set_title("Revenue by region (USD)")


def pie(ax, rng):
    labels = names(rng, 5)
    ax.pie(rng.integers(5, 40, 5), labels=labels, autopct="%1.0f%%")
    ax.set_title("Share of shipments")


KINDS = (("bar", bar), ("line", line), ("flow", flow), ("table", table), ("pie", pie))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=75)
    ap.add_argument("--chars", type=int, default=56000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = [json.loads(x) for x in open(a.docs, encoding="utf-8") if x.strip()]
    lines = []
    for i, (kind, draw) in enumerate(KINDS):
        rng = np.random.default_rng(a.seed * 100 + i)
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        draw(ax, rng)
        fig.tight_layout()
        name = f"img{i}-{kind}.png"
        fig.savefig(out / name, metadata={"Software": None})
        plt.close(fig)
        d = 3 + i
        if len(docs[d]["prompt"]) < a.chars:
            raise SystemExit(f"document {d} has {len(docs[d]['prompt'])} characters, fewer than --chars {a.chars}")
        lines.append({"id": f"img{i}-{kind}", "kind": kind, "image": name,
                      "image_sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
                      "docs": str(Path(a.docs).resolve()), "doc_index": d, "chars": a.chars, "seed": a.seed})
    (out / "media.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    for x in lines:
        print(x["id"], x["image_sha256"][:16], "doc", x["doc_index"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
