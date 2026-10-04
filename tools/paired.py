#!/usr/bin/env python3
"""paired.py -- compare two KL-mode reports doc by doc.

USAGE
    tools/paired.py A.json B.json [--corpus quick9.jsonl]

WHAT IT DOES
    Reads two --kld-out reports (radiance KL mode, docs/TOOLS.md lines 470-506; format
    from core/kld.cpp:562-579: "all" and "by_source" summaries) and prints:
      - per-doc (per source) NLL of each report's candidate and the paired difference
        B - A;
      - the mean paired difference with a bootstrap 95% CI: 10,000 resamples over the
        docs, fixed seed 0, percentile method;
      - each report's mean KL, 99th-percentile KL and top-1 agreement.

    Per-doc NLL comes from the report's by_source section (a source's mean NLL is the log
    of its ppl.candidate, exact: ppl = exp(mean NLL)). If the report lacks per-source NLL,
    per-doc means are computed from the .rows file written beside the report (<report
    path>.rows: [positions, 4] float32 in reference row order -- KL, the candidate's NLL,
    the reference's NLL, the argmax agreement; core/kld.cpp:586-590) and the corpus: one
    {"prompt", "score_from", "source", "tokens"} per line, in doc (reference row) order;
    each doc scores n - 1 - score_from positions. The corpus lines must carry "tokens"
    (the token count the doc was cut to -- what tools/kld_corpus.py writes); a line
    without it is an error, named.

    The two reports must cover the same sources; anything else is an error, named.
    Positive diff (B - A) means B is worse (higher NLL) than A.

    Standard library + numpy only.
"""

import argparse
import json
import math
import sys

import numpy as np

N_BOOTSTRAP = 10000
SEED = 0


def fail(msg):
    print(f"paired: FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        fail(f"cannot read {what} {path}: {e}")
    except json.JSONDecodeError as e:
        fail(f"{what} {path} is not JSON: {e}")


def load_report(path):
    rep = load_json(path, "report")
    if "all" not in rep:
        fail(f"{path}: no 'all' key -- not a KL-mode --kld-out report?")
    for key in ("kld", "top1_agreement"):
        if key not in rep["all"]:
            fail(f"{path}: no 'all.{key}' -- not a KL-mode --kld-out report?")
    return rep


def per_doc_nll_from_report(rep, path):
    """Mean candidate NLL per source, from by_source (the log of ppl.candidate)."""
    out = {}
    for source, s in sorted((rep.get("by_source") or {}).items()):
        if "nll" in s and s["nll"] is not None:
            out[source] = float(s["nll"])
            continue
        ppl = s.get("ppl") or {}
        if "candidate" not in ppl:
            fail(f"{path}: by_source[{source!r}] has neither nll nor ppl.candidate")
        out[source] = math.log(float(ppl["candidate"]))
    return out


def read_rows(path):
    """<path>.rows as an [n, 4] float32 array, in reference row order."""
    rows_path = path + ".rows"
    try:
        arr = np.fromfile(rows_path, dtype=np.float32)
    except OSError as e:
        fail(f"cannot read {rows_path}: {e}")
    if arr.size == 0 or arr.size % 4:
        fail(f"{rows_path}: {arr.size} floats is not a [positions, 4] f32 array")
    return arr.reshape(-1, 4)


def docs_from_corpus(corpus_path):
    """Doc order and scored-position counts (n - 1 - score_from) from the corpus JSONL."""
    try:
        f = open(corpus_path, encoding="utf-8")
    except OSError as e:
        fail(f"cannot read the corpus {corpus_path}: {e}")
    docs = []
    with f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError as e:
                fail(f"{corpus_path}:{lineno}: not JSON: {e}")
            for key in ("tokens", "score_from"):
                if key not in j:
                    fail(f'{corpus_path}:{lineno}: no "{key}" -- the rows fallback needs '
                         f"the scored-position count n - 1 - score_from")
            n = int(j["tokens"])
            score_from = int(j["score_from"])
            rows = n - 1 - score_from
            if rows < 1:
                fail(f"{corpus_path}:{lineno}: {n} tokens with score_from {score_from} "
                     f"leave nothing to score")
            docs.append({"source": j.get("source", f"doc{lineno}"), "rows": rows})
    if not docs:
        fail(f"the corpus holds no documents: {corpus_path}")
    return docs


def per_doc_nll_from_rows(path, corpus_path):
    """Mean candidate NLL per doc, from <path>.rows (column 1) and the corpus doc order."""
    rows = read_rows(path)
    docs = docs_from_corpus(corpus_path)
    total = sum(d["rows"] for d in docs)
    if total != len(rows):
        fail(f"{path}.rows holds {len(rows)} positions; the corpus says {total} "
             f"(sum of n - 1 - score_from)")
    out = {}
    offset = 0
    for d in docs:
        out[d["source"]] = float(rows[offset:offset + d["rows"], 1].mean())
        offset += d["rows"]
    return out


def per_doc_nll(rep, path, corpus_path):
    """Per-doc NLL: the report's by_source when it has one, else .rows + corpus."""
    from_report = per_doc_nll_from_report(rep, path)
    if from_report:
        return from_report
    if corpus_path is None:
        fail(f"{path}: the report has no per-source NLL and no --corpus was given "
             f"for the .rows fallback")
    return per_doc_nll_from_rows(path, corpus_path)


def bootstrap_ci(diffs):
    """95% percentile CI of the mean, 10,000 resamples over the docs, seed 0."""
    x = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, x.size, size=(N_BOOTSTRAP, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summary_line(name, rep, path):
    all_ = rep["all"]
    kld = all_["kld"]
    ppl = all_.get("ppl") or {}
    print(f"{name}: mean KL {kld.get('mean', float('nan')):.6f}  "
          f"99th KL {kld.get('p99', float('nan')):.6f}  "
          f"top-1 {100 * all_['top1_agreement']:.3f}%  "
          f"ppl {ppl.get('candidate', float('nan')):.4f} vs reference "
          f"{ppl.get('reference', float('nan')):.4f}  ({path})")


def main(argv=None):
    parser = argparse.ArgumentParser(description="compare two KL-mode reports doc by doc")
    parser.add_argument("report_a", help="first --kld-out report (A)")
    parser.add_argument("report_b", help="second --kld-out report (B)")
    parser.add_argument("--corpus", help="KL-mode corpus JSONL for the .rows fallback "
                        "(doc order, lines carry \"tokens\" and \"score_from\")")
    args = parser.parse_args(argv)

    rep_a = load_report(args.report_a)
    rep_b = load_report(args.report_b)
    nll_a = per_doc_nll(rep_a, args.report_a, args.corpus)
    nll_b = per_doc_nll(rep_b, args.report_b, args.corpus)

    only_a = sorted(set(nll_a) - set(nll_b))
    only_b = sorted(set(nll_b) - set(nll_a))
    if only_a or only_b:
        fail(f"the reports cover different sources; only in {args.report_a}: {only_a}, "
             f"only in {args.report_b}: {only_b}")
    sources = sorted(nll_a)
    if not sources:
        fail("no sources to pair")

    diffs = [nll_b[s] - nll_a[s] for s in sources]
    mean_diff = sum(diffs) / len(diffs)
    lo, hi = bootstrap_ci(diffs)

    print("per-doc NLL (each report's candidate) and the paired difference B - A:")
    print(f"  {'source':24s} {'nll(A)':>12s} {'nll(B)':>12s} {'B-A':>12s}")
    for s in sources:
        print(f"  {s:24s} {nll_a[s]:12.6f} {nll_b[s]:12.6f} {nll_b[s] - nll_a[s]:12.6f}")
    print(f"mean difference B - A: {mean_diff:.6f}  "
          f"(95% CI [{lo:.6f}, {hi:.6f}]; {N_BOOTSTRAP} bootstrap resamples over docs, "
          f"seed {SEED})")
    summary_line("A", rep_a, args.report_a)
    summary_line("B", rep_b, args.report_b)


if __name__ == "__main__":
    main()