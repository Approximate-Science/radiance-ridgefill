"""The class56 row rule (KVA quality mode), transcribed so the tools run without the research repo.

Source of truth: the research repo's fnlev/rules.py (classifier lines 48-71, row ranking 167-195) and
kva/select.py (token_bytes, _byte_map, rarity). Transcribed, not imported: the sidecar tool must run on a
machine that has only this repo. tests/test_kva_rules.py checks the transcription against the original when
KVA_RESEARCH_ROOT points at a checkout.

The rule, for one prompt window of ids:
  matches = rows whose token's coarse class is kept (cap, mixed, piece by default)
  k       = round(share * len(matches))            Python round: half to EVEN (2.5 -> 2, 1.5 -> 2)
  rows    = the k matches with the highest rarity (-log unigram frequency), ties by lower position
The per-id score table folds the class test into the rarity: score = rarity if kept, else -inf, so a row
matches iff its score is finite.
"""
import json
import math
import string
from pathlib import Path

import numpy as np

PUNCT = set(string.punctuation)
CLASSES = ("cap", "mixed", "piece")


def token_class(s):
    """fnlev/rules.py:48-60 verbatim: the class breakdown's fine class of one token's text."""
    t = s.strip()
    if not t:
        return "newline" if "\n" in s else "space"
    if any(c.isdigit() for c in t) and all(c.isdigit() or c in PUNCT for c in t):
        return "number"
    if all(c in PUNCT for c in t):
        return "punct"
    if t.isalpha():
        lead = s[0] in " \t\n"
        return ("word-start" if lead else "word-piece") + ("-cap" if t[0].isupper() else "")
    return "mixed"


def coarse(cls):
    """fnlev/rules.py:63-71 verbatim: lever 56's coarse class of a fine class, or None."""
    if cls in ("word-start-cap", "word-piece-cap"):
        return "cap"
    if cls == "mixed":
        return "mixed"
    if cls == "word-piece":
        return "piece"
    return None


def _byte_map():
    """kva/select.py _byte_map: GPT-2's printable-character -> byte map of byte-level BPE pieces."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_texts(tok):
    """Text of every tokenizer id, None for added/special tokens (kva/select.py token_bytes, then
    fnlev/rules.py:86's utf-8 decode with errors="replace": a lone byte of a split character reads U+FFFD)."""
    inv = _byte_map()
    added = set(tok.added_tokens_decoder)
    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
    out = []
    for i, s in enumerate(pieces):
        if s is None or i in added:
            out.append(None)
            continue
        raw = bytes(inv[c] for c in s) if all(c in inv for c in s) else s.encode()
        out.append(raw.decode("utf-8", errors="replace"))
    return out


def score_table(texts, logfreq, classes=CLASSES):
    """[len(logfreq)] f32: -logfreq[id] where the id's coarse class is kept, else -inf. Ids past the tokenizer
    (the model's padding rows) and added/special tokens are never kept. Negating an f32 is exact, so the
    ranking equals the original's fp64 ranking, ties included."""
    want = set(classes)
    score = np.full(len(logfreq), -np.inf, dtype=np.float32)
    for i, t in enumerate(texts):
        if t is not None and coarse(token_class(t)) in want:
            score[i] = -logfreq[i]
    return score


def select_rows(ids, score, share):
    """Rows of one window the rule keeps exact, ascending (fnlev/rules.py:167-195 for class56, rank rarity,
    with the window treated as the whole prompt)."""
    matches = [r for r, t in enumerate(ids) if math.isfinite(score[int(t)])]
    k = round(share * len(matches))
    order = sorted(matches, key=lambda r: (-float(score[int(ids[r])]), r))
    return sorted(order[:k])


def load_logfreq(path):
    """The unigram table's `logfreq` [vocab] f32 (natural-log frequency, add-one smoothed, all finite)."""
    from safetensors.numpy import load_file
    table = load_file(str(path))
    if "logfreq" not in table:
        raise SystemExit(f"{path}: no 'logfreq' tensor (keys {sorted(table)}); expected the KVA unigram table")
    return table["logfreq"]


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir))


def model_vocab(model_dir):
    """The model's vocab rows from <model_dir>/config.json (text_config first, as radiance reads it)."""
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found: --tokenizer must be a checkpoint directory with config.json "
                         "(the model's vocab size is read from it)")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    vocab = (cfg.get("text_config") or {}).get("vocab_size") or cfg.get("vocab_size")
    if not vocab:
        raise SystemExit(f"{path}: no vocab_size (nor text_config.vocab_size)")
    return int(vocab)
