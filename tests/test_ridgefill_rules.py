"""The transcribed class56 rule (tools/ridgefill_rules.py). Run: python -m pytest tests/

The cross-check against the original (fnlev/rules.py) runs only with RIDGEFILL_RESEARCH_ROOT (a checkout of the
research repo) and RIDGEFILL_TOKENIZER (the checkpoint directory); otherwise it is SKIPPED, never passed.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ridgefill_rules as R  # noqa: E402

INF = float("-inf")


def kept(text):
    return R.coarse(R.token_class(text))


def test_word_start_capital_is_cap():
    assert kept(" The") == "cap"


def test_word_piece_capital_is_cap():
    assert kept("The") == "cap"


def test_lower_word_piece_is_piece():
    assert kept("ing") == "piece"


def test_lower_word_start_is_not_kept():
    assert kept(" the") is None


def test_number_is_not_kept():
    assert R.token_class("3.14") == "number"
    assert kept("3.14") is None


def test_letters_with_digits_is_mixed():
    assert kept("x2") == "mixed"


def test_punctuation_space_and_newline_are_not_kept():
    assert [kept(","), kept(" "), kept("\n")] == [None, None, None]


def test_lone_utf8_byte_reads_replacement_char_and_is_mixed():
    # A byte-level piece holding only the first byte of a two-byte character: the original decodes it with
    # errors="replace", gets U+FFFD, and classes it mixed (kept). Faithful, if surprising.
    class Tok:
        added_tokens_decoder = {}

        def __len__(self):
            return 1

        def convert_ids_to_tokens(self, ids):
            return ["Ã"]          # GPT-2 byte map: "Ã" is byte 0xC3, a UTF-8 lead byte
    assert R.token_texts(Tok()) == ["�"]
    assert kept("�") == "mixed"


def test_share_times_matches_rounds_half_to_even():
    # Python round(): 0.25 * 6 = 1.5 -> 2 and 0.25 * 10 = 2.5 -> 2. A kernel using roundf() keeps 3 here.
    score = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    assert len(R.select_rows([0, 1, 2, 3, 4, 5], score, 0.25)) == 2
    assert len(R.select_rows(list(range(10)), score, 0.25)) == 2


def test_rarest_rows_kept_ties_by_lower_position():
    # Four matches, k = 2; rows 1, 2, 3 tie at the top score, so the two lowest positions win.
    score = np.array([5.0, 9.0], dtype=np.float32)
    assert R.select_rows([0, 1, 1, 1], score, 0.5) == [1, 2]


def test_minus_inf_rows_never_match():
    score = np.array([INF, 3.0], dtype=np.float32)
    assert R.select_rows([0, 0, 0, 1], score, 1.0) == [3]


def test_score_table_marks_padding_and_specials_minus_inf():
    logfreq = np.array([-5.0, -6.0, -7.0, -8.0], dtype=np.float32)
    score = R.score_table([" The", " the", None], logfreq)
    assert score[0] == 5.0
    assert [math.isinf(score[1]), math.isinf(score[2]), math.isinf(score[3])] == [True, True, True]


@pytest.mark.skipif(not (os.environ.get("RIDGEFILL_RESEARCH_ROOT") and os.environ.get("RIDGEFILL_TOKENIZER")),
                    reason="needs RIDGEFILL_RESEARCH_ROOT (research repo) and RIDGEFILL_TOKENIZER (checkpoint dir)")
def test_transcription_equals_fnlev_rules_over_the_whole_vocab():
    sys.path.insert(0, os.environ["RIDGEFILL_RESEARCH_ROOT"])
    from fnlev import rules as original
    tok = R.load_tokenizer(os.environ["RIDGEFILL_TOKENIZER"])
    ours = [None if t is None else R.coarse(R.token_class(t)) for t in R.token_texts(tok)]
    theirs = original.Rules(model_dir=os.environ["RIDGEFILL_TOKENIZER"], freq=None, tok=tok).cls
    assert ours == theirs
