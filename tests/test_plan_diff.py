"""tools/plan_diff.py on hand-written rad-convert / rad-info output. Run: python -m pytest tests/"""
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "dev"))   # the append route (A')
import plan_diff as D  # noqa: E402

PLAN = """info   plan     3 weight(s): 1 quantised by the recipe, 2 kept as the checkpoint holds them
debug rad_convert.cpp:652    blk.0.attn_k.weight          i8*bf16[1x128]      1.2 MiB  rtn clamp=sym,codes=i8
debug rad_convert.cpp:652    output_norm.weight           bf16                5.0 KiB  as is
debug rad_convert.cpp:652    ridgefill.st.24                    f32                 3.0 MiB  as is
info   --plan-only: about 4.2 MiB would be written; nothing was written to m.rad.
"""
HELD = """weight directory (2 entries, grouped by layer and movement unit)

  layer 0                2 tensor(s)        1.2 MiB   unit: one tensor
      blk.0.attn_k.weight   i8*bf16[1x128]   512x2560   1.2 MiB  @4096   rtn clamp=sym,codes=i8
          plane codes        i8           1.2 MiB  @4096
      output_norm.weight    bf16             2560       5.0 KiB  @1314816   checkpoint
"""


def write(tmp_path, plan=PLAN, held=HELD):
    (tmp_path / "plan.log").write_text(plan)
    (tmp_path / "held.txt").write_text(held)
    save_file({"ridgefill.st.24": torch.zeros(1)}, str(tmp_path / "s.safetensors"))
    return ["--plan", str(tmp_path / "plan.log"), "--container", str(tmp_path / "held.txt"),
            "--expect-new", str(tmp_path / "s.safetensors")]


def test_plan_rows_read_name_encoding_and_provenance():
    rows = D.plan_rows(PLAN)
    assert rows["blk.0.attn_k.weight"] == ("i8*bf16[1x128]", "rtn clamp=sym,codes=i8")
    assert rows["output_norm.weight"] == ("bf16", "checkpoint")


def test_container_rows_skip_plane_and_group_lines():
    assert sorted(D.container_rows(HELD)) == ["blk.0.attn_k.weight", "output_norm.weight"]


def test_only_the_expected_new_weights_passes(tmp_path):
    assert D.main(write(tmp_path)) == 0


def test_a_weight_missing_from_the_plan_fails(tmp_path):
    plan = PLAN.replace("output_norm.weight", "renamed_norm.weight")
    assert D.main(write(tmp_path, plan=plan)) == 1


def test_a_changed_encoding_fails(tmp_path):
    plan = PLAN.replace("rtn clamp=sym,codes=i8", "rtn clamp=full,codes=i8")
    assert D.main(write(tmp_path, plan=plan)) == 1


def test_an_expected_weight_absent_from_the_plan_fails(tmp_path):
    plan = "\n".join(line for line in PLAN.splitlines() if "ridgefill.st.24" not in line)
    assert D.main(write(tmp_path, plan=plan)) == 1


def test_unparseable_input_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="parsed 0 plan rows"):
        D.main(write(tmp_path, plan="nothing here\n"))


def test_after_the_append_the_container_holds_exactly_the_plan(tmp_path):
    held = HELD + "      ridgefill.st.24             f32              1           3.0 MiB  @9999999   checkpoint\n"
    args = write(tmp_path, held=held)
    assert D.main(args[:4]) == 0          # no --expect-new: nothing may be new


def test_without_expected_new_weights_any_new_weight_fails(tmp_path):
    assert D.main(write(tmp_path)[:4]) == 1
