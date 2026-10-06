# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
"""tools/ridgefill_report.py: the catalog covers every message the plugin prints, real logs get the right
diagnosis, and the integrity checks catch a damaged projector or plugin. The fixtures are lines from the
fresh-engine e2e run of 2026-10-06 (evidence/ridgefix-verify-20261006/release/e2e-0.1.0/logs, container paths)."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import ridgefill_report as R  # noqa: E402

FIX = REPO / "tests" / "fixtures" / "report"
PLACEHOLDER = re.compile(r"%[-+ 0#]*\d*(?:\.\d+)?(lld|ld|zu|d|u|x|g|f|s)")


def plugin_messages():
    """Every fprintf(stderr, ...) format string in the plugin's C++ (adjacent literals joined)."""
    out = []
    for path in sorted(list((REPO / "arch").glob("*.h")) + list((REPO / "arch").glob("*.cpp"))):
        src = path.read_text()
        for m in re.finditer(r"fprintf\(\s*stderr\s*,", src):
            i, text = m.end(), ""
            while (lit := re.compile(r'\s*"((?:[^"\\]|\\.)*)"').match(src, i)):
                text, i = text + lit.group(1), lit.end()
            out.append((f"{path.name}:{src.count(chr(10), 0, m.start()) + 1}", text))
    return out


def sample(fmt):
    """A line the format could print: strings X, integers 7, floats 1.5."""
    sub = {"s": "X", "g": "1.5", "f": "1.5"}
    return PLACEHOLDER.sub(lambda m: sub.get(m.group(1), "7"), fmt).replace("\\n", "").replace("%%", "%")


def lines(name):
    return (FIX / name).read_text().splitlines()


def test_every_message_the_plugin_prints_has_a_catalog_entry():
    messages = plugin_messages()
    assert len(messages) > 40   # the extractor found the plugin's messages, not nothing
    missing = [(where, fmt) for where, fmt in messages if R.classify(sample(fmt)) is None]
    assert missing == [], "add these to CATALOG in tools/ridgefill_report.py and docs/TROUBLESHOOTING.md"


def test_the_step_refusal_rf501_still_says_what_the_catalog_matches():
    src = re.sub(r'"\s*"', "", (REPO / "arch" / "ridgefill_step.h").read_text())   # join split literals
    assert "does not end on the delta net's chunk tile" in src


def test_catalog_codes_are_unique_and_documented():
    codes = [c for c, *_ in R.CATALOG]
    assert len(codes) == len(set(codes))
    doc = (REPO / "docs" / "TROUBLESHOOTING.md").read_text()
    assert [c for c in codes if c not in doc] == []


def test_no_projector_is_diagnosed_as_rf101():
    findings, _, errors = R.diagnose(lines("no-projector.txt"))
    assert R.headline(findings, errors, "quality").startswith("RF-101")


def test_a_corrupt_projector_is_diagnosed_as_rf103():
    findings, _, errors = R.diagnose(lines("corrupt-projector.txt"))
    assert R.headline(findings, errors, "quality").startswith("RF-103")


def test_a_retired_switch_is_diagnosed_as_rf303_with_the_engine_error_kept():
    findings, _, errors = R.diagnose(lines("retired-switch.txt"))
    assert R.headline(findings, errors, "quality").startswith("RF-303")
    assert any("rad_arch_declare returned invalid argument" in e for e in errors)


def test_a_working_quality_server_is_reported_running_with_its_steps_counted():
    findings, unclassified, errors = R.diagnose(lines("quality.txt"))
    counts = {f[0][0]: f[1] for f in findings}
    assert R.headline(findings, errors, "quality").startswith("RidgeFill is running")
    assert counts["RF-003"] == 3 and counts["RF-002"] == 2 and unclassified == []


def test_a_log_without_the_shadow_line_says_ridgefill_did_not_load():
    findings, _, errors = R.diagnose(lines("stock-no-plugin.txt"))
    assert R.headline(findings, errors, None).startswith("RidgeFill did not load")


def test_an_unknown_plugin_line_is_listed_not_dropped():
    _, unclassified, _ = R.diagnose(["radiance: qwen4exp_ridgefill: RidgeFill: something new happened"])
    assert unclassified == ["radiance: qwen4exp_ridgefill: RidgeFill: something new happened"]


def test_redact_hides_home_user_and_host():
    import getpass
    import socket
    text = f"{os.path.expanduser('~')}/models on {socket.gethostname()} as {getpass.getuser()}"
    out = R.redact(text)
    assert "<home>/models" in out and os.path.expanduser("~") not in out


def test_projector_check_names_a_corrupt_file(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"good")
    (tmp_path / "b.safetensors").write_bytes(b"flipped")
    files = {"a.safetensors": hashlib.sha256(b"good").hexdigest(), "b.safetensors": hashlib.sha256(b"bad").hexdigest()}
    (tmp_path / "ridgefill.json").write_text(json.dumps({"name": "p", "adapter": "qwen4exp", "split": 24, "files": files}))
    facts = dict(R.projector_facts(str(tmp_path)))
    assert facts["Projector files"] == "1/2 match the manifest; missing or corrupt: b.safetensors"


def test_plugin_check_flags_a_so_that_differs_from_sha256sums(tmp_path):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "ridgefill.so").write_bytes(b"binary")
    (tmp_path / "SHA256SUMS").write_text(f"{'0' * 64}  kernels/ridgefill.so\n")
    facts = dict(R.plugin_facts(str(tmp_path)))
    assert facts["kernels/ridgefill.so"].endswith("(DIFFERS from SHA256SUMS)")


def test_the_report_runs_end_to_end_on_a_log(tmp_path, capsys):
    assert R.main(["--log", str(FIX / "no-projector.txt"), "--plugin-dir", str(tmp_path),
                   "--out", str(tmp_path / "report.md")]) == 0
    report = (tmp_path / "report.md").read_text()
    assert report.startswith("# RidgeFill report") and "**RF-101:" in report and "## GPUs" in report
