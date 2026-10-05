"""tools/kva_projector.py on synthetic inputs and a synthetic container (no model files, no GPU).

The tiny container's two hashes are the constants tests/arch_static_test.cpp expects of the same bytes, which is what
ties the builder's canonical tokenizer / anchor forms to the plugin's (arch/kva_match.h).
Run: python -m pytest tests/
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kva_projector as P  # noqa: E402
import kva_rules as R  # noqa: E402
from test_kva_sidecar import FakeTokenizer, write_inputs  # noqa: E402

TINY_VOCAB = "3988fb447f719ad3fc2c75e5a0fa3daeb2b6a5e10e964744619d1dfbc6e95ff6"
TINY_ANCHOR = "be45cb2605bf36bebde684841a28f0fd43c69850a3dce5fedba69928ee3a8991"


def container(meta=None, entries=None, template=""):
    """A .rad of a few KiB: the string blob, metadata, one 16-byte plane per entry, the three-token vocab section
    (the third token's text empty) with one merge, and `template` as its chat template. With no arguments it holds
    the tokens, merge and entry bytes of tests/arch_static_test.cpp's tiny_container_bytes (laid out differently:
    the hashes are of what the bytes mean, not of where they sit)."""
    meta, entries = meta or {}, entries if entries is not None else ["anchor.weight"]
    strings = [b"tok_a", b"tok_b"] + [n.encode() for n in entries] + \
              [x.encode() for kv in meta.items() for x in kv] + ([template.encode()] if template else [])
    blob, off = b"\0", {}
    for s in strings:
        if s not in off:
            off[s] = len(blob)
            blob += s + b"\0"
    f = bytearray(1 << 16)
    kv_off, dir_off, plane_off = 4096, 8192, 12288
    data_off, str_off = 16384, 32768
    for i, (k, v) in enumerate(meta.items()):
        struct.pack_into("<QII8s", f, kv_off + 24 * i, off[k.encode()], P.P_STR, 0, struct.pack("<Q", off[v.encode()]))
    for i, name in enumerate(entries):
        e = [off[name.encode()], 0, 0, 0, 1, 16, 0, 0, 0, 0, 0, -1, -1, 0, 1, i, data_off + 4096 * i, 16, 0, 0]
        struct.pack_into("<QQQII6qiiiIQQQ2Q", f, dir_off + 136 * i, *e)
        struct.pack_into("<QQ", f, plane_off + 16 * i, data_off + 4096 * i, 16)
        f[data_off + 4096 * i:data_off + 4096 * i + 16] = bytes((x + i) % 256 for x in range(16))
    struct.pack_into("<IIII5Q6i2IQ4Q", f, 512, 1, 3, 1, 0, 640, 0, 664, 672, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     off.get(template.encode(), 0) if template else 0, 0, 0, 0, 0)
    struct.pack_into("<3Q", f, 640, off[b"tok_a"], off[b"tok_b"], 0)
    f[664:667] = bytes([1, 1, 3])
    struct.pack_into("<2I", f, 672, 0, 1)
    header = [P.RAD_MAGIC, P.RAD_FORMAT_VER, len(f), 0, 0, 0, 0, 0,
              str_off, len(blob), kv_off, len(meta), dir_off, len(entries), plane_off, len(entries), 0, 0, 512, 168,
              0, 0, data_off, len(f) - data_off] + [0] * 8
    struct.pack_into("<IIQ5Q16Q8Q", f, 0, *header)
    f[str_off:str_off + len(blob)] = blob
    return bytes(f)


def test_the_tiny_containers_hashes_are_the_plugins(tmp_path):
    """Same bytes as arch_static_test's tiny container (the C++ test asserts the same two constants)."""
    rad_path = tmp_path / "tiny.rad"
    rad_path.write_bytes(container())
    rad = P.Rad(rad_path)
    assert rad.vocab_sha256() == TINY_VOCAB
    assert rad.entry_sha256("anchor.weight") == TINY_ANCHOR == hashlib.sha256(bytes(range(16))).hexdigest()


SPEC = {"length": 64,
        "pattern": {"offsets": [0, 59], "tokens": ["<|a|>", "<|b|>"], "rule": "alternate, starting with the first token"},
        "dials": [{"offset": 60, "kwarg": "kva_share", "default": "0.25", "table": {"0.10": "<|c|>", "0.25": "<|d|>"}},
                  {"offset": 61, "kwarg": "kva_alpha", "default": "1.0", "table": {"0": "<|c|>", "1.0": "<|d|>"}},
                  {"offset": 62, "kwarg": "kva_tail", "default": "2048", "table": {"1024": "<|c|>", "2048": "<|d|>"}}],
        "end": {"offset": 63, "token": "<|e|>"}, "switch": {"kwarg": "kva", "on_value": "on"},
        "token_ids": {"<|a|>": 1, "<|b|>": 2, "<|c|>": 3, "<|d|>": 4, "<|e|>": 5}}
META = {"model_type": "qwen4_exp", "n_layers": "4", "n_embd": "4", "n_vocab": "12", "n_expert": "8", "hc_count": "1",
        "layer_types": "linear_attention linear_attention linear_attention full_attention",
        "full_attention_interval": "4", "linear_num_value_heads": "4", "linear_value_head_dim": "3",
        "linear_key_head_dim": "3"}
ANCHORS = ["blk.1.attn_hc_norm.weight", "blk.1.ffn_hc_norm.weight", "blk.2.attn_hc_norm.weight"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("projector")
    src = write_inputs(root)     # projector layers 2, 3 [4, 6+1]; corrections for 2, 3; 12 vocab rows
    rad = root / "model.rad"
    rad.write_bytes(container(META, ANCHORS + ["blk.2.ssm_inz.weight"], template="{{ messages }}"))
    info = root / "rad-info-v.txt"
    info.write_text("      blk.1.attn_hc_down.weight  fp8_e4m3*f32[1x128]  4x4  1 KiB  @0  rtn\n"
                    "      blk.2.ssm_inz.weight  i8*bf16[1x128]  4x4  1 KiB  @0  gptq\n"
                    "      blk.3.ffn_gate_up_exps.0.weight  i4  4x4  1 KiB  @0  gptq\n"
                    "      blk.0.attn_qg.weight  i8*bf16[1x128]  4x4  1 KiB  @0  gptq\n")
    spec = root / "spec.json"
    spec.write_text(json.dumps(SPEC))
    out = root / "projector"
    original = R.load_tokenizer
    R.load_tokenizer = lambda path: FakeTokenizer()
    try:
        P.main(["build", *src, "--container", str(rad), "--rad-info-v", str(info), "--spec", str(spec), "--out", str(out)])
    finally:
        R.load_tokenizer = original
    return out, rad


def test_the_folder_holds_the_layout_and_the_tensors(built):
    out, _ = built
    names = sorted(p.name for p in out.iterdir())
    assert names == ["README.md", "chat_template.jinja", "correction.safetensors", "kva.json", "proj.L2.safetensors",
                     "proj.L3.safetensors", "rowsel.safetensors"]
    with safe_open(str(out / "proj.L2.safetensors"), "pt") as f:
        assert sorted(f.keys()) == ["proj.2.bias", "proj.2.weight"]
        w = f.get_tensor("proj.2.weight")
        assert w.dtype == torch.bfloat16 and list(w.shape) == [4, 6]
        assert torch.equal(w, torch.arange(28, dtype=torch.float32).reshape(4, 7).bfloat16()[:, :6])
    with safe_open(str(out / "correction.safetensors"), "pt") as f:
        assert sorted(f.keys()) == ["st.2", "st.3"]
        assert torch.equal(f.get_tensor("st.2")[:2], torch.full((2, 3, 3), 2.0))      # rank 0: 8 / 4
        assert torch.equal(f.get_tensor("st.2")[2:], torch.full((2, 3, 3), 200.0))    # rank 1: 800 / 4
    with safe_open(str(out / "rowsel.safetensors"), "pt") as f:
        assert sorted(f.keys()) == ["score", "score_all", "score_none"]


def test_the_manifest_carries_the_fingerprint_and_every_files_hash(built):
    out, rad = built
    m = json.loads((out / "kva.json").read_text())
    assert m["format"] == 1 and m["adapter"] == "qwen4exp" and m["split"] == 2
    assert m["layers"] == {"2": "recurrent", "3": "full_attn"}
    assert m["stream_width"] == 6 and m["block_in_width"] == 4
    assert m["model"]["meta"] == META and m["model"]["vocab_sha256"] == TINY_VOCAB
    # the encodings: every non-expert weight of layers S-1.. (expert and lower layers left out)
    assert m["model"]["encodings"] == {"blk.1.attn_hc_down.weight": "fp8_e4m3*f32[1x128]",
                                       "blk.2.ssm_inz.weight": "i8*bf16[1x128]"}
    assert sorted(m["model"]["anchors"]) == ANCHORS
    r = P.Rad(rad)
    assert m["model"]["anchors"]["blk.1.ffn_hc_norm.weight"] == r.entry_sha256("blk.1.ffn_hc_norm.weight")
    assert m["model"]["anchors"]["blk.1.attn_hc_norm.weight"] != m["model"]["anchors"]["blk.1.ffn_hc_norm.weight"]
    assert m["tail"]["table"] == [1024, 2048] and m["rowsel"]["share_table"] == [0.1, 0.25]
    assert set(m["files"]) == {p.name for p in out.iterdir()} - {"kva.json"}
    for name, digest in m["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert m["marker"]["template_source_sha256"] == hashlib.sha256(b"{{ messages }}").hexdigest()


def test_the_template_is_the_models_own_with_the_marker_in_front(built):
    out, _ = built
    merged = (out / "chat_template.jinja").read_text()
    assert merged.endswith("{{ messages }}") and merged.startswith("{#- kva-marker v1")


def test_two_builds_are_byte_identical(built, tmp_path):
    out, rad = built
    again = tmp_path / "again"
    src = write_inputs(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    info = out.parent / "rad-info-v.txt"
    original = R.load_tokenizer
    R.load_tokenizer = lambda path: FakeTokenizer()
    try:
        P.main(["build", *src, "--container", str(rad), "--rad-info-v", str(info), "--spec", str(spec), "--out", str(again)])
    finally:
        R.load_tokenizer = original
    a, b = json.loads((out / "kva.json").read_text()), json.loads((again / "kva.json").read_text())
    assert a["files"] == b["files"]     # the source paths differ (fit hashes do not)


def test_a_container_without_the_metadata_is_refused(built, tmp_path):
    out, _ = built
    bare = tmp_path / "bare.rad"
    bare.write_bytes(container({"n_layers": "4"}, ANCHORS))
    with pytest.raises(SystemExit, match="metadata lacks"):
        P.fingerprint(P.Rad(bare), type("A", (), {"container": bare, "rad_info_v": ""})(), 2, [2, 3])


def test_quantise_i8_is_libquants_absmax_rule():
    """i8*bf16[1x128]: scale = absmax/127 rounded to bf16, a zero group scaled by 1, codes rounded half to even
    against the ROUNDED scale and clamped to +-127; dequantised within half a step of the map."""
    torch.manual_seed(7)
    w = torch.randn(32, 256, dtype=torch.float32).to(torch.bfloat16)
    w[3, 128:] = 0
    codes, scale, err = P.quantise_i8(w)
    assert codes.dtype == torch.int8 and codes.shape == (32, 256)
    assert scale.dtype == torch.bfloat16 and scale.shape == (32, 2)
    x = w.float().reshape(32, 2, 128)
    want = (x.abs().amax(-1) / 127).to(torch.bfloat16)
    want[3, 1] = 1.0
    assert torch.equal(scale, want)
    assert torch.equal(codes.reshape(32, 2, 128).float(),
                       torch.round(x / want.float().unsqueeze(-1)).clamp(-127, 127))
    assert codes.abs().max() <= 127 and (codes[3, 128:] == 0).all()
    deq = codes.float().reshape(32, 2, 128) * scale.float().unsqueeze(-1)
    assert ((deq - x).abs() <= scale.float().unsqueeze(-1) * 0.5 + 1e-7).all()
    assert err < 0.01
    with pytest.raises(SystemExit):
        P.quantise_i8(torch.zeros(4, 100, dtype=torch.bfloat16))


def test_int8_folder_keeps_every_other_file_and_names_its_source(tmp_path):
    """The int8 variant of a bf16 folder: one proj8 file a layer (codes, scale, the bias unchanged), every other
    file byte-identical, the manifest's projector dtype i8 with the source's manifest hash, every file hashed."""
    src, out = tmp_path / "bf16", tmp_path / "int8"
    src.mkdir()
    bias = torch.randn(16).to(torch.bfloat16)
    for layer in (4, 5):
        P.save({f"proj.{layer}.weight": torch.randn(16, 256).to(torch.bfloat16), f"proj.{layer}.bias": bias},
               src / f"proj.L{layer}.safetensors")
    (src / "README.md").write_text("hello")
    files = {"proj.L4.safetensors": "", "proj.L5.safetensors": "", "README.md": ""}
    manifest = {"format": 1, "split": 4, "projector": {"dtype": "bf16", "layout": "plain_nk",
                                                       "files": {"4": "proj.L4.safetensors", "5": "proj.L5.safetensors"}},
                "files": {n: P.S.sha256_file(src / n) for n in files}}
    (src / "kva.json").write_text(json.dumps(manifest))
    assert P.main(["int8", "--from", str(src), "--out", str(out)]) == 0
    m = json.loads((out / "kva.json").read_text())
    assert m["projector"]["dtype"] == "i8" and m["projector"]["encoding"] == "i8*bf16[1x128]"
    assert m["projector"]["files"] == {"4": "proj8.L4.safetensors", "5": "proj8.L5.safetensors"}
    assert m["projector"]["source"]["kva_json_sha256"] == P.S.sha256_file(src / "kva.json")
    assert sorted(m["files"]) == ["README.md", "proj8.L4.safetensors", "proj8.L5.safetensors"]
    assert all(m["files"][n] == P.S.sha256_file(out / n) for n in m["files"])
    assert (out / "README.md").read_text() == "hello"
    with safe_open(str(out / "proj8.L5.safetensors"), "pt") as f:
        assert sorted(f.keys()) == ["proj.5.bias", "proj.5.codes", "proj.5.scale"]
        assert torch.equal(f.get_tensor("proj.5.bias"), bias)
        assert f.get_tensor("proj.5.codes").dtype == torch.int8 and f.get_tensor("proj.5.scale").shape == (16, 2)
    with pytest.raises(SystemExit):   # an int8 folder is not a source
        P.main(["int8", "--from", str(out), "--out", str(tmp_path / "again")])
