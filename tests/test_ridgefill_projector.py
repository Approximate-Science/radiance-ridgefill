"""tools/ridgefill_projector.py on synthetic inputs and a synthetic container (no model files, no GPU).

The tiny container's two hashes are the constants tests/arch_static_test.cpp expects of the same bytes, which is what
ties the builder's canonical tokenizer / anchor forms to the plugin's (arch/ridgefill_match.h).
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
import ridgefill_projector as P  # noqa: E402
import ridgefill_rules as R  # noqa: E402
from test_ridgefill_sidecar import FakeTokenizer, write_inputs  # noqa: E402

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
        "dials": [{"offset": 60, "kwarg": "ridgefill_share", "default": "0.25", "table": {"0.10": "<|c|>", "0.25": "<|d|>"}},
                  {"offset": 61, "kwarg": "ridgefill_alpha", "default": "1.0", "table": {"0": "<|c|>", "1.0": "<|d|>"}},
                  {"offset": 62, "kwarg": "ridgefill_tail", "default": "2048", "table": {"1024": "<|c|>", "2048": "<|d|>"}}],
        "end": {"offset": 63, "token": "<|e|>"}, "switch": {"kwarg": "ridgefill", "on_value": "on"},
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
    assert names == ["README.md", "chat_template.jinja", "correction.safetensors", "proj.L2.safetensors",
                     "proj.L3.safetensors", "ridgefill.json", "rowsel.safetensors"]
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
    m = json.loads((out / "ridgefill.json").read_text())
    assert m["format"] == 1 and m["adapter"] == "qwen4exp" and m["split"] == 2
    # the credit leads the manifest, and every tensor file's header carries it too
    assert list(m)[:6] == ["format", "name", "authors", "license", "doi", "homepage"]
    assert m["name"] == "ridgefill-projector-qwen3.8-flash-next-bf16"
    assert m["authors"] == ["Dylan Johnston", "tcclaviger"] and m["license"] == "Apache-2.0"
    assert m["doi"] == "10.5281/zenodo.23179168"
    assert m["homepage"] == "https://huggingface.co/Dyluhn/ridgefill-projector-qwen3.8-flash-next-bf16"
    for name in (n for n in m["files"] if n.endswith(".safetensors")):
        with safe_open(str(out / name), "pt") as f:
            assert f.metadata() == {"format": "ridgefill-projector-1", "name": m["name"],
                                    "authors": "Dylan Johnston, tcclaviger", "license": "Apache-2.0",
                                    "doi": "10.5281/zenodo.23179168"}, name
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
    assert set(m["files"]) == {p.name for p in out.iterdir()} - {"ridgefill.json", "README.md"}   # documentation unlisted
    for name, digest in m["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert m["marker"]["template_source_sha256"] == hashlib.sha256(b"{{ messages }}").hexdigest()


def test_the_template_is_the_models_own_with_the_marker_in_front(built):
    out, _ = built
    merged = (out / "chat_template.jinja").read_text()
    assert merged.endswith("{{ messages }}") and merged.startswith("{#- ridgefill-marker v1")


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
    a, b = json.loads((out / "ridgefill.json").read_text()), json.loads((again / "ridgefill.json").read_text())
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
    (src / "ridgefill.json").write_text(json.dumps(manifest))
    assert P.main(["int8", "--from", str(src), "--out", str(out)]) == 0
    m = json.loads((out / "ridgefill.json").read_text())
    assert m["projector"]["dtype"] == "i8" and m["projector"]["encoding"] == "i8*bf16[1x128]"
    assert m["projector"]["files"] == {"4": "proj8.L4.safetensors", "5": "proj8.L5.safetensors"}
    assert m["projector"]["source"]["manifest_sha256"] == P.S.sha256_file(src / "ridgefill.json")
    assert m["name"] == "ridgefill-projector-qwen3.8-flash-next-i8" and m["authors"] == ["Dylan Johnston", "tcclaviger"]
    assert sorted(m["files"]) == ["proj8.L4.safetensors", "proj8.L5.safetensors"]   # README copied, never listed
    assert all(m["files"][n] == P.S.sha256_file(out / n) for n in m["files"])
    assert (out / "README.md").read_text() == "hello"
    with safe_open(str(out / "proj8.L5.safetensors"), "pt") as f:
        assert sorted(f.keys()) == ["proj.5.bias", "proj.5.codes", "proj.5.scale"]
        assert torch.equal(f.get_tensor("proj.5.bias"), bias)
        assert f.get_tensor("proj.5.codes").dtype == torch.int8 and f.get_tensor("proj.5.scale").shape == (16, 2)
    with pytest.raises(SystemExit):   # an int8 folder is not a source
        P.main(["int8", "--from", str(out), "--out", str(tmp_path / "again")])


def test_final_adds_the_mtp_map_from_the_fitted_source_only(tmp_path):
    """`final`: the folder copied byte for byte plus final.safetensors (weight [w, w] and bias [w] split from the
    source's `final` [w, w + 1]); refused when the source is not the file the folder was fitted from, or holds no
    `final`."""
    from safetensors.torch import save_file
    src, out = tmp_path / "folder", tmp_path / "with-final"
    src.mkdir()
    (src / "README.md").write_text("hello")
    proj = tmp_path / "proj.safetensors"
    fmap = torch.randn(32, 33).to(torch.bfloat16)
    save_file({"layer.4": torch.randn(8, 33).to(torch.bfloat16), "final": fmap}, str(proj))
    manifest = {"format": 1, "stream_width": 32, "fit": {"proj_sha256": P.S.sha256_file(proj)},
                "files": {"README.md": P.S.sha256_file(src / "README.md")}}
    (src / "ridgefill.json").write_text(json.dumps(manifest))
    assert P.main(["final", "--from", str(src), "--proj", str(proj), "--out", str(out)]) == 0
    m = json.loads((out / "ridgefill.json").read_text())
    assert m["final"] == {"file": "final.safetensors", "dtype": "bf16", "source_sha256": P.S.sha256_file(proj)}
    assert m["files"]["final.safetensors"] == P.S.sha256_file(out / "final.safetensors")
    assert (out / "README.md").read_text() == "hello"
    with safe_open(str(out / "final.safetensors"), "pt") as f:
        assert torch.equal(f.get_tensor("final.weight"), fmap[:, :-1])
        assert torch.equal(f.get_tensor("final.bias"), fmap[:, -1])
    other = tmp_path / "other.safetensors"
    save_file({"final": fmap}, str(other))
    with pytest.raises(SystemExit):   # not the folder's source
        P.main(["final", "--from", str(src), "--proj", str(other), "--out", str(tmp_path / "x")])
    nofinal = tmp_path / "nofinal.safetensors"
    save_file({"layer.4": torch.randn(8, 33).to(torch.bfloat16)}, str(nofinal))
    manifest["fit"]["proj_sha256"] = P.S.sha256_file(nofinal)
    (src / "ridgefill.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):   # no final map in it
        P.main(["final", "--from", str(src), "--proj", str(nofinal), "--out", str(tmp_path / "y")])


def test_reseal_unlists_documentation_and_touches_no_file(tmp_path):
    """reseal: an existing folder whose ridgefill.json lists README.md is rewritten to list everything but documentation;
    every other file is byte-identical and the rest of the manifest is unchanged. A changed weight is refused."""
    folder = tmp_path / "p"
    folder.mkdir()
    P.save({"proj.4.weight": torch.randn(4, 8).to(torch.bfloat16)}, folder / "proj8.L4.safetensors")
    (folder / "README.md").write_text("install flow")
    (folder / "notes.md").write_text("x")
    manifest = {"format": 1, "model": {"anchors": {"a": "00"}}, "files": {n: P.S.sha256_file(folder / n) for n in
                                                                           ("README.md", "notes.md", "proj8.L4.safetensors")}}
    (folder / "ridgefill.json").write_text(json.dumps(manifest))
    before = {n: P.S.sha256_file(folder / n) for n in ("README.md", "notes.md", "proj8.L4.safetensors")}
    assert P.main(["reseal", "--folder", str(folder)]) == 0
    m = json.loads((folder / "ridgefill.json").read_text())
    assert m["files"] == {"proj8.L4.safetensors": before["proj8.L4.safetensors"]}
    assert {k: v for k, v in m.items() if k != "files"} == {k: v for k, v in manifest.items() if k != "files"}
    assert before == {n: P.S.sha256_file(folder / n) for n in before}
    (folder / "README.md").write_text("---\nlicense: apache-2.0\n---\n# a model card")   # the hub's README: still fine
    assert P.main(["reseal", "--folder", str(folder)]) == 0
    P.save({"proj.4.weight": torch.zeros(4, 8).to(torch.bfloat16)}, folder / "proj8.L4.safetensors")
    with pytest.raises(SystemExit):
        P.main(["reseal", "--folder", str(folder)])


def test_credit_stamps_a_new_folder_and_changes_no_tensor_byte(tmp_path):
    """credit: a folder built before the credit (and before the rename: its manifest under the working name) becomes a
    NEW folder whose manifest leads with the credit and whose .safetensors headers carry it; every tensor's dtype,
    shape and bytes are the source's; the parked marker's template is rebuilt with the RidgeFill kwargs over the same
    base bytes; every other manifest field is kept. A changed source file and a non-empty --out are refused."""
    from safetensors.torch import save_file
    old = P.OLD_NAME                              # the working name, as the old folders spell it
    src, out = tmp_path / "old", tmp_path / "new"
    src.mkdir()
    codes = torch.randint(-127, 128, (8, 256), dtype=torch.int8)
    save_file({"proj.4.codes": codes, "proj.4.scale": torch.rand(8, 2).bfloat16()}, str(src / "proj8.L4.safetensors"),
              metadata={"format": old + "-projector-1"})
    save_file({"st.4": torch.randn(2, 3, 3)}, str(src / "correction.safetensors"), metadata={"format": old + "-projector-1"})
    spec = json.loads((Path(__file__).resolve().parent / "fixtures" / "ridgefill-marker-spec.json").read_text())
    spec = json.loads(json.dumps(spec).replace("ridgefill", old))
    base = b"{{ messages }} the model's own template"
    block = P.T.build_block(spec).replace("ridgefill-marker", old + "-marker")   # the old tool's marker comments
    (src / "chat_template.jinja").write_bytes(block.encode() + base)
    (src / "README.md").write_text("old docs")
    names = ["chat_template.jinja", "correction.safetensors", "proj8.L4.safetensors"]
    manifest = {"format": 1, "split": 4, "projector": {"dtype": "i8", "files": {"4": "proj8.L4.safetensors"},
                                                       "source": {"folder": "bf16", old + "_json_sha256": "ab" * 32}},
                "marker": {"spec": spec, "template_source_sha256": hashlib.sha256(base).hexdigest()},
                "files": {n: P.S.sha256_file(src / n) for n in names}}
    (src / P.OLD_MANIFEST).write_text(json.dumps(manifest))
    assert P.main(["credit", "--from", str(src), "--out", str(out)]) == 0
    m = json.loads((out / "ridgefill.json").read_text())
    assert list(m)[:6] == ["format", "name", "authors", "license", "doi", "homepage"]
    assert m["name"] == "ridgefill-projector-qwen3.8-flash-next-i8" and m["doi"] == "10.5281/zenodo.23179168"
    assert m["split"] == 4 and m["projector"]["files"] == manifest["projector"]["files"]
    assert m["projector"]["source"] == {"folder": "bf16", "manifest_sha256": "ab" * 32}
    assert m["rebuilt_from"] == {"folder": "old", "manifest_sha256": P.S.sha256_file(src / P.OLD_MANIFEST)}
    assert sorted(m["files"]) == names and all(m["files"][n] == P.S.sha256_file(out / n) for n in names)
    assert not (out / "README.md").exists() and not (out / P.OLD_MANIFEST).exists()
    for n in ("proj8.L4.safetensors", "correction.safetensors"):
        assert P.same_tensors(src / n, out / n)
        with safe_open(str(src / n), "pt") as a, safe_open(str(out / n), "pt") as b:
            assert b.metadata()["authors"] == "Dylan Johnston, tcclaviger" and b.metadata()["license"] == "Apache-2.0"
            for k in a.keys():
                assert torch.equal(a.get_tensor(k), b.get_tensor(k)), k
    with safe_open(str(out / "proj8.L4.safetensors"), "pt") as f:
        assert torch.equal(f.get_tensor("proj.4.codes"), codes)
    assert m["marker"]["spec"]["switch"]["kwarg"] == "ridgefill"
    assert [d["kwarg"] for d in m["marker"]["spec"]["dials"]] == ["ridgefill_share", "ridgefill_alpha", "ridgefill_tail"]
    merged = (out / "chat_template.jinja").read_bytes()
    assert merged == P.T.build_block(m["marker"]["spec"]).encode() + base and old.encode() not in merged
    with pytest.raises(SystemExit):   # --out must be new
        P.main(["credit", "--from", str(src), "--out", str(out)])
    (src / "correction.safetensors").write_bytes((src / "correction.safetensors").read_bytes()[:-4] + b"\0\0\0\0")
    with pytest.raises(SystemExit):   # a changed source is never credited
        P.main(["credit", "--from", str(src), "--out", str(tmp_path / "again")])


def test_stamp_keeps_the_data_section_and_same_tensors_catches_a_changed_byte(tmp_path):
    from safetensors.torch import save_file
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file({"x": torch.arange(12, dtype=torch.float32), "y": torch.ones(3, dtype=torch.bfloat16)}, str(a))
    P.stamp(a, b, P.st_metadata("n"))
    ra, rb = a.read_bytes(), b.read_bytes()
    (_, da), (_, db) = P.st_header(ra), P.st_header(rb)
    assert ra[da:] == rb[db:] and db % 8 == 0
    assert P.same_tensors(a, b) == ["x", "y"]
    b.write_bytes(rb[:-1] + bytes([rb[-1] ^ 1]))
    with pytest.raises(SystemExit):
        P.same_tensors(a, b)
