#!/usr/bin/env python3
"""A header-only stub of a Hugging Face safetensors checkpoint, plus extra shards, as one rad-convert input.

  stub_checkpoint.py --repo Qwen/Qwen3.8-Flash-Next --revision main --out DIR --extra SHARD [--extra SHARD ...]

Why: `rad-convert --reuse C --in-place` plans every weight the architecture declares from the INPUT namespace
(names, dtypes, shapes) before it copies anything from C, so appending the KVA tensors to a published container
needs the namespace of the checkpoint the container was made from (notes/sidecar.md §7). The weights themselves
are never read for a weight C already holds, so each shard here is its real safetensors header followed by a
hole of the real data length: apparent size = the original, disk use = the headers. Every non-safetensors file
of the repo (config, tokenizer, preprocessor configs, index) is downloaded as is. The extra shards are symlinked
in (relative paths) and added to the index's weight_map; a name clash with the checkpoint is refused.

Nothing is read from any shard's data region: the stub must never be fed to a run that writes a weight it
does not already hold in the --reuse container (those would quantise zeros). The `rad-convert --plan-only` diff
gate (notes/sidecar.md §7) is what checks that.

HF_ENDPOINT (default https://huggingface.co) and HF_TOKEN (optional, for gated repos) are read from the
environment. The revision is resolved to a commit and recorded in <out>/stub-manifest.json.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
TIMEOUT = 120
INDEX = "model.safetensors.index.json"


def get(url, byte_range=None):
    request = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    if byte_range:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        body = response.read()
    if byte_range and len(body) != byte_range[1] - byte_range[0] + 1:
        raise SystemExit(f"{url}: asked for bytes {byte_range}, got {len(body)} (server ignored the range?)")
    return body


def repo_files(repo, revision):
    """(commit sha, [{path, size}]) of every file in the repo at the revision."""
    info = json.loads(get(f"{ENDPOINT}/api/models/{repo}/revision/{revision}"))
    tree = json.loads(get(f"{ENDPOINT}/api/models/{repo}/tree/{info['sha']}"))
    return info["sha"], [e for e in tree if e["type"] == "file"]


def stub_shard(url, size, path):
    """Write the shard's real header and a hole up to its real size; returns the header's sha256."""
    header_len = int.from_bytes(get(url, (0, 7)), "little")
    header = get(url, (8, 8 + header_len - 1))
    tensors = {k: v for k, v in json.loads(header).items() if k != "__metadata__"}
    data_end = max(v["data_offsets"][1] for v in tensors.values())
    if 8 + header_len + data_end != size:
        raise SystemExit(f"{url}: header + data = {8 + header_len + data_end} B, but the file is {size} B")
    with open(path, "wb") as f:
        f.write(header_len.to_bytes(8, "little") + header)
        f.truncate(size)
    return hashlib.sha256(header).hexdigest()


def add_extra(out, index, extra):
    """Symlink an extra shard into the stub and name its tensors in the index."""
    extra = Path(extra).resolve()
    with open(extra, "rb") as f:
        names = [k for k in json.loads(f.read(int.from_bytes(f.read(8), "little"))) if k != "__metadata__"]
    clash = sorted(set(names) & set(index["weight_map"]))
    if clash:
        raise SystemExit(f"{extra.name}: {len(clash)} tensor name(s) already in the checkpoint, e.g. {clash[:3]}")
    link = out / extra.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(extra, out))
    index["weight_map"].update({name: extra.name for name in names})
    return len(names)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, help="Hugging Face model repo id")
    ap.add_argument("--revision", required=True, help="branch, tag or commit")
    ap.add_argument("--out", required=True, help="stub directory (created)")
    ap.add_argument("--extra", action="append", default=[], help="extra safetensors shard to add (repeatable)")
    args = ap.parse_args(argv)
    for extra in args.extra:
        if not Path(extra).is_file():
            raise SystemExit(f"--extra {extra} not found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    commit, files = repo_files(args.repo, args.revision)
    base = f"{ENDPOINT}/{args.repo}/resolve/{commit}"
    small = [e for e in files if not e["path"].endswith(".safetensors")]
    shards = [e for e in files if e["path"].endswith(".safetensors")]
    for e in small:
        (out / e["path"]).write_bytes(get(f"{base}/{e['path']}"))
    print(f"{args.repo}@{commit}: {len(small)} small files, {len(shards)} shards to stub", flush=True)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        jobs = {e["path"]: pool.submit(stub_shard, f"{base}/{e['path']}", e["size"], out / e["path"]) for e in shards}
        headers = {path: job.result() for path, job in jobs.items()}
    index = json.loads((out / INDEX).read_text(encoding="utf-8"))
    added = {Path(x).name: add_extra(out, index, x) for x in args.extra}
    (out / INDEX).write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    manifest = dict(repo=args.repo, revision=args.revision, commit=commit, endpoint=ENDPOINT,
                    small_files={e["path"]: e["size"] for e in small},
                    shards={e["path"]: dict(size=e["size"], header_sha256=headers[e["path"]]) for e in shards},
                    extra=added)
    (out / "stub-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(shards)} header-only shards ({sum(e['size'] for e in shards) / 2**30:.1f} GiB apparent), "
          f"extra tensors {added}; manifest stub-manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
