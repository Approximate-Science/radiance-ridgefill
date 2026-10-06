#!/bin/sh
# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
# ident_long.sh -- ident.sh with a LONG prompt (R89): is a server reproducible when RidgeFill approximates the
# prompt? ident.sh's two questions are too short for any approximate chunk; here each question follows a
# ~30K-token document, so a speed or quality server approximates most of it. One hash per (temperature,
# question), compared across restarts of the SAME configuration.
#
#   RK_DOCS=<ppl.jsonl> RK_PORT=8100 scripts/ident_long.sh     # against a server already up
#
# RK_DOCS: the quick quality set (its first doc of bucket 32k is the document). 200 tokens answered at
# temperatures 0, 0.7 and 1.0 with a fixed seed, thinking off (ident.sh's request otherwise).
P="${RK_PORT:-8100}"
[ -r "${RK_DOCS:-}" ] || { echo "ident_long.sh: RK_DOCS is not a readable file" >&2; exit 1; }
WORK=$(mktemp -d "${TMPDIR:-/tmp}/ident.XXXXXX"); trap 'rm -rf "$WORK"' EXIT INT TERM
jq -r 'select(.bucket == "32k") | .prompt' "$RK_DOCS" | head -c 120000 > "$WORK/doc.txt"
ask() {
  jq -n --rawfile d "$WORK/doc.txt" --arg q "$2" --argjson t "$1" \
     '{model:"m",messages:[{role:"user",content:($d + "\n\n" + $q)}],max_tokens:200,temperature:$t,seed:12345,
       chat_template_kwargs:{enable_thinking:false}}' > "$WORK/iq.json"
  curl -s -m 900 http://localhost:$P/v1/chat/completions -H "Content-Type: application/json" \
       -d @"$WORK/iq.json" > "$WORK/a.json"
  printf '%s/%s' "$(jq -r '.choices[0].message.content' "$WORK/a.json" | sha256sum | cut -c1-16)" \
                 "$(jq -r '.usage.prompt_tokens' "$WORK/a.json")"
}
Q1="Summarise the document above in three sentences."
Q2="List five specific facts stated in the document above."
for T in 0 0.7 1.0; do
  printf "temp %-4s  %s  %s\n" "$T" "$(ask $T "$Q1")" "$(ask $T "$Q2")"
done
