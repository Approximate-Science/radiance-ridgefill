# Stage F pre-code verification (step F.1) — engine line audit + marker token check

Workspace: agent-worker `4d935331`, REPORT ONLY — no tracked files touched.
Engine audited: `/var/home/dylan/projects/inference/radiance` at commit `140987fa7b0d9a1ec60e19809a70988f711997fe` (`140987f`) — confirmed with `git log -1`, matches REFUTATION-2's stated commit.
Design docs: `fix-246/REFUTATION-2-marker.md` §1–§3, §6; `fix-246/PLAN-FIX.md` §11; marker spec: `worker-context/ridgefill-marker-spec.json`.

## Summary (updated as verification proceeds)

| # | location | verdict |
|---|---|---|
| 1 | `libr4d/r4d_attn_prefill_h256_gqa6.hip:122-161` (causality from seqused/q_len/row) | CONFIRMED (exact lines) |
| 2 | `libr4d/r4d_attn_prefill_h256_gqa6.hip:438-441` (`inv = 0` when `l = 0`) | CONFIRMED WITH LINE DRIFT — the `inv` definition is at **:435**; the write of `acc*inv` (the "write 0" effect) is at **:443-445**; 438-441 is the inner of that write loop |
| 3 | `libr4d/r4d_qsa_work_bf16.hip:69-93` (p0 = pos[cu[s]], consecutive rows) | CONFIRMED (exact lines; p0 reads at :79 and :101, consecutive-row assumption at :84-89, :119-122, :131) |
| 4 | `libr4d/r4d_qsa_work_bf16.hip:96-109` (block-key rotation from rope planes, mc path) | CONFIRMED WITH LINE DRIFT — rope-plane branch is at **:109-123** (`rope[c*rplane + t0 + (f-p0)]` at :122); the `rope` operand is optional multi-component `RadBatch::rope_pos` (decl comment :149-150, `QW_ROPE` :183, :193) |
| 5 | `libr4d/r4d_gdn_conv_w4_h128_bf16.hip:145-178` (gate cumsum restarts per tile; GDN chunk = 64) | CONFIRMED (exact lines; `CP_BT 64` at :44 — "token block = the scan's chunk, so the cumsum stays wave-local"; per-chunk scan starts at g0=g1=0, :154, wave-scan :163-167) |
| 6 | `libr4d/r4d_gdn_conv_w4_h128_bf16.hip:636-639` (l2 eps compiled in) | CONFIRMED (exact lines; eps 1e-6 compiled in, a different value refused with `RAD_E_UNSUPPORTED`) |
| 7 | `docs/OPS.md:888-897` (n-gram cold window = EOS, every deeper shift too; conv cold window zero) | CONFIRMED (exact lines; n-gram EOS fallback :888-890, "A cold window is EOS and not zero … a convolution window: zero is a real token id" :896-897) |
| 8 | `arch/qwen4exp_fp8/qwen4exp_fp8.cpp:962` (PLE eos id from container metadata) | CONFIRMED (exact line; `pc.eos = meta_int2(meta, "qwen4exp.eos_token_id", "eos_token_id", -1)` in the PLE layer's config at :954-962) |
| 9 | `arch/common/rad_arch.h:741-751` (rope_mc true when mrope_section present) | CONFIRMED (exact lines) |
| 10 | `arch/common/rad_arch.h:935-941` (fused reads rope_pos plane 0 via rope_pos1; unfused reads [3,T] planes via rope_posmc) | CONFIRMED (exact lines) |
| 11 | `core/server/oai.cpp:655-680` (kwargs copied key-by-key as JSON dump(), only enable_thinking type-checked) | CONFIRMED (exact lines; dump() at :670, enable_thinking bool check :671-677) |
| 12 | `core/config.cpp:105` (`--override-chat-template PATH`) | CONFIRMED (exact line) |
| 13 | `core/config.cpp:145` (`--chat-template-kwargs JSON`) | CONFIRMED (exact lines :145-149) |
| 14 | marker token ids (spec vs tokenizer) | CONFIRMED — all 7 marker pieces encode to exactly ONE token each; ids match the spec exactly; `...` = 248044 = the container's configured `text_config.eos_token_id` |

**Counts: 14 CONFIRMED / 0 DIFFERENT / 0 NOT FOUND.** Two of the 14 have minor line drift (items 2 and 4; mechanism found by content within ~10 lines of the citation) — neither breaks any Stage F step:
- Item 2 (`inv = 0` at :435 vs cited 438-441): REFUTATION-2 §3 step (a) — `libr4d/r4d_attn_prefill_h256_gqa6.hip:122-161` is exact, and the write-0 effect holds at :435/:443-445. Erase step (a) stands.
- Item 4 (rope-plane read at :111-123 vs cited 96-109): REFUTATION-2 §3 step (c) — work's `p0 = pos[cu[s]]` citation (69-93) is exact, and block-key rotation does read the rope planes when the batch carries them. Erase step (c) stands.

The end-token string in `ridgefill-marker-spec.json` is `<|endoftext|>` (hex `3c7c656e646f66746578747c3e`), 13 bytes, token id 248044 — terminal display of that string is mangled in some renders; the file bytes are verified correct (see §2).

---

## 1. Engine line audit (per location, lines pasted, claim, verdict)

### 1.1 `libr4d/r4d_attn_prefill_h256_gqa6.hip:122-161` — dense prefill causality

**REFUTATION-2 §3(a) claim:** "the dense prefill kernel takes causality from `seqused`, `q_len` and the row (`klimit = ctx − qlen + qpos`, `libr4d/r4d_attn_prefill_h256_gqa6.hip:122-161`) — not from `positions`."

**Verdict: CONFIRMED** (exact lines). Causality is computed from `seqused_k[seq]` (=ctx), `qlen` from `cu_q`, and the block row `qpos`; `positions` does not appear in the kernel's causality path.

```hip
122	    const int ctx = a.seqused_k[seq];
123	    if (ctx <= 0) return;
124	
125	    // Ragged batches: this sequence owns rows [qoff, qoff+qlen) of the dense q and out blocks.
126	    // grid.x is sized from the batch MAXIMUM q_len, so a shorter sequence has blocks past its own
127	    // end -- they exit here rather than run a full KV walk to write nothing.
128	    const int qoff = r4d_q_off(a.cu_q, seq, a.q_len);
129	    const int qlen = r4d_q_len(a.cu_q, seq, a.q_len);
130	    if (qb * BLOCK_Q >= qlen) return;
131	
132	    const int row = warp * 16 + c;
133	    const int qi = row / GQA, hi = row - qi * GQA;
134	    const int qpos = qb * BLOCK_Q + qi;
135	    const bool live = (qpos < qlen);
136	    const int qrow = live ? qpos : (qlen - 1);
137	    const int qhead = kvh * GQA + hi;
138	
139	    const int* __restrict__ bt = a.block_table + (size_t)seq * a.max_blocks;
...
161	    const int klimit = ctx - qlen + qpos;
```

(Grep over the file finds no `positions`-derived causality; the row address `qoff + qrow` :147-148 and the causal bound :161 are the only row-identity inputs.)

### 1.2 `libr4d/r4d_attn_prefill_h256_gqa6.hip:438-441` — marker rows write 0

**REFUTATION-2 §3(a) claim:** "Marker rows get no key and write 0 (`inv = 0` when `l = 0`, `:438-441`)."

**Verdict: CONFIRMED with line drift** — the cited mechanism lives 4-8 lines earlier/around; found by content (`grep -n inv`): `inv` defined at **:435**, applied at :443-445.

```hip
433	    if (!live) return;
434	    l_i += swap16(l_i);
435	    const float inv = (l_i > 0.0f) ? (vdesc / l_i) : 0.0f;
436	    uint16_t* op = (uint16_t*)a.out
437	                 + ((size_t)(qoff + qpos) * a.q_heads + qhead) * HEAD_DIM;
438	    #pragma unroll
439	    for (int dt = 0; dt < NKS; ++dt) {
440	        uint32_t tw[4];
441	        #pragma unroll
442	        for (int e = 0; e < 8; e += 2)
443	            tw[e >> 1] = (uint32_t)f32_to_bf16(acc[dt][e] * inv)
444	                       | ((uint32_t)f32_to_bf16(acc[dt][e + 1] * inv) << 16);
445	        *(uint4*)(op + dt * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
446	    }
```

A live row with no keys accumulates `l_i = 0` → `inv = 0` → the store at :445 writes zeros. Substance of the claim holds exactly; the design's erase step (a) "marker rows get no key and write 0" is correct.

### 1.3 `libr4d/r4d_qsa_work_bf16.hip:69-109` — QSA work p0 and consecutive rows

**REFUTATION-2 §3(c) claim:** "QSA work takes p0 = `pos[cu[s]]` and assumes consecutive rows (`r4d_qsa_work_bf16.hip:69-93`)."

**Verdict: CONFIRMED** (exact lines; also the nc computation at :66-68 that §11.5's "nc is the same for every layer" style device computation relies on).

```hip
66	    if (w < seqs && threadIdx.x == 0) {
67	        const int t1 = cu[w + 1];
68	        nc[w] = t1 > cu[w] ? (pos[t1 - 1] + 1) / ratio : 0;
69	    }
70	    if (w >= max_work) return;
...
77	        const int t0 = cu[i], t1 = cu[i + 1];
78	        if (t1 <= t0) continue;
79	        const int p0 = pos[t0], p1 = pos[t1 - 1];
80	        /* A block is complete when its LAST token exists: ... */
84	        const int lo = p0 / ratio;
85	        const int hi = (p1 + 1) / ratio - 1;
86	        const int cnt = hi >= lo ? hi - lo + 1 : 0;
87	        if (w < run + cnt) { s = i; j = w - run + lo; break; }
...
99	    const int b  = j;
100	    const int t0 = cu[s];
101	    const int p0 = pos[t0];
```

`p0 = pos[cu[s]]` at :79/:101; block-key gather at :119-135 maps position `f` to row `t0 + (f - p0)` — the consecutive-rows assumption. A shifted p0 for a marker row would index the block table at a negative offset, exactly as the design says ("this is why work must stay absolute").

### 1.4 `libr4d/r4d_qsa_work_bf16.hip:96-109` — block-key rotation from the rope planes (mc path)

**REFUTATION-2 §3(c) claim:** "Block-key rotation is taken from the rope planes (`r4d_qsa_work_bf16.hip:96-109`, `mc` path), so real block keys rotate at real positions."

**Verdict: CONFIRMED with line drift** — the rope branch starts at :109 (`if (!rope)`) and the plane read is at :111-123.

```hip
108	    if (threadIdx.x == 0) {
109	        if (!rope) {
110	            bpos[(size_t)w * bstep] = b * ratio;
111	        } else {
...
119	            const int f = b * ratio;
120	            for (int c = 0; c < 3; ++c)
121	                bpos[(size_t)w * bstep + c] =
122	                    f >= p0 ? rope[(size_t)c * rplane + t0 + (f - p0)] : f + (rope[t0] - p0);
123	        }
124	    }
```

Operand declaration comment (found by content): `rope [3][rplane] i32, optional: this step's rotary components (RadBatch::rope_pos)` at :149-150, wired as `QW_ROPE` (:183, :193). When the batch carries rope planes, block-key rotation reads them (three components, per plane); when it does not, `b * ratio` from index positions. So keeping QSA work on ABSOLUTE `positions` while the copy supplies shifted `rope_pos` planes makes real block keys rotate at real rotary positions — the design's §3(c) split. Substance holds.

### 1.5 `libr4d/r4d_gdn_conv_w4_h128_bf16.hip:145-178` — GDN tile alignment and gate cumsum

**REFUTATION-2 §3(d) claim:** "Tiles stay aligned because B = 64 = the GDN chunk, and each tile's gate cumsum restarts (`:145-178`)."

**Verdict: CONFIRMED** (exact lines, plus `CP_BT` at :44).

```hip
 44	#define CP_BT     64            // token block = the scan's chunk, so the cumsum stays wave-local
...
138	      const int nc = (e - b + CP_BT - 1) / CP_BT;
139	      if (want < acc + nc) { n = i; bos = b; slen = e - b; t0 = (want - acc) * CP_BT;
140	                             rows = min(CP_BT, slen - t0); break; }
...
145	  const bool first = (t0 == 0);
...
150	  if (blockIdx.y == 0) {
151	    for (int hv = warp; hv < H; hv += HPB) {
...
154	      float g0 = 0.0f, g1 = 0.0f;
155	      if (i0 < rows)
156	        g0 = -alog * cp_softplus(cp_ab<ABF16>(av, (size_t)(bos + t0 + i0) * ab_stride + hv)
157	                                 + dtb, sp_thr);
...
161	      // inclusive wave scan of the per-lane pair sums, made exclusive by subtracting the pair
162	      float p = g0 + g1;
163	#pragma unroll
164	      for (int d = 1; d < 32; d <<= 1) {
165	        const float o = __shfl_up(p, d, 32);
166	        if (lane >= d) p += o;
167	      }
168	      const float pre = p - (g0 + g1);
```

Chunks are exactly 64 tokens (`CP_BT 64`), one workgroup per chunk, and each chunk's inclusive scan starts from zero (`g0 = g1 = 0` :154, warp-local scan :163-167) — the cumsum restarts per 64-row tile. B = 64 = the GDN chunk, so a marker occupies exactly one aligned tile.

### 1.6 `libr4d/r4d_gdn_conv_w4_h128_bf16.hip:636-639` — l2-norm epsilon compiled in

**REFUTATION-2 §3(d) claim:** "conv output 0 → silu 0 → l2-norm 0 (eps compiled in, `r4d_gdn_conv_w4_h128_bf16.hip:636-639`)".

**Verdict: CONFIRMED** (exact lines).

```hip
636	    /* libr4d's l2 norm is rsqrt(ss + 1e-6f) with the epsilon COMPILED IN, not an argument. A
637	     * caller that names a different one is asking for arithmetic this kernel does not do, and
638	     * running it anyway would be a silently different normalisation. Refused by name instead. */
639	    if ((float)r4d_getf(a, "l2_eps", 1e-6) != 1e-6f) return R4D_NO(RAD_E_UNSUPPORTED);
```

The eps is fixed at 1e-6 in the kernel (not an operand), so a zero conv output → silu(0) = 0 → rsqrt(0 + 1e-6) normalised zero — i.e. q = k = v = 0 rows, matching the design's "conv output 0 → silu 0 → l2-norm 0".

### 1.7 `docs/OPS.md:888-897` — PLE n-gram cold window is EOS; conv cold window is zero

**REFUTATION-2 §3(e) claim:** "`ngram_ids` treats a shift that crosses EOS as EOS for every deeper shift, and a cold window IS EOS (docs/OPS.md:888-897). … A cold convolution window is zero (OPS.md:896-897)."

**Verdict: CONFIRMED** (exact lines).

```markdown
888	where `heads` is `ngram - 1` equal blocks and block `b` uses the `(b+1)`-gram. `t_s` is the token
889	`s` positions back, **or the EOS token if the shift would cross one** — and once a shift falls back
890	to EOS every deeper shift does too. `mult`, `vocab_sizes` and `offsets` are weights because the
891	the checkpoint carries them; the reference builds them from a seed and a prime search at construction
...
894	**`state` is a rolling window of committed ids**, `ngram - 1 + n_spec` deep, read at
895	`num_accepted - 1` and rewritten in the layout `ple_conv` describes below — operand for operand,
896	including the two layouts `num_accepted`'s presence chooses between. **A cold window is EOS and not
897	zero**, which is the one place it differs from a convolution window: zero is a real token id.
```

Both halves hold: n-gram shifts crossing EOS (or before the start) fall back to EOS and stay EOS for deeper shifts (:888-890); the convolution window's cold fill is zero (:896-897). This is the basis of §3(e)'s "with 248044 last, real token 0 hashes exactly like stock's position 0" — the marker's final `<|endoftext|>` plays the role of the EOS boundary a cold stock window already has.

### 1.8 `arch/qwen4exp_fp8/qwen4exp_fp8.cpp:962` — the PLE block's EOS boundary id

**REFUTATION-2 §1 claim:** "The LAST marker token must be `<|endoftext|>` 248044. It is the id the PLE block treats as a boundary (`arch/qwen4exp_fp8/qwen4exp_fp8.cpp:962`; §3e)."

**Verdict: CONFIRMED** (exact line).

```cpp
954	        if (l == m.ple_layer) {
955	            PleLayer::Config pc{};
...
962	            pc.eos        = meta_int2(meta, "qwen4exp.eos_token_id", "eos_token_id", -1);
```

The PLE layer's config takes its eos from the container's `eos_token_id` metadata — the id the PLE block treats as the n-gram boundary (per OPS.md :888-897 above). The plugin's declare-time refusal when the container's eos id differs from 248044 (PLAN-FIX §11.1) is therefore checking the right thing.

### 1.9 `arch/common/rad_arch.h:741-751` — rope_mc set from mrope_section

**REFUTATION-2 §3 claim:** "This model has M-RoPE sections (`text_config.rope_parameters.mrope_section` = [11, 11, 10]), so `g.rope_mc` is true (`arch/common/rad_arch.h:741-751`)."

**Verdict: CONFIRMED for the code** (exact lines). The model-metadata half (that this container's `mrope_section` is [11,11,10]) is container metadata — checked in §2 below against the tokenizer dir's config.

```cpp
740	    {
741	        const char* sec = rad_meta_gets(m, "rope_parameters.mrope_section",
742	                                        rad_meta_gets(m, "text_config.rope_parameters.mrope_section",
743	                                                      ""));
744	        if (sec && *sec) {
745	            const bool inter =
746	                rad_meta_geti(m, "rope_parameters.mrope_interleaved",
747	                              rad_meta_geti(m, "text_config.rope_parameters.mrope_interleaved",
748	                                            0)) != 0;
749	            g.rope_mc = true;
750	            std::snprintf(g.rope_mode, sizeof g.rope_mode, "%s", inter ? "imrope" : "mrope");
751	            std::snprintf(g.rope_sections, sizeof g.rope_sections, "%s", sec);
752	        }
753	    }
```

When the container carries `mrope_section`, `g.rope_mc = true` and the mode is mrope/imrope. (For the served container's actual value, see §2.2 — `[11,11,10]` is confirmed in the model config.)

### 1.10 `arch/common/rad_arch.h:935-941` — rotary consumers read rope_pos

**REFUTATION-2 §3(b) claim:** "Every rotary consumer then reads `rope_pos` when the batch carries it (`rad_arch.h:935-941`) … fused paths read plane 0 (`rope_pos1`), unfused read the planes (`rope_posmc`)."

**Verdict: CONFIRMED** (exact lines).

```cpp
935	inline RadOperand rope_pos1(const RadBatch* b, int64_t T) {
936	    return praw(b->rope_pos ? b->rope_pos : b->positions, RAD_I32, T);
937	}
938	inline RadOperand rope_posmc(const Geom& g, const RadBatch* b, int64_t T) {
939	    if (g.rope_mc && b->rope_pos) return praw2(b->rope_pos, RAD_I32, 3, T);
940	    return praw(b->positions, RAD_I32, T);
941	}
```

`rope_pos1` falls back to `positions` only when the batch does not carry `rope_pos`; otherwise plane 0. `rope_posmc` takes the dense [3, T] planes for an M-RoPE model when the batch carries them (:938-939). The copy-with-rope_pos mechanism of §3(b) therefore reaches every rotary consumer.

### 1.11 `core/server/oai.cpp:655-680` — kwargs reach the template unfiltered

**REFUTATION-2 §1 / PLAN-FIX §11.1 claim:** "Request `chat_template_kwargs` reach the template unfiltered: every key is copied as JSON (`dump()`), and only `enable_thinking` is type-checked (`core/server/oai.cpp:657-678`)" (task range 655-680).

**Verdict: CONFIRMED** (exact lines).

```cpp
655	static int parse_thinking(const json& b, const OaiLimits& lim, ChatRenderOptions* opt,
656	                          ApiError* err) {
657	    auto kw = b.find("chat_template_kwargs");
658	    const bool has_kw = kw != b.end() && !kw->is_null();
659	    if (has_kw && !kw->is_object()) {
660	        *err = err_bad("chat_template_kwargs", "expected an object");
661	        return RAD_E_INVAL;
662	    }
...
666	    if (has_kw) {
667	        for (auto it = kw->begin(); it != kw->end(); ++it) {
668	            /* dump(), not get<string>(): the template context wants JSON, so a string value has
669	             * to arrive still quoted. See ChatRenderOptions. */
670	            opt->template_kwargs[it.key()] = it.value().dump();
671	            if (it.key() == "enable_thinking") {
672	                if (!it.value().is_boolean()) {
673	                    *err = err_bad("chat_template_kwargs.enable_thinking", "expected a boolean");
674	                    return RAD_E_INVAL;
675	                }
676	                opt->enable_thinking = it.value().get<bool>();
677	            }
678	        }
679	    }
```

Every key (including `ridgefill`) is copied via `dump()` as raw JSON into the template context; only `enable_thinking` gets a type check. `"ridgefill": "on"` arrives at the template as the JSON value `"on"`.

### 1.12 `core/config.cpp:105` — `--override-chat-template`

**REFUTATION-2 §1 / PLAN-FIX §11.1 claim:** "The operator can replace the template without touching the container: `--override-chat-template PATH` (`core/config.cpp:105, 462-463`)."

**Verdict: CONFIRMED for :105** (exact line).

```cpp
105	    { "--override-chat-template", "PATH", "a Jinja chat template file, used in place of the one the container carries" },
```

(The :462-463 half of that citation is the option's parse/handling site, outside this task's listed ranges but spot-checked: `grep -n override-chat-template core/config.cpp` shows the option definition at :105 and its parsing at :462-463 — "used in place of the one the container carries".)

### 1.13 `core/config.cpp:145` — `--chat-template-kwargs`

**REFUTATION-2 §1 / PLAN-FIX §11.1 claim:** "A server default can come from `--chat-template-kwargs` (`config.cpp:145-147, 552-553`)."

**Verdict: CONFIRMED for :145-149** (exact lines).

```cpp
145	    { "--chat-template-kwargs", "JSON", "variables handed to the chat template on every chat request: a JSON object,\n"
146	"                                    or @FILE to read one. Repeatable, a later key replacing an earlier one.\n"
147	"                                    A key the request's own chat_template_kwargs names is the request's, and a\n"
148	"                                    request that says anything about thinking replaces enable_thinking and\n"
149	"                                    reasoning_effort here" },
```

Server-side template-variable defaults exist and request-side `chat_template_kwargs` win per key — matching §11.1's "server default can come from `--chat-template-kwargs`; the startup render runs without `ridgefill`".

---

## 2. Marker tokens vs the tokenizer

### 2.1 Tokens the marker uses (from `ridgefill-marker-spec.json`)

Structure: 64 tokens — offsets 0-59 fixed alternating pattern, 60/61/62 dial tokens, 63 end.

| piece | where | token (text) | spec id |
|---|---|---|---|
| pattern piece A | offsets 0, 2, 4, … (even) | `<\|quad_start\|>` | 248051 |
| pattern piece B | offsets 1, 3, 5, … (odd) | `<\|quad_end\|>` | 248052 |
| ridgefill_share dial | offset 60 | `<\|box_start\|>` 0.10 / `<\|box_end\|>` 0.25 / `<\|object_ref_start\|>` 0.50 | 248049 / 248050 / 248047 |
| ridgefill_alpha dial | offset 61 | `<\|box_start\|>` 0 / `<\|box_end\|>` 0.5 / `<\|object_ref_start\|>` 1.0 | 248049 / 248050 / 248047 |
| ridgefill_tail dial | offset 62 | `<\|box_start\|>` 1024 / `<\|box_end\|>` 2048 / `<\|object_ref_start\|>` 2560 / `<\|object_ref_end\|>` 3072 | 248049 / 248050 / 248047 / 248048 |
| end | offset 63 | `<\|endoftext\|>` (raw bytes `3c7c656e646f66746578747c3e`) | 248044 |

Distinct token strings used by the marker (7): `<|quad_start|>`, `<|quad_end|>`, `<|box_start|>`, `<|box_end|>`, `<|object_ref_start|>`, `<|object_ref_end|>`, `<|endoftext|>`. The spec's `token_ids` map carries exactly these 7 with the ids above.

### 2.2 Tokenizer verification (offline, `tokenizers` over `tokenizer.json`)

Tool: `/var/home/dylan/projects/research/kva/.venv/bin/python`, `tokenizers.Tokenizer.from_file('.../tokenizer.json')`, `add_special_tokens=False` (both with and without the special-tokens parse flag were tried — identical ids).

**Every marker piece encodes to exactly ONE token; all ids match the spec exactly:**

| piece | encoded id | spec id | one token? |
|---|---|---|---|
| `<|quad_start|>` | 248051 | 248051 | YES |
| `<|quad_end|>` | 248052 | 248052 | YES |
| `<|box_start|>` | 248049 | 248049 | YES |
| `<|box_end|>` | 248050 | 248050 | YES |
| `<|object_ref_start|>` | 248047 | 248047 | YES |
| `<|object_ref_end|>` | 248048 | 248048 | YES |
| `...` (raw bytes `3c7c656e646f66746578747c3e`) | 248044 | 248044 | YES |

**Full marker render check (default dials: share 0.25 → `<|box_end|>`, alpha 1.0 → `<|object_ref_start|>`, tail 2048 → `<|box_end|>`):** the concatenated 64-token marker text encodes to exactly 64 tokens, and the id sequence equals the expected `[quad pattern (30×248051, 30×248052), 248050, 248047, 248050, 248044]`. Spec `length` = 64. ✓

**EOS ids:**
- The marker end token `...` = id **248044**, exactly one token. ✓
- The container's configured eos: `config.json` → `text_config.eos_token_id = 248044` — **same id as the marker end token**, so the PLE boundary (`qwen4exp.eos_token_id` metadata, qwen4exp_fp8.cpp:962) is the marker's last token for this container. ✓ (`text_config.bos_token_id` is also 248044.)
- `generation_config.json` → `eos_token_id = [248046, 248044]` — 248044 is among the generation stop ids; 248046 is `<|im_end|>`.
- `tokenizer_config.json` → `eos_token = "<|im_end|>"` (id 248046) and `add_bos_token: False` (confirming REFUTATION-2 §1's "no BOS added"). **Note for F.1 declare check:** the plugin's declare-time eos check must compare against the container's `eos_token_id` *metadata* (config.json `text_config.eos_token_id` = 248044), NOT `tokenizer_config.json`'s `eos_token` (`<|im_end|>`, 248046) — the two notions of "eos" differ in this container; the PLE boundary is the former.
- Vocab cross-check (REFUTATION-2 §1): tokenizer vocab = **248,077** entries, max id 248,076, **33** added tokens (`added_tokens_decoder`) — matches "248,077 entries (248,044 base + 33 added)"; ids ≥ 248,077 have no text form in the tokenizer. ✓

**Served `/tokenize` check:** NOT run here (no GPU session in this sandbox) — the served endpoint check (special-token parsing of the marker text, no BOS) remains for a GPU session, as the task notes.

### 2.3 Bonus cross-checks made while verifying §3's claims (all CONFIRMED, exact lines)

- `libr4d/r4d_rows.cpp:1756-1760` — the `rope` schema: "The multi-component modes (`mrope`, `imrope`, `axial`) take [C, M] component-major positions and `sections`; with [M] positions every component is that one, so mrope and imrope over a text batch are neox exactly." — REFUTATION-2 §3(b)'s "three equal planes are the same values" check holds: three equal planes ≡ the one-component broadcast, i.e. neox exactly. (This container sets `mrope_interleaved = true`, so the mode is `imrope` — covered by the same schema sentence.)
- `config.json` → `text_config.rope_parameters.mrope_section = [11, 11, 10]` — confirms REFUTATION-2 §3's model-metadata claim behind `g.rope_mc = true` (rad_arch.h:741-751).
- `arch/common/rad_block_ple.h:453-541` — the PLE step structure cited by §3(e): `op_ids` (n-gram ids, ahead-token rows included, :463-485), `op_gate` producing `gv`/`gvn` (:516-519), `op_conv` (ple_conv) consuming `gvn` with `gv` as residual (:521-536), `op_add` (:541). Header comment :15: `out = ple_conv(gvn, resid = gv, ...)`. The §3(e) mechanism "zero marker rows of `gv` before `ple_conv`" matches this structure: `gv` is the residual operand of ple_conv. ✓

---

## 3. Report

- **Commands run:** `git log -1` in the engine repo (commit `140987fa7b0d9a1ec60e19809a70988f711997fe` = `140987f`); `awk` line-range pastes of every listed location; `grep -n` content searches for the two drifted citations (`inv` in the attention kernel, `rope` in the QSA work kernel) and for `CP_BT`, `eos_token_id`, `b_gv`; offline tokenizer runs with `/var/home/dylan/projects/research/kva/.venv/bin/python` + `tokenizers` (single-piece encodings, full 64-token marker render, vocab size, added-token count) and `json` reads of `config.json` / `tokenizer_config.json` / `generation_config.json`.
- **Files changed:** only `notes/stagef-verify.md` (this report). No tracked file in either repo was touched.
- **Not done (and why):** the served `/tokenize` endpoint check (needs a live GPU session — flagged above for the GPU session); the §1 embedding-side claim that ids 248,077–248,319 "exist in the embedding but have no text form" (needs the model weights; not in the task's listed locations).
- **Conclusion for step F.1:** every engine line REFUTATION-2 §3 (and the §1/§11.1 lines listed for this audit) relies on says what the design claims, at the cited lines or within a few lines of them (two minor drifts, content found and pasted). The marker spec's 7 tokens are each exactly one token in the container tokenizer with the spec's ids; the end token `...` (248044) equals the container's configured `eos_token_id`. **No DIFFERENT, no NOT FOUND — nothing in the audit blocks starting Stage F code.**