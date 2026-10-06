/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* adapter_core_test.cpp -- the core serves a model it was not written for (notes/adapter-split-spec.md §4).
 *
 * A toy adapter, whole in this file: four dense attention layers, a plain 64-wide residual (wide =
 * n_embd), no recurrent state, no MoE, tile 1, every block taking a pre-normed input. Its "model" is
 * three buffers and no ops; every step hook only counts its calls. The file includes the core
 * (arch/ridgefill_core.h) and the shared fakes, and NOTHING of qwen4exp or of radiance's arch sources: its
 * target has no ${RADIANCE_SRC}/arch on the include path, so the build itself is the proof that no
 * core header reaches for the in-tree architecture (tests/core_purity.cmake is the grep half).
 *
 * Together with arch_static_test (qwen4exp through the same core), this is the core serving two
 * adapters that share no block type -- the second deliberately trivial. What the toy pins: the
 * planner's capability gates (no stream, no straddle), the projector folder's match and refusals
 * against a second model's facts, the tail facts, and that a dense declare drops the recurrent and
 * MoE capabilities OUT (no correction, no drop, no probes) rather than declaring them wrong.
 */
#if defined(RAD_ARCH_NO_EXPORTS) || defined(QWEN4EXP_ADAPTER_H)
#error "adapter_core_test must not see the qwen4exp plugin"
#endif

#include "rad_test.h"
#include "rad_fake.h"

#define RIDGEFILL_RADIANCE_VERSION "0.0.0-test"
#include "ridgefill_core.h"
#include "folder_fixture.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace ridgefill;

/* ==================================================================== the toy adapter */
namespace {

constexpr int64_t kLayers = 4, kEmbd = 64, kVocab = 3, kToySplit = 2;

/* The toy model: its stream, block input and logits; no ops at all. */
struct ToyModel { rad_buf h = 0, x = 0, logits = 0; };
ToyModel g_toy[MAX_RANKS];

/* How often the core called each hook in the last step (the toy's blocks issue nothing). */
std::map<std::string, int> g_calls;

int toy_model_declare(RadBuilder* b, const RadBuildCtx* ctx) {
    ToyModel& m = g_toy[ctx->rank];
    m.h = decl_b(b, "toy_h", RAD_BF16, {ctx->max_tok, kEmbd});
    m.x = decl_b(b, "toy_x", RAD_BF16, {ctx->max_tok, kEmbd});
    m.logits = decl_b(b, "toy_logits", RAD_F32, {ctx->max_tok, kVocab});
    return m.h && m.x && m.logits ? RAD_OK : RAD_E_INVAL;
}

void toy_conn(RadCtx*, int64_t, bool, bool, int64_t, int64_t, int64_t) { ++g_calls["conn"]; }
void toy_late_block(RadCtx*, const RidgeFill&, int64_t, const RadBatch*, const Pass&, StateDump*, Path path,
                    int64_t, int64_t) { ++g_calls[std::string("late_block.") + kPathNames[path]]; }
void toy_ffn(RadCtx*, const RidgeFill&, int64_t, const RadBatch*, int64_t, int64_t, rad_op drop, rad_buf) {
    ++g_calls[drop ? "ffn.drop" : "ffn"];
}
void toy_prologue(RadCtx*, const RadBatch*) { ++g_calls["prologue"]; }
void toy_stock_layer(RadCtx*, const RidgeFill&, int64_t, const RadBatch*, bool probes) {
    ++g_calls[probes ? "stock_layer.probes" : "stock_layer"];
}
void toy_epilogue(RadCtx*, const RadBatch*) { ++g_calls["epilogue"]; }
void toy_stock_step(RadCtx*, const RadBatch*) { ++g_calls["stock_step"]; }

const std::vector<const char*> kNoStraddle(kLayers, "the toy's attention has no per-row sparse form");

RidgeFillAdapter toy_adapter(int rank) {
    const ToyModel& m = g_toy[rank];
    RidgeFillAdapter a;
    a.log_name = "toy_ridgefill";
    a.match_name = "toy";
    a.shadow_so = "toy.so";
    a.n_layer = kLayers;
    a.n_embd = kEmbd;
    a.n_vocab_all = kVocab;
    a.n_vocab = kVocab;
    a.wide = kEmbd;          /* a plain residual: the projector reads the block width */
    a.tile = 1;              /* no recurrent block: every bulk end is valid */
    a.split_lo = 0;          /* nothing enters the stream late */
    a.probe_depth = 0;       /* no routed FFN to stream: top_k 0 */
    a.min_tail = 256;
    a.default_tail = 1024;
    a.full.assign(kLayers, 1);
    a.ext_in.assign(kLayers, 1);
    a.calibrated.assign(kLayers, 0);
    a.routed.assign(kLayers, 0);
    a.straddle_lack = kNoStraddle;
    a.qsa_exact_to.assign(kLayers, 0);
    a.buf_stream = m.h;
    a.buf_x = m.x;
    a.buf_logits = m.logits;
    a.conn = &toy_conn;
    a.late_block = &toy_late_block;
    a.ffn = &toy_ffn;
    a.prologue = &toy_prologue;
    a.stock_layer = &toy_stock_layer;
    a.epilogue = &toy_epilogue;
    a.stock_step = &toy_stock_step;
    return a;                /* no declare_model / declare_codes / decl_state_ops / captures: absent */
}

/* The toy plugin's declare: its own graph, then the core over its facts. */
int toy_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    RAD_ARCH_TRY(toy_model_declare(b, ctx));
    RidgeFill& k = g_ridgefill[ctx->rank];
    k = RidgeFill{};
    k.ad = toy_adapter(ctx->rank);
    return core_declare(b, meta, ctx, k);
}

/* ==================================================================== fixtures */

const char* kToyKeys[] = { "toy_heads" };
const char* kToyVals[] = { "4" };

RadModelMeta toy_meta() {
    RadModelMeta m{};
    m.arch_id = "toy"; m.name = "toy-4"; m.quant = "";
    m.n_layers = (int)kLayers; m.n_embd = (int)kEmbd; m.n_vocab = (int)kVocab;
    m.n_kv = 1; m.kv_key = kToyKeys; m.kv_val = kToyVals;
    return m;
}

RadBuildCtx toy_ctx() {
    RadBuildCtx c{};
    c.rank = 0; c.world_size = 1;
    c.max_tok = 2048; c.max_seqs = 8; c.max_ctx = 49152; c.max_spec = 0;
    c.scope = "";
    return c;
}

/* The toy's projector folder: maps proj.L.weight [64, 64] bf16 + .bias for layers 2 and 3, a row
 * table, no correction (no recurrent layer). `edit` changes it before the plugin sees it. */
void hold_toy(const std::string& adapter = "toy", const std::string& meta_val = "4",
              const std::string& vocab = kTinyVocab) {
    reset_projector();
    g_test_folder = Folder{};
    g_test_folder.place = { "/test/toy-projector", "the test", tiny_container() };
    const std::string text = R"({"format": 1, "adapter": ")" + adapter + R"(", "split": 2,
        "model": {"arch_id": "toy", "name": "toy-4", "meta": {"toy_heads": ")" + meta_val + R"("},
                  "vocab_sha256": ")" + vocab + R"("},
        "files": {"proj.L2.safetensors": "unused by the test folder"}})";
    json_parse(text.data(), text.size(), &g_test_folder.manifest);
    for (int64_t l = kToySplit; l < kLayers; ++l) {
        add_tensor("proj." + std::to_string(l) + ".weight", RAD_BF16, {kEmbd, kEmbd});
        add_tensor("proj." + std::to_string(l) + ".bias", RAD_BF16, {kEmbd});
    }
    add_tensor("score", RAD_F32, {kVocab});
    g_folder_for_test = &g_test_folder;
}

struct Declared { RadBuilder b; int st = RAD_OK; std::string log; };

Declared declare_toy(const char* mode) {
    Declared d;
    Env env({{"RADIANCE_RIDGEFILL", mode}});
    const RadModelMeta meta = toy_meta();
    const RadBuildCtx ctx = toy_ctx();
    d.log = stderr_of([&] { d.st = toy_declare(&d.b, &meta, &ctx); });
    return d;
}

/* A step of sequences with query lengths `q`, the first `D` decoding, the last followed by `ahead`
 * prompt tokens (the toy declares no KV group, so the batch carries none). */
RadBatch toy_step(const std::vector<int32_t>& q, int64_t D, int64_t ahead) {
    static int32_t ids[8192], pos[8192], cu[17], qlen[16], ctxl[16], acc[16];
    RadBatch b{};
    int64_t T = 0, DT = 0;
    int32_t qd = 0, qp = 0, qmax = 0;
    for (size_t i = 0; i < q.size(); ++i) {
        for (int32_t j = 0; j < q[i]; ++j) { ids[T + j] = 1; pos[T + j] = 4096 + j; }
        T += q[i];
        cu[i + 1] = (int32_t)T;
        qlen[i] = q[i]; ctxl[i] = 4096; acc[i] = 0;
        ((int64_t)i < D ? qd : qp) = std::max((int64_t)i < D ? qd : qp, q[i]);
        qmax = std::max(qmax, q[i]);
        if ((int64_t)i < D) DT += q[i];
    }
    const int64_t n = (int64_t)q.size();
    b.phase = D == 0 ? RAD_PHASE_PREFILL : D == n ? RAD_PHASE_DECODE : RAD_PHASE_MIXED;
    b.n_tok = T; b.n_seq = n; b.n_ahead = ahead;
    b.token_ids = ids; b.positions = pos; b.cu_seqlens = cu;
    b.q_lens = qlen; b.ctx_lens = ctxl; b.num_accepted = acc;
    b.max_q_len = qmax; b.max_ctx_len = 4096;
    b.n_seq_decode = D; b.n_tok_decode = DT;
    b.max_q_len_decode = qd; b.max_q_len_prefill = qp;
    return b;
}

std::vector<std::string> ops_named(const RadBuilder& b) {
    std::vector<std::string> v;
    for (const RecOp& o : b.ops) v.push_back(o.op);
    return v;
}

/* The planner's minimum bulk rows is a host-placement cost gate; these cases are about capabilities. */
[[maybe_unused]] const int g_min_bulk_off = setenv("RADIANCE_RIDGEFILL_MIN_BULK_ROWS", "0", 1);

}  /* namespace */

/* ==================================================================== the cases */

/* 1. THE PLANNER THROUGH A DENSE MODEL'S FACTS. With no routed layer nothing streams, and with no
 * per-row attention nothing straddles: a whole-bulk chunk in speed is still lean (the cache writers
 * only), and every other approximate shape falls to the model's own step -- except plumb, whose
 * masked path is the exact oracle and needs neither. */
TEST(the_planner_serves_a_dense_toy) {
    struct Want { const char* mode; std::vector<int32_t> q; int64_t D, ahead; int path; int64_t b; };
    const Want rows[] = {
        {"speed",   {2048},       0, 4096, PATH_LEAN,   2048},   /* whole chunk >= the 1,024 tail ahead */
        {"speed",   {2048},       0,  512, PATH_STOCK,  0},      /* straddles: no per-row attention */
        {"quality", {2048},       0, 4096, PATH_STOCK,  0},      /* masked needs the stream */
        {"plumb",   {2048},       0, 4096, PATH_MASKED, 2048},   /* the oracle path needs nothing */
        {"plumb",   {2048},       0,  512, PATH_MASKED, 1536},   /* tile 1: b = 2048 - (1024 - 512) */
        {"speed",   {1, 1, 2046}, 2, 4096, PATH_STOCK,  0},      /* decoders need stream and straddle */
    };
    for (const Want& w : rows) {
        hold_toy();
        const Declared d = declare_toy(w.mode);
        REQUIRE_EQ(d.st, RAD_OK);
        const RidgeFill& k = g_ridgefill[0];
        CHECK(k.have_proj);
        CHECK(!can_stream(k));
        CHECK(!k.straddle_layers);
        const RadBatch b = toy_step(w.q, w.D, w.ahead);
        const Pass p = derive(k, &b);
        CHECK_EQ(p.path, w.path);
        CHECK_EQ(p.b, w.b);
        CHECK(!p.stream);
    }
}

/* 2. THE FOLDER AGAINST THE TOY'S FACTS: its own folder is taken at split 2; another adapter's, other
 * metadata, another tokenizer, a misshapen map and a missing map are each refused by name, and a
 * refused folder declares nothing past the toy's graph. */
TEST(the_toy_projector_folder_is_matched_and_refused) {
    hold_toy();
    Declared ok = declare_toy("speed");
    REQUIRE_EQ(ok.st, RAD_OK);
    CHECK(g_ridgefill[0].have_proj);
    CHECK_EQ(g_ridgefill[0].split, kToySplit);
    CHECK(has(ok.log, "radiance: toy_ridgefill: RidgeFill: projector /test/toy-projector"));

    struct Bad { const char* why; std::function<void()> hold; const char* says; };
    const Bad bad[] = {
        {"another adapter", [] { hold_toy("qwen4exp"); }, "the projector is for adapter 'qwen4exp'"},
        {"other metadata",  [] { hold_toy("toy", "5"); }, "metadata 'toy_heads' is '4' in this model and '5'"},
        {"another tokenizer", [] { hold_toy("toy", "4", std::string(64, '0')); }, "the tokenizer differs"},
        {"a misshapen map", [] { hold_toy(); add_tensor("proj.2.weight", RAD_BF16, {kEmbd, kEmbd / 2}); },
         "proj.2.weight [64, 64] / .bias bf16 is missing or misshapen"},
        {"a hole in the maps", [] { hold_toy(); g_test_folder.tensors.erase("proj.3.weight"); },
         "proj.3.weight [64, 64] / .bias bf16 is missing or misshapen"},
    };
    for (const Bad& x : bad) {
        x.hold();
        const Declared d = declare_toy("speed");
        CHECK_EQ(d.st, RAD_OK);   /* refused = served stock, not a failed start */
        CHECK(!g_ridgefill[0].have_proj);
        CHECK(has(d.log, "REFUSED"));
        if (!has(d.log, x.says)) std::fprintf(stderr, "  [%s] log was: %s\n", x.why, d.log.c_str());
        CHECK(has(d.log, x.says));
        CHECK(d.b.ops.empty());
    }
}

/* 3. THE TAIL FACTS ARE THE ADAPTER'S: the toy's default tail is 1,024 and its floor 256, so 384 --
 * refused under qwen4exp's 512 -- serves here, and 128 is refused naming the toy's floor. */
TEST(the_toy_config_takes_its_tail_from_the_adapter) {
    hold_toy();
    Declared d = declare_toy("speed");
    REQUIRE_EQ(d.st, RAD_OK);
    CHECK_EQ(g_ridgefill[0].cfg.tail, 1024);
    {
        Env tail({{"RADIANCE_RIDGEFILL_TAIL", "384"}});
        hold_toy();
        d = declare_toy("speed");
        CHECK_EQ(d.st, RAD_OK);
        CHECK_EQ(g_ridgefill[0].cfg.tail, 384);
    }
    {
        Env tail({{"RADIANCE_RIDGEFILL_TAIL", "128"}});
        hold_toy();
        d = declare_toy("speed");
        CHECK_EQ(d.st, RAD_E_INVAL);
        CHECK(has(d.log, "radiance: toy_ridgefill: ridgefill.tail is 128 tokens; the shortest exact tail this "
                         "method was measured at is 256"));
    }
}

/* 4. A DENSE DECLARE ADDS ONLY THE GENERIC OPS: after the toy's (empty) graph, speed declares one
 * projector GEMM per late layer (N 64, K 64), the ring's copy, the h_S copy, the select and the mask --
 * no correction, no decay sums, no drop, no probes, no quantiser, no hazard (no recurrent layer to
 * carry its slot). A lean step issues exactly the projector GEMMs and the ring around them, and calls
 * the late block's LEAN hook once a late layer. Plumb declares the mask alone and copies nothing. */
TEST(the_toy_declare_adds_only_generic_ops) {
    hold_toy();
    Declared d = declare_toy("speed");
    REQUIRE_EQ(d.st, RAD_OK);
    const RidgeFill& k = g_ridgefill[0];
    const std::vector<std::string> want = {"ridgefill_gemm_nt_bias", "ridgefill_gemm_nt_bias", "cast", "cast",
                                           "ridgefill_select", "ridgefill_mask"};
    CHECK(ops_named(d.b) == want);
    for (int64_t l = kToySplit; l < kLayers; ++l) {
        const RecOp& g = d.b.ops[(size_t)k.op_proj[(size_t)l] - 1];
        CHECK_EQ(g.op, std::string("ridgefill_gemm_nt_bias"));
        for (const RecParam& p : g.p) {
            if (p.key == "N") CHECK_EQ(p.ival, kEmbd);
            if (p.key == "K") CHECK_EQ(p.ival, kEmbd);
        }
    }
    CHECK_EQ(k.op_drop, (rad_op)0);
    CHECK_EQ(k.b_zeros, (rad_buf)0);
    CHECK_EQ(k.quant.op, (rad_op)0);
    CHECK_EQ(k.op_hazard, (rad_op)0);
    CHECK(k.kv_applied == 0 && k.kv_rho == 0);
    CHECK(d.b.kv_groups.empty());

    /* a lean step */
    const RadBatch b = toy_step({2048}, 0, 4096);
    g_calls.clear();
    RadCtx c;
    c.batch = &b;
    g_ctx = &c;
    core_step(&c, &b);
    g_ctx = nullptr;
    CHECK(c.step_fail.empty());
    CHECK_EQ(c.device_calls, 0);
    std::map<rad_op, int> issued;
    for (const RecIssue& i : c.issues)
        if (i.op < kLane) ++issued[i.op];
    CHECK_EQ(issued[k.op_proj[2]], 1);
    CHECK_EQ(issued[k.op_proj[3]], 1);
    CHECK_EQ(issued[k.op_ring], 2);       /* layer 2's map before the layers, layer 3's after layer 2's GEMM */
    CHECK_EQ(issued.size(), (size_t)3);   /* nothing else: no mask, no select, no state op */
    CHECK_EQ(g_calls["prologue"], 1);
    CHECK_EQ(g_calls["stock_layer"], (int)kToySplit);
    CHECK_EQ(g_calls["late_block.lean"], (int)(kLayers - kToySplit));
    CHECK_EQ(g_calls["epilogue"], 1);
    CHECK_EQ(g_calls["stock_step"], 0);
    CHECK_EQ(g_calls["conn"], 0);         /* the lean path reads and writes no connection */

    /* a stock step: the toy's own */
    const RadBatch s = toy_step({2048}, 0, 512);
    g_calls.clear();
    RadCtx c2;
    g_ctx = &c2;
    core_step(&c2, &s);
    g_ctx = nullptr;
    CHECK(c2.issues.empty());
    CHECK_EQ(g_calls["stock_step"], 1);

    /* plumb */
    hold_toy();
    g_mem.copies.clear();
    d = declare_toy("plumb");
    REQUIRE_EQ(d.st, RAD_OK);
    CHECK(ops_named(d.b) == std::vector<std::string>{"ridgefill_mask"});
    CHECK(g_mem.copies.empty());
    CHECK(has(d.log, "rank 0 holds nothing"));
}

/* 5. THE CORE COMPILES WITHOUT THE IN-TREE ARCHITECTURE: this TU is the proof (see the header and
 * the #error above); the case pins that the toy's hooks, and only they, served the steps above. */
TEST(no_core_header_names_the_in_tree_architecture) {
    hold_toy();
    REQUIRE_EQ(declare_toy("plumb").st, RAD_OK);
    const RadBatch b = toy_step({2048}, 0, 4096);
    g_calls.clear();
    RadCtx c;
    g_ctx = &c;
    core_step(&c, &b);
    g_ctx = nullptr;
    /* plumb's masked layer: the toy's conn read/write twice a late layer, its block, its ffn (no drop) */
    CHECK_EQ(g_calls["conn"], 4 * (int)(kLayers - kToySplit));
    CHECK_EQ(g_calls["late_block.masked"], (int)(kLayers - kToySplit));
    CHECK_EQ(g_calls["ffn"], (int)(kLayers - kToySplit));
    CHECK_EQ(g_calls["ffn.drop"], 0);
    CHECK_EQ(std::string(g_ridgefill[0].ad.log_name), std::string("toy_ridgefill"));
}

RAD_TEST_MAIN()
