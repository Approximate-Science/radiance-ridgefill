/* arch_static_test.cpp -- the KVA arch plugin's declared graph and issued sequence, asserted with no
 * GPU, no model file and no core (R11, arch half). Modelled on radiance's tests/arch_test.cpp: a
 * recording fake builder and a recording fake RadCtx, both included plugins called directly.
 *
 * THE ORACLE IS THE IN-TREE PLUGIN. `off`, and any container without kva.* weights, must declare
 * exactly what qwen4exp_fp8 declares and issue exactly what it issues (R6, R7 at the graph level);
 * the comparisons below are element by element against qwen4exp_fp8::declare / ::step run on a
 * second builder, never against numbers typed here.
 *
 * The fake is this file's own rather than core/build's, as radiance's is: a plugin that only works
 * against one builder has an undeclared dependency.
 */
#define RAD_ARCH_NO_EXPORTS 1

#include "rad_test.h"

#include "rad_builder.h"
#include "rad_device.h"
#include "rad_runtime.h"

#include <unistd.h>

#include <cstdarg>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <set>
#include <string>
#include <vector>

/* ==================================================================== the recording builder */
namespace {
struct RecParam { std::string key; int kind = 0; long long ival = 0, ihi = 0; double dval = 0; std::string sval; };
struct RecOp {
    std::string op;
    std::vector<RecParam> p;
    std::vector<rad_weight> w;
    std::vector<rad_buf> reads, writes;
};
struct RecIssue { rad_op op = 0; std::vector<RadOperand> opd; int64_t n = 0; };
}  /* namespace */

struct RadBuilder {
    std::vector<std::pair<std::string, RadWeightDecl>> weights;
    std::vector<std::pair<std::string, RadBufDecl>>    bufs;
    std::vector<RecOp>                                 ops;
    std::vector<std::string>                           maps, notes, kv_groups;
    std::set<std::string>                              refuse;   /* ops no kernel serves */
    std::set<rad_buf>                                  concurrent;
    std::vector<std::pair<int, rad_kvgroup>>           binds;
    /* What rad_weight_encoding answers: for a kva.* key, the source named by it or by it plus a
     * '.'-suffix ("kva.proj.4" holds kva.proj.4.weight and .bias, "kva.rowsel.score" does not hold
     * kva.rowsel.score_none); for any other key, the first source CONTAINING it, as radiance's own
     * arch_test matches ("ffn_gate_up_exps"). */
    std::vector<std::pair<std::string, RadEncoding>>   encs;
};

struct RadCtx {
    const RadBatch*       batch = nullptr;
    int                   rank = 0, world = 1;
    std::vector<RecIssue> issues;
    std::string           step_fail;
    int                   device_calls = 0;   /* a step that moves bytes itself is not stock */
    /* Each device read: its source and how many issues preceded it -- which is what pins a
     * capture to the point in the step it claims to read. */
    std::vector<std::pair<const void*, size_t>> reads;
};

static RadCtx* g_ctx = nullptr;   /* for the device API, which carries no context */

extern "C" {
rad_weight rad_decl_weight(RadBuilder* b, const char* name, const RadWeightDecl* d) {
    b->weights.push_back({name, *d});
    return (rad_weight)b->weights.size();
}
int rad_weight_encoding(RadBuilder* b, const char* source, RadEncoding* out, int64_t*, uint32_t*) {
    for (const auto& [key, e] : b->encs) {
        const size_t n = key.size();
        const bool hit = !source ? false
                       : key.rfind("kva.", 0) == 0
                           ? !std::strncmp(source, key.c_str(), n) && (!source[n] || source[n] == '.')
                           : std::strstr(source, key.c_str()) != nullptr;
        if (hit) { *out = e; return RAD_OK; }
    }
    return RAD_E_NOTFOUND;
}
rad_buf rad_decl_buffer(RadBuilder* b, const char* name, const RadBufDecl* d) {
    b->bufs.push_back({name, *d});
    return (rad_buf)b->bufs.size();
}
rad_op rad_decl_op(RadBuilder* b, const char* op, const RadParam* p, int n_p,
                   const rad_weight* w, int n_w) {
    RecOp r;
    r.op = op;
    for (int i = 0; i < n_p; ++i)
        r.p.push_back({p[i].key ? p[i].key : "", p[i].kind, p[i].ival, p[i].ihi, p[i].dval,
                       p[i].sval ? p[i].sval : ""});
    for (int i = 0; i < n_w; ++i) r.w.push_back(w[i]);
    b->ops.push_back(r);
    return b->refuse.count(r.op) ? RAD_NULL_HANDLE : (rad_op)b->ops.size();
}
int rad_op_reads(RadBuilder* b, rad_op h, const rad_buf* x, int n) {
    if (!h) return RAD_E_INVAL;
    b->ops[h - 1].reads.insert(b->ops[h - 1].reads.end(), x, x + n);
    return RAD_OK;
}
int rad_op_writes(RadBuilder* b, rad_op h, const rad_buf* x, int n) {
    if (!h) return RAD_E_INVAL;
    b->ops[h - 1].writes.insert(b->ops[h - 1].writes.end(), x, x + n);
    return RAD_OK;
}
rad_kvgroup rad_decl_kv_group(RadBuilder* b, const char* name, const RadKVGroupDecl*) {
    b->kv_groups.push_back(name);
    return (rad_kvgroup)b->kv_groups.size();
}
int64_t rad_kv_block_size(RadBuilder*, rad_kvgroup) { return 4; }
int rad_bind_layer_kv(RadBuilder* b, int layer, rad_kvgroup g) { b->binds.push_back({layer, g}); return RAD_OK; }
int rad_decl_name_map(RadBuilder* b, const RadNameMap* m) { b->maps.push_back(m->declared); return RAD_OK; }
void rad_note(RadBuilder* b, const char* fmt, ...) {
    char buf[1024];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    b->notes.push_back(buf);
}
int rad_declare_logits(RadBuilder*, rad_buf) { return RAD_OK; }
int rad_buf_concurrent(RadBuilder* b, rad_buf h) { b->concurrent.insert(h); return RAD_OK; }
int rad_weight_shard_span(RadBuilder*, rad_weight, int64_t, int64_t) { return RAD_OK; }
int rad_declare_drafter(RadBuilder*, const RadDrafterDecl*) { return RAD_OK; }
int rad_declare_encoder(RadBuilder*, const RadEncoderDecl*) { return RAD_OK; }

long long rad_meta_geti(const RadModelMeta* m, const char* key, long long dflt) {
    for (int i = 0; i < m->n_kv; ++i) if (!std::strcmp(m->kv_key[i], key)) return atoll(m->kv_val[i]);
    return dflt;
}
double rad_meta_getf(const RadModelMeta* m, const char* key, double dflt) {
    for (int i = 0; i < m->n_kv; ++i) if (!std::strcmp(m->kv_key[i], key)) return atof(m->kv_val[i]);
    return dflt;
}
const char* rad_meta_gets(const RadModelMeta* m, const char* key, const char* dflt) {
    for (int i = 0; i < m->n_kv; ++i) if (!std::strcmp(m->kv_key[i], key)) return m->kv_val[i];
    return dflt;
}

int rad_issue(RadCtx* c, rad_op op, const RadOperand* opd, int n_opd, int64_t n) {
    c->issues.push_back({op, std::vector<RadOperand>(opd, opd + n_opd), n});
    return RAD_OK;
}
int rad_step_fail(RadCtx* c, const char* what) { c->step_fail = what ? what : ""; return RAD_E_SHAPE; }
int rad_rank(RadCtx* c) { return c->rank; }
RadStream rad_stream(RadCtx*) { return nullptr; }
int rad_route_report(RadCtx*, int, const RadRouting*) { return RAD_OK; }
int32_t* rad_route_counts(RadCtx*, int, int64_t) { return nullptr; }
void* rad_buf_ptr(RadCtx*, rad_buf b) { return (void*)(uintptr_t)(b * 64 + 16); }

void* rad_dev_alloc(int64_t, int) { ++g_ctx->device_calls; return nullptr; }
void  rad_dev_free(void*, int) { ++g_ctx->device_calls; }
/* The debug paths' device reads. rad_buf_ptr hands out small fake addresses (below), which read as
 * zeros; a batch field is real host memory here and is copied, so a capture sees the batch's own
 * ids and positions. */
int   rad_stream_sync(RadStream) { ++g_ctx->device_calls; return RAD_OK; }
int   rad_memcpy_async(void* dst, const void* src, int64_t n, RadStream) {
    ++g_ctx->device_calls;
    g_ctx->reads.push_back({src, g_ctx->issues.size()});
    if ((uintptr_t)src < (1u << 20)) std::memset(dst, 0, (size_t)n);
    else                             std::memcpy(dst, src, (size_t)n);
    return RAD_OK;
}
int   rad_memset_async(void*, int, int64_t, RadStream) { ++g_ctx->device_calls; return RAD_E_UNSUPPORTED; }
}  /* extern "C" */

/* ==================================================================== the plugin under test */
/* The release the guard compares against; the build sets it for the plugin (arch/CMakeLists.txt). */
#define KVA_RADIANCE_VERSION "0.0.0-test"
#include "qwen4exp_kva.cpp"

using rad::arch::bcol;
using rad::arch::brows;
using rad::arch::kv_cache;
using rad::arch::praw2;
using rad::arch::praw;
using rad::arch::brow_slice;
using rad::arch::MoeFP8;

/* ==================================================================== fixtures */
namespace {

/* Qwen3.8-Flash-Next's metadata cut to eight layers, as radiance's arch_test cuts it: full
 * attention at 3 and 7, delta net elsewhere. The projector below starts at layer 4, so the late
 * layers are three delta-net layers (4, 5, 6) and one attention layer (7). */
const char* kKeys[] = {
    "full_attention_interval", "hc_count", "hc_lowrank", "heads_per_ngram", "hidden_act",
    "indexer_budget", "indexer_compress_ratio", "indexer_head_dim", "indexer_n_heads",
    "layer_types", "linear_conv_kernel_dim", "linear_key_head_dim", "linear_num_key_heads",
    "linear_num_value_heads", "linear_value_head_dim", "make_ngram_vocab_size_divisible_by",
    "moe_intermediate_size", "mtp_num_hidden_layers", "ngram_size", "ngram_vocab_size_base",
    "num_experts", "num_experts_per_tok", "output_gate_type", "partial_rotary_factor",
    "ple_conv_kernel_size", "ple_embed_dim", "ple_layer_ids", "shared_expert_intermediate_size",
    "split_ngram_parts", "rope_parameters.partial_rotary_factor", "eos_token_id",
};
const char* kVals[] = {
    "4", "4", "320", "8", "silu",
    "2048", "4", "128", "4",
    "linear_attention linear_attention linear_attention full_attention "
    "linear_attention linear_attention linear_attention full_attention",
    "4", "128", "16",
    "48", "128", "128",
    "640", "1", "3", "20000000",
    "512", "10", "sigmoid", "0.25",
    "4", "2560", "2", "640",
    "128", "0.25", "248044",
};
constexpr int kN = (int)(sizeof(kKeys) / sizeof(kKeys[0]));
constexpr int kSplit = 4;

RadModelMeta flash_next_meta() {
    RadModelMeta m{};
    m.arch_id = "qwen4exp"; m.name = "test-q38-flashnext"; m.quant = "";
    m.n_layers = 8; m.n_embd = 2560; m.n_head = 24; m.n_head_kv = 2; m.head_dim = 256;
    m.n_ff = 640; m.n_vocab = 248320; m.n_ctx_train = 262144;
    m.rms_eps = 1e-6f; m.rope_theta = 1e7f; m.rope_scale = 1.0f;
    m.n_kv = kN; m.kv_key = kKeys; m.kv_val = kVals;
    return m;
}

/* The served deployment's bounds: 2048-token steps, MTP off, 49152 context. */
RadBuildCtx served_ctx(int rank = 0, int world = 1) {
    RadBuildCtx c{};
    c.rank = rank; c.world_size = world;
    c.max_tok = 2048; c.max_seqs = 8; c.max_ctx = 49152; c.max_spec = 0;
    c.scope = "";
    return c;
}

/* Every kva.* tensor the sidecar writes, for layers kSplit..7. */
void hold_kva(RadBuilder& b, std::initializer_list<const char*> bases) {
    for (const char* base : bases)
        for (int l = kSplit; l < 8; ++l)
            b.encs.push_back({std::string(base) + "." + std::to_string(l),
                              rad_enc_plain(std::strstr(base, ".st") ? RAD_F32 : RAD_BF16)});
}
void hold_score(RadBuilder& b, const char* name) {
    b.encs.push_back({name, rad_enc_plain(RAD_F32)});
}

/* The SERVED container's formats (data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe): four-bit rotated
 * experts, int8 trunk linears, E4M3-row connection mixes, the indexer projection left bf16. Under
 * it the connection read writes the int8 codes itself (codes_i8), the in-tree linears declare no
 * quantiser, and the fill must quantise the projected block input -- the path the engine runs. */
void served(RadBuilder& b) {
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    const RadEncoding row8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 128);
    b.encs.push_back({"ffn_gate_up_exps", w4});
    b.encs.push_back({"ffn_down_exps", w4});
    for (const char* k : {"attn_qg.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                          "ssm_inz.weight", "ssm_out.weight", "ffn_gate_up_shexp.weight",
                          "ffn_down_shexp.weight", "output.weight"})
        b.encs.push_back({k, i8});
    b.encs.push_back({"_hc_down.weight", row8});
    b.encs.push_back({"_hc_up.weight", row8});
    b.encs.push_back({"qsa_qk", rad_enc_plain(RAD_BF16)});
}

/* setenv for one case, undone on scope exit, so cases cannot leak switches into each other. */
struct Env {
    std::vector<std::string> set;
    Env(std::initializer_list<std::pair<const char*, const char*>> kv) {
        for (const auto& [k, v] : kv) { setenv(k, v, 1); set.push_back(k); }
    }
    ~Env() { for (const auto& k : set) unsetenv(k.c_str()); }
};

/* What a call wrote to stderr: the refusals are fprintf'd, as the in-tree plugin's are. */
std::string stderr_of(const std::function<void()>& fn) {
    fflush(stderr);
    FILE* tmp = tmpfile();
    const int saved = dup(2);
    dup2(fileno(tmp), 2);
    fn();
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    std::string out;
    rewind(tmp);
    char buf[4096];
    for (size_t n; (n = fread(buf, 1, sizeof buf, tmp)) > 0;) out.append(buf, n);
    fclose(tmp);
    return out;
}

bool has(const std::string& s, const char* what) { return s.find(what) != std::string::npos; }

const RadWeightDecl* weight(const RadBuilder& b, const std::string& name) {
    for (const auto& [n, d] : b.weights) if (n == name) return &d;
    return nullptr;
}

/* The declared graph is the same when every op, parameter, weight, buffer and group agrees. */
void check_same_graph(const RadBuilder& a, const RadBuilder& b) {
    REQUIRE_EQ(b.ops.size(), a.ops.size());
    for (size_t i = 0; i < a.ops.size(); ++i) {
        CHECK_EQ(b.ops[i].op, a.ops[i].op);
        REQUIRE_EQ(b.ops[i].p.size(), a.ops[i].p.size());
        for (size_t k = 0; k < a.ops[i].p.size(); ++k) {
            CHECK_EQ(b.ops[i].p[k].key, a.ops[i].p[k].key);
            CHECK_EQ(b.ops[i].p[k].ival, a.ops[i].p[k].ival);
            CHECK_EQ(b.ops[i].p[k].ihi, a.ops[i].p[k].ihi);
            CHECK_EQ(b.ops[i].p[k].sval, a.ops[i].p[k].sval);
        }
        CHECK(b.ops[i].w == a.ops[i].w);
        CHECK(b.ops[i].reads == a.ops[i].reads);
        CHECK(b.ops[i].writes == a.ops[i].writes);
    }
    REQUIRE_EQ(b.weights.size(), a.weights.size());
    for (size_t i = 0; i < a.weights.size(); ++i) CHECK_EQ(b.weights[i].first, a.weights[i].first);
    REQUIRE_EQ(b.bufs.size(), a.bufs.size());
    for (size_t i = 0; i < a.bufs.size(); ++i) CHECK_EQ(b.bufs[i].first, a.bufs[i].first);
    CHECK(b.kv_groups == a.kv_groups);
    CHECK(b.maps == a.maps);
}

}  /* namespace */

/* ==================================================================== off is the in-tree plugin */

/* No kva.* weights, no switch: the KVA declare is the in-tree declare, op for op, on both ranks of a
 * TP2 deployment, with the MTP head declared, for a bf16 container and for the served formats. */
TEST(off_declares_exactly_the_in_tree_graph) {
    RadModelMeta meta = flash_next_meta();
    for (int world : {1, 2})
        for (int rank = 0; rank < world; ++rank)
            for (int spec : {0, 3})
                for (bool fmt : {false, true}) {
                RadBuildCtx c = served_ctx(rank, world);
                c.max_spec = spec;
                RadBuilder stock, kva;
                if (fmt) { served(stock); served(kva); }
                REQUIRE_EQ(qwen4exp_fp8::declare(&stock, &meta, &c), RAD_OK);
                REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
                CHECK(stock.ops.size() > 100);
                check_same_graph(stock, kva);
            }
}

/* The container HOLDS every kva.* tensor and the mode is off: still the in-tree graph. Declaring
 * the projector anyway would place 1.26 GiB a rank and move experts between tiers. */
TEST(off_with_kva_weights_held_still_declares_the_in_tree_graph) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    Env env({{"RADIANCE_KVA", "off"}});
    RadBuilder stock, kva;
    hold_kva(kva, {"kva.proj", "kva.st"});
    hold_score(kva, "kva.rowsel.score");
    REQUIRE_EQ(qwen4exp_fp8::declare(&stock, &meta, &c), RAD_OK);
    REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
    check_same_graph(stock, kva);
}

/* A batch's KV entries for every group the builder declared, with distinct fake device pointers:
 * without them the blocks see no group and skip work (rad_qsa.h returns early), and the comparison
 * would be over a shorter sequence than a real step issues. */
struct Batch {
    std::vector<RadKVGroupBatch> kv;
    RadBatch b{};
};

Batch make_batch(const RadBuilder& bld, bool prefill) {
    static int32_t ids[64], pos[64], cu[3] = {0, 32, 64}, out[2] = {31, 63};
    static int32_t table[256], used[8], slot[64], state[8], qlen[8], ctxl[8], acc[8];
    Batch x;
    for (size_t g = 0; g < bld.kv_groups.size(); ++g) {
        RadKVGroupBatch k{};
        k.group = (rad_kvgroup)(g + 1);
        k.slot_mapping = slot; k.block_table = table; k.block_table_pitch = 32;
        k.seqused = used; k.state_index = state; k.state_index_pitch = 1;
        k.block_size = 4; k.max_blocks = 32;
        x.kv.push_back(k);
    }
    RadBatch& b = x.b;
    b.phase = prefill ? RAD_PHASE_PREFILL : RAD_PHASE_DECODE;
    b.n_tok = prefill ? 64 : 2;
    b.n_seq = 2;
    b.token_ids = ids; b.positions = pos; b.cu_seqlens = cu;
    b.n_out = 2; b.out_ids = prefill ? out : cu;
    b.q_lens = qlen; b.ctx_lens = ctxl; b.num_accepted = acc;
    b.max_q_len = prefill ? 32 : 1;
    b.max_ctx_len = prefill ? 32 : 100;
    b.n_seq_decode = prefill ? 0 : 2;
    b.n_tok_decode = prefill ? 0 : 2;
    b.n_kv_groups = (int)x.kv.size();
    b.kv = x.kv.data();
    return x;
}

std::vector<RecIssue> issues_of(void (*step)(RadCtx*, const RadBatch*), const RadBatch& b,
                                int* device_calls) {
    RadCtx c;
    c.batch = &b;
    g_ctx = &c;
    step(&c, &b);
    g_ctx = nullptr;
    *device_calls = c.device_calls;
    return c.issues;
}

bool same_operand(const RadOperand& a, const RadOperand& b) {
    return a.kind == b.kind && a.dtype == b.dtype && a.handle == b.handle && a.raw == b.raw &&
           a.offset == b.offset && a.rows == b.rows && a.cols == b.cols;
}

bool same_issue(const RecIssue& a, const RecIssue& b) {
    bool same = a.op == b.op && a.n == b.n && a.opd.size() == b.opd.size();
    for (size_t k = 0; same && k < a.opd.size(); ++k) same = same_operand(a.opd[k], b.opd[k]);
    return same;
}

/* How many positions of two issue sequences differ (a length difference counts every extra one). */
int differ(const std::vector<RecIssue>& a, const std::vector<RecIssue>& b) {
    int n = (int)(a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) n += !same_issue(a[i], b[i]);
    return n;
}

/* `off` issues the in-tree step's sequence, op for op and operand for operand, for a prefill
 * batch and a decode batch, and touches the device API not at all. */
TEST(off_step_issues_the_in_tree_sequence) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    for (bool fmt : {false, true})
        for (bool prefill : {true, false}) {
            RadBuilder stock, kva;
            if (fmt) { served(stock); served(kva); }
            REQUIRE_EQ(qwen4exp_fp8::declare(&stock, &meta, &c), RAD_OK);
            Batch bs = make_batch(stock, prefill);
            int dev_stock = 0, dev_kva = 0;
            const std::vector<RecIssue> want = issues_of(qwen4exp_fp8::step, bs.b, &dev_stock);
            REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
            Batch bk = make_batch(kva, prefill);
            const std::vector<RecIssue> got = issues_of(qwen4exp_kva::step, bk.b, &dev_kva);
            CHECK(want.size() > 100);
            CHECK_EQ(dev_kva, 0);
            CHECK_EQ(dev_stock, 0);
            CHECK_EQ(differ(got, want), 0);
        }
}

/* ==================================================================== the fitted tensors */

/* rad-convert's view: every copy the model holds is declared, with the shapes, dtypes, shards,
 * access class and group the plan states, and nothing is issued that the in-tree graph does not. */
TEST(declare_all_declares_every_held_copy_and_no_op) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx(1, 2);   /* rank 1 of 2: per-rank extents show */
    Env env({{"RADIANCE_KVA_DECLARE", "all"}});
    RadBuilder stock, kva;
    hold_kva(kva, {"kva.proj", "kva.projr", "kva.st", "kva.stswap", "kva.str"});
    hold_score(kva, "kva.rowsel.score");
    hold_score(kva, "kva.rowsel.score_all");
    REQUIRE_EQ(qwen4exp_fp8::declare(&stock, &meta, &c), RAD_OK);
    REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
    CHECK_EQ(kva.ops.size(), stock.ops.size());
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[1];
    int proj = 0, st = 0;
    for (int l = 0; l < 8; ++l) {
        for (const char* p : {"kva.proj.", "kva.projr."}) {
            const RadWeightDecl* w = weight(kva, p + std::to_string(l) + ".weight");
            const RadWeightDecl* bias = weight(kva, p + std::to_string(l) + ".bias");
            CHECK_EQ(w != nullptr, l >= kSplit);
            CHECK_EQ(bias != nullptr, l >= kSplit);
            if (!w || !bias) continue;
            ++proj;
            CHECK_EQ(w->dtype, (uint32_t)RAD_BF16);
            CHECK_EQ(w->shape[0], m.g.n_embd);
            CHECK_EQ(w->shape[1], m.hccfg.hc * m.g.n_embd);
            CHECK_EQ(bias->shape[0], m.g.n_embd);
            CHECK_EQ(w->shard, RAD_SHARD_NONE);
            CHECK_EQ(w->access, RAD_ACCESS_PER_TOKEN);
            CHECK_EQ(w->optional, 1);
            CHECK_EQ(w->group.layer, l);
        }
        for (const char* s : {"kva.st.", "kva.stswap.", "kva.str."}) {
            const RadWeightDecl* w = weight(kva, s + std::to_string(l));
            CHECK_EQ(w != nullptr, l >= kSplit && !m.layers[(size_t)l].full);
            if (!w) continue;
            ++st;
            CHECK_EQ(w->dtype, (uint32_t)RAD_F32);
            CHECK_EQ(w->shape[0], m.gcfg.n_head_v * 2);   /* every head: replicated */
            CHECK_EQ(w->shape[1], m.gcfg.head_v);
            CHECK_EQ(w->shape[2], m.gcfg.head_k);
            CHECK_EQ(w->shard, RAD_SHARD_NONE);   /* the loader has no ROW share of rank 3 */
            CHECK_EQ(w->optional, 1);
        }
    }
    CHECK_EQ(proj, 2 * (8 - kSplit));
    CHECK_EQ(st, 3 * 3);
    const RadWeightDecl* sc = weight(kva, "kva.rowsel.score");
    REQUIRE(sc != nullptr);
    CHECK_EQ(sc->shape[0], meta.n_vocab);
    CHECK_EQ(sc->shard, RAD_SHARD_NONE);
    CHECK(weight(kva, "kva.rowsel.score_all") != nullptr);
    CHECK(weight(kva, "kva.rowsel.score_none") == nullptr);   /* not held, not declared */
}

/* A serving mode declares the SELECTED copy only -- the controls stay off the card -- and the
 * kernel ops it will issue. */
TEST(speed_declares_the_selected_copy_and_its_kernel_ops) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    for (const char* which : {"shipped", "swap"}) {
        Env env({{"RADIANCE_KVA", "speed"}, {"RADIANCE_KVA_ST", which}});
        RadBuilder kva;
        hold_kva(kva, {"kva.proj", "kva.projr", "kva.st", "kva.stswap"});
        CHECK_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
        const bool swap = !std::strcmp(which, "swap");
        CHECK(weight(kva, "kva.proj.4.weight") != nullptr);
        CHECK(weight(kva, "kva.projr.4.weight") == nullptr);
        CHECK_EQ(weight(kva, "kva.st.4") != nullptr, !swap);
        CHECK_EQ(weight(kva, "kva.stswap.4") != nullptr, swap);
        CHECK(weight(kva, "kva.rowsel.score") == nullptr);
        const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
        CHECK_EQ(k.split, (int64_t)kSplit);
        CHECK(k.have_proj && k.have_st && !k.have_rowsel);
        int undo = 0, apply = 0;
        for (const RecOp& o : kva.ops) {
            if (o.op != "kva_state_correct") continue;
            REQUIRE_EQ(o.w.size(), (size_t)1);
            const std::string& wn = kva.weights[o.w[0] - 1].first;
            CHECK(wn.rfind(swap ? "kva.stswap." : "kva.st.", 0) == 0);
            for (const RecParam& p : o.p)
                if (p.key == "mode") { undo += p.sval == "undo"; apply += p.sval == "apply"; }
        }
        CHECK_EQ(undo, 3);    /* delta-net layers 4, 5, 6 */
        CHECK_EQ(apply, 3);
    }
}


/* Quality declares the selected row table and the kva_mask op over it, with the rule the switch
 * names; no cap and no compacted-row buffers exist any more (PLAN-FIX DD-F). */
TEST(quality_declares_the_selected_row_table_and_kva_mask) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    Env env({{"RADIANCE_KVA", "quality"}, {"RADIANCE_KVA_ROWSEL_TABLE", "none"},
             {"RADIANCE_KVA_ROWSEL", "random"}});
    RadBuilder kva;
    hold_kva(kva, {"kva.proj", "kva.st"});
    for (const char* t : {"kva.rowsel.score", "kva.rowsel.score_none", "kva.rowsel.score_all"})
        hold_score(kva, t);
    CHECK_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
    CHECK(weight(kva, "kva.rowsel.score_none") != nullptr);
    CHECK(weight(kva, "kva.rowsel.score") == nullptr);
    CHECK(weight(kva, "kva.rowsel.score_all") == nullptr);
    int masks = 0;
    for (const RecOp& o : kva.ops) {
        CHECK(o.op != "kva_rowsel");
        if (o.op != "kva_mask") continue;
        ++masks;
        REQUIRE_EQ(o.w.size(), (size_t)1);
        CHECK_EQ(kva.weights[o.w[0] - 1].first, std::string("kva.rowsel.score_none"));
        for (const RecParam& p : o.p) {
            CHECK(p.key != "cap");
            if (p.key == "mode") CHECK_EQ(p.sval, std::string("random"));
            if (p.key == "M")    CHECK_EQ(p.ihi, 2048LL);
            if (p.key == "share") CHECK_EQ(p.dval, 0.25);
        }
    }
    CHECK_EQ(masks, 1);
    for (const auto& [name, d] : kva.bufs)
        CHECK(name != "kva_rows_idx" && name != "kva_h_rows" && name != "kva_x_rows" && name != "kva_y_rows");
}

/* ==================================================================== the masked path: declare */

/* Declares stock and KVA side by side on the served formats with every kva.* tensor held. */
struct Pair {
    RadBuilder stock, kva;
    int        st = RAD_OK;
};
void declare_pair(Pair& p, const char* mode, int rank = 0, int world = 1, int64_t max_out_rows = 0) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx(rank, world);
    c.max_out_rows = max_out_rows;
    served(p.stock);
    served(p.kva);
    hold_kva(p.kva, {"kva.proj", "kva.st"});
    hold_score(p.kva, "kva.rowsel.score");
    REQUIRE_EQ(qwen4exp_fp8::declare(&p.stock, &meta, &c), RAD_OK);
    Env env({{"RADIANCE_KVA", mode}});
    p.st = qwen4exp_kva::declare(&p.kva, &meta, &c);
}

/* R93 -- THE LEVER'S PREMISE: in every routed layer the FIRST op naming one of the layer's EXPERT
 * weights (the stager's unit, RadWeightGroup.expert >= 0) is its gate-up GEMM -- the handle a probe
 * issues -- and the LAST is its routed down GEMM: the protected experts are layer weights and the
 * calibration tap names none. Nothing KVA declares names an expert weight. */
TEST(each_routed_layers_expert_span_is_gate_up_to_down) {
    Pair p;
    declare_pair(p, "quality");
    REQUIRE_EQ(p.st, RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    std::vector<size_t> first(8, 0), last(8, 0);
    for (size_t i = 0; i < p.kva.ops.size(); ++i)
        for (rad_weight w : p.kva.ops[i].w) {
            const RadWeightDecl& d = p.kva.weights[w - 1].second;
            if (d.group.expert < 0 || d.group.layer < 0 || d.group.layer >= 8) continue;
            CHECK(i < p.stock.ops.size());
            size_t& f = first[(size_t)d.group.layer];
            f = f ? std::min(f, i + 1) : i + 1;
            last[(size_t)d.group.layer] = std::max(last[(size_t)d.group.layer], i + 1);
        }
    for (int l = 0; l < 8; ++l) {
        CHECK_EQ(first[(size_t)l], (size_t)m.layers[(size_t)l].mlp.op_gu);
        CHECK_EQ(last[(size_t)l], (size_t)m.layers[(size_t)l].mlp.op_dn);
    }
}

/* Every buffer the masked path owns takes the whole program, and so does every in-tree buffer an op
 * declared after the graph touches (PLAN D4): the mask and bounds always; the layer-S stream, the
 * projected input with its codes and the routing ids when the mode projects. */
TEST(the_masked_paths_buffers_take_the_whole_program) {
    for (const char* mode : {"plumb", "speed", "quality"}) {
        Pair p;
        declare_pair(p, mode);
        REQUIRE_EQ(p.st, RAD_OK);
        const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
        const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
        const bool project = std::strcmp(mode, "plumb") != 0;
        CHECK(p.kva.concurrent.count(k.b_mask) && p.kva.concurrent.count(k.b_bounds));
        CHECK_EQ(k.b_hs != 0, project);
        CHECK_EQ(k.op_select != 0 && k.op_drop != 0 && k.op_cast != 0, project);
        for (rad_buf h : {k.b_hs, k.xp.x, k.xp.q8, k.xp.s8, m.b_eids, m.b_h, m.a_x.x})
            if (project) CHECK(p.kva.concurrent.count(h) == 1);
        CHECK_EQ(k.xp.q8_fed, project && m.a_x.q8_fed);
        CHECK(p.kva.concurrent.count(k.b_zeros) && p.kva.concurrent.count(k.b_probe));
        CHECK_EQ(k.n_zeros, m.layers[0].mlp.c.n_expert + 1);
    }
}

/* ==================================================================== the decision (R52') */

/* A step of sequences with query lengths `q`, the first `D` of them decoding, `ctx` tokens before
 * each, the last followed by `ahead` prompt tokens. Every KV group gets its own slot rows. */
struct Shape {
    std::vector<int32_t> q;
    int64_t D = 0, ahead = 0;
    int32_t ctx = 4096;
};
Batch make_step(const RadBuilder& bld, const Shape& s) {
    static int32_t ids[4096], pos[4096], cu[17], slot[4096], table[16 * 1024];
    static int32_t used[16], state[64 * 48], qlen[16], ctxl[16], acc[16];
    Batch x = make_batch(bld, true);
    for (size_t g = 0; g < x.kv.size(); ++g) {
        RadKVGroupBatch& k = x.kv[g];
        k.slot_mapping = slot; k.block_table = table; k.block_table_pitch = 1024;
        k.seqused = used; k.state_index = state + 48 * g; k.state_index_pitch = 3; k.max_blocks = 1024;
    }
    for (int i = 0; i < 64 * 48; ++i) state[i] = i % 7;
    RadBatch& b = x.b;
    const int64_t n = (int64_t)s.q.size();
    int64_t T = 0, DT = 0;
    int32_t qd = 0, qp = 0, qmax = 0;
    cu[0] = 0;
    for (int64_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < s.q[(size_t)i]; ++j) { ids[T + j] = 1000 + (int32_t)(T + j); pos[T + j] = s.ctx + j; }
        T += s.q[(size_t)i];
        cu[i + 1] = (int32_t)T;
        qlen[i] = s.q[(size_t)i]; ctxl[i] = s.ctx; acc[i] = 0;
        (i < s.D ? qd : qp) = std::max(i < s.D ? qd : qp, s.q[(size_t)i]);
        qmax = std::max(qmax, s.q[(size_t)i]);
        if (i < s.D) DT += s.q[(size_t)i];
    }
    b.kv = x.kv.data();
    b.phase = s.D == 0 ? RAD_PHASE_PREFILL : s.D == n ? RAD_PHASE_DECODE : RAD_PHASE_MIXED;
    b.n_tok = T; b.n_seq = n; b.n_ahead = s.ahead;
    b.token_ids = ids; b.positions = pos; b.cu_seqlens = cu;
    b.n_out = 0; b.out_ids = nullptr;
    b.q_lens = qlen; b.ctx_lens = ctxl; b.num_accepted = acc;
    b.max_q_len = qmax; b.max_ctx_len = s.ctx;
    b.n_seq_decode = s.D; b.n_tok_decode = DT;
    b.max_q_len_decode = qd; b.max_q_len_prefill = qp;
    return x;
}

/* One row of the truth table: a mode, a shape and its edits, and the hand-computed answer
 * (PLAN-FIX §2 with T = 2048, G = 64). `stream` is the default stage (auto, every masked pass). */
struct Row {
    const char* why;
    const char* mode;
    Shape       shape;
    std::function<void(RadBatch&)> edit;
    int         path;
    int64_t     b, s_lb;
    bool        stream;
};

TEST(the_approximate_decision_truth_table) {
    using qwen4exp_kva::PATH_STOCK;
    using qwen4exp_kva::PATH_LEAN;
    using qwen4exp_kva::PATH_MASKED;
    const auto none = [](RadBatch&) {};
    const std::vector<Row> rows = {
        {"whole bulk chunk, speed", "speed", {{2048}, 0, 2048}, none, PATH_LEAN, 2048, 0, false},
        {"whole bulk chunk, quality", "quality", {{2048}, 0, 2048}, none, PATH_MASKED, 2048, 0, true},
        {"whole bulk chunk, plumb", "plumb", {{2048}, 0, 2048}, none, PATH_MASKED, 2048, 0, true},
        {"more than T ahead", "speed", {{2048}, 0, 4096}, none, PATH_LEAN, 2048, 0, false},
        {"T-1 ahead: one tile exact", "speed", {{2048}, 0, 2047}, none, PATH_MASKED, 1984, 0, true},
        {"half ahead", "speed", {{2048}, 0, 1024}, none, PATH_MASKED, 1024, 0, true},
        {"64 ahead", "quality", {{2048}, 0, 64}, none, PATH_MASKED, 64, 0, true},
        {"1 ahead: no bulk", "speed", {{2048}, 0, 1}, none, PATH_STOCK, 0, 0, false},
        {"final chunk", "speed", {{2048}, 0, 0}, none, PATH_STOCK, 0, 0, false},
        {"mode off", "off", {{2048}, 0, 2048}, none, PATH_STOCK, 0, 0, false},
        {"draft pass", "speed", {{2048}, 0, 2048}, [](RadBatch& b) { b.draft_pass = 1; }, PATH_STOCK, 0, 0, false},
        {"encoder pass", "speed", {{2048}, 0, 2048}, [](RadBatch& b) { b.enc = 1; }, PATH_STOCK, 0, 0, false},
        {"media rows", "quality", {{2048}, 0, 2048}, [](RadBatch& b) { b.n_mm_rows = 5; }, PATH_STOCK, 0, 0, false},
        {"mixed rope", "speed", {{2048}, 0, 2048},
         [](RadBatch& b) { static int32_t rp[3]; b.rope_pos = rp; b.rope_mixed = 1; }, PATH_STOCK, 0, 0, false},
        {"one decoder beside", "speed", {{1, 1984}, 1, 2048}, none, PATH_MASKED, 1985, 1, true},
        {"eight decoders beside", "speed", {{1, 1, 1, 1, 1, 1, 1, 1, 1984}, 8, 2048}, none, PATH_MASKED, 1992, 8, true},
        {"eight decoders, straddle", "quality", {{1, 1, 1, 1, 1, 1, 1, 1, 1984}, 8, 1000}, none, PATH_MASKED, 904, 8, true},
        {"two prefills, last long", "speed", {{64, 1984}, 0, 2048}, none, PATH_MASKED, 2048, 64, true},
        {"two prefills, last short", "speed", {{1984, 64}, 0, 2048}, none, PATH_MASKED, 2048, 64, true},
        {"two prefills, no bulk above s_lb", "speed", {{1984, 64}, 0, 100}, none, PATH_STOCK, 0, 0, false},
        {"three prefills and a decoder", "quality", {{1, 1024, 512, 448}, 1, 2048}, none, PATH_MASKED, 1985, 961, true},
        {"pure decode", "speed", {{1, 1}, 2, 0}, none, PATH_STOCK, 0, 0, false},
    };
    for (const Row& r : rows) {
        Pair p;
        declare_pair(p, r.mode);
        REQUIRE_EQ(p.st, RAD_OK);
        Batch x = make_step(p.kva, r.shape);
        r.edit(x.b);
        const qwen4exp_kva::Pass got = qwen4exp_kva::derive(qwen4exp_kva::g_kva[0], &x.b);
        if (got.path != r.path || got.b != r.b || got.s_lb != r.s_lb || got.stream != r.stream)
            std::fprintf(stderr, "    row '%s': path %d b %lld s_lb %lld stream %d\n", r.why, got.path,
                         (long long)got.b, (long long)got.s_lb, (int)got.stream);
        CHECK_EQ(got.path, r.path);
        CHECK_EQ(got.b, r.b);
        CHECK_EQ(got.s_lb, r.s_lb);
        CHECK_EQ(got.stream, r.stream);
    }
}

/* R73 -- KL MODE IS EXACT UNLESS THE SWITCH IS SET: a declare that sizes logits for every prompt
 * row (max_out_rows > 0) serves every pass stock, because bulk rows' logits are not the model's;
 * RADIANCE_KVA_SCORE_BULK=1 says the caller scores the exact tail only, and the pass approximates. */
TEST(kl_mode_serves_stock_unless_score_bulk) {
    for (bool bulk : {false, true}) {
        Env e(bulk ? std::initializer_list<std::pair<const char*, const char*>>{{"RADIANCE_KVA_SCORE_BULK", "1"}}
                   : std::initializer_list<std::pair<const char*, const char*>>{});
        Pair p;
        declare_pair(p, "quality", 0, 1, 4096);
        REQUIRE_EQ(p.st, RAD_OK);
        Batch x = make_step(p.kva, {{128}, 0, 2048});
        const qwen4exp_kva::Pass got = qwen4exp_kva::derive(qwen4exp_kva::g_kva[0], &x.b);
        CHECK_EQ(got.path, bulk ? qwen4exp_kva::PATH_MASKED : qwen4exp_kva::PATH_STOCK);
        CHECK_EQ(qwen4exp_kva::g_kva[0].out_rows_ok, bulk);
    }
}

/* The stager lever's switch: stock never streams, and the row threshold gates auto. */
TEST(the_stage_switch_and_threshold_gate_the_lever) {
    struct Case { const char* stage; const char* rows; Shape s; bool stream; };
    for (const Case& c : {Case{"stock", nullptr, {{2048}, 0, 2047}, false},
                          Case{"auto", "63", {{2048}, 0, 2047}, false},    /* 64 exact rows */
                          Case{"auto", "64", {{2048}, 0, 2047}, true},
                          Case{"auto", "8", {{1, 1, 1, 1, 1, 1, 1, 1, 1984}, 8, 2048}, true}}) {
        Env e(c.rows ? std::initializer_list<std::pair<const char*, const char*>>{{"RADIANCE_KVA_STAGE", c.stage}, {"RADIANCE_KVA_STAGE_ROWS", c.rows}}
                     : std::initializer_list<std::pair<const char*, const char*>>{{"RADIANCE_KVA_STAGE", c.stage}});
        Pair p;
        declare_pair(p, "quality");
        REQUIRE_EQ(p.st, RAD_OK);
        Batch x = make_step(p.kva, c.s);
        CHECK_EQ(qwen4exp_kva::derive(qwen4exp_kva::g_kva[0], &x.b).stream, c.stream);
    }
}

/* ==================================================================== issue sequences */

struct Run {
    std::vector<RecIssue> issues;
    std::string           log;
    int                   device_calls = 0;
    std::vector<std::pair<const void*, size_t>> reads;
};

Run run_step(void (*step)(RadCtx*, const RadBatch*), const RadBatch& b, int rank = 0) {
    Run r;
    RadCtx c;
    c.batch = &b;
    c.rank = rank;
    g_ctx = &c;
    r.log = stderr_of([&] { step(&c, &b); });
    g_ctx = nullptr;
    r.issues = c.issues;
    r.device_calls = c.device_calls;
    r.reads = c.reads;
    return r;
}

int count(const std::string& s, const char* what) {
    int n = 0;
    for (size_t at = s.find(what); at != std::string::npos; at = s.find(what, at + 1)) ++n;
    return n;
}

/* Where each late layer's issues start: its connection read in a stock or masked run, its projector
 * in a lean run; the mixer's read ends the last one. */
std::vector<size_t> starts(const std::vector<RecIssue>& v, const std::vector<rad_op>& first, rad_op end) {
    std::vector<size_t> at;
    for (rad_op h : first) {
        size_t i = 0;
        while (i < v.size() && v[i].op != h) ++i;
        at.push_back(i);
    }
    size_t i = 0;
    while (i < v.size() && v[i].op != end) ++i;
    at.push_back(i);
    return at;
}

std::vector<RecIssue> slice(const std::vector<RecIssue>& v, size_t a, size_t b) {
    return std::vector<RecIssue>(v.begin() + (long)std::min(a, v.size()),
                                 v.begin() + (long)std::min(b, v.size()));
}

/* What a masked pass adds to the in-tree step (PLAN-FIX §8), as switches the expectation reads. */
struct Want {
    int64_t b = 0, s_lb = 0, rho_rows = 0;
    bool    stream = false, project = false, correct = false, rho = false, split = false;
    bool    scored = false;
};

/* A late group's slot row for the step's last sequence (index row n_seq-1, pitch 3). */
RadOperand last_row(const Batch& x, rad_kvgroup g) {
    return praw2(x.b.kv[g - 1].state_index + (x.b.n_seq - 1) * 3, RAD_I32, 1, 3);
}

std::vector<RadOperand> correction_operands(const Batch& x, int l, int rank = 0) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
    RadOperand heads = RAD_W(k.st[(size_t)l]);
    heads.offset = (int64_t)rank * m.gcfg.n_head_v * m.gcfg.head_v * m.gcfg.head_k;
    heads.rows = m.gcfg.n_head_v;
    return {kv_cache(m.kv_state, l), last_row(x, m.kv_state), kv_cache(k.kv_applied, l),
            last_row(x, k.kv_applied), heads, k.kv_rho ? kv_cache(k.kv_rho, l) : RAD_NONE,
            k.kv_rho ? last_row(x, k.kv_rho) : RAD_NONE, brows(k.b_bounds, 2)};
}

/* The in-tree scan, as the masked layer issues it for the last sequence (and, before it, the
 * other prefill sequences): cu and state rows narrowed, split at the bulk end when asked. */
void push_scans(std::vector<RecIssue>& out, const RecIssue& scan, const Batch& x, int l, const Want& w,
                int rank) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
    const qwen4exp_fp8::Layer& lay = m.layers[(size_t)l];
    const RadBatch& b = x.b;
    const int32_t* st = b.kv[m.kv_state - 1].state_index;
    const int64_t D = b.n_seq_decode, S = b.n_seq, P = S - D;
    RecIssue r = scan;
    if (P > 1) {
        r.opd[7] = praw(b.cu_seqlens + D, RAD_I32, P);
        r.opd[10] = praw2(st + D * 3, RAD_I32, P - 1, 3);
        out.push_back(r);
    }
    r.opd[7] = w.split ? brow_slice(k.b_bounds, 0, 2, 1) : praw(b.cu_seqlens + S - 1, RAD_I32, 2);
    r.opd[10] = praw2(st + (S - 1) * 3, RAD_I32, 1, 3);
    out.push_back(r);
    if (w.rho)
        out.push_back({k.op_rho[(size_t)l],
                       {bcol(lay.gdn.w.ab, 0, lay.gdn.cfg.n_head_v, w.rho_rows), brows(k.b_mask, w.rho_rows),
                        RAD_W(lay.gdn.w_a_log), RAD_W(lay.gdn.w_dt_bias), kv_cache(k.kv_rho, l),
                        last_row(x, k.kv_rho), brows(k.b_bounds, 1)}, w.rho_rows});
    if (w.correct) out.push_back({k.op_apply[(size_t)l], correction_operands(x, l, rank), 1});
    if (!w.split) return;
    r.opd[7] = brow_slice(k.b_bounds, 2, 2, 1);
    out.push_back(r);
}

/* The projected block input, its codes and the select, right after the connection read. */
void push_projection(std::vector<RecIssue>& out, const Batch& x, int l, const Want& w, int rank) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
    const int64_t T = x.b.n_tok, rows = w.b - w.s_lb, n = m.g.n_embd, wide = m.hccfg.hc * n;
    out.push_back({k.op_proj[(size_t)l], {brow_slice(k.b_hs, w.s_lb, rows, wide), RAD_W(k.proj_w[(size_t)l]),
                   RAD_W(k.proj_b[(size_t)l]), RAD_NONE, brow_slice(k.xp.x, w.s_lb, rows, n)}, rows});
    out.push_back({k.quant.op, {brow_slice(k.xp.x, w.s_lb, rows, n), brow_slice(k.xp.q8, w.s_lb, rows, n),
                   brow_slice(k.xp.s8, w.s_lb, rows, n / 128)}, rows});
    out.push_back({k.op_select, {brows(k.b_mask, T), brows(k.xp.x, T), brows(k.xp.q8, T), brows(k.xp.s8, T),
                   brows(m.a_x.x, T), brows(m.a_x.q8, T), brows(m.a_x.s8, T)}, T});
}

/* The routing with the bulk rows' slots dropped: the fused top-k+sort becomes the pair, with the
 * drop between (the fused form has no point between the two). */
void push_route(std::vector<RecIssue>& out, const RecIssue& r, const qwen4exp_fp8::Layer& lay,
                const Batch& x, int rank) {
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
    const MoeFP8& e = lay.mlp;
    const int64_t T = x.b.n_tok, kk = e.c.top_k;
    const RecIssue drop{k.op_drop, {brows(k.b_mask, T), brow_slice(e.w.ids, 0, T, kk)}, T};
    if (r.op == e.op_topk) { out.push_back(r); out.push_back(drop); return; }
    out.push_back({e.op_topk, {r.opd[0], r.opd[1], r.opd[2]}, r.n});
    out.push_back(drop);
    out.push_back({e.op_scatter, {r.opd[1], r.opd[3], r.opd[4], r.opd[5]}, r.n});
}

void push_layer(std::vector<RecIssue>& out, const std::vector<RecIssue>& seg, const Batch& x, int l,
                const Want& w, int rank) {
    const qwen4exp_fp8::Layer& lay = qwen4exp_fp8::g_model[rank].layers[(size_t)l];
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
    for (RecIssue r : seg) {
        if (r.op == lay.hc_mix.op_read) {
            out.push_back(r);
            if (w.project) push_projection(out, x, l, w, rank);
        } else if (!lay.full && r.op == lay.gdn.op_conv_prep) {
            if (w.correct) out.push_back({k.op_undo[(size_t)l], correction_operands(x, l, rank), 1});
            out.push_back(r);
        } else if (!lay.full && r.op == lay.gdn.op_scan) {
            push_scans(out, r, x, l, w, rank);
        } else if (w.project && (r.op == lay.mlp.op_topk || r.op == lay.mlp.op_topk_scatter)) {
            push_route(out, r, lay, x, rank);
        } else {
            out.push_back(r);
        }
    }
}

/* A stager probe as the oracle sees it: layer x's own stock gate-up issue, over one token, with the
 * routing replaced by the zero offsets and its output by the probe rows (notes/impl.md §2). */
RecIssue probe_of(const std::vector<RecIssue>& stock, int x, int rank) {
    const MoeFP8& e = qwen4exp_fp8::g_model[rank].layers[(size_t)x].mlp;
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
    RecIssue r;
    for (const RecIssue& i : stock) if (i.op == e.op_gu) { r = i; break; }
    r.n = 1;
    r.opd[0].rows = 1;
    r.opd[1].rows = 1;
    r.opd[4] = brows(k.b_zeros, e.c.top_k);
    r.opd[5].handle = k.b_zeros;
    r.opd[8] = brows(k.b_probe, e.c.top_k);
    return r;
}

/* THE ORACLE FOR EVERY MASKED PASS: the in-tree step's own issues on this batch, with exactly the
 * substitutions the masked path names -- the mask ahead of layer 0; on a streaming pass the probes
 * of layers S-1.. behind layer S-3's gate-up GEMM; the stream copy before layer S; and in each late
 * layer the projection selected in after the connection read, the correction around the last
 * sequence's scan and the bulk rows dropped from the routing. Built from qwen4exp_fp8::step, not
 * from the plugin. */
std::vector<RecIssue> masked_expected(const Batch& x, const Want& w, int rank = 0) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
    const RadBatch& b = x.b;
    const int64_t T = b.n_tok;
    const std::vector<RecIssue> stock = run_step(qwen4exp_fp8::step, b, rank).issues;
    std::vector<rad_op> reads;
    for (int l = kSplit; l < 8; ++l) reads.push_back(m.layers[(size_t)l].hc_mix.op_read);
    const std::vector<size_t> at = starts(stock, reads, m.mixer.op_read);
    std::vector<RecIssue> out;
    for (size_t i = 0; i < at[0]; ++i) {
        if (stock[i].op == m.layers[0].hc_mix.op_read)
            out.push_back({k.op_mask, {praw(b.cu_seqlens + b.n_seq - 1, RAD_I32, 2), praw(b.token_ids, RAD_I32, w.b),
                           praw(b.positions, RAD_I32, w.b), w.scored ? RAD_W(k.score) : RAD_NONE,
                           brows(k.b_mask, T), brows(k.b_bounds, 4), brows(k.b_zeros, k.n_zeros)}, w.b});
        out.push_back(stock[i]);
        if (w.stream && stock[i].op == m.layers[kSplit - 3].mlp.op_gu)
            for (int p = kSplit - 1; p < 8; ++p) out.push_back(probe_of(stock, p, rank));
    }
    if (w.project) {
        const int64_t wide = m.hccfg.hc * m.g.n_embd, rows = w.b - w.s_lb;
        out.push_back({k.op_cast, {brow_slice(m.b_h, w.s_lb, rows, wide), brow_slice(k.b_hs, w.s_lb, rows, wide)}, rows});
    }
    for (int l = kSplit; l < 8; ++l)
        push_layer(out, slice(stock, at[(size_t)(l - kSplit)], at[(size_t)(l - kSplit + 1)]), x, l, w, rank);
    for (size_t i = at.back(); i < stock.size(); ++i) out.push_back(stock[i]);
    return out;
}

/* Prints the first difference, so a failing case says where. */
int differ_at(const std::vector<RecIssue>& got, const std::vector<RecIssue>& want) {
    const int n = differ(got, want);
    for (size_t i = 0; n && i < std::max(got.size(), want.size()); ++i)
        if (i >= got.size() || i >= want.size() || !same_issue(got[i], want[i])) {
            std::fprintf(stderr, "    first difference at issue %zu of %zu/%zu: op %u vs %u\n", i,
                         got.size(), want.size(), i < got.size() ? got[i].op : 0u,
                         i < want.size() ? want[i].op : 0u);
            break;
        }
    return n;
}

/* PLUMB IS THE STOCK STEP THROUGH THE MASKED PATH: with the stager lever off (stage stock) it is the
 * in-tree step plus the one kva_mask issue (mode all: every row exact, no projection, no correction,
 * no drop); with the lever on, the probes ride behind layer S-3's gate-up GEMM (R94's static half);
 * with a forced split, each late delta-net layer's scan is two scans over the bounds pairs (R47's
 * static half). A 64-row chunk keeps the fused top-k (no drop). */
TEST(plumb_is_the_stock_step_through_the_masked_path) {
    struct Case { const char* stage; const char* force; int64_t T; bool stream, split; };
    for (const Case& c : {Case{"stock", nullptr, 128, false, false}, Case{"auto", nullptr, 128, true, false},
                          Case{"auto", "64", 128, true, true}, Case{"stock", nullptr, 64, false, false}}) {
        Env e(c.force ? std::initializer_list<std::pair<const char*, const char*>>{{"RADIANCE_KVA_STAGE", c.stage}, {"RADIANCE_KVA_FORCE_SPLIT", c.force}}
                      : std::initializer_list<std::pair<const char*, const char*>>{{"RADIANCE_KVA_STAGE", c.stage}});
        Pair p;
        declare_pair(p, "plumb");
        REQUIRE_EQ(p.st, RAD_OK);
        Batch x = make_step(p.kva, {{(int32_t)c.T}, 0, 2048});
        Want w;
        w.b = c.split ? c.T - 64 : c.T;
        w.stream = c.stream;
        w.split = c.split;
        const Run got = run_step(qwen4exp_kva::step, x.b);
        CHECK_EQ(differ_at(got.issues, masked_expected(x, w)), 0);
        CHECK_EQ(count(got.log, "kva: approximate step"), 1);
        CHECK_EQ(got.device_calls, 0);
    }
}

/* QUALITY MASKS ROWS IN PLACE THROUGH THE IN-TREE LAYER (PLAN-FIX §3): a whole bulk chunk at TP1
 * and on rank 0 of TP2 (where the writes carry the all-reduce), and a 64-row chunk (whose stock
 * routing is the fused form). The correction scales by the decay sums over [s, b). */
TEST(quality_masks_rows_in_place_through_the_in_tree_layer) {
    for (auto [world, T] : {std::pair<int, int32_t>{1, 128}, {2, 128}, {1, 64}}) {
        Pair p;
        declare_pair(p, "quality", 0, world);
        REQUIRE_EQ(p.st, RAD_OK);
        Batch x = make_step(p.kva, {{T}, 0, 2048});
        Want w;
        w.b = T; w.rho_rows = T;
        w.stream = w.project = w.correct = w.rho = w.scored = true;
        const Run got = run_step(qwen4exp_kva::step, x.b);
        CHECK_EQ(differ_at(got.issues, masked_expected(x, w)), 0);
        CHECK_EQ(count(got.log, "kva: approximate step"), 1);
        CHECK_EQ(got.device_calls, 0);
    }
}

/* #6 -- THE STRADDLING CHUNK (PLAN-FIX §4, DD-C): 64 tokens ahead of a 128-row chunk with T = 2048,
 * so the bulk ends at row 64. split: scan [s, b), decay sums over [s, b), apply, scan [b, e); end:
 * one scan, decay sums over the whole chunk, apply after. Speed mode on the same shape is masked
 * too (the lean fill has no exact rows) and applies without decay sums. */
TEST(a_straddling_chunk_splits_the_last_scan_at_the_bulk_end) {
    struct Case { const char* mode; const char* straddle; bool split, rho; int64_t rho_rows; };
    for (const Case& c : {Case{"quality", "split", true, true, 64}, Case{"quality", "end", false, true, 128},
                          Case{"speed", "split", true, false, 0}}) {
        Env e({{"RADIANCE_KVA_STRADDLE", c.straddle}});
        Pair p;
        declare_pair(p, c.mode);
        REQUIRE_EQ(p.st, RAD_OK);
        Batch x = make_step(p.kva, {{128}, 0, 1984});
        Want w;
        w.b = 64; w.rho_rows = c.rho_rows;
        w.stream = w.project = w.correct = true;
        w.rho = c.rho;
        w.split = c.split;
        w.scored = !std::strcmp(c.mode, "quality");
        const Run got = run_step(qwen4exp_kva::step, x.b);
        CHECK_EQ(differ_at(got.issues, masked_expected(x, w)), 0);
        CHECK(has(got.log, c.split ? ", split)" : ", end)"));
    }
}

/* R53' SUBSET -- DECODERS AND OTHER PREFILLS BESIDE THE APPROXIMATED CHUNK: the in-tree decode half
 * and the other prefill sequences' scan are issued as stock; only the last sequence is corrected
 * (M = 1 on its own index row), and its bulk superset starts at s_lb. */
TEST(a_mixed_step_corrects_only_the_last_sequence) {
    struct Case { Shape s; int64_t b, s_lb; };
    for (const Case& c : {Case{{{1, 1, 128}, 2, 2048}, 130, 2}, Case{{{64, 128}, 0, 2048}, 192, 64},
                          Case{{{1, 64, 128}, 1, 2048}, 193, 65}}) {
        Pair p;
        declare_pair(p, "quality");
        REQUIRE_EQ(p.st, RAD_OK);
        Batch x = make_step(p.kva, c.s);
        Want w;
        w.b = c.b; w.s_lb = c.s_lb; w.rho_rows = c.b;
        w.stream = w.project = w.correct = w.rho = w.scored = true;
        const Run got = run_step(qwen4exp_kva::step, x.b);
        CHECK_EQ(differ_at(got.issues, masked_expected(x, w)), 0);
        CHECK_EQ(count(got.log, "kva: approximate step"), 1);
    }
}

/* A ONE-SEQUENCE CHUNK OFF THE DELTA NET'S TILE FAILS THE STEP BY NAME: the scheduler never cuts one,
 * and splitting a tile would be silently wrong. A forced split off the tile (R47's negative
 * control) is served, and said at declare. */
TEST(a_chunk_off_the_tile_fails_the_step_by_name) {
    Pair p;
    declare_pair(p, "quality");
    REQUIRE_EQ(p.st, RAD_OK);
    Batch x = make_step(p.kva, {{100}, 0, 2048});
    RadCtx c;
    c.batch = &x.b;
    g_ctx = &c;
    qwen4exp_kva::step(&c, &x.b);
    g_ctx = nullptr;
    CHECK(has(c.step_fail, "tile"));
    CHECK(c.issues.empty());
    Env e({{"RADIANCE_KVA_FORCE_SPLIT", "100"}});
    Pair q;
    std::string err = stderr_of([&] { declare_pair(q, "plumb"); });
    REQUIRE_EQ(q.st, RAD_OK);
    CHECK(has(err, "OFF the delta net's tile"));
    Batch y = make_step(q.kva, {{128}, 0, 2048});
    const Run got = run_step(qwen4exp_kva::step, y.b);
    CHECK(!got.issues.empty());
}

/* ==================================================================== the lean fill */

void add_linear(std::set<rad_op>& s, const rad::arch::LinearFP8& l) {
    for (rad_op h : {l.op, l.op_q, l.op_m1}) if (h) s.insert(h);
}

/* The handles of the pieces the lean fill issues for late layer `l`: the delta net's projections
 * and recurrence, or the indexer's block-key half (`keys`) and the attention's K/V path (`kv`). */
struct Pieces { std::set<rad_op> gdn, keys, kv; };
Pieces pieces(const qwen4exp_fp8::Layer& l) {
    Pieces p;
    add_linear(p.gdn, l.gdn.in);
    for (rad_op h : {l.gdn.op_ab, l.gdn.op_conv_prep, l.gdn.op_kkt, l.gdn.op_scan}) p.gdn.insert(h);
    add_linear(p.keys, l.qsa.proj);
    for (rad_op h : {l.qsa.op_work, l.qsa.op_bkey, l.qsa.op_tail}) p.keys.insert(h);
    add_linear(p.kv, l.attn.kp);
    add_linear(p.kv, l.attn.vp);
    for (rad_op h : {l.attn.op_k_norm, l.attn.op_rope_k, l.attn.op_kv_store}) p.kv.insert(h);
    return p;
}

/* THE LEAN FILL (speed, one sequence, whole chunk bulk) FILLS A LATE LAYER WITH ITS CACHE-WRITING
 * OPS AND NOTHING ELSE: the projector into `x`, the quantiser, then the stock block's own delta-net
 * projections and recurrence -- or indexer block-key half and K/V path -- in stock order with stock
 * operands. No connection read or write, no MoE, no query path, no mask, no alternate. */
TEST(speed_fills_late_layers_with_their_cache_writing_ops_only) {
    Pair p;
    declare_pair(p, "speed");
    REQUIRE_EQ(p.st, RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
    REQUIRE(k.quant.op != 0);                          /* the served formats need codes */
    CHECK(m.a_x.q8_fed);
    Batch x = make_step(p.kva, {{128}, 0, 2048});
    const Run want = run_step(qwen4exp_fp8::step, x.b);
    const Run got = run_step(qwen4exp_kva::step, x.b);
    std::vector<rad_op> reads, projs;
    for (int l = kSplit; l < 8; ++l) {
        reads.push_back(m.layers[(size_t)l].hc_mix.op_read);
        projs.push_back(k.op_proj[(size_t)l]);
    }
    const std::vector<size_t> ws = starts(want.issues, reads, m.mixer.op_read);
    const std::vector<size_t> gs = starts(got.issues, projs, m.mixer.op_read);
    CHECK_EQ(differ(slice(got.issues, 0, gs[0]), slice(want.issues, 0, ws[0])), 0);
    CHECK_EQ(differ(slice(got.issues, gs.back(), got.issues.size()),
                    slice(want.issues, ws.back(), want.issues.size())), 0);
    for (const RecIssue& r : got.issues) {   /* no mask, no probe */
        CHECK(r.op != k.op_mask);
        for (const RadOperand& o : r.opd) CHECK(o.kind != RAD_OPK_BUF || o.handle != k.b_zeros);
    }
    for (int l = kSplit; l < 8; ++l) {
        const size_t i = (size_t)(l - kSplit);
        const qwen4exp_fp8::Layer& lay = m.layers[(size_t)l];
        const Pieces pc = pieces(lay);
        std::vector<RecIssue> seg = slice(got.issues, gs[i], gs[i + 1]);
        REQUIRE(seg.size() > 2);
        CHECK_EQ(seg[0].op, k.op_proj[(size_t)l]);
        CHECK_EQ(seg[0].opd[0].handle, m.b_h);
        CHECK_EQ(seg[0].opd[1].handle, k.proj_w[(size_t)l]);
        CHECK_EQ(seg[0].opd[4].handle, m.a_x.x);
        CHECK_EQ(seg[1].op, k.quant.op);
        CHECK_EQ(seg[1].opd[1].handle, m.a_x.q8);
        std::vector<RecIssue> keep;
        for (const RecIssue& r : slice(want.issues, ws[i], ws[i + 1]))
            if (lay.full ? (pc.keys.count(r.op) || pc.kv.count(r.op)) : pc.gdn.count(r.op))
                keep.push_back(r);
        CHECK(keep.size() >= (lay.full ? 9u : 5u));
        if (!lay.full) {
            /* Stage 4: undo right before the conv/scan, apply right after, on this layer's slots;
             * no bounds on the lean path (its sequence's whole chunk is bulk). */
            std::vector<RadOperand> opd = correction_operands(x, l);
            opd[7] = RAD_NONE;
            size_t conv = 0;
            while (conv < keep.size() && keep[conv].op != lay.gdn.op_conv_prep) ++conv;
            keep.insert(keep.begin() + (long)conv, RecIssue{k.op_undo[(size_t)l], opd, 1});
            keep.push_back(RecIssue{k.op_apply[(size_t)l], opd, 1});
        }
        CHECK_EQ(differ(slice(seg, 2, seg.size()), keep), 0);
    }
    CHECK_EQ(count(got.log, "kva: approximate step"), 1);
    CHECK(has(got.log, "lean"));
    CHECK_EQ(got.device_calls, 0);
}

/* The correction is wired only where the model holds one, with the strength the switch names,
 * and its slot group binds exactly the late delta-net layers. Without kva.st.* nothing of it is
 * declared or issued (speed without correction, Stage 3's arm). */
TEST(the_correction_is_declared_and_issued_only_when_held) {
    for (bool held : {true, false}) {
        RadModelMeta meta = flash_next_meta();
        RadBuildCtx c = served_ctx();
        RadBuilder kva;
        served(kva);
        if (held) hold_kva(kva, {"kva.proj", "kva.st"});
        else      hold_kva(kva, {"kva.proj"});
        Env env({{"RADIANCE_KVA", "speed"}, {"RADIANCE_KVA_ALPHA", "0.5"}});
        REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
        const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
        CHECK_EQ(k.kv_applied != 0, held);
        CHECK_EQ(std::count(kva.kv_groups.begin(), kva.kv_groups.end(), "kv_kva_applied"), held ? 1 : 0);
        int bound = 0;
        for (const auto& [layer, g] : kva.binds) {
            if (g != k.kv_applied || !held) continue;
            ++bound;
            CHECK(layer >= kSplit && !qwen4exp_fp8::g_model[0].layers[(size_t)layer].full);
        }
        CHECK_EQ(bound, held ? 3 : 0);
        for (const RecOp& o : kva.ops)
            if (o.op == "kva_state_correct")
                for (const RecParam& q : o.p) if (q.key == "alpha") CHECK_EQ(q.dval, 0.5);
        Batch bk = make_step(kva, {{128}, 0, 2048});
        const Run r = run_step(qwen4exp_kva::step, bk.b);
        int corrections = 0;
        for (const RecIssue& i : r.issues)
            corrections += i.op && kva.ops[i.op - 1].op == "kva_state_correct";
        CHECK_EQ(corrections, held ? 2 * 3 : 0);
    }
}

/* AT TP2 EACH RANK HANDS THE CORRECTION ITS OWN VALUE HEADS: the weight is the whole [48, V, K]
 * tensor on every rank (no ROW share of a rank-3 weight exists), sliced at issue to rows
 * [rank*H, rank*H + H) -- the contiguous head split of the delta net's own weights. */
TEST(at_tp2_each_rank_corrects_its_own_heads) {
    for (int rank : {0, 1}) {
        Pair p;
        declare_pair(p, "speed", rank, 2);
        REQUIRE_EQ(p.st, RAD_OK);
        const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
        const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
        const RadWeightDecl* w = weight(p.kva, "kva.st.4");
        REQUIRE(w != nullptr);
        CHECK_EQ(w->shard, RAD_SHARD_NONE);
        CHECK_EQ(w->shape[0], m.gcfg.n_head_v * 2);
        Batch bk = make_step(p.kva, {{128}, 0, 2048});
        const Run r = run_step(qwen4exp_kva::step, bk.b, rank);
        int seen = 0;
        for (const RecIssue& i : r.issues) {
            if (i.op != k.op_undo[kSplit] && i.op != k.op_apply[kSplit]) continue;
            ++seen;
            REQUIRE_EQ(i.opd.size(), (size_t)8);
            CHECK_EQ(i.opd[4].handle, k.st[kSplit]);
            CHECK_EQ(i.opd[4].rows, m.gcfg.n_head_v);
            CHECK_EQ(i.opd[4].offset, (int64_t)rank * m.gcfg.n_head_v * m.gcfg.head_v * m.gcfg.head_k);
        }
        CHECK_EQ(seen, 2);
    }
}

/* plumb runs the late layers exactly and issues no correction, so it places none; it declares no
 * projector op, no select, no drop and no stream copy either (it still declares the projector's
 * weights: S is the lowest projected layer). */
TEST(plumb_declares_no_correction) {
    Pair p;
    declare_pair(p, "plumb");
    REQUIRE_EQ(p.st, RAD_OK);
    CHECK(weight(p.kva, "kva.st.4") == nullptr);
    CHECK(weight(p.kva, "kva.proj.4.weight") != nullptr);
    for (size_t i = p.stock.ops.size(); i < p.kva.ops.size(); ++i) {
        const std::string& op = p.kva.ops[i].op;
        CHECK(op == "kva_mask" || op == "moe_gemm_q");
    }
}

/* ==================================================================== Stage 6 captures */



/* A scratch directory, removed with the object. */
struct TempDir {
    std::filesystem::path path;
    TempDir() {
        std::string t = (std::filesystem::temp_directory_path() / "kva_capture_XXXXXX").string();
        path = mkdtemp(t.data()) ? t : std::string();
    }
    ~TempDir() { if (!path.empty()) std::filesystem::remove_all(path); }
};

std::string slurp(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/* The `shape` of an .npy file's header, and its descr ("<u2", "<i4", "<f4"). */
std::vector<int64_t> npy_shape(const std::filesystem::path& p, std::string* descr) {
    const std::string s = slurp(p);
    std::vector<int64_t> shape;
    const size_t d = s.find("'descr': '"), sh = s.find("'shape': (");
    if (s.size() < 10 || d == std::string::npos || sh == std::string::npos) return shape;
    *descr = s.substr(d + 10, 3);
    for (size_t i = sh + 10; i < s.size() && s[i] != ')';) {
        if (std::isdigit((unsigned char)s[i])) { size_t e = 0; shape.push_back(std::stoll(s.substr(i), &e)); i += e; }
        else ++i;
    }
    return shape;
}

/* The one file in `dir` whose name starts with `head` and ends with `tail`. */
std::filesystem::path find_file(const std::filesystem::path& dir, const std::string& head,
                                const std::string& tail) {
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.rfind(head, 0) == 0 && n.size() >= tail.size() &&
            n.compare(n.size() - tail.size(), tail.size(), tail) == 0)
            return e.path();
    }
    return {};
}

/* RADIANCE_KVA_CAPTURE (off): on rank 0 a single-sequence prefill chunk issues the stock sequence
 * exactly and writes the layout notes/arch.md documents -- the stream entering S and every late
 * layer's block input at the rows whose position is a multiple of 8, all ids and positions, one
 * jsonl line. Rank 1 and a two-sequence step capture nothing and copy nothing. */
TEST(capture_records_an_exact_chunk_and_issues_the_stock_sequence) {
    TempDir dir;
    REQUIRE(!dir.path.empty());
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    RadBuilder stock, kva;
    served(stock);
    served(kva);
    hold_kva(kva, {"kva.proj", "kva.st"});
    REQUIRE_EQ(qwen4exp_fp8::declare(&stock, &meta, &c), RAD_OK);
    Env env({{"RADIANCE_KVA_CAPTURE", dir.path.c_str()}});
    REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
    CHECK_EQ(kva.ops.size(), stock.ops.size());
    CHECK_EQ(qwen4exp_kva::g_kva[0].split, (int64_t)kSplit);
    Batch bs = make_step(stock, {{128}, 0, 300, 4100});   /* positions 4100..4227 */
    Batch bk = make_step(kva, {{128}, 0, 300, 4100});
    const Run want = run_step(qwen4exp_fp8::step, bs.b);
    const Run got = run_step(qwen4exp_kva::step, bk.b);
    CHECK_EQ(differ(got.issues, want.issues), 0);
    CHECK(got.device_calls > 0);
    /* WHERE each read sits: b_h just before layer S's first issue, and x just after each late
     * layer's connection read -- before the block overwrites it with its output. */
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    auto first_issue = [&](rad_op h) {
        size_t i = 0;
        while (i < got.issues.size() && got.issues[i].op != h) ++i;
        return i;
    };
    std::vector<size_t> x_reads, h_reads;
    for (const auto& [src, at] : got.reads) {
        if (src == rad_buf_ptr(nullptr, m.a_x.x)) x_reads.push_back(at);
        if (src == rad_buf_ptr(nullptr, m.b_h)) h_reads.push_back(at);
    }
    REQUIRE_EQ(h_reads.size(), (size_t)1);
    CHECK_EQ(h_reads[0], first_issue(m.layers[kSplit].hc_mix.op_read));
    REQUIRE_EQ(x_reads.size(), (size_t)(8 - kSplit));
    for (int l = kSplit; l < 8; ++l)
        CHECK_EQ(x_reads[(size_t)(l - kSplit)], first_issue(m.layers[(size_t)l].hc_mix.op_read) + 1);
    const std::string head = "chunk.p4100.h";
    std::string descr;
    const std::filesystem::path bnd = find_file(dir.path, head, ".boundary.npy");
    REQUIRE(!bnd.empty());
    CHECK(npy_shape(bnd, &descr) == std::vector<int64_t>({16, 10240}));
    CHECK_EQ(descr, std::string("<u2"));
    for (int l = kSplit; l < 8; ++l) {
        const std::filesystem::path bi = find_file(dir.path, head, ".bi." + std::to_string(l) + ".npy");
        REQUIRE(!bi.empty());
        CHECK(npy_shape(bi, &descr) == std::vector<int64_t>({16, 2560}));
    }
    CHECK(find_file(dir.path, head, ".bi.3.npy").empty());
    const std::filesystem::path rows = find_file(dir.path, head, ".rows.npy");
    REQUIRE(!rows.empty());
    CHECK(npy_shape(rows, &descr) == std::vector<int64_t>({16}));
    CHECK_EQ(descr, std::string("<i4"));
    const std::string rb = slurp(rows);
    int32_t first = -1, second = -1;
    std::memcpy(&first, rb.data() + rb.size() - 64, 4);
    std::memcpy(&second, rb.data() + rb.size() - 60, 4);
    CHECK_EQ(first, 4);    /* 4100 + 4 = 4104 is the first multiple of 8 */
    CHECK_EQ(second, 12);
    CHECK(npy_shape(find_file(dir.path, head, ".ids.npy"), &descr) == std::vector<int64_t>({128}));
    CHECK(npy_shape(find_file(dir.path, head, ".pos.npy"), &descr) == std::vector<int64_t>({128}));
    const std::string line = slurp(dir.path / "capture.jsonl");
    CHECK_EQ(count(line, "\n"), 1);
    CHECK(has(line, "\"chunk_start\": 4100") && has(line, "\"split\": 4") && has(line, "\"rows\": 16") &&
          has(line, "\"layers\": [4, 5, 6, 7]") && has(line, "\"stride\": 8"));
    /* rank 1, and a two-sequence step: the stock step, nothing read */
    const Run r1 = run_step(qwen4exp_kva::step, bk.b, 1);
    CHECK_EQ(r1.device_calls, 0);
    Batch two = make_step(kva, {{128}, 0, 300, 4100});
    two.b.n_seq = 2;
    CHECK_EQ(run_step(qwen4exp_kva::step, two.b).device_calls, 0);
    CHECK_EQ(count(slurp(dir.path / "capture.jsonl"), "\n"), 1);
}


/* RADIANCE_KVA_CAPTURE_STATE: on an approximate chunk each late delta-net layer's slot is copied
 * right after its scan and BEFORE the correction's apply; on an exact chunk after the step. Each is
 * one kva_state_read issue on the layer's own slot, and the step's other issues are unchanged. */
TEST(state_capture_copies_each_late_state_before_the_apply) {
    TempDir dir;
    REQUIRE(!dir.path.empty());
    Pair p;
    declare_pair(p, "speed");
    REQUIRE_EQ(p.st, RAD_OK);
    std::vector<std::vector<RecIssue>> plain;   /* the same chunks without the switch */
    for (int64_t ahead : {2048, 100}) {
        Batch ref = make_step(p.kva, {{128}, 0, ahead, 2048});
        plain.push_back(run_step(qwen4exp_kva::step, ref.b).issues);
    }
    const RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    RadBuilder kva;
    served(kva);
    hold_kva(kva, {"kva.proj", "kva.st"});
    hold_score(kva, "kva.rowsel.score");
    Env env({{"RADIANCE_KVA", "speed"}, {"RADIANCE_KVA_CAPTURE_STATE", dir.path.c_str()}});
    REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
    REQUIRE(k.op_state_read != 0);
    CHECK(kva.concurrent.count(k.b_state) == 1);
    for (int64_t ahead : {2048, 100}) {   /* approximate, then exact */
        Batch bk = make_step(kva, {{128}, 0, ahead, 2048});
        const std::vector<RecIssue>& want = plain[ahead == 2048 ? 0 : 1];
        const Run got = run_step(qwen4exp_kva::step, bk.b);
        std::vector<RecIssue> reads, rest;
        for (const RecIssue& i : got.issues) (i.op == k.op_state_read ? reads : rest).push_back(i);
        CHECK_EQ(differ(rest, want), 0);    /* the sequence without the copies is the plain one */
        REQUIRE_EQ(reads.size(), (size_t)3);
        for (size_t j = 0; j < reads.size(); ++j) {
            const int l = std::vector<int>{4, 5, 6}[j];
            CHECK_EQ(reads[j].n, 1);
            CHECK(same_operand(reads[j].opd[0], kv_cache(m.kv_state, l)));
            CHECK(same_operand(reads[j].opd[1], praw2(bk.b.kv[m.kv_state - 1].state_index, RAD_I32, 1, 3)));
            CHECK_EQ(reads[j].opd[2].handle, k.b_state);
        }
        if (ahead == 2048) {   /* each copy sits between its layer's scan and its apply */
            for (int l : {4, 5, 6}) {
                size_t scan = 0, read = 0, apply = 0;
                for (size_t i = 0; i < got.issues.size(); ++i) {
                    if (got.issues[i].op == m.layers[(size_t)l].gdn.op_scan) scan = i;
                    if (got.issues[i].op == k.op_state_read &&
                        same_operand(got.issues[i].opd[0], kv_cache(m.kv_state, l))) read = i;
                    if (got.issues[i].op == k.op_apply[(size_t)l]) apply = i;
                }
                CHECK(scan < read && read < apply);
                CHECK_EQ(read, scan + 1);
            }
        }
    }
    const std::string lines = slurp(dir.path / "state.jsonl");
    CHECK_EQ(count(lines, "\n"), 2);
    CHECK_EQ(count(lines, "\"approximate\": true"), 1);
    CHECK_EQ(count(lines, "\"approximate\": false"), 1);
    CHECK(has(lines, "\"layers\": [4, 5, 6]") && has(lines, "\"last_position\": 2175"));
    std::string descr;
    const std::filesystem::path f = find_file(dir.path, "state.p2048.h", ".r0.npy");
    REQUIRE(!f.empty());
    CHECK(npy_shape(f, &descr) == std::vector<int64_t>({3, m.gcfg.n_head_v, 128, 128}));
    CHECK_EQ(descr, std::string("<f4"));
}


/* Speed declares the projector per late layer and one quantiser, after the whole in-tree graph and

/* Every op a serving mode adds is declared after the whole in-tree graph and nothing in between:
 * the in-tree op list is a prefix of the KVA one. Speed adds the projector per late layer, one
 * quantiser, the correction pair per late delta-net layer, the mask, the stream copy, the select,
 * the drop and one alternate down handle per routed layer from S-1. */
TEST(speed_adds_its_ops_after_the_in_tree_graph) {
    Pair p;
    declare_pair(p, "speed");
    REQUIRE_EQ(p.st, RAD_OK);
    REQUIRE(p.kva.ops.size() > p.stock.ops.size());
    for (size_t i = 0; i < p.stock.ops.size(); ++i) CHECK_EQ(p.kva.ops[i].op, p.stock.ops[i].op);
    std::map<std::string, int> n;
    for (size_t i = p.stock.ops.size(); i < p.kva.ops.size(); ++i) {
        const RecOp& o = p.kva.ops[i];
        ++n[o.op];
        if (o.op != "gemm_nt_bias") continue;
        for (const RecParam& q : o.p) {
            if (q.key == "N") CHECK_EQ(q.ival, 2560LL);
            if (q.key == "K") CHECK_EQ(q.ival, 10240LL);
            if (q.key == "M") CHECK_EQ(q.ihi, 2048LL);
        }
    }
    CHECK_EQ(n["gemm_nt_bias"], 8 - kSplit);
    CHECK_EQ(n["quant_act_i8g"], 1);
    CHECK_EQ(n["kva_state_correct"], 2 * 3);
    CHECK_EQ(n["kva_mask"], 1);
    CHECK_EQ(n["cast"], 1);
    CHECK_EQ(n["kva_select"], 1);
    CHECK_EQ(n["kva_drop_rows"], 1);
    CHECK_EQ(n["moe_gemm_q"], 0);
    CHECK_EQ(n["kva_rho_update"], 0);
    int total = 0;
    for (const auto& [op, c] : n) total += c;
    CHECK_EQ(total, (int)(p.kva.ops.size() - p.stock.ops.size()));
}

/* A SIZING DECLARE describes the same graph with only rows smaller and leaves the step's state
 * alone (radiance core/engine_bringup.cpp:598-660). Its weight list is the real one name for name
 * -- which is what lets the masked path reuse the real declare's in-tree weight handles (A_log,
 * dt_bias, the experts' tables) under a sizing declare -- and its buffers are the real ones with
 * only dim 0 smaller. */
TEST(a_sizing_declare_matches_the_real_one) {
    for (const char* mode : {"plumb", "speed", "quality"}) {
        Pair p;
        declare_pair(p, mode);
        REQUIRE_EQ(p.st, RAD_OK);
        RadModelMeta meta = flash_next_meta();
        RadBuildCtx c = served_ctx();
        c.max_tok = 256;
        c.shape_probe = 1;
        RadBuilder small;
        served(small);
        hold_kva(small, {"kva.proj", "kva.st"});
        hold_score(small, "kva.rowsel.score");
        Env env({{"RADIANCE_KVA", mode}});
        REQUIRE_EQ(qwen4exp_kva::declare(&small, &meta, &c), RAD_OK);
        REQUIRE_EQ(small.ops.size(), p.kva.ops.size());
        int fixed_differ = 0;
        for (size_t i = 0; i < small.ops.size(); ++i) {
            CHECK_EQ(small.ops[i].op, p.kva.ops[i].op);
            REQUIRE_EQ(small.ops[i].p.size(), p.kva.ops[i].p.size());
            CHECK(small.ops[i].w == p.kva.ops[i].w);
            for (size_t j = 0; j < small.ops[i].p.size(); ++j) {
                const RecParam& a = p.kva.ops[i].p[j];
                const RecParam& q = small.ops[i].p[j];
                const bool cap = (p.kva.ops[i].op == "qsa_work" && a.key == "work") ||
                                 (p.kva.ops[i].op == "attn_paged_gate_quant" && a.key == "max_seqs");
                if (a.kind == RAD_P_RANGE || (cap && q.ival <= a.ival)) continue;
                fixed_differ += a.ival != q.ival || a.sval != q.sval || a.dval != q.dval;
            }
        }
        CHECK_EQ(fixed_differ, 0);
        REQUIRE_EQ(small.weights.size(), p.kva.weights.size());
        for (size_t i = 0; i < small.weights.size(); ++i)
            CHECK_EQ(small.weights[i].first, p.kva.weights[i].first);
        REQUIRE_EQ(small.bufs.size(), p.kva.bufs.size());
        for (size_t i = 0; i < small.bufs.size(); ++i) {
            CHECK_EQ(small.bufs[i].first, p.kva.bufs[i].first);
            CHECK(small.bufs[i].second.shape[0] <= p.kva.bufs[i].second.shape[0]);
            for (uint32_t d = 1; d < p.kva.bufs[i].second.rank; ++d)
                CHECK_EQ(small.bufs[i].second.shape[d], p.kva.bufs[i].second.shape[d]);
        }
        CHECK(small.kv_groups == p.kva.kv_groups);
        CHECK_EQ(qwen4exp_fp8::g_model[0].g.max_tok, 2048);
        CHECK_EQ(qwen4exp_kva::g_kva[0].op_proj.size(), (size_t)8);
        CHECK(qwen4exp_kva::g_kva[0].b_zeros != 0);
    }
}

/* ==================================================================== refusals (R31) */

int refused(std::initializer_list<std::pair<const char*, const char*>> env, RadBuilder& b,
            std::string* err, int64_t max_tok = 2048) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    c.max_tok = max_tok;
    Env e(env);
    int st = RAD_OK;
    *err = stderr_of([&] { st = qwen4exp_kva::declare(&b, &meta, &c); });
    return st;
}

/* R84: a tail longer than a step refuses with both numbers and names the operator route. */
TEST(a_tail_longer_than_a_step_is_refused_with_both_numbers) {
    RadBuilder b;
    hold_kva(b, {"kva.proj", "kva.st"});
    std::string err;
    CHECK(refused({{"RADIANCE_KVA", "speed"}, {"RADIANCE_KVA_TAIL", "4096"}}, b, &err) < 0);
    CHECK(has(err, "4096") && has(err, "2048"));
    RadBuilder b2;
    hold_kva(b2, {"kva.proj", "kva.st"});
    CHECK(refused({{"RADIANCE_KVA", "speed"}}, b2, &err, 1024) < 0);
    CHECK(has(err, "2048") && has(err, "1024"));
    RadBuilder b3;
    hold_kva(b3, {"kva.proj", "kva.st"});
    CHECK_EQ(refused({{"RADIANCE_KVA", "speed"}, {"RADIANCE_KVA_TAIL", "1024"}}, b3, &err, 1024), RAD_OK);
}

TEST(a_tail_below_the_measured_minimum_is_refused) {
    RadBuilder b;
    hold_kva(b, {"kva.proj"});
    std::string err;
    CHECK(refused({{"RADIANCE_KVA", "speed"}, {"RADIANCE_KVA_TAIL", "256"}}, b, &err) < 0);
    CHECK(has(err, "256") && has(err, "512"));
}

TEST(quality_without_the_row_table_is_refused) {
    RadBuilder b;
    hold_kva(b, {"kva.proj", "kva.st"});
    std::string err;
    CHECK(refused({{"RADIANCE_KVA", "quality"}}, b, &err) < 0);
    CHECK(has(err, "kva.rowsel.score"));
}

/* Each kva.so op a mode issues is refused by name when no kernel library serves it -- plumb's mask
 * included. (`cast` and `moe_gemm_q` are in-tree ops too: refusing them fails the in-tree declare
 * first, which is its own refusal.) */
TEST(a_mode_whose_kva_op_no_kernel_serves_is_refused_by_name) {
    struct Case { const char* mode; const char* op; };
    for (const Case& c : {Case{"speed", "kva_state_correct"}, Case{"quality", "kva_mask"},
                          Case{"plumb", "kva_mask"}, Case{"speed", "kva_select"},
                          Case{"speed", "kva_drop_rows"}, Case{"quality", "kva_rho_update"}}) {
        RadBuilder b;
        hold_kva(b, {"kva.proj", "kva.st"});
        hold_score(b, "kva.rowsel.score");
        b.refuse = {c.op};
        std::string err;
        CHECK_EQ(refused({{"RADIANCE_KVA", c.mode}}, b, &err), RAD_E_UNSUPPORTED);
        CHECK(has(err, c.op) && has(err, "kva.so"));
    }
}

TEST(a_mode_with_no_projector_held_is_refused) {
    RadBuilder b;
    std::string err;
    CHECK_EQ(refused({{"RADIANCE_KVA", "plumb"}}, b, &err), RAD_E_UNSUPPORTED);
    CHECK(has(err, "kva.proj"));
}

TEST(a_projector_with_a_hole_is_refused) {
    RadBuilder b;
    for (int l : {4, 5, 7})
        b.encs.push_back({"kva.proj." + std::to_string(l), rad_enc_plain(RAD_BF16)});
    std::string err;
    CHECK(refused({{"RADIANCE_KVA", "speed"}}, b, &err) < 0);
    CHECK(has(err, "kva.proj.6.weight"));
}

TEST(a_switch_with_an_unknown_value_is_refused_naming_the_values) {
    struct Case { const char* name; const char* value; const char* allowed; };
    for (const Case& c : {Case{"RADIANCE_KVA", "fast", "off|plumb|speed|quality"},
                          Case{"RADIANCE_KVA_ST", "flipped", "shipped|swap|refit"},
                          Case{"RADIANCE_KVA_STAGE", "never", "auto|stock"},
                          Case{"RADIANCE_KVA_STRADDLE", "middle", "split|end"},
                          Case{"RADIANCE_KVA_SCORE_BULK", "yes", "1 or unset"},
                          Case{"RADIANCE_KVA_FORCE_SPLIT", "0", "a positive row count"},
                          Case{"RADIANCE_KVA_FORCE_STREAM", "2", "1 or unset"}}) {
        RadBuilder b;
        hold_kva(b, {"kva.proj"});
        std::string err;
        const bool mode = !std::strcmp(c.name, "RADIANCE_KVA");
        CHECK_EQ(mode ? refused({{c.name, c.value}}, b, &err)
                      : refused({{"RADIANCE_KVA", "speed"}, {c.name, c.value}}, b, &err), RAD_E_INVAL);
        CHECK(has(err, c.allowed) && has(err, c.name));
    }
}

/* R81 -- THE DEFAULT IS OFF AND THE CONTAINER'S kva.mode IS NOT A SWITCH: a container that still says
 * kva.mode=quality, with RADIANCE_KVA unset, declares exactly the in-tree graph, and rank 0's real
 * declare says once that the metadata mode is ignored. */
TEST(the_container_mode_is_ignored_and_said_so) {
    static const char* keys[kN + 1];
    static const char* vals[kN + 1];
    for (int i = 0; i < kN; ++i) { keys[i] = kKeys[i]; vals[i] = kVals[i]; }
    keys[kN] = "kva.mode";
    vals[kN] = "quality";
    RadModelMeta meta = flash_next_meta();
    meta.n_kv = kN + 1; meta.kv_key = keys; meta.kv_val = vals;
    for (int rank : {0, 1}) {
        RadBuildCtx c = served_ctx(rank, 2);
        RadBuilder stock, kva;
        served(stock);
        served(kva);
        hold_kva(kva, {"kva.proj", "kva.st"});
        hold_score(kva, "kva.rowsel.score");
        REQUIRE_EQ(qwen4exp_fp8::declare(&stock, &meta, &c), RAD_OK);
        int st = RAD_E_INVAL;
        const std::string err = stderr_of([&] { st = qwen4exp_kva::declare(&kva, &meta, &c); });
        REQUIRE_EQ(st, RAD_OK);
        check_same_graph(stock, kva);
        CHECK_EQ(qwen4exp_kva::g_kva[rank].cfg.mode, (int)qwen4exp_kva::MODE_OFF);
        CHECK_EQ(count(err, "kva.mode=quality is ignored"), rank == 0 ? 1 : 0);
    }
}

/* The fill feeds blocks their input already normed and skips the n-gram layer: a split at or below
 * the PLE layer is refused (the projector would predict from a stream without the PLE). */
TEST(a_split_at_or_below_the_ple_layer_is_refused) {
    RadBuilder b;
    for (int l = 1; l < 8; ++l)
        b.encs.push_back({"kva.proj." + std::to_string(l), rad_enc_plain(RAD_BF16)});
    std::string err;
    CHECK_EQ(refused({{"RADIANCE_KVA", "speed"}}, b, &err), RAD_E_UNSUPPORTED);
    CHECK(has(err, "n-gram"));
}

TEST(capture_refuses_a_serving_mode) {
    RadBuilder b;
    hold_kva(b, {"kva.proj"});
    std::string err;
    CHECK_EQ(refused({{"RADIANCE_KVA", "speed"}, {"RADIANCE_KVA_CAPTURE", "x"}}, b, &err), RAD_E_INVAL);
    CHECK(has(err, "RADIANCE_KVA=off"));
}

TEST(state_capture_without_kva_state_read_is_refused) {
    RadBuilder b;
    hold_kva(b, {"kva.proj"});
    b.refuse = {"kva_state_read"};
    std::string err;
    CHECK_EQ(refused({{"RADIANCE_KVA_CAPTURE_STATE", "x"}}, b, &err), RAD_E_UNSUPPORTED);
    CHECK(has(err, "kva_state_read"));
}

/* ==================================================================== the release guard (R83) */

/* The scan counts NUL-delimited copies only: a copy inside a longer string, or at a file's very
 * start with no NUL in front, is not one; two adjacent copies sharing a NUL are two; a copy that
 * straddles the scan's 1 MiB block boundary is found once. An unreadable file is -1. */
TEST(the_release_scan_counts_nul_delimited_copies) {
    TempDir dir;
    REQUIRE(!dir.path.empty());
    auto write = [&](const char* name, const std::string& bytes) {
        std::ofstream(dir.path / name, std::ios::binary) << bytes;
        return (dir.path / name).string();
    };
    const std::string z(1, '\0');
    CHECK_EQ(qwen4exp_kva::count_version(write("one", "abc" + z + "1.0.8" + z + "x").c_str(), "1.0.8"), 1);
    CHECK_EQ(qwen4exp_kva::count_version(write("none", "x" + z + "1.0.80" + z + "11.0.8" + z).c_str(), "1.0.8"), 0);
    CHECK_EQ(qwen4exp_kva::count_version(write("start", "1.0.8" + z + "q").c_str(), "1.0.8"), 0);
    CHECK_EQ(qwen4exp_kva::count_version(write("two", z + "1.0.8" + z + "1.0.8" + z).c_str(), "1.0.8"), 2);
    std::string big((1 << 20) - 3, 'a');
    big += z + "1.0.8" + z;
    CHECK_EQ(qwen4exp_kva::count_version(write("straddle", big).c_str(), "1.0.8"), 1);
    CHECK_EQ(qwen4exp_kva::count_version((dir.path / "absent").c_str(), "1.0.8"), -1);
}

/* The log names the releases the engine does carry: every NUL-delimited d.d.d string, once each. */
TEST(the_found_releases_are_listed) {
    TempDir dir;
    REQUIRE(!dir.path.empty());
    const std::string z(1, '\0'), p = (dir.path / "e").string();
    std::ofstream(p, std::ios::binary) << "x" + z + "1.0.9" + z + "0.46.1" + z + "1.0" + z + "v1.2.3" + z +
                                          "1..2" + z + "1.0.9" + z;
    CHECK_EQ(qwen4exp_kva::releases_in(p.c_str()), std::string("1.0.9, 0.46.1"));
}

/* FIPS 180-4's own vectors, so a logged sha256 names the binary it claims to. */
TEST(sha256_matches_the_fips_vectors) {
    TempDir dir;
    REQUIRE(!dir.path.empty());
    auto sha = [&](const std::string& bytes) {
        const std::string p = (dir.path / "f").string();
        std::ofstream(p, std::ios::binary) << bytes;
        return qwen4exp_kva::sha256_file(p.c_str());
    };
    CHECK_EQ(sha(""), std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    CHECK_EQ(sha("abc"), std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
    CHECK_EQ(sha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
             std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
}

/* With no forward taken the plugin serves itself; the forward table is empty at load. */
TEST(the_forward_table_starts_empty) {
    CHECK(qwen4exp_kva::g_forward.declare == nullptr && qwen4exp_kva::g_forward.step == nullptr &&
          qwen4exp_kva::g_forward.probe == nullptr);
    CHECK(qwen4exp_kva::find_shadowed("").empty() || std::getenv("RADIANCE_HOME") != nullptr);
}

RAD_TEST_MAIN()
