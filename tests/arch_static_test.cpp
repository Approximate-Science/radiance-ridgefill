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
#include <functional>
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
int   rad_stream_sync(RadStream) { ++g_ctx->device_calls; return RAD_E_UNSUPPORTED; }
int   rad_memcpy_async(void*, const void*, int64_t, RadStream) { ++g_ctx->device_calls; return RAD_E_UNSUPPORTED; }
int   rad_memset_async(void*, int, int64_t, RadStream) { ++g_ctx->device_calls; return RAD_E_UNSUPPORTED; }
}  /* extern "C" */

/* ==================================================================== the plugin under test */
#include "qwen4exp_kva.cpp"

using rad::arch::kv_cache;
using rad::arch::praw2;

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

/* Quality declares the selected row table and the kva_rowsel op over it, with the cap the share
 * derives (0.25 * 2048 = 512) and the rule the switch names. */
TEST(quality_declares_the_selected_row_table_and_kva_rowsel) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    Env env({{"RADIANCE_KVA", "quality"}, {"RADIANCE_KVA_ROWSEL_TABLE", "none"},
             {"RADIANCE_KVA_ROWSEL", "random"}});
    RadBuilder kva;
    hold_kva(kva, {"kva.proj", "kva.st"});
    for (const char* t : {"kva.rowsel.score", "kva.rowsel.score_none", "kva.rowsel.score_all"})
        hold_score(kva, t);
    int st = RAD_OK;
    const std::string err = stderr_of([&] { st = qwen4exp_kva::declare(&kva, &meta, &c); });
    CHECK_EQ(st, RAD_E_UNSUPPORTED);
    CHECK(has(err, "not implemented"));
    CHECK(weight(kva, "kva.rowsel.score_none") != nullptr);
    CHECK(weight(kva, "kva.rowsel.score") == nullptr);
    CHECK(weight(kva, "kva.rowsel.score_all") == nullptr);
    int rowsel = 0;
    for (const RecOp& o : kva.ops) {
        if (o.op != "kva_rowsel") continue;
        ++rowsel;
        REQUIRE_EQ(o.w.size(), (size_t)1);
        CHECK_EQ(kva.weights[o.w[0] - 1].first, std::string("kva.rowsel.score_none"));
        for (const RecParam& p : o.p) {
            if (p.key == "cap")  CHECK_EQ(p.ival, 512LL);
            if (p.key == "mode") CHECK_EQ(p.sval, std::string("random"));
            if (p.key == "M")    CHECK_EQ(p.ihi, 2048LL);
        }
    }
    CHECK_EQ(rowsel, 1);
}

/* ==================================================================== the fill path (Stage 3) */

/* ONE prefill chunk of one sequence, T rows, with `ahead` prompt tokens after it and `ctx` before.
 * 128 rows is past qk_fuse_rows (64), where the stock attention takes the unfused prologue too. */
Batch one_prefill(const RadBuilder& bld, int64_t T, int64_t ahead, int32_t ctx) {
    static int32_t ids[2048], pos[2048], cu[2] = {0, 0}, slot[2048], table[4096];
    static int32_t used[1], state[3], qlen[1], ctxl[1], acc[1];
    Batch x = make_batch(bld, true);
    for (RadKVGroupBatch& k : x.kv) {
        k.slot_mapping = slot; k.block_table = table; k.block_table_pitch = 1024;
        k.seqused = used; k.state_index = state; k.state_index_pitch = 3; k.max_blocks = 1024;
    }
    cu[1] = (int32_t)T;
    RadBatch& b = x.b;
    b.kv = x.kv.data();
    b.n_tok = T; b.n_seq = 1; b.n_ahead = ahead;
    b.token_ids = ids; b.positions = pos; b.cu_seqlens = cu;
    b.n_out = 0; b.out_ids = nullptr;
    b.q_lens = qlen; b.ctx_lens = ctxl; b.num_accepted = acc;
    b.max_q_len = (int32_t)T; b.max_ctx_len = ctx;
    return x;
}

struct Run {
    std::vector<RecIssue> issues;
    std::string           log;
    int                   device_calls = 0;
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
    return r;
}

int count(const std::string& s, const char* what) {
    int n = 0;
    for (size_t at = s.find(what); at != std::string::npos; at = s.find(what, at + 1)) ++n;
    return n;
}

/* Where each late layer's issues start: its connection read in a stock or plumb run, its projector
 * in a speed run; the mixer's read ends the last one. */
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

void add_linear(std::set<rad_op>& s, const rad::arch::LinearFP8& l) {
    for (rad_op h : {l.op, l.op_q, l.op_m1}) if (h) s.insert(h);
}

/* The handles of the pieces the fill issues for late layer `l`: the delta net's projections and
 * recurrence, or the indexer's block-key half (`keys`) and the attention's K/V path (`kv`). */
struct Pieces { std::set<rad_op> gdn, keys, select, kv; };
Pieces pieces(const qwen4exp_fp8::Layer& l) {
    Pieces p;
    add_linear(p.gdn, l.gdn.in);
    for (rad_op h : {l.gdn.op_ab, l.gdn.op_conv_prep, l.gdn.op_kkt, l.gdn.op_scan}) p.gdn.insert(h);
    add_linear(p.keys, l.qsa.proj);
    for (rad_op h : {l.qsa.op_work, l.qsa.op_bkey, l.qsa.op_tail}) p.keys.insert(h);
    for (rad_op h : {l.qsa.op_qprep, l.qsa.op_norm, l.qsa.op_rope, l.qsa.op_score, l.qsa.op_select})
        if (h) p.select.insert(h);
    add_linear(p.kv, l.attn.kp);
    add_linear(p.kv, l.attn.vp);
    for (rad_op h : {l.attn.op_k_norm, l.attn.op_rope_k, l.attn.op_kv_store}) p.kv.insert(h);
    return p;
}

/* Declares stock and KVA side by side on the served formats with every kva.* tensor held. */
struct Pair {
    RadBuilder stock, kva;
    int        st = RAD_OK;
};
void declare_pair(Pair& p, const char* mode, int rank = 0, int world = 1) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx(rank, world);
    served(p.stock);
    served(p.kva);
    hold_kva(p.kva, {"kva.proj", "kva.st"});
    REQUIRE_EQ(qwen4exp_fp8::declare(&p.stock, &meta, &c), RAD_OK);
    Env env({{"RADIANCE_KVA", mode}});
    p.st = qwen4exp_kva::declare(&p.kva, &meta, &c);
}

/* ONLY a prefill chunk of ONE sequence with T prompt tokens after it is filled; a mixed step, two
 * prefills, a draft pass, a speculative step and a chunk inside the last T all issue the stock
 * sequence exactly. The approximate one logs its one line on rank 0 alone. */
TEST(only_single_sequence_bulk_chunks_are_approximated) {
    Pair p;
    declare_pair(p, "speed");
    REQUIRE_EQ(p.st, RAD_OK);
    const std::vector<std::function<void(RadBatch&)>> stock_shapes = {
        [](RadBatch& b) { b.n_ahead = 2047; },                                  /* inside the tail */
        [](RadBatch& b) { b.n_seq = 2; },                                       /* two prefills */
        [](RadBatch& b) { b.phase = RAD_PHASE_MIXED; b.n_seq = 2; b.n_seq_decode = 1; b.n_tok_decode = 1; },
        [](RadBatch& b) { b.n_spec = 1; },                                      /* a verify step */
        [](RadBatch& b) { b.phase = RAD_PHASE_DECODE; },                        /* decode by phase */
    };
    for (const auto& shape : stock_shapes) {
        Batch bs = one_prefill(p.stock, 128, 2048, 128);
        Batch bk = one_prefill(p.kva, 128, 2048, 128);
        shape(bs.b);
        shape(bk.b);
        const Run want = run_step(qwen4exp_fp8::step, bs.b);
        const Run got = run_step(qwen4exp_kva::step, bk.b);
        CHECK_EQ(differ(got.issues, want.issues), 0);
        CHECK_EQ(count(got.log, "kva: approximate step"), 0);
    }
    Batch bk = one_prefill(p.kva, 128, 2048, 128);
    const Run r0 = run_step(qwen4exp_kva::step, bk.b, 0);
    const Run r1 = run_step(qwen4exp_kva::step, bk.b, 1);
    CHECK_EQ(count(r0.log, "kva: approximate step"), 1);
    CHECK_EQ(count(r1.log, "kva: approximate step"), 0);
    CHECK_EQ(r0.device_calls, 0);
}

/* PLUMB IS THE STOCK STEP, issue for issue and operand for operand, with exactly two moves, both
 * inside one late attention layer's block: the indexer's query prep and selection after its
 * block-key half, and the attention's K/V path ahead of its query path. Every other position --
 * the prologue, layers below S, each late delta-net layer, every connection and MoE, the
 * epilogue -- is identical. Dense (short context) and sparse (past the QSA exactness bound). */
TEST(plumb_reproduces_the_stock_step_with_two_moves_inside_a_block) {
    Pair p;
    declare_pair(p, "plumb");
    REQUIRE_EQ(p.st, RAD_OK);
    CHECK_EQ(p.kva.ops.size(), p.stock.ops.size());   /* plumb adds no op */
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    for (int32_t ctx : {128, 4096}) {
        Batch bs = one_prefill(p.stock, 128, 2048, ctx);
        Batch bk = one_prefill(p.kva, 128, 2048, ctx);
        const Run want = run_step(qwen4exp_fp8::step, bs.b);
        const Run got = run_step(qwen4exp_kva::step, bk.b);
        REQUIRE_EQ(got.issues.size(), want.issues.size());
        std::vector<rad_op> reads;
        for (int l = kSplit; l < 8; ++l) reads.push_back(m.layers[(size_t)l].hc_mix.op_read);
        const std::vector<size_t> at = starts(want.issues, reads, m.mixer.op_read);
        CHECK_EQ(differ(slice(got.issues, 0, at[0]), slice(want.issues, 0, at[0])), 0);
        CHECK_EQ(differ(slice(got.issues, at.back(), got.issues.size()),
                        slice(want.issues, at.back(), want.issues.size())), 0);
        for (int l = kSplit; l < 8; ++l) {
            const qwen4exp_fp8::Layer& lay = m.layers[(size_t)l];
            std::vector<RecIssue> seg = slice(want.issues, at[(size_t)(l - kSplit)], at[(size_t)(l - kSplit + 1)]);
            if (lay.full) {
                /* the documented order: read, keys, select, K/V, the rest of the block, the rest */
                const Pieces pc = pieces(lay);
                const rad_op write = lay.hc_mix.op_write;
                bool after_write = false;
                std::vector<std::pair<int, RecIssue>> keyed;
                for (const RecIssue& i : seg) {
                    after_write = after_write || i.op == write;
                    const int g = i.op == lay.hc_mix.op_read ? 0 : pc.keys.count(i.op) ? 1
                                : pc.select.count(i.op) ? 2 : pc.kv.count(i.op) ? 3 : after_write ? 5 : 4;
                    keyed.push_back({g, i});
                }
                std::stable_sort(keyed.begin(), keyed.end(),
                                 [](const auto& a, const auto& b) { return a.first < b.first; });
                seg.clear();
                for (const auto& k : keyed) seg.push_back(k.second);
            }
            CHECK_EQ(differ(slice(got.issues, at[(size_t)(l - kSplit)], at[(size_t)(l - kSplit + 1)]), seg), 0);
        }
        CHECK_EQ(count(got.log, "kva: approximate step"), 1);
    }
}

/* SPEED FILLS A LATE LAYER WITH ITS CACHE-WRITING OPS AND NOTHING ELSE: the projector into `x`,
 * the quantiser, then the stock block's own delta-net projections and recurrence -- or indexer
 * block-key half and K/V path -- in stock order with stock operands. No connection read or
 * write, no MoE, no query path, no attention, no output projection for layers >= S. Below S and
 * after the late layers the step is the stock one. */
TEST(speed_fills_late_layers_with_their_cache_writing_ops_only) {
    Pair p;
    declare_pair(p, "speed");
    REQUIRE_EQ(p.st, RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
    REQUIRE(k.quant.op != 0);                          /* the served formats need codes */
    CHECK(m.a_x.q8_fed);
    for (rad_buf h : {m.b_h, m.a_x.x, m.a_x.cq(), m.a_x.cs()}) CHECK(p.kva.concurrent.count(h) == 1);
    Batch bs = one_prefill(p.stock, 128, 2048, 4096);
    Batch bk = one_prefill(p.kva, 128, 2048, 4096);
    const Run want = run_step(qwen4exp_fp8::step, bs.b);
    const Run got = run_step(qwen4exp_kva::step, bk.b);
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
            /* Stage 4: undo right before the conv/scan, apply right after, on this layer's slots. */
            const RadOperand state = kv_cache(m.kv_state, l), applied = kv_cache(k.kv_applied, l);
            const RadOperand idx = praw2(bk.b.kv[m.kv_state - 1].state_index, RAD_I32, 1, 3);
            RadOperand heads = RAD_W(k.st[(size_t)l]);   /* rank 0 of 1: every head, from 0 */
            heads.rows = m.gcfg.n_head_v;
            const RecIssue undo{k.op_undo[(size_t)l], {state, applied, idx, heads, RAD_NONE}, 1};
            const RecIssue apply{k.op_apply[(size_t)l], {state, applied, idx, heads, RAD_NONE}, 1};
            size_t conv = 0;
            while (conv < keep.size() && keep[conv].op != lay.gdn.op_conv_prep) ++conv;
            keep.insert(keep.begin() + (long)conv, undo);
            keep.push_back(apply);   /* the scan is the block's last cache-writing piece */
        }
        CHECK_EQ(differ(slice(seg, 2, seg.size()), keep), 0);
    }
    CHECK_EQ(count(got.log, "kva: approximate step"), 1);
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
        Batch bk = one_prefill(kva, 128, 2048, 128);
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
        Batch bk = one_prefill(p.kva, 128, 2048, 128);
        const Run r = run_step(qwen4exp_kva::step, bk.b, rank);
        int seen = 0;
        for (const RecIssue& i : r.issues) {
            if (i.op != k.op_undo[kSplit] && i.op != k.op_apply[kSplit]) continue;
            ++seen;
            REQUIRE_EQ(i.opd.size(), (size_t)5);
            CHECK_EQ(i.opd[3].handle, k.st[kSplit]);
            CHECK_EQ(i.opd[3].rows, m.gcfg.n_head_v);
            CHECK_EQ(i.opd[3].offset, (int64_t)rank * m.gcfg.n_head_v * m.gcfg.head_v * m.gcfg.head_k);
        }
        CHECK_EQ(seen, 2);
    }
}

/* plumb runs the late layers exactly and issues no correction, so it places none. */
TEST(plumb_declares_no_correction) {
    Pair p;
    declare_pair(p, "plumb");
    REQUIRE_EQ(p.st, RAD_OK);
    CHECK(weight(p.kva, "kva.st.4") == nullptr);
    CHECK(weight(p.kva, "kva.proj.4.weight") != nullptr);
}

/* Speed declares the projector per late layer and one quantiser, after the whole in-tree graph and
 * nothing in between: the in-tree op list is a prefix of the KVA one. */
TEST(speed_adds_its_ops_after_the_in_tree_graph) {
    Pair p;
    declare_pair(p, "speed");
    REQUIRE_EQ(p.st, RAD_OK);
    REQUIRE(p.kva.ops.size() > p.stock.ops.size());
    for (size_t i = 0; i < p.stock.ops.size(); ++i) CHECK_EQ(p.kva.ops[i].op, p.stock.ops[i].op);
    int proj = 0, quant = 0, correct = 0;
    for (size_t i = p.stock.ops.size(); i < p.kva.ops.size(); ++i) {
        const RecOp& o = p.kva.ops[i];
        proj += o.op == "gemm_nt_bias";
        quant += o.op == "quant_act_i8g";
        correct += o.op == "kva_state_correct";
        if (o.op != "gemm_nt_bias") continue;
        for (const RecParam& q : o.p) {
            if (q.key == "N") CHECK_EQ(q.ival, 2560LL);
            if (q.key == "K") CHECK_EQ(q.ival, 10240LL);
            if (q.key == "M") CHECK_EQ(q.ihi, 2048LL);
        }
    }
    CHECK_EQ(proj, 8 - kSplit);
    CHECK_EQ(quant, 1);
    CHECK_EQ(correct, 2 * 3);
}

/* A SIZING DECLARE describes the same graph with only rows smaller and leaves the step's state
 * alone (radiance core/engine_bringup.cpp:598-660); the cap stays the real declare's. */
TEST(a_speed_sizing_declare_matches_the_real_one) {
    Pair p;
    declare_pair(p, "speed");
    REQUIRE_EQ(p.st, RAD_OK);
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    c.max_tok = 256;
    c.shape_probe = 1;
    RadBuilder small;
    served(small);
    hold_kva(small, {"kva.proj", "kva.st"});
    Env env({{"RADIANCE_KVA", "speed"}});
    REQUIRE_EQ(qwen4exp_kva::declare(&small, &meta, &c), RAD_OK);
    REQUIRE_EQ(small.ops.size(), p.kva.ops.size());
    int fixed_differ = 0;
    for (size_t i = 0; i < small.ops.size(); ++i) {
        CHECK_EQ(small.ops[i].op, p.kva.ops[i].op);
        REQUIRE_EQ(small.ops[i].p.size(), p.kva.ops[i].p.size());
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
    CHECK_EQ(qwen4exp_fp8::g_model[0].g.max_tok, 2048);
    CHECK_EQ(qwen4exp_kva::g_kva[0].op_proj.size(), (size_t)8);
    CHECK(qwen4exp_kva::g_kva[0].op_proj[kSplit] != 0);
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

TEST(a_tail_longer_than_a_step_is_refused_with_both_numbers) {
    RadBuilder b;
    hold_kva(b, {"kva.proj", "kva.st"});
    std::string err;
    CHECK(refused({{"RADIANCE_KVA", "speed"}, {"RADIANCE_KVA_TAIL", "4096"}}, b, &err) < 0);
    CHECK(has(err, "4096") && has(err, "2048"));
    CHECK(!has(err, "not implemented"));
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

TEST(speed_with_a_correction_and_no_kva_so_is_refused_by_name) {
    RadBuilder b;
    hold_kva(b, {"kva.proj", "kva.st"});
    b.refuse = {"kva_state_correct"};
    std::string err;
    CHECK_EQ(refused({{"RADIANCE_KVA", "speed"}}, b, &err), RAD_E_UNSUPPORTED);
    CHECK(has(err, "kva_state_correct") && has(err, "kva.so"));
    CHECK(!has(err, "not implemented"));
}

TEST(quality_with_no_kva_so_is_refused_by_name) {
    RadBuilder b;
    hold_kva(b, {"kva.proj", "kva.st"});
    hold_score(b, "kva.rowsel.score");
    b.refuse = {"kva_rowsel"};
    std::string err;
    CHECK_EQ(refused({{"RADIANCE_KVA", "quality"}}, b, &err), RAD_E_UNSUPPORTED);
    CHECK(has(err, "kva_rowsel") && has(err, "kva.so"));
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
    RadBuilder b;
    std::string err;
    CHECK_EQ(refused({{"RADIANCE_KVA", "fast"}}, b, &err), RAD_E_INVAL);
    CHECK(has(err, "off|plumb|speed|quality"));
    RadBuilder b2;
    CHECK_EQ(refused({{"RADIANCE_KVA_ST", "flipped"}, {"RADIANCE_KVA", "speed"}}, b2, &err),
             RAD_E_INVAL);
    CHECK(has(err, "shipped|swap|refit"));
}

/* The share sweep's override re-derives the cap: ceil(share * max_tok / 64) * 64, never above
 * max_tok. */
TEST(the_row_cap_derives_from_the_share) {
    RadModelMeta meta = flash_next_meta();
    for (auto [share, want] : {std::pair<const char*, int64_t>{"0.25", 512}, {"0.10", 256},
                               {"0.5", 1024}, {"1", 2048}}) {
        Env env({{"RADIANCE_KVA_SHARE", share}});
        qwen4exp_kva::Config cfg;
        REQUIRE_EQ(qwen4exp_kva::read_config(&meta, 2048, &cfg), RAD_OK);
        CHECK_EQ(cfg.cap, want);
    }
    qwen4exp_kva::Config cfg;
    REQUIRE_EQ(qwen4exp_kva::read_config(&meta, 2000, &cfg), RAD_OK);
    CHECK_EQ(cfg.cap, (int64_t)512);   /* ceil(500 / 64) * 64 */
    Env bad({{"RADIANCE_KVA_SHARE", "1.5"}});
    CHECK_EQ(qwen4exp_kva::read_config(&meta, 2048, &cfg), RAD_E_INVAL);
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

RAD_TEST_MAIN()
