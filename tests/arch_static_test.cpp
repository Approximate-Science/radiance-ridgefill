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
    /* What rad_weight_encoding answers: the first key that is the source name, or a prefix of it
     * ending at a '.' -- so "kva.proj.4" holds kva.proj.4.weight and .bias, and "kva.rowsel.score"
     * does not hold kva.rowsel.score_none. */
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
        if (source && !std::strncmp(source, key.c_str(), n) && (!source[n] || source[n] == '.')) {
            *out = e;
            return RAD_OK;
        }
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
int rad_bind_layer_kv(RadBuilder*, int, rad_kvgroup) { return RAD_OK; }
int rad_decl_name_map(RadBuilder* b, const RadNameMap* m) { b->maps.push_back(m->declared); return RAD_OK; }
void rad_note(RadBuilder* b, const char* fmt, ...) {
    char buf[1024];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    b->notes.push_back(buf);
}
int rad_declare_logits(RadBuilder*, rad_buf) { return RAD_OK; }
int rad_buf_concurrent(RadBuilder*, rad_buf) { return RAD_OK; }
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
 * TP2 deployment and with the MTP head declared. */
TEST(off_declares_exactly_the_in_tree_graph) {
    RadModelMeta meta = flash_next_meta();
    for (int world : {1, 2})
        for (int rank = 0; rank < world; ++rank)
            for (int spec : {0, 3}) {
                RadBuildCtx c = served_ctx(rank, world);
                c.max_spec = spec;
                RadBuilder stock, kva;
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

/* `off` issues the in-tree step's sequence, op for op and operand for operand, for a prefill
 * batch and a decode batch, and touches the device API not at all. */
TEST(off_step_issues_the_in_tree_sequence) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    for (bool prefill : {true, false}) {
        RadBuilder stock, kva;
        REQUIRE_EQ(qwen4exp_fp8::declare(&stock, &meta, &c), RAD_OK);
        Batch bs = make_batch(stock, prefill);
        int dev_stock = 0, dev_kva = 0;
        const std::vector<RecIssue> want = issues_of(qwen4exp_fp8::step, bs.b, &dev_stock);
        REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
        Batch bk = make_batch(kva, prefill);
        const std::vector<RecIssue> got = issues_of(qwen4exp_kva::step, bk.b, &dev_kva);
        CHECK(want.size() > 100);
        CHECK_EQ(dev_kva, dev_stock);
        CHECK_EQ(dev_kva, 0);
        REQUIRE_EQ(got.size(), want.size());
        int differ = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            bool same = got[i].op == want[i].op && got[i].n == want[i].n &&
                        got[i].opd.size() == want[i].opd.size();
            for (size_t k = 0; same && k < want[i].opd.size(); ++k)
                same = same_operand(got[i].opd[k], want[i].opd[k]);
            differ += !same;
        }
        CHECK_EQ(differ, 0);
    }
}

/* ==================================================================== the fitted tensors */

/* rad-convert's view: every copy the model holds is declared, with the shapes, dtypes, shards,
 * access class and group the plan states, and nothing is issued that the in-tree graph does not. */
TEST(declare_all_declares_every_held_copy_and_no_op) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx(1, 2);   /* rank 1 of 2: the correction is row-sharded */
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
            CHECK_EQ(w->shape[0], m.gcfg.n_head_v);   /* this rank's heads */
            CHECK_EQ(w->shape[1], m.gcfg.head_v);
            CHECK_EQ(w->shape[2], m.gcfg.head_k);
            CHECK_EQ(w->shard, RAD_SHARD_ROW);
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
 * kernel ops it will issue. Until the fill path exists it then refuses, by name. */
TEST(speed_declares_the_selected_copy_and_its_kernel_ops) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    for (const char* which : {"shipped", "swap"}) {
        Env env({{"RADIANCE_KVA", "speed"}, {"RADIANCE_KVA_ST", which}});
        RadBuilder kva;
        hold_kva(kva, {"kva.proj", "kva.projr", "kva.st", "kva.stswap"});
        int st = RAD_OK;
        const std::string err = stderr_of([&] { st = qwen4exp_kva::declare(&kva, &meta, &c); });
        CHECK_EQ(st, RAD_E_UNSUPPORTED);
        CHECK(has(err, "not implemented"));
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

RAD_TEST_MAIN()
