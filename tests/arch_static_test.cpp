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
/* The second lane, recorded in the issue list as pseudo-ops (only the staging ring switches lanes). */
constexpr rad_op kLane = 0xFFFF0000u, kJoin = 0xFFFF1000u;
int rad_lane(RadCtx* c, int lane) { c->issues.push_back({kLane, {}, lane}); return RAD_OK; }
int rad_lane_join(RadCtx* c, int from, int to) { c->issues.push_back({kJoin, {}, from * 16 + to}); return RAD_OK; }
int rad_step_fail(RadCtx* c, const char* what) { c->step_fail = what ? what : ""; return RAD_E_SHAPE; }
int rad_rank(RadCtx* c) { return c->rank; }
RadStream rad_stream(RadCtx*) { return nullptr; }
int rad_route_report(RadCtx*, int, const RadRouting*) { return RAD_OK; }
int32_t* rad_route_counts(RadCtx*, int, int64_t) { return nullptr; }
void* rad_buf_ptr(RadCtx*, rad_buf b) { return (void*)(uintptr_t)(b * 64 + 16); }

/* THE PROJECTOR'S DEVICE MEMORY (kva_projector.h). VRAM is a fake address range nothing reads: a
 * copy into it is recorded, not made. Host-mapped memory is real (the plugin writes it on the
 * host), and its device view is a different fake address, as a real device view is. Declare runs
 * with no RadCtx, so only a step's device calls are counted against the step. */
constexpr uintptr_t kFakeVram = 0x7e0000000000ull, kFakeVramEnd = 0x7f0000000000ull;
constexpr uintptr_t kDeviceView = 0x10000000000ull;
struct Upload { void* dst; const void* src; int64_t bytes; };
struct FakeMem { int vram = 0, host = 0; std::vector<Upload> copies; } g_mem;
void* rad_dev_alloc(int64_t n, int kind) {
    if (g_ctx) ++g_ctx->device_calls;
    if (kind == RAD_MEM_HOST_MAPPED) { ++g_mem.host; return std::calloc(1, (size_t)n); }
    static uintptr_t next = kFakeVram;
    ++g_mem.vram;
    void* p = (void*)next;
    next += ((uintptr_t)n + (1u << 20)) & ~(uintptr_t)0xFFFF;
    return p;
}
void  rad_dev_free(void* p, int kind) {
    if (g_ctx) ++g_ctx->device_calls;
    if (kind == RAD_MEM_HOST_MAPPED) std::free(p);
}
void* rad_dev_host_ptr(void* p) { return p; }
void* rad_dev_device_ptr(void* p) { return (unsigned char*)p + kDeviceView; }
int   rad_stream_create(RadStream* s, int) { *s = (RadStream)&g_mem; return RAD_OK; }
void  rad_stream_destroy(RadStream) {}
const char* rad_dev_last_error(void) { return "(fake)"; }
/* The debug paths' device reads. rad_buf_ptr hands out small fake addresses (below), which read as
 * zeros; a batch field is real host memory here and is copied, so a capture sees the batch's own
 * ids and positions. */
int   rad_stream_sync(RadStream) { if (g_ctx) ++g_ctx->device_calls; return RAD_OK; }
int   rad_memcpy_async(void* dst, const void* src, int64_t n, RadStream) {
    if ((uintptr_t)dst >= kFakeVram && (uintptr_t)dst < kFakeVramEnd) {
        g_mem.copies.push_back({dst, src, n});
        return RAD_OK;
    }
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
using rad::arch::AttnGatedFP8;
using rad::arch::GdnFP8;
using rad::arch::bcol_at;

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

/* ==================================================================== the projector folder */

/* A container of a few KiB (abi/rad_format.h): three tokens (the third with empty text), one merge
 * and one 16-byte entry. tools/kva_projector.py's test builds the same bytes and expects the same
 * two hashes (kTinyVocab, kTinyAnchor), which is what ties the two canonical forms together. */
const char* kTinyVocab  = "3988fb447f719ad3fc2c75e5a0fa3daeb2b6a5e10e964744619d1dfbc6e95ff6";
const char* kTinyAnchor = "be45cb2605bf36bebde684841a28f0fd43c69850a3dce5fedba69928ee3a8991";

std::vector<unsigned char> tiny_container_bytes() {
    std::vector<unsigned char> f(4096, 0);
    auto put = [&](size_t at, const void* p, size_t n) { std::memcpy(f.data() + at, p, n); };
    RadFileHeader h{};
    h.magic = RAD_MAGIC; h.version = RAD_FORMAT_VER; h.file_bytes = 4096;
    const char blob[] = "\0tok_a\0tok_b\0anchor.weight";   /* offsets 0, 1, 7, 13 */
    h.str_off = 256; h.str_bytes = sizeof blob;
    h.vocab_off = 512; h.vocab_bytes = 168;
    h.dir_off = 1024; h.dir_count = 1;
    h.plane_off = 1280; h.plane_count = 1;
    h.data_off = 2048; h.data_bytes = 2048;
    put(0, &h, sizeof h);
    put(256, blob, sizeof blob);
    RadVocabHeader v{};
    v.kind = RAD_TOK_BPE; v.n_tokens = 3; v.n_merges = 1;
    v.tok_text_off = 640; v.tok_type_off = 664; v.merge_off = 672;
    put(512, &v, sizeof v);
    const uint64_t text[3] = { 1, 7, 0 };
    const uint8_t type[3] = { 1, 1, 3 };
    const uint32_t merge[2] = { 0, 1 };
    put(640, text, sizeof text); put(664, type, sizeof type); put(672, merge, sizeof merge);
    RadFileEntry e{};
    e.name = 13; e.rank = 1; e.shape[0] = 16; e.layer = -1; e.expert = -1; e.n_planes = 1;
    e.offset = 2048; e.bytes = 16;
    put(1024, &e, sizeof e);
    const RadFilePlane pl{ 2048, 16 };
    put(1280, &pl, sizeof pl);
    for (int i = 0; i < 16; ++i) f[2048 + (size_t)i] = (unsigned char)i;
    return f;
}

/* The tiny container on disk, once per process; its path. */
const std::string& tiny_container() {
    static std::string path;
    if (path.empty()) {
        char t[] = "/tmp/kva_tiny_XXXXXX";
        const int fd = mkstemp(t);
        const std::vector<unsigned char> f = tiny_container_bytes();
        if (fd >= 0 && write(fd, f.data(), f.size()) == (ssize_t)f.size()) path = t;
        if (fd >= 0) close(fd);
    }
    return path;
}

/* Bytes every test tensor points at (zeros, one projector map's worth): the uploads copy from them. */
const unsigned char* tensor_bytes() {
    static std::vector<unsigned char> z((size_t)2560 * 10240 * 2, 0);
    return z.data();
}

/* The manifest the test folder carries: this test model's name, a few of its metadata keys, the
 * tiny container's tokenizer hash; `model` may add members to the model block. */
qwen4exp_kva::Json test_manifest(const std::string& model = "") {
    const std::string text = std::string(R"({"format": 1, "adapter": "qwen4exp", "split": 4,
        "model": {"arch_id": "qwen4exp", "name": "test-q38-flashnext",
                  "meta": {"hc_count": "4", "linear_num_value_heads": "48"},
                  "vocab_sha256": ")") + kTinyVocab + "\"" + model + R"(},
        "files": {"proj.L4.safetensors": "unused by the test folder"}})";
    qwen4exp_kva::Json j;
    qwen4exp_kva::json_parse(text.data(), text.size(), &j);
    return j;
}

qwen4exp_kva::Folder g_test_folder;

/* The plugin forgets every folder and copy; the next declare looks again (every case starts so). */
void reset_projector() {
    qwen4exp_kva::g_folder_for_test = nullptr;
    qwen4exp_kva::g_i8_rows_for_test = nullptr;
    qwen4exp_kva::g_loaded = qwen4exp_kva::Loaded{};
    qwen4exp_kva::free_uploads();
    g_mem.copies.clear();
}

void add_tensor(const std::string& name, uint32_t dtype, std::vector<int64_t> shape) {
    qwen4exp_kva::FolderTensor t;
    t.data = tensor_bytes();
    t.dtype = dtype;
    t.shape = shape;
    int64_t n = 1;
    for (int64_t e : shape) n *= e;
    t.bytes = rad_dtype_bytes(dtype, n);
    g_test_folder.tensors[name] = t;
}

/* A fresh projector folder holding, for layers kSplit..7, what `bases` name: "kva.proj" the maps
 * and biases, "kva.st" the delta-net layers' corrections (layers 4, 5, 6). The builder argument is
 * the old container-weight call's and is not read: nothing comes from the container now. */
void hold_kva(RadBuilder&, std::initializer_list<const char*> bases, const std::string& model = "") {
    reset_projector();
    g_test_folder = qwen4exp_kva::Folder{};
    g_test_folder.place = { "/test/projector", "the test", tiny_container() };
    g_test_folder.manifest = test_manifest(model);
    for (const char* base : bases)
        for (int l = kSplit; l < 8; ++l) {
            const std::string L = std::to_string(l);
            if (!std::strcmp(base, "kva.proj")) {
                add_tensor("proj." + L + ".weight", RAD_BF16, {2560, 4 * 2560});
                add_tensor("proj." + L + ".bias", RAD_BF16, {2560});
            } else if (l != 7) {
                add_tensor("st." + L, RAD_F32, {48, 128, 128});
            }
        }
    qwen4exp_kva::g_folder_for_test = &g_test_folder;
}

/* What an appended container still holds (the retired route): the plugin declares none of it. */
void hold_appended_weights(RadBuilder& b) {
    for (const char* base : {"kva.proj", "kva.st"})
        for (int l = kSplit; l < 8; ++l)
            b.encs.push_back({std::string(base) + "." + std::to_string(l),
                              rad_enc_plain(std::strstr(base, ".st") ? RAD_F32 : RAD_BF16)});
    b.encs.push_back({"kva.rowsel.score", rad_enc_plain(RAD_F32)});
}

/* Adds a row table to the folder hold_kva made: "kva.rowsel.score[_none|_all]" -> score[...]. */
void hold_score(RadBuilder&, const char* name) {
    add_tensor(std::string(name).substr(std::strlen("kva.rowsel.")), RAD_F32, {248320});
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
    std::vector<std::pair<std::string, std::string>> saved;   /* name, value before ("\x01" = unset) */
    Env(std::initializer_list<std::pair<const char*, const char*>> kv) {   /* a null value unsets */
        for (const auto& [k, v] : kv) {
            const char* old = std::getenv(k);
            saved.push_back({k, old ? old : "\x01"});
            if (v) setenv(k, v, 1);
            else   unsetenv(k);
        }
    }
    ~Env() {
        for (auto it = saved.rbegin(); it != saved.rend(); ++it)
            if (it->second == "\x01") unsetenv(it->first.c_str());
            else                       setenv(it->first.c_str(), it->second.c_str(), 1);
    }
};

/* The static cases run chunks of 64 to 2,048 rows; the planner's default gate (RADIANCE_KVA_MIN_BULK_ROWS,
 * 1,024 bulk rows: a host-streamed projector's fixed cost) would send the small ones to the stock step. The
 * suite runs with it off, as it did before the gate; the gate and its default are their own case. */
[[maybe_unused]] const int g_min_bulk_off = setenv("RADIANCE_KVA_MIN_BULK_ROWS", "0", 1);

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
    for (int world : {1, 2, 4})   /* TP4 too (R87): a world this box cannot run */
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

/* A projector folder is THERE and the mode is off: still the in-tree graph, and the folder is not
 * even looked for -- nothing read, nothing copied (placing it would cost 1.26 GiB a rank). */
TEST(off_with_a_projector_folder_still_declares_the_in_tree_graph) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    Env env({{"RADIANCE_KVA", "off"}});
    RadBuilder stock, kva;
    hold_kva(kva, {"kva.proj", "kva.st"});
    hold_score(kva, "kva.rowsel.score");
    REQUIRE_EQ(qwen4exp_fp8::declare(&stock, &meta, &c), RAD_OK);
    REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
    check_same_graph(stock, kva);
    CHECK(!qwen4exp_kva::g_loaded.tried);
    CHECK_EQ(g_mem.copies.size(), (size_t)0);
}

/* A mode is asked and the folder is missing, or not this model's, or misshapen: the folder is
 * refused by name and the declare is the in-tree graph exactly (DD-K: refuse only what cannot run,
 * and serve stock). Also when the container still holds appended kva.* weights: none is declared. */
TEST(a_mode_without_a_usable_projector_serves_the_in_tree_graph) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    struct Case { const char* what; std::function<void(RadBuilder&)> setup; const char* said; };
    const Case cases[] = {
        { "no folder", [](RadBuilder&) { reset_projector(); }, "no projector folder" },
        { "another arch", [](RadBuilder& b) {
              hold_kva(b, {"kva.proj", "kva.st"});
              g_test_folder.manifest.obj[1].second.str = "llama"; }, "is for adapter 'llama'" },
        { "other dims", [](RadBuilder& b) {
              hold_kva(b, {"kva.proj", "kva.st"}, R"(, "meta": {"hc_count": "8"})"); }, "metadata 'hc_count' is '4'" },
        { "another tokenizer", [](RadBuilder& b) {
              hold_kva(b, {"kva.proj", "kva.st"}, R"(, "vocab_sha256": "00")"); }, "the tokenizer differs" },
        { "a misshapen map", [](RadBuilder& b) {
              hold_kva(b, {"kva.proj", "kva.st"});
              add_tensor("proj.6.weight", RAD_BF16, {2560, 2560}); }, "proj.6.weight" },
        { "a hole in the maps", [](RadBuilder& b) {
              hold_kva(b, {"kva.proj", "kva.st"});
              g_test_folder.tensors.erase("proj.6.weight"); }, "proj.6.weight" },
        { "a split at the n-gram layer", [](RadBuilder& b) {
              hold_kva(b, {"kva.proj", "kva.st"});
              g_test_folder.manifest.obj[2].second.num = 1; }, "its split 1 is not a late layer" },
        { "half a correction", [](RadBuilder& b) {
              hold_kva(b, {"kva.proj", "kva.st"});
              g_test_folder.tensors.erase("st.5"); }, "covers 2 of the 3" },
        { "quality without a row table", [](RadBuilder& b) { hold_kva(b, {"kva.proj", "kva.st"}); },
          "holds none; serving stock" },
    };
    for (const Case& k : cases) {
        RadBuilder stock, kva;
        served(stock); served(kva);
        k.setup(kva);
        hold_appended_weights(kva);
        REQUIRE_EQ(qwen4exp_fp8::declare(&stock, &meta, &c), RAD_OK);
        Env env({{"RADIANCE_KVA", "quality"}});
        int st = -1;
        const std::string log = stderr_of([&] { st = qwen4exp_kva::declare(&kva, &meta, &c); });
        CHECK_EQ(st, RAD_OK);
        check_same_graph(stock, kva);
        CHECK(has(log, k.said));
        CHECK(has(log, "serving stock"));
        CHECK_EQ(g_mem.copies.size(), (size_t)0);
        if (!has(log, k.said)) std::fprintf(stderr, "    case '%s' said: %s\n", k.what, log.c_str());
    }
}

/* The same model fitted on another variant -- another trunk encoding, other base weights, another
 * name: a WARNING naming both sides, and KVA runs (DD-K). */
TEST(a_projector_fitted_on_another_variant_warns_and_runs) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    RadBuilder kva;
    served(kva);
    hold_kva(kva, {"kva.proj", "kva.st"}, R"(, "encodings": {"blk.5.ssm_inz.weight": "bf16",
             "blk.5.attn_hc_down.weight": "fp8_e4m3*f32[1x128]"}, "anchors": {"anchor.weight": "11"})");
    g_test_folder.manifest.obj[3].second.obj[1].second.str = "another-quant";
    Env env({{"RADIANCE_KVA", "speed"}});
    int st = -1;
    const std::string log = stderr_of([&] { st = qwen4exp_kva::declare(&kva, &meta, &c); });
    CHECK_EQ(st, RAD_OK);
    CHECK(qwen4exp_kva::g_kva[0].have_proj);
    CHECK(has(log, "WARNING: projector /test/projector: weight blk.5.ssm_inz.weight is i8*bf16[1x128] "
                   "here; the projector was fitted on bf16"));
    CHECK(has(log, "blk.5.attn_hc_down.weight is fp8_e4m3*f32[1x128]") == false);
    CHECK(has(log, ("tensor anchor.weight hashes " + std::string(kTinyAnchor) + " here and 11").c_str()));
    CHECK(has(log, "named 'test-q38-flashnext' and the projector was fitted on 'another-quant'"));
    CHECK(has(log, "encodings 1/2, anchors 0/1"));
    CHECK(has(log, "3 warning(s)"));
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

/* The container route's switches are retired with it (A'): refused by name, so an old command line
 * cannot silently serve something else. */
TEST(the_container_routes_switches_are_refused_by_name) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    for (const char* name : {"RADIANCE_KVA_DECLARE", "RADIANCE_KVA_PROJ", "RADIANCE_KVA_ST"}) {
        RadBuilder kva;
        hold_kva(kva, {"kva.proj", "kva.st"});
        Env env({{"RADIANCE_KVA", "speed"}, {name, "all"}});
        int st = RAD_OK;
        const std::string log = stderr_of([&] { st = qwen4exp_kva::declare(&kva, &meta, &c); });
        CHECK_EQ(st, RAD_E_INVAL);
        CHECK(has(log, (std::string(name) + "=all is retired with the container append").c_str()));
    }
    /* Retired with the vram placement (Dylan, Stage E): any value, the documented ones included. */
    for (auto [name, value] : {std::pair<const char*, const char*>{"RADIANCE_KVA_PROJ_PLACE", "vram"},
                               {"RADIANCE_KVA_PROJ_PLACE", "host"}, {"RADIANCE_KVA_PROJ_RING", "1"},
                               {"RADIANCE_KVA_PROJ_RING", "0"}}) {
        RadBuilder kva;
        hold_kva(kva, {"kva.proj", "kva.st"});
        Env env({{"RADIANCE_KVA", "quality"}, {name, value}});
        int st = RAD_OK;
        const std::string log = stderr_of([&] { st = qwen4exp_kva::declare(&kva, &meta, &c); });
        CHECK_EQ(st, RAD_E_INVAL);
        CHECK(has(log, (std::string(name) + "=" + value + " is retired: the projector is always streamed "
                        "from host memory through the staging ring").c_str()));
    }
}


/* A serving mode takes its tensors from the folder: no weight is declared (the graph's weights are
 * the in-tree ones), the kernel ops take the tensors as inputs, and the real declare copies what
 * the mode reads -- the projector maps, this rank's correction heads -- once, into one block. */
TEST(speed_takes_the_folder_and_declares_its_kernel_ops) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    Env env({{"RADIANCE_KVA", "speed"}});
    RadBuilder stock, kva;
    hold_kva(kva, {"kva.proj", "kva.st"});
    hold_score(kva, "kva.rowsel.score");
    hold_appended_weights(kva);
    REQUIRE_EQ(qwen4exp_fp8::declare(&stock, &meta, &c), RAD_OK);
    const std::string log = stderr_of([&] { CHECK_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK); });
    CHECK_EQ(kva.weights.size(), stock.weights.size());
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
    CHECK_EQ(k.split, (int64_t)kSplit);
    CHECK(k.have_proj && k.have_st && !k.have_rowsel);
    CHECK(has(log, "matches test-q38-flashnext: arch ok, metadata 2/2, tokenizer ok, encodings 0/0, anchors 0/0"));
    int undo = 0, apply = 0, gemm = 0;
    for (const RecOp& o : kva.ops) {
        if (o.op == "kva_gemm_nt_bias" || o.op == "kva_state_correct" || o.op == "kva_mask") CHECK(o.w.empty());
        gemm += o.op == "kva_gemm_nt_bias";
        if (o.op != "kva_state_correct") continue;
        for (const RecParam& q : o.p)
            if (q.key == "mode") { undo += q.sval == "undo"; apply += q.sval == "apply"; }
    }
    CHECK_EQ(undo, 3);    /* delta-net layers 4, 5, 6 */
    CHECK_EQ(apply, 3);
    CHECK_EQ(gemm, 8 - kSplit);
    /* no VRAM copies: the maps, biases and corrections are written into the host block on the host; the
     * VRAM block is the ring's one slot */
    CHECK_EQ(g_mem.copies.size(), (size_t)0);
    for (int l = 0; l < 8; ++l) {
        CHECK_EQ(k.proj_w[(size_t)l].kind == RAD_OPK_RAW, l >= kSplit);
        CHECK_EQ(k.st[(size_t)l].kind == RAD_OPK_RAW, l >= kSplit && l != 7);
    }
    CHECK_EQ(k.score.kind, (int)RAD_OPK_NONE);
    /* A second declare of the same configuration copies nothing again. */
    RadBuilder again;
    hold_appended_weights(again);
    CHECK_EQ(qwen4exp_kva::declare(&again, &meta, &c), RAD_OK);
    CHECK_EQ(g_mem.copies.size(), (size_t)0);
}

/* What each RAW operand points at is the copy of exactly its tensor: the projector map [n, hc*n]
 * bf16, its bias, THIS RANK's value heads of the correction (rows [rank*H, rank*H + H) of the
 * [48, V, K] tensor, the delta net's own contiguous head split) and the selected row table; 256-byte
 * aligned in one block. At TP2, each rank its own block. */
TEST(the_raw_operands_point_at_their_tensors_copies) {
    RadModelMeta meta = flash_next_meta();
    for (int world : {1, 2, 4})   /* TP4 (R87): each rank 12 of the 48 value heads */
        for (int rank = 0; rank < world; ++rank) {
            RadBuildCtx c = served_ctx(rank, world);
            Env env({{"RADIANCE_KVA", "quality"}, {"RADIANCE_KVA_ROWSEL_TABLE", "all"}});
            RadBuilder kva;
            served(kva);
            hold_kva(kva, {"kva.proj", "kva.st"});
            for (const char* t : {"kva.rowsel.score", "kva.rowsel.score_none", "kva.rowsel.score_all"})
                hold_score(kva, t);
            /* distinct source bytes per tensor, so a copy can be traced to its tensor */
            int64_t at = 0;
            for (auto& [name, t] : g_test_folder.tensors) { t.data = tensor_bytes() + at; at += 256; }
            REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
            const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
            const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
            const int64_t heads = m.gcfg.n_head_v * m.gcfg.head_v * m.gcfg.head_k;
            auto copy_of = [&](const RadOperand& o) -> const Upload* {
                for (const Upload& u : g_mem.copies) if (u.dst == o.raw) return &u;
                return nullptr;
            };
            auto src = [&](const std::string& n) { return g_test_folder.tensors[n].data; };
            const qwen4exp_kva::Upload& up = qwen4exp_kva::g_upload[rank];
            const int64_t row = 10240 * 2;
            for (int l = kSplit; l < 8; ++l) {
                const std::string L = std::to_string(l);
                const RadOperand& w = k.proj_w[(size_t)l];
                /* the GEMM reads the slot this layer takes turns on; the ring copies the layer's host
                 * row block there: the map's 2,560 rows, then the bias in row 2,560 */
                CHECK(w.dtype == RAD_BF16 && w.rows == 2560 && w.cols == 10240);
                CHECK((uintptr_t)w.raw >= kFakeVram && (uintptr_t)w.raw == (uintptr_t)k.ring_dst[(size_t)l].raw);
                CHECK_EQ((uintptr_t)k.proj_b[(size_t)l].raw, (uintptr_t)w.raw + 2560u * row);
                CHECK_EQ(((uintptr_t)w.raw - kFakeVram) % 256, 0u);
                const unsigned char* block = (const unsigned char*)k.ring_src[(size_t)l].raw - kDeviceView;
                CHECK(block >= (const unsigned char*)up.host && block < (const unsigned char*)up.host + up.host_bytes);
                CHECK(std::memcmp(block, src("proj." + L + ".weight"), 2560 * row) == 0);
                CHECK(std::memcmp(block + 2560 * row, src("proj." + L + ".bias"), 2560 * 2) == 0);
                if (l == 7) continue;
                /* this rank's value heads of the correction, in the host block, read through its device view */
                const unsigned char* st = (const unsigned char*)k.st[(size_t)l].raw - kDeviceView;
                CHECK(st >= (const unsigned char*)up.host && st + heads * 4 <= (const unsigned char*)up.host + up.host_bytes);
                CHECK(std::memcmp(st, src("st." + L) + rank * heads * 4, (size_t)heads * 4) == 0);
                CHECK(k.st[(size_t)l].rows == m.gcfg.n_head_v && k.st[(size_t)l].cols == m.gcfg.head_v * m.gcfg.head_k);
            }
            const unsigned char* sc = (const unsigned char*)k.score.raw - kDeviceView;
            CHECK(sc >= (const unsigned char*)up.host && sc + 248320 * 4 <= (const unsigned char*)up.host + up.host_bytes);
            CHECK(std::memcmp(sc, src("score_all"), 248320 * 4) == 0 && k.score.rows == 248320);
            CHECK(g_mem.copies.empty());
        }
}

/* The projector maps live in host-mapped memory (written on the host, read through its device view by
 * the ring's copy), nothing of them in VRAM but the two ring slots; the correction and the row table
 * stay in VRAM. */
TEST(the_maps_live_in_host_mapped_memory) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    Env env({{"RADIANCE_KVA", "quality"}});
    RadBuilder kva;
    served(kva);
    hold_kva(kva, {"kva.proj", "kva.st"});
    hold_score(kva, "kva.rowsel.score");
    const int hosts = g_mem.host;
    const std::string log = stderr_of([&] { CHECK_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK); });
    CHECK_EQ(g_mem.host, hosts + 1);
    CHECK(g_mem.copies.empty());   /* the corrections and the row table are host-block pieces too */
    const qwen4exp_kva::Upload& u = qwen4exp_kva::g_upload[0];
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
    for (int l = kSplit; l < 8; ++l) {
        const uintptr_t w = (uintptr_t)k.ring_src[(size_t)l].raw;
        CHECK(w >= (uintptr_t)u.host + kDeviceView && w < (uintptr_t)u.host + kDeviceView + (uintptr_t)u.host_bytes);
    }
    CHECK(u.host_bytes >= (int64_t)(8 - kSplit) * (2561LL * 10240 * 2) + 3 * (24LL * 128 * 128 * 4) + 248320 * 4);
    CHECK_EQ(u.vram_bytes, 2561LL * 10240 * 2);   /* the ring's one slot, nothing else */
    CHECK(has(log, "MiB host-mapped (bf16 maps, correction, row table)"));
    CHECK((uintptr_t)k.st[kSplit].raw >= (uintptr_t)u.host + kDeviceView && (uintptr_t)k.score.raw >= (uintptr_t)u.host + kDeviceView);
}

/* ==================================================================== the masked path: declare */

/* Declares stock and KVA side by side on the served formats with every kva.* tensor held. */
struct Pair {
    RadBuilder stock, kva;
    int        st = RAD_OK;
};
void declare_pair(Pair& p, const char* mode, int rank = 0, int world = 1, int64_t max_out_rows = 0,
                  int max_spec = 0) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx(rank, world);
    c.max_out_rows = max_out_rows;
    c.max_spec = max_spec;
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
    int32_t n_spec = 0;   /* draft depth of the step: each decode entry verifies 1 + n_spec rows */
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
    b.n_tok = T; b.n_seq = n; b.n_ahead = s.ahead; b.n_spec = s.n_spec;
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
    using qwen4exp_kva::PATH_STRADDLE;
    using qwen4exp_kva::PATH_DECODERS;
    const auto none = [](RadBatch&) {};
    const std::vector<Row> rows = {
        {"whole bulk chunk, speed", "speed", {{2048}, 0, 2048}, none, PATH_LEAN, 2048, 0, false},
        {"whole bulk chunk, quality", "quality", {{2048}, 0, 2048}, none, PATH_MASKED, 2048, 0, true},
        {"whole bulk chunk, plumb", "plumb", {{2048}, 0, 2048}, none, PATH_MASKED, 2048, 0, true},
        {"more than T ahead", "speed", {{2048}, 0, 4096}, none, PATH_LEAN, 2048, 0, false},
        {"T-1 ahead, speed: tail-only straddle", "speed", {{2048}, 0, 2047}, none, PATH_STRADDLE, 1984, 0, false},
        {"half ahead, speed: tail-only straddle", "speed", {{2048}, 0, 1024}, none, PATH_STRADDLE, 1024, 0, false},
        {"T-1 ahead, quality: one tile exact streams", "quality", {{2048}, 0, 2047}, none, PATH_MASKED, 1984, 0, true},
        {"half ahead, quality, rows <= 64: 1024 exact rows run exact", "quality", {{2048}, 0, 1024}, none, PATH_STOCK, 0, 0, false},
        {"64 ahead, quality", "quality", {{2048}, 0, 64}, none, PATH_STOCK, 0, 0, false},
        {"straddle at short context, speed: dense attention", "speed", {{2048}, 0, 1024, 0}, none, PATH_STOCK, 0, 0, false},
        {"straddle at short context, one tile", "speed", {{2048}, 0, 2047, 0}, none, PATH_MASKED, 1984, 0, true},
        {"1 ahead: no bulk", "speed", {{2048}, 0, 1}, none, PATH_STOCK, 0, 0, false},
        {"final chunk", "speed", {{2048}, 0, 0}, none, PATH_STOCK, 0, 0, false},
        {"mode off", "off", {{2048}, 0, 2048}, none, PATH_STOCK, 0, 0, false},
        {"draft pass", "speed", {{2048}, 0, 2048}, [](RadBatch& b) { b.draft_pass = 1; }, PATH_STOCK, 0, 0, false},
        {"encoder pass", "speed", {{2048}, 0, 2048}, [](RadBatch& b) { b.enc = 1; }, PATH_STOCK, 0, 0, false},
        {"media rows", "quality", {{2048}, 0, 2048}, [](RadBatch& b) { b.n_mm_rows = 5; }, PATH_STOCK, 0, 0, false},
        {"mixed rope", "speed", {{2048}, 0, 2048},
         [](RadBatch& b) { static int32_t rp[3]; b.rope_pos = rp; b.rope_mixed = 1; }, PATH_STOCK, 0, 0, false},
        {"one decoder beside", "speed", {{1, 1984}, 1, 2048}, none, PATH_DECODERS, 1985, 1, true},
        {"eight decoders beside", "speed", {{1, 1, 1, 1, 1, 1, 1, 1, 1984}, 8, 2048}, none, PATH_DECODERS, 1992, 8, true},
        {"decoders beside a straddle: 1,025 exact rows > the table's 64", "speed", {{1, 1984}, 1, 1024}, none, PATH_STOCK, 0, 0, false},
        {"decoders beside, short context: dense attention", "speed", {{1, 1984}, 1, 2048, 0}, none, PATH_MASKED, 1985, 1, true},
        {"one decoder beside, quality", "quality", {{1, 1984}, 1, 2048}, none, PATH_MASKED, 1985, 1, true},
        {"eight decoders, straddle: 1096 exact rows run exact", "quality", {{1, 1, 1, 1, 1, 1, 1, 1, 1984}, 8, 1000}, none, PATH_STOCK, 0, 0, false},
        {"two prefills, last long", "speed", {{64, 1984}, 0, 2048}, none, PATH_MASKED, 2048, 64, true},
        {"two prefills, last short", "speed", {{1984, 64}, 0, 2048}, none, PATH_MASKED, 2048, 64, true},
        {"two prefills, no bulk above s_lb", "speed", {{1984, 64}, 0, 100}, none, PATH_STOCK, 0, 0, false},
        {"three prefills and a decoder: 961 exact rows", "quality", {{1, 1024, 512, 448}, 1, 2048}, none, PATH_STOCK, 0, 0, false},
        {"pure decode", "speed", {{1, 1}, 2, 0}, none, PATH_STOCK, 0, 0, false},
    };
    Env guard({{"RADIANCE_KVA_STAGE_ROWS", "64"}});   /* the threshold's rows; the default is the next case */
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

/* A.1 -- THE PLANNER'S OWN CONTRACT, with no adapter in front of it (kva_plan.h is model-agnostic): the
 * tail-only straddle is a speed path. Quality and plumb on a straddling chunk whose adapter could
 * straddle still take the masked path or the exact step, never the straddle. */
TEST(the_planner_straddles_only_in_speed) {
    using namespace qwen4exp_kva;
    PlanIn in;
    in.eligible = in.stream_ok = in.straddle_ok = true;
    in.n_tok = 2048; in.n_seq = 1; in.q_prefill = 2048; in.n_ahead = 1024;
    const PlanConfig pc;
    for (int mode : {PLAN_PLUMB, PLAN_QUALITY}) {
        in.mode = mode;
        CHECK_EQ(plan_pass(in, pc).path, PATH_MASKED);   /* masked (streamed by default), never the straddle */
    }
    in.mode = PLAN_SPEED;
    CHECK_EQ(plan_pass(in, pc).path, PATH_STRADDLE);
    /* beside decoders, whole bulk: the decoders path is speed's alone, even when an adapter claims
     * the per-row attention form for every mode */
    in.n_tok = 2049; in.n_seq = 2; in.n_seq_decode = 1; in.n_tok_decode = 1; in.q_prefill = 2048; in.n_ahead = 2048;
    CHECK_EQ(plan_pass(in, pc).path, PATH_DECODERS);
    for (int mode : {PLAN_PLUMB, PLAN_QUALITY}) {
        in.mode = mode;
        CHECK_EQ(plan_pass(in, pc).path, PATH_MASKED);
    }
}

/* R56 (host placement) -- A PASS APPROXIMATES ONLY WITH ENOUGH BULK ROWS: a decoder-shared prompt's
 * 64-row checkpoint remainder runs the stock step when the projector lives in host memory (every
 * approximate pass streams the whole projector), the 1,984-row chunk beside it still approximates;
 * the bound is inclusive and is the bulk superset b - s_lb, a keyed number. Plumb (the oracle) is
 * never held back. Default 1,024 (the projector is always in host memory); the switch overrides it. */
TEST(a_host_placed_projector_approximates_only_passes_with_enough_bulk_rows) {
    using namespace qwen4exp_kva;
    PlanIn in;
    in.eligible = in.stream_ok = true;
    in.n_seq = 2; in.n_seq_decode = 1; in.n_tok_decode = 1; in.n_ahead = 2048;
    PlanConfig pc;
    pc.min_bulk_rows = 1024;
    struct Case { int mode; int64_t bulk; int path; };
    for (const Case& c : {Case{PLAN_QUALITY, 64, PATH_STOCK}, Case{PLAN_SPEED, 64, PATH_STOCK},
                          Case{PLAN_QUALITY, 1023, PATH_STOCK}, Case{PLAN_QUALITY, 1024, PATH_MASKED},
                          Case{PLAN_SPEED, 1984, PATH_MASKED}, Case{PLAN_PLUMB, 64, PATH_MASKED}}) {
        in.mode = c.mode;
        in.n_tok = 1 + c.bulk; in.q_prefill = c.bulk;
        CHECK_EQ(plan_pass(in, pc).path, c.path);
    }
    {
        Env e({{"RADIANCE_KVA", "quality"}, {"RADIANCE_KVA_MIN_BULK_ROWS", nullptr}});
        Config cfg;
        RadModelMeta meta = flash_next_meta();
        REQUIRE_EQ(read_config(&meta, &cfg), RAD_OK);
        CHECK_EQ(cfg.min_bulk_rows, 1024);
    }
    Env e({{"RADIANCE_KVA", "quality"}, {"RADIANCE_KVA_MIN_BULK_ROWS", "0"}});
    Config cfg;
    RadModelMeta meta = flash_next_meta();
    REQUIRE_EQ(read_config(&meta, &cfg), RAD_OK);
    CHECK_EQ(cfg.min_bulk_rows, 0);
}

/* DD-A's T_ck floor (Stage C) -- a chunk that writes a checkpoint keeps its last T_ck rows exact
 * (rounded up to the tile); a chunk that writes none is untouched; 0 is off; it composes with a
 * capped n_ahead (the smaller bulk end wins). The switch is read at declare. */
TEST(a_checkpoint_writing_chunk_keeps_its_last_t_ck_rows_exact) {
    using namespace qwen4exp_kva;
    PlanIn in;
    in.mode = PLAN_QUALITY; in.eligible = in.stream_ok = true;
    in.n_tok = 2048; in.n_seq = 1; in.q_prefill = 2048;
    PlanConfig pc;
    struct Case { int64_t floor, ckpts, ahead, b; };
    for (const Case& c : {Case{0, 1, 2048, 2048}, Case{512, 0, 2048, 2048}, Case{512, 1, 2048, 1536},
                          Case{500, 1, 2048, 1536}, Case{1024, 1, 2048, 1024}, Case{512, 1, 1536, 1536},
                          Case{512, 1, 1024, 1024}}) {
        pc.ckpt_floor = c.floor; in.n_checkpoints = c.ckpts; in.n_ahead = c.ahead;
        CHECK_EQ(plan_pass(in, pc).b, c.b);
    }
    Env e({{"RADIANCE_KVA", "quality"}, {"RADIANCE_KVA_CKPT_FLOOR", "512"}});
    Config cfg;
    RadModelMeta meta = flash_next_meta();
    REQUIRE_EQ(read_config(&meta, &cfg), RAD_OK);
    CHECK_EQ(cfg.ckpt_floor, 512);
    Pair p;
    declare_pair(p, "quality");
    REQUIRE_EQ(p.st, RAD_OK);
    Batch x = make_step(p.kva, {{128}, 0, 2048});
    x.b.n_checkpoints = 1;
    CHECK_EQ(qwen4exp_kva::derive(qwen4exp_kva::g_kva[0], &x.b).path, (int)PATH_STOCK);   /* 128 rows: all floor */
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

/* THE DEFAULT STREAMS EVERY MASKED PASS (Stage A.1's R96 and guard trade, notes/impl.md): the rows the
 * truth table above sends to the stock step under a 64-row threshold take the masked path, streaming. */
TEST(the_default_streams_every_masked_pass) {
    using qwen4exp_kva::PATH_MASKED;
    const auto none = [](RadBatch&) {};
    const std::vector<Row> rows = {
        {"half ahead, quality: 1024 exact rows stream", "quality", {{2048}, 0, 1024}, none, PATH_MASKED, 1024, 0, true},
        {"64 ahead, quality", "quality", {{2048}, 0, 64}, none, PATH_MASKED, 64, 0, true},
        {"straddle at short context, speed: dense attention", "speed", {{2048}, 0, 1024, 0}, none, PATH_MASKED, 1024, 0, true},
        {"eight decoders, straddle", "quality", {{1, 1, 1, 1, 1, 1, 1, 1, 1984}, 8, 1000}, none, PATH_MASKED, 904, 8, true},
        {"three prefills and a decoder: 961 exact rows", "quality", {{1, 1024, 512, 448}, 1, 2048}, none, PATH_MASKED, 1985, 961, true},
    };
    for (const Row& r : rows) {
        Pair p;
        declare_pair(p, r.mode);
        REQUIRE_EQ(p.st, RAD_OK);
        CHECK_EQ(qwen4exp_kva::g_kva[0].cfg.stage_rows, INT64_MAX);
        Batch x = make_step(p.kva, r.shape);
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

/* The stager lever's switch: stock never streams, and the row threshold gates auto. */
TEST(the_stage_switch_and_threshold_gate_the_lever) {
    struct Case { const char* stage; const char* rows; Shape s; bool stream; };
    for (const Case& c : {Case{"stock", nullptr, {{2048}, 0, 2047}, false},
                          Case{"auto", "63", {{2048}, 0, 2047}, false},    /* 64 exact rows */
                          Case{"auto", "64", {{2048}, 0, 2047}, true},
                          Case{"auto", "8", {{1, 1, 1, 1, 1, 1, 1, 1, 1984}, 8, 2048}, true},
                          Case{"auto", "", {{2048}, 0, 1024}, true}}) {   /* default: 1024 exact rows stream */
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
    std::vector<RecIssue> issues;   /* the KVA step's without the staging ring's lanes and copies */
    std::vector<RecIssue> all;      /* every issue, the ring's included */
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
    r.all = c.issues;
    /* THE RING IS ALWAYS ON (the projector streams from host memory), and the in-tree oracle has no
     * ring: the KVA step's lane switches, joins and copies are set aside here and checked on their
     * own (the_staging_ring_copies_each_map_a_layer_ahead_on_lane_1). Its op is declared after the
     * whole in-tree graph, so no in-tree handle can share its number. */
    const rad_op ring = step == qwen4exp_kva::step ? qwen4exp_kva::g_kva[rank].op_ring : 0;
    for (const RecIssue& i : c.issues)
        if (i.op != kLane && i.op != kJoin && (!ring || i.op != ring)) r.issues.push_back(i);
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
    const RadOperand heads = k.st[(size_t)l];   /* the_raw_operands_point_at_their_tensors_copies */
    return {kv_cache(m.kv_state, l), last_row(x, m.kv_state), kv_cache(k.kv_applied, l),
            last_row(x, k.kv_applied), heads, k.kv_rho ? kv_cache(k.kv_rho, l) : RAD_NONE,
            k.kv_rho ? last_row(x, k.kv_rho) : RAD_NONE, brows(k.b_bounds, 2)};
}

/* DD-A's hazard op as an approximate pass ends it (kva_hazard.h): the last sequence's cu pair, the
 * positions, the bulk bounds (the cu pair itself on the lean path), the span T - n_ahead as an
 * extent when positive, the meta slot of the last sequence, this rank's counter. None in plumb. */
void push_hazard(std::vector<RecIssue>& out, const Batch& x, bool lean, int rank = 0) {
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
    if (!k.op_hazard) return;
    const RadBatch& b = x.b;
    const int64_t span = k.cfg.tail - b.n_ahead;
    const RadOperand cu = praw(b.cu_seqlens + b.n_seq - 1, RAD_I32, 2);
    out.push_back({k.op_hazard, {cu, praw(b.positions, RAD_I32, b.n_tok), lean ? cu : brows(k.b_bounds, 2),
                   span > 0 ? praw(b.positions, RAD_I32, span) : RAD_NONE, kv_cache(k.kv_meta, k.meta_layer),
                   last_row(x, k.kv_meta), praw(qwen4exp_kva::g_hazard_dev[rank], RAD_F32, 1)}, 1});
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
    out.push_back({k.op_proj[(size_t)l], {brow_slice(k.b_hs, w.s_lb, rows, wide), k.proj_w[(size_t)l],
                   k.proj_b[(size_t)l], RAD_NONE, brow_slice(k.xp.x, w.s_lb, rows, n)}, rows});
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
                           praw(b.positions, RAD_I32, w.b), w.scored ? k.score : RAD_NONE,
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
    push_hazard(out, x, false, rank);
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

/* #6 -- THE STRADDLING CHUNK, masked (PLAN-FIX §4, DD-C): 64 tokens ahead of a 128-row chunk with
 * T = 2048, so the bulk ends at row 64 and 64 rows are exact (one tile: the pass still streams).
 * split: scan [s, b), decay sums over [s, b), apply, scan [b, e); end: one scan, decay sums over the
 * whole chunk, apply after. Speed on the same shape with the tail-only path off (TAIL_ONLY=0) is
 * masked too and applies without decay sums. */
TEST(a_straddling_chunk_splits_the_last_scan_at_the_bulk_end) {
    struct Case { const char* mode; const char* straddle; bool split, rho; int64_t rho_rows; };
    Env tail_only_off({{"RADIANCE_KVA_TAIL_ONLY", "0"}});
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

/* What an in-tree helper issues, on a scratch context: the oracle for the hand-issued pieces. */
template <class F>
std::vector<RecIssue> issues_by(F f) {
    RadCtx cx;
    f(&cx);
    return cx.issues;
}

void add_linear(std::set<rad_op>& s, const rad::arch::LinearFP8& l) {
    for (rad_op h : {l.op, l.op_q, l.op_m1}) if (h) s.insert(h);
}

/* The stock issues of one late layer segment whose handle is in `ops`, in stock order. */
std::vector<RecIssue> pick(const std::vector<RecIssue>& seg, const std::set<rad_op>& ops) {
    std::vector<RecIssue> out;
    for (const RecIssue& r : seg) if (ops.count(r.op)) out.push_back(r);
    return out;
}

void append(std::vector<RecIssue>& out, const std::vector<RecIssue>& more) {
    out.insert(out.end(), more.begin(), more.end());
}

/* A TAIL-ONLY STRADDLE LAYER as the oracle sees it (A.1): the in-tree helpers called with the tail's
 * row range (connection read/write, the MoE pass, the output linears), the stock K/V, indexer, query
 * and delta-net front issues over every row, and the split scan around the correction. */
std::vector<RecIssue> straddle_expected(const std::vector<RecIssue>& seg, const Batch& x, int l, int64_t b,
                                        int rank) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
    const qwen4exp_fp8::Layer& lay = m.layers[(size_t)l];
    const RadBatch& bt = x.b;
    const int64_t T = bt.n_tok, rows = T - b, n = m.g.n_embd;
    std::vector<RecIssue> out = issues_by([&](RadCtx* cx) { lay.hc_mix.read(cx, T, b, rows); });
    out.push_back({k.op_proj[(size_t)l], {brows(m.b_h, b), k.proj_w[(size_t)l], k.proj_b[(size_t)l],
                   RAD_NONE, brows(m.a_x.x, b)}, b});
    out.push_back({k.quant.op, {brows(m.a_x.x, b), brows(m.a_x.q8, b), brow_slice(m.a_x.s8, 0, b, n / 128)}, b});
    if (lay.full) {
        const AttnGatedFP8& a = lay.attn;
        const int64_t qw = a.g.q_dim(), hd = a.g.head_dim;
        append(out, issues_by([&](RadCtx* cx) { lay.qsa.step(cx, a.w.h, &bt); }));
        std::set<rad_op> kv, q;
        add_linear(kv, a.kp);
        add_linear(kv, a.vp);
        for (rad_op h : {a.op_k_norm, a.op_rope_k, a.op_kv_store}) kv.insert(h);
        add_linear(q, a.qg);
        for (rad_op h : {a.op_q_norm, a.op_rope_q}) q.insert(h);
        append(out, pick(seg, kv));
        append(out, pick(seg, q));
        out.push_back({a.op_attn_gq, {brow_slice(a.w.q, b, rows, qw), kv_cache(a.kv, a.layer),
                       brow_slice(a.qsa_sel, b, rows, a.qsa_topk + 1), brow_slice(a.qsa_sequ, b, rows, 1),
                       RAD_NONE, RAD_NONE, RAD_NONE, brow_slice(a.w.attn.x, b, rows, qw),
                       bcol_at(a.w.qg, b, a.g.n_head * 2 * hd, hd, hd, rows),
                       brow_slice(a.w.attn.cq(), b, rows, qw), brow_slice(a.w.attn.cs(), b, rows, qw / 128)}, 1});
        append(out, issues_by([&](RadCtx* cx) { a.o.step(cx, a.w.attn, a.w.h.x, T, b, rows); }));
        if (a.op_ar && !rad::arch::ar_taken(a.g, T, a.ar_out, a.ar_out_take))
            out.push_back({a.op_ar, {brow_slice(a.w.h.x, b, rows, n), RAD_NONE}, rows * n});
    } else {
        const GdnFP8& d = lay.gdn;
        std::set<rad_op> front;
        add_linear(front, d.in);
        front.insert(d.op_ab);
        append(out, pick(seg, front));
        Want w;
        w.b = b; w.correct = true; w.split = true;
        for (const RecIssue& r : seg) {
            if (r.op == d.op_conv_prep) out.push_back({k.op_undo[(size_t)l], correction_operands(x, l, rank), 1});
            if (r.op == d.op_conv_prep || r.op == d.op_kkt) out.push_back(r);
            if (r.op == d.op_scan) push_scans(out, r, x, l, w, rank);
        }
        const int64_t cd = d.cfg.conv_dim(), vd = d.cfg.v_dim();
        out.push_back({d.op_gnorm, {brow_slice(d.w.o.x, b, rows, vd), bcol_at(d.w.in.x, b, cd + vd, cd, vd, rows),
                       RAD_W(d.w_out_norm), brow_slice(d.w.o.x, b, rows, vd)}, rows});
        append(out, issues_by([&](RadCtx* cx) { d.q_o.step(cx, T, b, rows); }));
        append(out, issues_by([&](RadCtx* cx) { d.out.step(cx, d.w.o, d.w.h.x, T, b, rows); }));
        if (d.op_ar && !rad::arch::ar_taken(d.g, T, d.ar_out, d.ar_out_take))
            out.push_back({d.op_ar, {brow_slice(d.w.h.x, b, rows, n), RAD_NONE}, rows * n});
    }
    append(out, issues_by([&](RadCtx* cx) { lay.hc_mix.write(cx, T, b, rows); }));
    append(out, issues_by([&](RadCtx* cx) { lay.hc_ffn.read(cx, T, b, rows); }));
    append(out, issues_by([&](RadCtx* cx) { lay.mlp.pass(cx, T, b, rows); }));
    append(out, issues_by([&](RadCtx* cx) { lay.hc_ffn.write(cx, T, b, rows); }));
    return out;
}

/* SPEED BESIDE DECODERS, one late layer as the oracle sees it: the in-tree helpers over the decoder
 * rows [0, DT) (connection read/write, MoE pass, output linears), the attention's indexer, K/V and
 * query path over every row and its per-row sparse attention over [0, DT); the delta net's
 * projections, decode half, conv/kkt and corrected scan as stock issues them, its output projection
 * over [0, DT); the bulk rows' projection from the layer-S stream into `x` and its codes. */
std::vector<RecIssue> decoders_layer_expected(const std::vector<RecIssue>& seg, const Batch& x, int l, int rank) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
    const qwen4exp_fp8::Layer& lay = m.layers[(size_t)l];
    const RadBatch& bt = x.b;
    const int64_t T = bt.n_tok, DT = bt.n_tok_decode, n = m.g.n_embd, wide = m.hccfg.hc * n, bulk = T - DT;
    std::vector<RecIssue> out = issues_by([&](RadCtx* cx) { lay.hc_mix.read(cx, T, 0, DT); });
    out.push_back({k.op_proj[(size_t)l], {brow_slice(m.b_h, DT, bulk, wide), k.proj_w[(size_t)l], k.proj_b[(size_t)l],
                   RAD_NONE, brow_slice(m.a_x.x, DT, bulk, n)}, bulk});
    out.push_back({k.quant.op, {brow_slice(m.a_x.x, DT, bulk, n), brow_slice(m.a_x.q8, DT, bulk, n),
                   brow_slice(m.a_x.s8, DT, bulk, n / 128)}, bulk});
    if (lay.full) {
        const AttnGatedFP8& a = lay.attn;
        const int64_t qw = a.g.q_dim(), hd = a.g.head_dim;
        append(out, issues_by([&](RadCtx* cx) { lay.qsa.step(cx, a.w.h, &bt); }));
        std::set<rad_op> kv, q;
        add_linear(kv, a.kp);
        add_linear(kv, a.vp);
        for (rad_op h : {a.op_k_norm, a.op_rope_k, a.op_kv_store}) kv.insert(h);
        add_linear(q, a.qg);
        for (rad_op h : {a.op_q_norm, a.op_rope_q}) q.insert(h);
        append(out, pick(seg, kv));
        append(out, pick(seg, q));
        out.push_back({a.op_attn_gq, {brow_slice(a.w.q, 0, DT, qw), kv_cache(a.kv, a.layer),
                       brow_slice(a.qsa_sel, 0, DT, a.qsa_topk + 1), brow_slice(a.qsa_sequ, 0, DT, 1),
                       RAD_NONE, RAD_NONE, RAD_NONE, brow_slice(a.w.attn.x, 0, DT, qw),
                       bcol_at(a.w.qg, 0, a.g.n_head * 2 * hd, hd, hd, DT),
                       brow_slice(a.w.attn.cq(), 0, DT, qw), brow_slice(a.w.attn.cs(), 0, DT, qw / 128)}, 1});
        append(out, issues_by([&](RadCtx* cx) { a.o.step(cx, a.w.attn, a.w.h.x, T, 0, DT); }));
        if (a.op_ar && !rad::arch::ar_taken(a.g, T, a.ar_out, a.ar_out_take))
            out.push_back({a.op_ar, {brow_slice(a.w.h.x, 0, DT, n), RAD_NONE}, DT * n});
    } else {
        const GdnFP8& d = lay.gdn;
        std::set<rad_op> front, half;
        add_linear(front, d.in);
        front.insert(d.op_ab);
        for (rad_op h : {d.op_cr, d.op_conv_update, d.op_recur}) if (h) half.insert(h);
        append(out, pick(seg, front));
        append(out, pick(seg, half));
        Want w;
        w.b = T; w.correct = true;
        for (const RecIssue& r : seg) {
            if (r.op == d.op_conv_prep) out.push_back({k.op_undo[(size_t)l], correction_operands(x, l, rank), 1});
            if (r.op == d.op_conv_prep || r.op == d.op_kkt) out.push_back(r);
            if (r.op == d.op_scan) push_scans(out, r, x, l, w, rank);
        }
        append(out, issues_by([&](RadCtx* cx) { d.out.step(cx, d.w.o, d.w.h.x, T, 0, DT); }));
        if (d.op_ar && !rad::arch::ar_taken(d.g, T, d.ar_out, d.ar_out_take))
            out.push_back({d.op_ar, {brow_slice(d.w.h.x, 0, DT, n), RAD_NONE}, DT * n});
    }
    append(out, issues_by([&](RadCtx* cx) { lay.hc_mix.write(cx, T, 0, DT); }));
    append(out, issues_by([&](RadCtx* cx) { lay.hc_ffn.read(cx, T, 0, DT); }));
    append(out, issues_by([&](RadCtx* cx) { lay.mlp.pass(cx, T, 0, DT); }));
    append(out, issues_by([&](RadCtx* cx) { lay.hc_ffn.write(cx, T, 0, DT); }));
    return out;
}

/* The whole step of speed beside decoders: the in-tree step's issues on this batch up to layer S
 * with the mask ahead of layer 0 (zeros and bounds) and the stager probes behind layer S-3's gate-up
 * (the decoders' late experts stream), then decoders_layer_expected per late layer, the epilogue stock. */
std::vector<RecIssue> decoders_expected(const Batch& x, int rank = 0) {
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
    const RadBatch& b = x.b;
    const std::vector<RecIssue> stock = run_step(qwen4exp_fp8::step, b, rank).issues;
    std::vector<rad_op> reads;
    for (int l = kSplit; l < 8; ++l) reads.push_back(m.layers[(size_t)l].hc_mix.op_read);
    const std::vector<size_t> at = starts(stock, reads, m.mixer.op_read);
    std::vector<RecIssue> out;
    for (size_t i = 0; i < at[0]; ++i) {
        if (stock[i].op == m.layers[0].hc_mix.op_read)
            out.push_back({k.op_mask, {praw(b.cu_seqlens + b.n_seq - 1, RAD_I32, 2), praw(b.token_ids, RAD_I32, b.n_tok),
                           praw(b.positions, RAD_I32, b.n_tok), RAD_NONE, brows(k.b_mask, b.n_tok),
                           brows(k.b_bounds, 4), brows(k.b_zeros, k.n_zeros)}, b.n_tok});
        out.push_back(stock[i]);
        if (stock[i].op == m.layers[kSplit - 3].mlp.op_gu)
            for (int p = kSplit - 1; p < 8; ++p) out.push_back(probe_of(stock, p, rank));
    }
    for (int l = kSplit; l < 8; ++l)
        append(out, decoders_layer_expected(slice(stock, at[(size_t)(l - kSplit)], at[(size_t)(l - kSplit + 1)]), x, l, rank));
    for (size_t i = at.back(); i < stock.size(); ++i) out.push_back(stock[i]);
    push_hazard(out, x, false, rank);
    return out;
}

/* R55 -- SPEED BESIDE DECODERS keeps the lean fill for its bulk rows and runs the full late blocks
 * over the decoder rows only, at TP1 and on rank 0 of TP2 (where the connection writes carry the
 * all-reduce), one decoder and four verifying 1 + 3 rows; the log names the path; nothing reaches
 * the device. RADIANCE_KVA_MASK=all and TAIL_ONLY=0 keep the masked path (the controls/oracle). */
TEST(speed_beside_decoders_runs_full_late_blocks_over_the_decoder_rows_only) {
    struct Case { int world; Shape s; };
    for (const Case& c : {Case{1, {{1, 128}, 1, 2048}}, Case{2, {{1, 128}, 1, 2048}},
                          Case{2, {{4, 4, 4, 4, 128}, 4, 2048, 4096, 3}}}) {
        Pair p;
        declare_pair(p, "speed", 0, c.world, 0, c.s.n_spec);
        REQUIRE_EQ(p.st, RAD_OK);
        Batch x = make_step(p.kva, c.s);
        const Run got = run_step(qwen4exp_kva::step, x.b);
        CHECK_EQ(differ_at(got.issues, decoders_expected(x)), 0);
        CHECK(has(got.log, "decoders, stage stream"));
        CHECK_EQ(got.device_calls, 0);
    }
    for (auto [name, value] : {std::pair<const char*, const char*>{"RADIANCE_KVA_MASK", "all"}, {"RADIANCE_KVA_TAIL_ONLY", "0"}}) {
        Env e({{name, value}});
        Pair p;
        declare_pair(p, "speed");
        REQUIRE_EQ(p.st, RAD_OK);
        Batch x = make_step(p.kva, {{1, 128}, 1, 2048});
        CHECK_EQ(qwen4exp_kva::derive(qwen4exp_kva::g_kva[0], &x.b).path, qwen4exp_kva::PATH_MASKED);
    }
}

/* A.1 -- A SPEED STRADDLE RUNS ITS LATE BLOCKS OVER THE TAIL ROWS ONLY: 64 tokens ahead of a 128-row
 * chunk (b = 64). The mask (for the split scan's bounds) ahead of layer 0, layers 0..S-1 stock with
 * no probes (a tile of exact rows already reads most experts: the pass stages), then per late layer
 * the tail-only issues above; the epilogue stock. TP1 and rank 0 of TP2 (writes carry the
 * all-reduce). */
TEST(a_speed_straddle_runs_its_late_blocks_over_the_tail_rows_only) {
    for (int world : {1, 2, 4}) {
        Pair p;
        declare_pair(p, "speed", 0, world);
        REQUIRE_EQ(p.st, RAD_OK);
        const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
        const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
        REQUIRE(k.straddle_layers);
        Batch x = make_step(p.kva, {{128}, 0, 1984});
        const qwen4exp_kva::Pass pass = qwen4exp_kva::derive(k, &x.b);
        REQUIRE_EQ(pass.path, qwen4exp_kva::PATH_STRADDLE);
        CHECK_EQ(pass.b, 64);
        CHECK(!pass.stream);
        const std::vector<RecIssue> stock = run_step(qwen4exp_fp8::step, x.b).issues;
        std::vector<rad_op> reads;
        for (int l = kSplit; l < 8; ++l) reads.push_back(m.layers[(size_t)l].hc_mix.op_read);
        const std::vector<size_t> at = starts(stock, reads, m.mixer.op_read);
        std::vector<RecIssue> want;
        for (size_t i = 0; i < at[0]; ++i) {
            if (stock[i].op == m.layers[0].hc_mix.op_read)
                want.push_back({k.op_mask, {praw(x.b.cu_seqlens, RAD_I32, 2), praw(x.b.token_ids, RAD_I32, 64),
                                praw(x.b.positions, RAD_I32, 64), RAD_NONE, brows(k.b_mask, 128),
                                brows(k.b_bounds, 4), brows(k.b_zeros, k.n_zeros)}, 64});
            want.push_back(stock[i]);
        }
        for (int l = kSplit; l < 8; ++l)
            append(want, straddle_expected(slice(stock, at[(size_t)(l - kSplit)], at[(size_t)(l - kSplit + 1)]),
                                           x, l, 64, 0));
        for (size_t i = at.back(); i < stock.size(); ++i) want.push_back(stock[i]);
        push_hazard(want, x, false);
        const Run got = run_step(qwen4exp_kva::step, x.b);
        CHECK_EQ(differ_at(got.issues, want), 0);
        CHECK(has(got.log, "straddle, stage stock, split)"));
        CHECK_EQ(got.device_calls, 0);
    }
}

/* R53' SUBSET -- DECODERS AND OTHER PREFILLS BESIDE THE APPROXIMATED CHUNK: the in-tree decode half
 * and the other prefill sequences' scan are issued as stock; only the last sequence is corrected
 * (M = 1 on its own index row), and its bulk superset starts at s_lb. */
TEST(a_mixed_step_corrects_only_the_last_sequence) {
    struct Case { Shape s; int64_t b, s_lb; };
    Env stream_rows({{"RADIANCE_KVA_STAGE_ROWS", "4096"}});   /* the issue shape, not the guard */
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

/* R53' -- THE FULL MIXED SET: D in {1, 4} decoders, verifying one row or 1 + n_spec 3 rows each, beside
 * one or two prefill chunks, the last whole-bulk or straddling, in speed and quality. Each step is the
 * in-tree step (its decode half over [0, DT) with num_accepted, the other prefill's scan as stock)
 * with only the masked path's substitutions; the correction and the decay sums take M = 1 on index
 * row n_seq - 1; the drop and every connection op run at M = n_tok. */
TEST(r53_mixed_steps_are_the_in_tree_step_with_only_the_last_sequence_masked) {
    struct Case { Shape s; int64_t b, s_lb; };
    Env stream_rows({{"RADIANCE_KVA_STAGE_ROWS", "4096"}});   /* the issue shape, not the guard */
    const std::vector<Case> cases = {
        {{{1, 128}, 1, 2048}, 129, 1},
        {{{1, 1, 1, 1, 128}, 4, 2048}, 132, 4},
        {{{4, 128}, 1, 2048, 4096, 3}, 132, 4},
        {{{4, 4, 4, 4, 128}, 4, 2048, 4096, 3}, 144, 16},
        {{{1, 64, 128}, 1, 2048}, 193, 65},
        {{{4, 4, 4, 4, 64, 128}, 4, 2048, 4096, 3}, 208, 80},
        {{{1, 1, 1, 1, 128}, 4, 1984}, 68, 4},                  /* straddle: b = n_tok - 64 */
        {{{4, 4, 4, 4, 64, 128}, 4, 1984, 4096, 3}, 144, 80},   /* straddle beside a prefill */
    };
    for (const char* mode : {"quality", "speed"})
        for (const Case& c : cases) {
            const bool quality = !std::strcmp(mode, "quality");
            Pair p;
            declare_pair(p, mode, 0, 1, 0, c.s.n_spec);
            REQUIRE_EQ(p.st, RAD_OK);
            Batch x = make_step(p.kva, c.s);
            const qwen4exp_kva::Pass pass = qwen4exp_kva::derive(qwen4exp_kva::g_kva[0], &x.b);
            const bool decoders = !quality && c.b == x.b.n_tok && x.b.n_seq - x.b.n_seq_decode == 1;
            CHECK_EQ(pass.path, decoders ? qwen4exp_kva::PATH_DECODERS : qwen4exp_kva::PATH_MASKED);
            if (decoders) {
                const Run got = run_step(qwen4exp_kva::step, x.b);
                CHECK_EQ(differ_at(got.issues, decoders_expected(x)), 0);
                CHECK(has(got.log, "decoders, stage stream"));
                continue;
            }
            Want w;
            w.b = c.b; w.s_lb = c.s_lb; w.rho_rows = c.b;
            w.stream = w.project = w.correct = true;
            w.rho = w.scored = quality;
            w.split = c.b < x.b.n_tok;
            const Run got = run_step(qwen4exp_kva::step, x.b);
            CHECK_EQ(differ_at(got.issues, masked_expected(x, w)), 0);
            CHECK_EQ(count(got.log, "kva: approximate step"), 1);
            CHECK_EQ(got.device_calls, 0);
        }
}

/* R54'S NEGATIVE CONTROL (RADIANCE_KVA_MASK=all): the mask op is declared in `step` mode (every row
 * before the bulk end approximated, decoders included) and the projector covers the step from row
 * 0, so the decoders' rows are projected, selected and dropped like bulk rows; otherwise the issue
 * sequence is the masked path's. It is said loudly at declare, and plumb (which projects nothing)
 * refuses it. */
TEST(the_mask_all_control_approximates_every_row_before_the_bulk_end) {
    Env e({{"RADIANCE_KVA_MASK", "all"}});
    Pair p;
    const std::string log = stderr_of([&] { declare_pair(p, "quality"); });
    REQUIRE_EQ(p.st, RAD_OK);
    CHECK(has(log, "DEBUG RADIANCE_KVA_MASK=all"));
    int step_mode = 0;
    for (const RecOp& o : p.kva.ops)
        for (const RecParam& q : o.p) step_mode += o.op == "kva_mask" && q.key == "mode" && q.sval == "step";
    CHECK_EQ(step_mode, 1);
    Batch x = make_step(p.kva, {{1, 1, 128}, 2, 2048});
    Want w;
    w.b = 130; w.s_lb = 0; w.rho_rows = 130;
    w.stream = w.project = w.correct = w.rho = true;
    const Run got = run_step(qwen4exp_kva::step, x.b);
    CHECK_EQ(differ_at(got.issues, masked_expected(x, w)), 0);
    CHECK(has(got.log, "s_lb 0,"));
    Pair q;
    const std::string err = stderr_of([&] { declare_pair(q, "plumb"); });
    CHECK_EQ(q.st, RAD_E_INVAL);
    CHECK(has(err, "RADIANCE_KVA_MASK=all"));
}

/* DD-A's INSTRUMENT (Stage C, R65): speed and quality declare one LINEAR meta group bound to the
 * first late delta-net layer (so checkpoints snapshot it), the kva_hazard op and a host-mapped counter;
 * plumb and off declare none. A stock pass whose last sequence still has tail ahead (n_ahead < T)
 * ends with the op, counting only (no bounds); a stock pass with T or more ahead and a decode-only
 * step issue nothing. The counter is read on the host for the log alone. */
TEST(the_hazard_instrument_counts_on_tail_passes_and_records_on_approximate_ones) {
    Pair p;
    declare_pair(p, "quality");
    REQUIRE_EQ(p.st, RAD_OK);
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
    REQUIRE(k.op_hazard != 0 && k.kv_meta != 0);
    CHECK_EQ(std::count(p.kva.kv_groups.begin(), p.kva.kv_groups.end(), "kv_kva_meta"), 1);
    CHECK_EQ(k.meta_layer, kSplit);
    int binds = 0;
    for (auto [layer, g] : p.kva.binds)
        if (g == k.kv_meta) { CHECK_EQ(layer, kSplit); ++binds; }
    CHECK_EQ(binds, 1);
    REQUIRE(qwen4exp_kva::g_hazard_dev[0] != nullptr);
    Batch tail = make_step(p.kva, {{128}, 0, 0});          /* the final chunk: stock, n_ahead 0 */
    const Run got = run_step(qwen4exp_kva::step, tail.b);
    std::vector<RecIssue> want = run_step(qwen4exp_fp8::step, tail.b).issues;
    want.push_back({k.op_hazard, {praw(tail.b.cu_seqlens, RAD_I32, 2), praw(tail.b.positions, RAD_I32, 128), RAD_NONE,
                    praw(tail.b.positions, RAD_I32, k.cfg.tail), kv_cache(k.kv_meta, k.meta_layer),
                    last_row(tail, k.kv_meta), praw(qwen4exp_kva::g_hazard_dev[0], RAD_F32, 1)}, 1});
    CHECK_EQ(differ_at(got.issues, want), 0);
    for (const Shape& s : {Shape{{128}, 0, 4096}, Shape{{1, 1}, 2, 0}}) {   /* T+ ahead but stock; decode only */
        Batch x = make_step(p.kva, s);
        if (s.D == 0) x.b.n_mm_rows = 1;                 /* a media step: stock whatever is ahead */
        CHECK_EQ(differ(run_step(qwen4exp_kva::step, x.b).issues, run_step(qwen4exp_fp8::step, x.b).issues), 0);
    }
    float* host = (float*)rad_dev_host_ptr(qwen4exp_kva::g_hazard_dev[0]);
    REQUIRE(host != nullptr);
    *host = qwen4exp_kva::g_hazard_logged[0] + 1144.0f;
    CHECK(has(run_step(qwen4exp_kva::step, tail.b).log, "kva: hazard 1144 positions"));
    CHECK(!has(run_step(qwen4exp_kva::step, tail.b).log, "kva: hazard"));   /* said once */
    Pair q;
    declare_pair(q, "plumb");
    REQUIRE_EQ(q.st, RAD_OK);
    CHECK(qwen4exp_kva::g_kva[0].op_hazard == 0);
    CHECK_EQ(std::count(q.kva.kv_groups.begin(), q.kva.kv_groups.end(), "kv_kva_meta"), 0);
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
    std::vector<RecIssue> tail = slice(want.issues, ws.back(), want.issues.size());
    push_hazard(tail, x, true);
    CHECK_EQ(differ(slice(got.issues, gs.back(), got.issues.size()), tail), 0);
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
        CHECK(seg[0].opd[1].kind == RAD_OPK_RAW && seg[0].opd[1].raw == k.proj_w[(size_t)l].raw);
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

/* AT TP2 EACH RANK HANDS THE CORRECTION ITS OWN VALUE HEADS: each rank copied rows [rank*H,
 * rank*H + H) of the folder's [48, V, K] tensor at declare (the_raw_operands_point_at_their_tensors_
 * copies), and its undo and apply issue exactly that copy. */
TEST(at_tp2_each_rank_corrects_its_own_heads) {
    for (int rank : {0, 1}) {
        Pair p;
        declare_pair(p, "speed", rank, 2);
        REQUIRE_EQ(p.st, RAD_OK);
        const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
        const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[rank];
        Batch bk = make_step(p.kva, {{128}, 0, 2048});
        const Run r = run_step(qwen4exp_kva::step, bk.b, rank);
        int seen = 0;
        for (const RecIssue& i : r.issues) {
            if (i.op != k.op_undo[kSplit] && i.op != k.op_apply[kSplit]) continue;
            ++seen;
            REQUIRE_EQ(i.opd.size(), (size_t)8);
            CHECK(i.opd[4].kind == RAD_OPK_RAW && i.opd[4].raw == k.st[kSplit].raw);
            CHECK_EQ(i.opd[4].rows, m.gcfg.n_head_v);
        }
        CHECK_EQ(seen, 2);
    }
}

/* plumb runs the late layers exactly and issues no correction, so it places none; it declares no
 * projector op, no select, no drop and no stream copy either, and copies nothing of the folder (it
 * takes only S, the split, from it). */
TEST(plumb_declares_no_correction) {
    Pair p;
    declare_pair(p, "plumb");
    REQUIRE_EQ(p.st, RAD_OK);
    CHECK_EQ(p.kva.weights.size(), p.stock.weights.size());
    CHECK_EQ(g_mem.copies.size(), (size_t)0);
    CHECK_EQ(qwen4exp_kva::g_kva[0].split, (int64_t)kSplit);
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
    RadBuilder kva;   /* the same folder and copies as the plain declare's */
    served(kva);
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

/* R61's instrument -- RADIANCE_KVA_CAPTURE_STATE ON A MIXED STEP copies, after the step, every
 * sequence's state of every late delta-net layer (sequence-major, its own index row), in off (the
 * stock step) and on a masked approximate step alike; the step's own issues are unchanged. */
TEST(a_mixed_step_capture_copies_every_sequences_late_states_after_the_step) {
    for (const char* mode : {"off", "quality"}) {
        TempDir dir;
        REQUIRE(!dir.path.empty());
        Env env({{"RADIANCE_KVA_STAGE_ROWS", "4096"}, {"RADIANCE_KVA_CAPTURE_STATE", dir.path.c_str()}});
        Pair q;
        declare_pair(q, mode);
        REQUIRE_EQ(q.st, RAD_OK);
        const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
        const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
        Batch x = make_step(q.kva, {{1, 1, 128}, 2, 2048});
        Want w;   /* quality: the masked step of a_mixed_step_corrects_only_the_last_sequence */
        w.b = 130; w.s_lb = 2; w.rho_rows = 130;
        w.stream = w.project = w.correct = w.rho = w.scored = true;
        const std::vector<RecIssue> plain = std::strcmp(mode, "off") ? masked_expected(x, w)
                                                                     : run_step(qwen4exp_fp8::step, x.b).issues;
        const Run got = run_step(qwen4exp_kva::step, x.b);
        REQUIRE(got.issues.size() == plain.size() + 9);
        CHECK_EQ(differ_at(std::vector<RecIssue>(got.issues.begin(), got.issues.begin() + (long)plain.size()), plain), 0);
        for (int seq = 0; seq < 3; ++seq)
            for (int j = 0; j < 3; ++j) {
                const RecIssue& r = got.issues[plain.size() + (size_t)(seq * 3 + j)];
                CHECK_EQ(r.op, k.op_state_read);
                CHECK(same_operand(r.opd[0], kv_cache(m.kv_state, std::vector<int>{4, 5, 6}[(size_t)j])));
                CHECK(same_operand(r.opd[1], praw2(x.b.kv[m.kv_state - 1].state_index + seq * 3, RAD_I32, 1, 3)));
            }
        const std::string lines = slurp(dir.path / "mixed.jsonl");
        CHECK_EQ(count(lines, "\n"), 1);
        CHECK(has(lines, "\"n_seq\": 3, \"n_seq_decode\": 2, \"cu\": [0, 1, 2, 130]"));
        CHECK(has(lines, std::strcmp(mode, "off") ? "\"approximate\": true" : "\"approximate\": false"));
        std::string descr;
        const std::filesystem::path f = find_file(dir.path, "mixed.p", ".r0.npy");
        REQUIRE(!f.empty());
        CHECK(npy_shape(f, &descr) == std::vector<int64_t>({3, 3, m.gcfg.n_head_v, 128, 128}));
    }
}

/* Gate 1's instrument -- RADIANCE_KVA_DUMP_LOGITS writes, after a pass with output rows, each rank's
 * logits rows and what each row is (its sequence by cu_seqlens, position, token), in any mode, off
 * included, and changes nothing the step issues. */
TEST(the_logits_dump_names_each_rows_sequence_and_changes_no_issue) {
    TempDir dir;
    REQUIRE(!dir.path.empty());
    Env e({{"RADIANCE_KVA_DUMP_LOGITS", dir.path.c_str()}});
    Pair p;
    declare_pair(p, "off");
    REQUIRE_EQ(p.st, RAD_OK);
    Batch x = make_step(p.kva, {{1, 1, 128}, 2, 0});
    static int32_t out_ids[3] = {0, 1, 129};
    x.b.n_out = 3;
    x.b.out_ids = out_ids;
    const Run got = run_step(qwen4exp_kva::step, x.b);
    CHECK_EQ(differ(got.issues, run_step(qwen4exp_fp8::step, x.b).issues), 0);
    const std::string lines = slurp(dir.path / "logits.jsonl");
    CHECK_EQ(count(lines, "\n"), 1);
    CHECK(has(lines, "\"rows\": [[0, 0, 0, 4096, 1000], [1, 1, 1, 4096, 1001], [2, 129, 2, 4223, 1129]]"));
    std::string descr;
    const std::filesystem::path f = find_file(dir.path, "logits.p", ".r0.npy");
    REQUIRE(!f.empty());
    CHECK(npy_shape(f, &descr) == std::vector<int64_t>({3, qwen4exp_fp8::g_model[0].g.n_vocab}));
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
        if (o.op != "kva_gemm_nt_bias") continue;
        for (const RecParam& q : o.p) {
            if (q.key == "N") CHECK_EQ(q.ival, 2560LL);
            if (q.key == "K") CHECK_EQ(q.ival, 10240LL);
            if (q.key == "M") CHECK_EQ(q.ihi, 2048LL);
        }
    }
    CHECK_EQ(n["kva_gemm_nt_bias"], 8 - kSplit);
    CHECK_EQ(n["quant_act_i8g"], 1);
    CHECK_EQ(n["kva_state_correct"], 2 * 3);
    CHECK_EQ(n["kva_mask"], 1);
    CHECK_EQ(n["cast"], 2);   /* the layer-S stream into h_S, and the staging ring's copy (always on) */
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
    {   /* a sizing declare alone (no real one before it in this process) copies nothing */
        RadModelMeta meta = flash_next_meta();
        RadBuildCtx c = served_ctx();
        RadBuilder real;
        REQUIRE_EQ(qwen4exp_fp8::declare(&real, &meta, &c), RAD_OK);   /* the in-tree model only */
        c.max_tok = 256;
        c.shape_probe = 1;
        RadBuilder small;
        served(small);
        hold_kva(small, {"kva.proj", "kva.st"});
        hold_score(small, "kva.rowsel.score");
        Env env({{"RADIANCE_KVA", "quality"}});
        CHECK_EQ(qwen4exp_kva::declare(&small, &meta, &c), RAD_OK);
        CHECK(g_mem.copies.empty() && !qwen4exp_kva::g_upload[0].done);
    }
    for (const char* mode : {"plumb", "speed", "quality"}) {
        Pair p;
        declare_pair(p, mode);
        REQUIRE_EQ(p.st, RAD_OK);
        RadModelMeta meta = flash_next_meta();
        RadBuildCtx c = served_ctx();
        c.max_tok = 256;
        c.shape_probe = 1;
        RadBuilder small;   /* the real declare's folder and copies stay */
        served(small);
        Env env({{"RADIANCE_KVA", mode}});
        const size_t copies = g_mem.copies.size();
        const void* block = qwen4exp_kva::g_upload[0].vram;
        REQUIRE_EQ(qwen4exp_kva::declare(&small, &meta, &c), RAD_OK);
        CHECK_EQ(g_mem.copies.size(), copies);   /* a sizing declare copies nothing */
        CHECK(qwen4exp_kva::g_upload[0].vram == block);
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

/* R84 / R100: a tail of two steps less one tile or more refuses with the numbers -- no chunk could
 * ever hold a provably-bulk row -- and every tail up to it is served (2C - G = 4032 at C = 2048). */
TEST(a_tail_of_two_steps_less_a_tile_or_more_is_refused) {
    struct Case { const char* tail; int64_t max_tok; bool ok; };
    for (const Case& c : {Case{"4096", 2048, false}, Case{"4033", 2048, false}, Case{"4032", 2048, true},
                          Case{"2560", 2048, true}, Case{"2048", 1024, false}, Case{"1984", 1024, true},
                          Case{"1024", 1024, true}}) {
        RadBuilder b;
        hold_kva(b, {"kva.proj", "kva.st"});
        std::string err;
        const int st = refused({{"RADIANCE_KVA", "speed"}, {"RADIANCE_KVA_TAIL", c.tail}}, b, &err, c.max_tok);
        CHECK_EQ(st == RAD_OK, c.ok);
        if (!c.ok) CHECK(has(err, c.tail) && has(err, std::to_string(c.max_tok).c_str()) && has(err, "tile"));
    }
}

/* R100 -- A TAIL PAST THE STEP (T = 2560, C = 2048): a capped n_ahead proves only C tokens ahead, so a
 * full chunk's first n_tok - ceil_64(T - C) = n_tok - 512 rows are bulk and the rest exact; with
 * fewer than T - C + 64 ahead nothing is provable and the pass is stock. Never the lean fill (some
 * rows of every chunk are exact): speed takes the tail-only straddle, quality the masked path when its
 * exact rows may stream (here forced by a high row threshold) and the exact step otherwise. */
TEST(a_tail_past_the_step_approximates_the_provable_rows) {
    using qwen4exp_kva::PATH_STOCK;
    using qwen4exp_kva::PATH_MASKED;
    using qwen4exp_kva::PATH_STRADDLE;
    struct Case { const char* mode; const char* rows; Shape s; int path; int64_t b, s_lb; };
    for (const Case& c : {Case{"speed", "64", {{2048}, 0, 2048}, PATH_STRADDLE, 1536, 0},
                          Case{"speed", "64", {{2048}, 0, 1024}, PATH_STRADDLE, 512, 0},
                          Case{"speed", "64", {{2048}, 0, 576}, PATH_STRADDLE, 64, 0},
                          Case{"speed", "64", {{2048}, 0, 575}, PATH_STOCK, 0, 0},
                          Case{"quality", "64", {{2048}, 0, 2048}, PATH_STOCK, 0, 0},
                          Case{"quality", "4096", {{2048}, 0, 2048}, PATH_MASKED, 1536, 0},
                          Case{"quality", "", {{2048}, 0, 2048}, PATH_MASKED, 1536, 0},   /* the default */
                          Case{"speed", "4096", {{1, 1, 1, 1, 1, 1, 1, 1, 1984}, 8, 2048}, PATH_MASKED, 1480, 8}}) {
        Env e({{"RADIANCE_KVA_TAIL", "2560"}, {"RADIANCE_KVA_STAGE_ROWS", c.rows}});
        Pair p;
        declare_pair(p, c.mode);
        REQUIRE_EQ(p.st, RAD_OK);
        Batch x = make_step(p.kva, c.s);
        const qwen4exp_kva::Pass got = qwen4exp_kva::derive(qwen4exp_kva::g_kva[0], &x.b);
        CHECK_EQ(got.path, c.path);
        CHECK_EQ(got.b, c.b);
        CHECK_EQ(got.s_lb, c.s_lb);
    }
}

TEST(a_tail_below_the_measured_minimum_is_refused) {
    RadBuilder b;
    hold_kva(b, {"kva.proj"});
    std::string err;
    CHECK(refused({{"RADIANCE_KVA", "speed"}, {"RADIANCE_KVA_TAIL", "256"}}, b, &err) < 0);
    CHECK(has(err, "256") && has(err, "512"));
}

/* Each kva.so op a mode issues is refused by name when no kernel library serves it -- plumb's mask
 * included. (`cast` and `moe_gemm_q` are in-tree ops too: refusing them fails the in-tree declare
 * first, which is its own refusal.) */
TEST(a_mode_whose_kva_op_no_kernel_serves_is_refused_by_name) {
    struct Case { const char* mode; const char* op; };
    for (const Case& c : {Case{"speed", "kva_state_correct"}, Case{"quality", "kva_mask"},
                          Case{"plumb", "kva_mask"}, Case{"speed", "kva_select"},
                          Case{"speed", "kva_drop_rows"}, Case{"quality", "kva_rho_update"},
                          Case{"speed", "kva_gemm_nt_bias"}}) {
        RadBuilder b;
        hold_kva(b, {"kva.proj", "kva.st"});
        hold_score(b, "kva.rowsel.score");
        b.refuse = {c.op};
        std::string err;
        CHECK_EQ(refused({{"RADIANCE_KVA", c.mode}}, b, &err), RAD_E_UNSUPPORTED);
        CHECK(has(err, c.op) && has(err, "kva.so"));
    }
}

TEST(a_switch_with_an_unknown_value_is_refused_naming_the_values) {
    struct Case { const char* name; const char* value; const char* allowed; };
    for (const Case& c : {Case{"RADIANCE_KVA", "fast", "off|plumb|speed|quality"},
                          Case{"RADIANCE_KVA_ROWSEL_TABLE", "rare", "class|none|all"},
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

/* A capture with no folder takes S from RADIANCE_KVA_CAPTURE_SPLIT, never from the container: a
 * kva.split key in the model's metadata is not read (nothing kva.* is, but kva.mode's notice). */
TEST(a_capture_without_a_folder_takes_its_split_from_the_env_not_the_container) {
    static const char* keys[kN + 1];
    static const char* vals[kN + 1];
    for (int i = 0; i < kN; ++i) { keys[i] = kKeys[i]; vals[i] = kVals[i]; }
    keys[kN] = "kva.split";
    vals[kN] = "4";
    RadModelMeta meta = flash_next_meta();
    meta.n_kv = kN + 1; meta.kv_key = keys; meta.kv_val = vals;
    RadBuildCtx c = served_ctx();
    for (const char* split : {"", "4"}) {
        RadBuilder b;
        Env env({{"RADIANCE_KVA_CAPTURE", "/nonexistent"}, {"RADIANCE_KVA_CAPTURE_SPLIT", split}});
        int st = RAD_OK;
        const std::string err = stderr_of([&] { st = qwen4exp_kva::declare(&b, &meta, &c); });
        CHECK_EQ(st, *split ? RAD_OK : RAD_E_UNSUPPORTED);
        if (*split) CHECK_EQ(qwen4exp_kva::g_kva[0].split, (int64_t)kSplit);
        else CHECK(has(err, "RADIANCE_KVA_CAPTURE_SPLIT is unset"));
    }
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
    CHECK(qwen4exp_kva::find_shadowed("", qwen4exp_kva::kShadowSo).empty() || std::getenv("RADIANCE_HOME") != nullptr);
}

/* THE STAGING RING (DD-L, one slot since Stage E): each late layer's map + bias row block is copied
 * into the ring's one VRAM slot on lane 1 -- layer S's at the start of the pass, layer li+1's right
 * after layer li's GEMM, behind a join that waits for that GEMM to have been handed to lane 0 -- and
 * lane 0 waits for a layer's copy before its GEMM, which reads the slot. */
TEST(the_staging_ring_copies_each_map_a_layer_ahead_on_lane_1) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    Env env({{"RADIANCE_KVA", "speed"}});
    RadBuilder kva;
    served(kva);
    hold_kva(kva, {"kva.proj", "kva.st"});
    REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
    const qwen4exp_kva::Upload& u = qwen4exp_kva::g_upload[0];
    REQUIRE(k.op_ring != 0);
    CHECK_EQ(kva.ops[k.op_ring - 1].op, std::string("cast"));
    const int64_t block = 2561LL * 10240 * 2;
    /* the row blocks, then this rank's correction heads (a few MiB) */
    CHECK(u.host_bytes > (8 - kSplit) * block && u.host_bytes < (8 - kSplit) * block + (16 << 20));
    CHECK_EQ(u.vram_bytes, block);
    for (int l = kSplit; l < 8; ++l) {
        const uintptr_t slot = (uintptr_t)k.proj_w[(size_t)l].raw, other = (uintptr_t)k.proj_w[(size_t)(l ^ 1)].raw;
        CHECK(slot >= kFakeVram && slot == other);   /* one slot, every layer in turn */
        CHECK_EQ((uintptr_t)k.proj_b[(size_t)l].raw, slot + 2560u * 10240 * 2);
        CHECK_EQ((uintptr_t)k.ring_dst[(size_t)l].raw, slot);
        CHECK_EQ(k.ring_src[(size_t)l].rows, 2561);
        CHECK((uintptr_t)k.ring_src[(size_t)l].raw >= (uintptr_t)u.host + kDeviceView);
    }
    Batch bk = make_step(kva, {{2048}, 0, 2048});   /* a lean speed pass */
    const Run r = run_step(qwen4exp_kva::step, bk.b);
    std::vector<std::string> seq;   /* the ring's events and the projector GEMMs, in issue order */
    for (const RecIssue& i : r.all) {
        if (i.op == kLane) seq.push_back("lane" + std::to_string(i.n));
        else if (i.op == kJoin) seq.push_back("join" + std::to_string(i.n / 16) + std::to_string(i.n % 16));
        else if (i.op == k.op_ring) {
            int l = -1;
            for (int x = kSplit; x < 8; ++x) if (i.opd[0].raw == k.ring_src[(size_t)x].raw) l = x;
            seq.push_back("copy" + std::to_string(l));
            CHECK(i.opd[1].raw == k.ring_dst[(size_t)l].raw && i.n == 2561);
        } else if (i.op && i.op == k.op_proj[(size_t)kSplit]) seq.push_back("gemm4");
        else for (int x = kSplit + 1; x < 8; ++x) if (i.op == k.op_proj[(size_t)x]) seq.push_back("gemm" + std::to_string(x));
    }
    /* one slot: lane 0 waits for layer l's copy, its GEMM reads the slot, then lane 1 (after lane 0) copies
     * layer l + 1 into it */
    const std::vector<std::string> want = {
        "join01", "lane1", "copy4", "lane0",
        "join10", "gemm4", "join01", "lane1", "copy5", "lane0",
        "join10", "gemm5", "join01", "lane1", "copy6", "lane0",
        "join10", "gemm6", "join01", "lane1", "copy7", "lane0",
        "join10", "gemm7" };
    CHECK(seq == want);
    if (seq != want) for (const std::string& e : seq) std::fprintf(stderr, " %s", e.c_str());
}

/* ==================================================================== the int8 projector (R79) */

/* FAKE INT8 GEMM ROWS, standing in for kva.so's forwards of libr4d's: each describes a stored form
 * 256 bytes longer than the plane (so a copy's size shows whether the hook was asked) and writes
 * every plane byte XOR 0x5A, then 0x77 padding. `tag` lets two rows disagree. */
struct FakeI8 { int layouts = 0, relayouts = 0; const char* tag = "test.i8"; } g_fake_i8;

long long param(const RadParam* p, int n_p, const char* key) {
    for (int i = 0; i < n_p; ++i) if (!std::strcmp(p[i].key, key)) return p[i].ival;
    return 0;
}
int fake_i8_layout(const RadParam* p, int n_p, int opd, const RadEncoding* enc, const int* sel,
                   const RadTensor* pl, int n, RadLayout* out) {
    ++g_fake_i8.layouts;
    const bool ok = n == 1 && (opd == 2 || opd == 3) && enc->plane[sel[0]].dtype == (opd == 2 ? RAD_I8 : RAD_BF16) &&
                    param(p, n_p, "group") == 128 && pl[0].shape[0] == param(p, n_p, "N");
    if (!ok) return RAD_E_DTYPE;
    *out = RadLayout{};
    out->tag = opd == 2 ? g_fake_i8.tag : "test.i8.scale";
    out->bytes = rad_dtype_bytes(pl[0].dtype, pl[0].shape[0] * pl[0].shape[1]) + 256;
    return RAD_OK;
}
int fake_i8_relayout(const RadParam* p, int n_p, int opd, const RadEncoding* enc, const int* sel,
                     const RadTensor* pl, int n, void* dst, int64_t bytes) {
    RadLayout L;
    const char* tag = g_fake_i8.tag;
    if (fake_i8_layout(p, n_p, opd, enc, sel, pl, n, &L) != RAD_OK || L.bytes != bytes) return RAD_E_SHAPE;
    g_fake_i8.tag = tag;
    ++g_fake_i8.relayouts;
    const unsigned char* s = (const unsigned char*)pl[0].data;
    for (int64_t i = 0; i < bytes; ++i) ((unsigned char*)dst)[i] = i < bytes - 256 ? (unsigned char)(s[i] ^ 0x5A) : 0x77;
    return RAD_OK;
}
RadKernelInfo fake_i8_row(const char* name) {
    RadKernelInfo k{};
    k.name = name;
    k.op = "kva_gemm_nt_q";
    k.domain = RAD_DOMAIN_DEVICE;
    k.layout = fake_i8_layout;
    k.relayout = fake_i8_relayout;
    return k;
}
RadKernelInfo g_i8_m16 = fake_i8_row("fake_i8_m16"), g_i8_tiled = fake_i8_row("fake_i8_tiled");
std::vector<const RadKernelInfo*> g_i8_rows = { &g_i8_m16, &g_i8_tiled };

/* hold_kva's folder with the maps in int8 (codes + scale + the bf16 bias) and the manifest saying
 * so; the fake int8 rows stand in for kva.so. */
void hold_kva_i8(RadBuilder& b) {
    hold_kva(b, {"kva.st"});
    const std::string text = std::string(R"({"format": 1, "adapter": "qwen4exp", "split": 4,
        "projector": {"dtype": "i8", "layout": "i8_row128"},
        "model": {"arch_id": "qwen4exp", "name": "test-q38-flashnext",
                  "meta": {"hc_count": "4", "linear_num_value_heads": "48"},
                  "vocab_sha256": ")") + kTinyVocab + R"("},
        "files": {"proj8.L4.safetensors": "unused by the test folder"}})";
    g_test_folder.manifest = qwen4exp_kva::Json{};
    qwen4exp_kva::json_parse(text.data(), text.size(), &g_test_folder.manifest);
    for (int l = kSplit; l < 8; ++l) {
        const std::string L = std::to_string(l);
        add_tensor("proj." + L + ".codes", RAD_I8, {2560, 4 * 2560});
        add_tensor("proj." + L + ".scale", RAD_BF16, {2560, 4 * 2560 / 128});
        add_tensor("proj." + L + ".bias", RAD_BF16, {2560});
    }
    g_fake_i8 = FakeI8{};
    qwen4exp_kva::g_i8_rows_for_test = &g_i8_rows;
}

/* The int8 folder's declare: the quantiser over the stream, the bias add, one int8 GEMM a late layer
 * (no bf16 GEMM), the stream's codes and scales taking the whole program; each map copied in the
 * stored form the GEMM rows' hooks made (their size, their bytes), shaped as the GEMM reads it. */
TEST(an_int8_folder_uploads_its_maps_in_the_gemms_stored_form) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    Env env({{"RADIANCE_KVA", "quality"}});
    RadBuilder kva;
    served(kva);
    hold_kva_i8(kva);
    hold_score(kva, "kva.rowsel.score");
    int64_t at = 0;   /* distinct source bytes per tensor */
    for (auto& [name, t] : g_test_folder.tensors) { t.data = tensor_bytes() + at; at += 256; }
    const std::string log = stderr_of([&] { CHECK_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK); });
    const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
    REQUIRE(k.int8);
    std::map<std::string, int> n;
    for (const RecOp& o : kva.ops) {
        ++n[o.op];
        if (o.op != "kva_gemm_nt_q") continue;
        CHECK(o.w.empty());
        for (const RecParam& q : o.p) {
            if (q.key == "M") CHECK(q.kind == RAD_P_RANGE && q.ival == 1 && q.ihi == 2048);
            if (q.key == "N") CHECK_EQ(q.ival, 2560);
            if (q.key == "K") CHECK_EQ(q.ival, 10240);
            if (q.key == "group") CHECK_EQ(q.ival, 128);
            if (q.key == "dtype") CHECK_EQ(q.sval, std::string("i8a8"));
        }
    }
    CHECK_EQ(n["kva_gemm_nt_q"], 8 - kSplit);
    CHECK_EQ(n["kva_gemm_nt_bias"], 0);
    CHECK_EQ(kva.ops[k.op_quant8 - 1].op, std::string("quant_act_i8g"));
    CHECK_EQ(kva.ops[k.op_bias - 1].op, std::string("add"));
    for (const RecParam& q : kva.ops[k.op_quant8 - 1].p) if (q.key == "n") CHECK_EQ(q.ival, 10240);
    for (const RecParam& q : kva.ops[k.op_bias - 1].p) if (q.key == "n") CHECK_EQ(q.ival, 2560);
    CHECK_EQ(kva.bufs[k.b_q8 - 1].second.dtype, (uint32_t)RAD_I8);
    CHECK_EQ(kva.bufs[k.b_s8 - 1].second.dtype, (uint32_t)RAD_F32);
    CHECK(kva.concurrent.count(k.b_q8) && kva.concurrent.count(k.b_s8));
    CHECK_EQ(g_fake_i8.relayouts, 2 * (8 - kSplit));
    /* The row block, from the hook's sizes (each 256 bytes past the plane): codes at 0, scales at the
     * next 256-byte boundary, the bias after them; whole rows of hc*n bf16. */
    const int64_t codes = 2560LL * 10240 + 256, scales = 2560LL * 80 * 2 + 256;
    const int64_t scale_at = (codes + 255) / 256 * 256, bias_at = (scale_at + scales + 255) / 256 * 256;
    const int64_t rows = (bias_at + 5120 + 20479) / 20480;
    auto relaid = [](const unsigned char* got, const unsigned char* src, int64_t plane) {
        bool ok = true;
        for (int64_t i = 0; i < plane + 256 && ok; ++i) ok = got[i] == (i < plane ? (unsigned char)(src[i] ^ 0x5A) : 0x77);
        return ok;
    };
    for (int l = kSplit; l < 8; ++l) {
        const std::string L = std::to_string(l);
        const RadOperand &w = k.proj_w[(size_t)l], &sc = k.proj_s[(size_t)l], &b = k.proj_b[(size_t)l];
        CHECK(w.dtype == RAD_I8 && w.rows == 2560 && w.cols == 10240 && w.raw == k.ring_dst[(size_t)l].raw);
        CHECK(sc.dtype == RAD_BF16 && sc.rows == 2560 && sc.cols == 80);
        CHECK_EQ((uintptr_t)sc.raw, (uintptr_t)w.raw + (uintptr_t)scale_at);
        CHECK_EQ((uintptr_t)b.raw, (uintptr_t)w.raw + (uintptr_t)bias_at);
        CHECK_EQ(k.ring_src[(size_t)l].rows, rows);
        CHECK_EQ(k.ring_dst[(size_t)l].rows, rows);
        const unsigned char* block = (const unsigned char*)k.ring_src[(size_t)l].raw - kDeviceView;
        CHECK(relaid(block, g_test_folder.tensors["proj." + L + ".codes"].data, 2560LL * 10240));
        CHECK(relaid(block + scale_at, g_test_folder.tensors["proj." + L + ".scale"].data, 2560LL * 80 * 2));
        CHECK(std::memcmp(block + bias_at, g_test_folder.tensors["proj." + L + ".bias"].data, 5120) == 0);
    }
    CHECK(has(log, "(int8 maps, correction, row table)"));
    CHECK(kva.notes.size() && has(kva.notes.back(), "(int8, streamed from host)"));
}

/* Each issue as text, with this declare's handles named: the KVA ops by role, every other op and
 * buffer by its declared name (op names numbered by occurrence), projector memory as "vram". Two
 * declares of different folders then compare op for op although their handles are numbered apart. */
std::vector<std::string> named(const std::vector<RecIssue>& v, const RadBuilder& b, const qwen4exp_kva::Kva& k) {
    std::map<rad_op, std::string> ops;
    std::map<std::string, int> seen;
    for (size_t i = 0; i < b.ops.size(); ++i) ops[(rad_op)(i + 1)] = b.ops[i].op + "#" + std::to_string(seen[b.ops[i].op]++);
    for (int l = kSplit; l < 8; ++l) ops[k.op_proj[(size_t)l]] = "PROJ" + std::to_string(l);
    if (k.op_quant8) ops[k.op_quant8] = "QUANT8";
    if (k.op_bias) ops[k.op_bias] = "BIAS";
    std::vector<std::string> out;
    for (const RecIssue& r : v) {
        if (r.op && r.op == k.op_ring) {   /* the row block differs by dtype: the layer it moves is what matters */
            int l = -1;
            for (int x = kSplit; x < 8; ++x) if (r.opd[1].raw == k.ring_dst[(size_t)x].raw && r.opd[0].raw == k.ring_src[(size_t)x].raw) l = x;
            out.push_back("RING layer " + std::to_string(l));
            continue;
        }
        std::string t = (ops.count(r.op) ? ops[r.op] : std::to_string(r.op)) + " n" + std::to_string(r.n);
        for (const RadOperand& o : r.opd) {
            const bool vram = o.kind == RAD_OPK_RAW && (uintptr_t)o.raw >= kFakeVram && (uintptr_t)o.raw < kFakeVramEnd;
            t += " |" + std::to_string(o.kind) + "/" + std::to_string(o.dtype) + "/";
            t += o.kind == RAD_OPK_BUF && o.handle ? b.bufs[o.handle - 1].first : vram ? std::string("vram")
               : std::to_string(o.handle) + "@" + std::to_string((uintptr_t)o.raw);
            t += "/" + std::to_string(o.offset) + "/" + std::to_string(o.rows) + "/" + std::to_string(o.cols);
        }
        out.push_back(t);
    }
    return out;
}

/* A buffer operand of declare `from` re-pointed at the buffer of the same name in declare `to`. */
RadOperand rebuf(RadOperand o, const RadBuilder& from, const RadBuilder& to) {
    if (o.kind != RAD_OPK_BUF || !o.handle) return o;
    for (size_t i = 0; i < to.bufs.size(); ++i)
        if (to.bufs[i].first == from.bufs[o.handle - 1].first) { o.handle = (uint32_t)(i + 1); return o; }
    o.handle = 0;
    return o;
}

/* The bf16 run's issues, named, with each projector GEMM replaced by what the int8 projector issues:
 * at layer S the stream's codes from the GEMM's input rows, then per layer the int8 GEMM over those
 * codes into the same destination, then the bias added in place. */
std::vector<std::string> as_int8(const std::vector<RecIssue>& bf16, const RadBuilder& bb, const qwen4exp_kva::Kva& kb,
                                 const RadBuilder& b8, const qwen4exp_kva::Kva& k8) {
    std::vector<std::string> out;
    for (const RecIssue& r : bf16) {
        int l = -1;
        for (int x = kSplit; x < 8; ++x) if (r.op == kb.op_proj[(size_t)x]) l = x;
        /* int8 keeps no h_S (no MTP here): the bf16 pass's copy into it has no int8 counterpart, and the codes
         * are made from b_h's same rows, which still hold the layer-S stream at layer S */
        if (r.op == kb.op_cast && r.opd.size() > 1 && r.opd[1].kind == RAD_OPK_BUF && r.opd[1].handle == kb.b_hs) continue;
        if (l < 0) { out.push_back(named({r}, bb, kb)[0]); continue; }
        RadOperand src = rebuf(r.opd[0], bb, b8);
        const RadOperand dst = rebuf(r.opd[4], bb, b8);
        if (r.opd[0].kind == RAD_OPK_BUF && r.opd[0].handle == kb.b_hs) src.handle = qwen4exp_fp8::g_model[0].b_h;
        RadOperand q = src, s = src;
        q.handle = k8.b_q8;
        s.handle = k8.b_s8;
        s.offset = src.offset / 128;
        std::vector<RecIssue> v;
        if (l == kSplit) v.push_back({k8.op_quant8, {src, q, s}, r.n});
        v.push_back({k8.op_proj[(size_t)l], {q, s, k8.proj_w[(size_t)l], k8.proj_s[(size_t)l], dst, RAD_NONE, RAD_NONE}, r.n});
        v.push_back({k8.op_bias, {dst, k8.proj_b[(size_t)l], dst}, r.n});
        for (const std::string& t : named(v, b8, k8)) out.push_back(t);
    }
    return out;
}

/* THE INT8 PASS IS THE BF16 PASS WITH THE PROJECTION SWAPPED, on every path that projects (lean,
 * tail-only straddle, masked, speed beside decoders): same batch, same everything else, op for op; the stream quantised
 * once, at layer S, from exactly the rows the GEMMs read. */
TEST(the_int8_projector_quantises_the_stream_once_then_gemm_and_bias_a_layer) {
    struct Case { const char* mode; Shape s; int path; };
    for (const Case& cs : {Case{"speed", {{2048}, 0, 2048}, qwen4exp_kva::PATH_LEAN},
                           Case{"speed", {{128}, 0, 1984}, qwen4exp_kva::PATH_STRADDLE},
                           Case{"quality", {{128}, 0, 2048}, qwen4exp_kva::PATH_MASKED},
                           Case{"quality", {{128}, 0, 1984}, qwen4exp_kva::PATH_MASKED},
                           Case{"quality", {{1, 64, 128}, 1, 2048}, qwen4exp_kva::PATH_MASKED},   /* s_lb 65 */
                           Case{"speed", {{1, 1984}, 1, 2048}, qwen4exp_kva::PATH_DECODERS}}) {
        RadModelMeta meta = flash_next_meta();
        RadBuildCtx c = served_ctx();
        Env env({{"RADIANCE_KVA", cs.mode}});
        RadBuilder bf, i8;
        served(bf);
        served(i8);
        hold_kva(bf, {"kva.proj", "kva.st"});
        hold_score(bf, "kva.rowsel.score");
        REQUIRE_EQ(qwen4exp_kva::declare(&bf, &meta, &c), RAD_OK);
        const qwen4exp_kva::Kva kb = qwen4exp_kva::g_kva[0];
        Batch x = make_step(bf, cs.s);
        REQUIRE_EQ(qwen4exp_kva::derive(kb, &x.b).path, cs.path);
        const Run rb = run_step(qwen4exp_kva::step, x.b);
        hold_kva_i8(i8);
        hold_score(i8, "kva.rowsel.score");
        REQUIRE_EQ(qwen4exp_kva::declare(&i8, &meta, &c), RAD_OK);
        const qwen4exp_kva::Kva& k8 = qwen4exp_kva::g_kva[0];
        REQUIRE(k8.int8 && !kb.int8);
        /* the slot follows the folder's block (1,280 rows of codes, 20 of scales, 1 of bias), and no h_S */
        CHECK_EQ(qwen4exp_kva::g_upload[0].vram_bytes, 1301LL * 10240 * 2);
        CHECK(k8.b_hs == 0 && kb.b_hs != 0);
        const Run r8 = run_step(qwen4exp_kva::step, x.b);
        const std::vector<std::string> got_t = named(r8.all, i8, k8), want_t = as_int8(rb.all, bf, kb, i8, k8);
        int quants = 0, bad = 0;
        for (const RecIssue& r : r8.issues) quants += r.op == k8.op_quant8;
        CHECK_EQ(quants, 1);
        CHECK_EQ(got_t.size(), want_t.size());
        for (size_t i = 0; i < got_t.size() && i < want_t.size(); ++i)
            if (got_t[i] != want_t[i] && bad++ < 3)
                std::fprintf(stderr, "  %s/%d issue %zu:\n    got  %s\n    want %s\n", cs.mode, cs.path, i,
                             got_t[i].c_str(), want_t[i].c_str());
        CHECK_EQ(bad, 0);
        /* and the int8 issues' projector operands are this layer's own copies */
        for (const RecIssue& r : r8.issues)
            for (int l = kSplit; l < 8; ++l) {
                if (r.op == k8.op_proj[(size_t)l])
                    CHECK(r.opd[2].raw == k8.proj_w[(size_t)l].raw && r.opd[3].raw == k8.proj_s[(size_t)l].raw);
                if (r.op == k8.op_bias && r.opd[1].raw == k8.proj_b[(size_t)l].raw) CHECK(same_operand(r.opd[0], r.opd[2]));
            }
        CHECK_EQ(r8.device_calls, 0);
    }
}

/* No int8 GEMM row in the loaded libraries, or rows that would read different stored bytes (one
 * upload must serve whichever row a step's M selects): the folder cannot run, startup fails naming
 * why. A folder whose dtype is neither is refused as one that cannot run on this model. */
TEST(an_int8_folder_without_one_agreed_stored_form_is_refused_by_name) {
    RadModelMeta meta = flash_next_meta();
    RadBuildCtx c = served_ctx();
    Env env({{"RADIANCE_KVA", "speed"}});
    RadKernelInfo other = fake_i8_row("fake_other");
    other.layout = [](const RadParam* p, int n_p, int opd, const RadEncoding* e, const int* s, const RadTensor* pl,
                      int n, RadLayout* out) {
        const int st = fake_i8_layout(p, n_p, opd, e, s, pl, n, out);
        if (opd == 2) out->tag = "test.i8.other";
        return st;
    };
    const std::vector<const RadKernelInfo*> none, disagree = { &g_i8_m16, &other };
    for (const auto* rows : { &none, &disagree }) {
        RadBuilder kva;
        served(kva);
        hold_kva_i8(kva);
        qwen4exp_kva::g_i8_rows_for_test = rows;
        int st = RAD_OK;
        const std::string log = stderr_of([&] { st = qwen4exp_kva::declare(&kva, &meta, &c); });
        CHECK(st != RAD_OK);
        CHECK(has(log, "cannot load the int8 projector"));
        CHECK(has(log, rows == &none ? "no kernel library offers the int8 GEMM" : "store the map differently"));
    }
    RadBuilder stock, kva;
    served(stock);
    served(kva);
    hold_kva_i8(kva);
    std::string text = R"({"format": 1, "adapter": "qwen4exp", "split": 4, "projector": {"dtype": "i4"},
        "model": {"arch_id": "qwen4exp", "name": "test-q38-flashnext", "meta": {"hc_count": "4"},
                  "vocab_sha256": ")" + std::string(kTinyVocab) + R"("}, "files": {}})";
    g_test_folder.manifest = qwen4exp_kva::Json{};
    qwen4exp_kva::json_parse(text.data(), text.size(), &g_test_folder.manifest);
    REQUIRE_EQ(qwen4exp_fp8::declare(&stock, &meta, &c), RAD_OK);
    const std::string log = stderr_of([&] { CHECK_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK); });
    check_same_graph(stock, kva);
    CHECK(has(log, "its projector dtype 'i4' is neither bf16 nor i8"));
}

/* ==================================================================== the MTP final map (Stage D, R71) */

/* The final map's bytes: 210 MB with a pattern, so each block's copy can be traced to its rows. */
const unsigned char* final_bytes() {
    static std::vector<unsigned char> v;
    if (v.empty()) {
        v.resize((size_t)10240 * 10240 * 2 + 10240 * 2);
        for (size_t i = 0; i < v.size(); ++i) v[i] = (unsigned char)(i * 2654435761u >> 13);
    }
    return v.data();
}

/* hold_kva's folder plus final.weight [hc*n, hc*n] and final.bias [hc*n], and the manifest saying so. */
void hold_kva_final(RadBuilder& b) {
    hold_kva(b, {"kva.proj", "kva.st"});
    hold_score(b, "kva.rowsel.score");
    const std::string text = std::string(R"({"format": 1, "adapter": "qwen4exp", "split": 4,
        "final": {"file": "final.safetensors", "dtype": "bf16"},
        "model": {"arch_id": "qwen4exp", "name": "test-q38-flashnext",
                  "meta": {"hc_count": "4", "linear_num_value_heads": "48"},
                  "vocab_sha256": ")") + kTinyVocab + R"("},
        "files": {"final.safetensors": "unused by the test folder"}})";
    g_test_folder.manifest = qwen4exp_kva::Json{};
    qwen4exp_kva::json_parse(text.data(), text.size(), &g_test_folder.manifest);
    for (auto [name, shape, at] : {std::tuple<const char*, std::vector<int64_t>, int64_t>{"final.weight", {10240, 10240}, 0},
                                   {"final.bias", {10240}, 10240LL * 10240 * 2}}) {
        qwen4exp_kva::FolderTensor t;
        t.data = final_bytes() + at;
        t.dtype = RAD_BF16;
        t.shape = shape;
        t.bytes = rad_dtype_bytes(RAD_BF16, shape.size() == 2 ? shape[0] * shape[1] : shape[0]);
        g_test_folder.tensors[name] = t;
    }
}

/* With MTP (max_spec > 0) the final map is declared, uploaded as hc row blocks of [n + 1, hc*n] -- block i holds
 * rows i*n .. i*n + n of the map and that slice of the bias -- and ridden by the ring after the last late layer;
 * without MTP, or with RADIANCE_KVA_FINAL=off, nothing of it is declared or held. */
TEST(the_final_map_is_held_and_declared_only_with_mtp) {
    RadModelMeta meta = flash_next_meta();
    for (auto [spec, sw, want] : {std::tuple<int, const char*, bool>{3, "on", true}, {0, "on", false}, {3, "off", false}}) {
        RadBuildCtx c = served_ctx();
        c.max_spec = spec;
        Env env({{"RADIANCE_KVA", "quality"}, {"RADIANCE_KVA_FINAL", sw}});
        RadBuilder kva;
        served(kva);
        hold_kva_final(kva);
        REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
        const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
        CHECK_EQ(k.want_final, want);
        CHECK_EQ(k.op_final != 0, want);
        CHECK_EQ(k.b_final != 0, want);
        CHECK_EQ(k.ring_end, (int64_t)(8 + (want ? 4 : 0)));
        CHECK_EQ(k.final_w.size(), (size_t)(want ? 4 : 0));
        if (!want) continue;
        CHECK_EQ(kva.ops[k.op_final - 1].op, std::string("kva_gemm_nt_bias"));
        CHECK(kva.concurrent.count(k.b_final));
        const int64_t row = 10240 * 2;
        for (int i = 0; i < 4; ++i) {
            const unsigned char* block = (const unsigned char*)k.ring_src[(size_t)(8 + i)].raw - kDeviceView;
            CHECK_EQ(k.ring_src[(size_t)(8 + i)].rows, 2561);
            CHECK(std::memcmp(block, final_bytes() + (size_t)i * 2560 * row, (size_t)2560 * row) == 0);
            CHECK(std::memcmp(block + 2560 * row, final_bytes() + 10240LL * row + (size_t)i * 2560 * 2, 2560 * 2) == 0);
            /* the ring's one slot, and the GEMM reads it */
            CHECK(k.ring_dst[(size_t)(8 + i)].raw == k.ring_dst[7].raw);
            CHECK(k.final_w[(size_t)i].raw == k.ring_dst[(size_t)(8 + i)].raw);
            CHECK_EQ((uintptr_t)k.final_b[(size_t)i].raw, (uintptr_t)k.final_w[(size_t)i].raw + 2560u * row);
        }
    }
}

/* R71: with MTP, every approximate path ends with the final map -- hc GEMMs over the bulk rows' layer-S stream
 * (h_S on the masked path, b_h's rows on lean and straddle), block i into columns i*n .. i*n + n of kva_final --
 * and the predicted rows go into b_h through the mask (masked, straddle) or by a row copy (lean), right before
 * the epilogue's connection read. Everything else is the pass without the map, op for op; the ring carries the
 * hc blocks after the last layer, each copied while the one before computes. */
TEST(with_mtp_the_bulk_rows_take_the_predicted_final_stream_before_the_epilogue) {
    struct Case { const char* mode; Shape s; int path; };
    for (const Case& cs : {Case{"speed", {{2048}, 0, 2048}, qwen4exp_kva::PATH_LEAN},
                           Case{"speed", {{128}, 0, 1984}, qwen4exp_kva::PATH_STRADDLE},
                           Case{"quality", {{1, 64, 128}, 1, 2048}, qwen4exp_kva::PATH_MASKED}}) {
        RadModelMeta meta = flash_next_meta();
        RadBuildCtx c = served_ctx();
        c.max_spec = 3;
        const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
        std::vector<std::string> runs[2];
        Run r[2];
        RadBuilder b[2];
        for (int on = 0; on < 2; ++on) {
            Env env({{"RADIANCE_KVA", cs.mode}, {"RADIANCE_KVA_FINAL", on ? "on" : "off"}});
            served(b[on]);
            hold_kva_final(b[on]);
            REQUIRE_EQ(qwen4exp_kva::declare(&b[on], &meta, &c), RAD_OK);
            Batch x = make_step(b[on], cs.s);
            const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
            const qwen4exp_kva::Pass p = qwen4exp_kva::derive(k, &x.b);
            REQUIRE_EQ(p.path, cs.path);
            r[on] = run_step(qwen4exp_kva::step, x.b);
            runs[on] = named(r[on].issues, b[on], k);
            if (!on) continue;
            /* the final segment: right before the mixer's read */
            const std::vector<RecIssue>& v = r[on].issues;
            size_t mix = 0;
            while (mix < v.size() && v[mix].op != m.mixer.op_read) ++mix;
            REQUIRE(mix >= 5 && mix < v.size());
            const int64_t T = x.b.n_tok, r0 = p.path == qwen4exp_kva::PATH_MASKED ? p.s_lb : 0;
            const int64_t rows = p.path == qwen4exp_kva::PATH_LEAN ? T : p.b - r0;
            for (int i = 0; i < 4; ++i) {
                const RecIssue& g = v[mix - 5 + (size_t)i];
                CHECK_EQ(g.op, k.op_final);
                CHECK_EQ(g.n, rows);
                CHECK(same_operand(g.opd[0], brow_slice(p.path == qwen4exp_kva::PATH_MASKED ? k.b_hs : m.b_h, r0, rows, 10240)));
                CHECK(g.opd[1].raw == k.final_w[(size_t)i].raw && g.opd[2].raw == k.final_b[(size_t)i].raw);
                CHECK(same_operand(g.opd[4], bcol_at(k.b_final, r0, 10240, i * 2560, 2560, rows)));
            }
            const RecIssue& in = v[mix - 1];
            if (p.path == qwen4exp_kva::PATH_LEAN) {
                CHECK_EQ(in.op, k.op_cast);
                CHECK(same_operand(in.opd[1], brows(m.b_h, T)));
            } else {
                CHECK_EQ(in.op, k.op_select);
                CHECK(same_operand(in.opd[0], brows(k.b_mask, T)) && same_operand(in.opd[4], brows(m.b_h, T)));
            }
            /* the ring: blocks 8 .. 11 copied after layer 7, each waited for before its GEMM */
            std::vector<std::string> seq;
            for (const RecIssue& i : r[on].all) {
                if (i.op == kJoin) seq.push_back("join" + std::to_string(i.n / 16) + std::to_string(i.n % 16));
                else if (i.op == k.op_ring) {
                    int j = -1;
                    for (int y = kSplit; y < 12; ++y) if (i.opd[0].raw == k.ring_src[(size_t)y].raw) j = y;
                    seq.push_back("copy" + std::to_string(j));
                } else if (i.op == k.op_final) seq.push_back("final");
            }
            const std::vector<std::string> tail(seq.end() - 15, seq.end());
            const std::vector<std::string> want = { "join10", "final", "join01", "copy9", "join10", "final", "join01",
                                                    "copy10", "join10", "final", "join01", "copy11", "join10", "final" };
            CHECK(std::vector<std::string>(tail.end() - 14, tail.end()) == want);
            CHECK(std::find(seq.begin(), seq.end(), "copy8") != seq.end());
        }
        /* everything but the final segment is the pass without the map */
        std::vector<std::string> on_wo;
        for (const std::string& t : runs[1])
            if (t.rfind("kva_gemm_nt_bias#" + std::to_string(8 - kSplit), 0) != 0) on_wo.push_back(t);
        on_wo.erase(std::remove_if(on_wo.begin(), on_wo.end(), [](const std::string& t) {
            return t.find("kva_final") != std::string::npos; }), on_wo.end());
        CHECK(on_wo == runs[0]);
        CHECK_EQ(r[1].device_calls, 0);
    }
}

/* ==================================================================== the folder on disk */

/* A safetensors file of the given tensors (all zeros), as safetensors writes one. */
std::string safetensors(const std::vector<std::tuple<std::string, std::string, std::vector<int64_t>, int>>& ts) {
    std::string header = "{", data;
    for (const auto& [name, dtype, shape, esize] : ts) {
        int64_t n = 1;
        std::string dims;
        for (int64_t e : shape) { n *= e; dims += (dims.empty() ? "" : ",") + std::to_string(e); }
        header += (header.size() > 1 ? "," : "") + std::string("\"") + name + "\":{\"dtype\":\"" + dtype +
                  "\",\"shape\":[" + dims + "],\"data_offsets\":[" + std::to_string(data.size()) + "," +
                  std::to_string(data.size() + (size_t)(n * esize)) + "]}";
        data.append((size_t)(n * esize), '\0');
    }
    header += ",\"__metadata__\":{\"x\":\"y\"}}";
    std::string out(8, '\0');
    const uint64_t len = header.size();
    std::memcpy(out.data(), &len, 8);
    return out + header + data;
}

std::string sha_of(const std::string& bytes) {
    return qwen4exp_kva::sha256_bytes((const unsigned char*)bytes.data(), bytes.size());
}

/* A folder of two files, one tensor each; its manifest lists both with their hashes. */
void write_folder(const std::filesystem::path& dir, const std::string& files_override = "") {
    const std::string a = safetensors({{"proj.4.weight", "BF16", {2, 8}, 2}, {"proj.4.bias", "BF16", {2}, 2}});
    const std::string b = safetensors({{"score", "F32", {5}, 4}});
    std::ofstream(dir / "proj.L4.safetensors", std::ios::binary) << a;
    std::ofstream(dir / "rowsel.safetensors", std::ios::binary) << b;
    std::ofstream(dir / "README.md") << "hello";
    const std::string files = files_override.empty()
        ? "{\"proj.L4.safetensors\": \"" + sha_of(a) + "\", \"rowsel.safetensors\": \"" + sha_of(b) +
              "\", \"README.md\": \"" + sha_of("hello") + "\"}"
        : files_override;
    std::ofstream(dir / "kva.json") << "{\"format\": 1, \"files\": " << files << "}";
}


/* THE STRADDLE'S DOWNGRADE IS SAID AT STARTUP (orchestrator, Stage E): speed's tail-only straddle needs every
 * late attention layer's per-row sparse form; without it straddle chunks take the masked path, which only each
 * step log's path field showed. A container with no indexer (indexer_n_heads 0) declares no selection: one line
 * names layer 7, the missing piece and the consequence, in speed only; the published geometry says nothing. */
RadModelMeta flash_next_meta_without_indexer() {
    static std::vector<const char*> vals;
    if (vals.empty()) {
        vals.assign(kVals, kVals + kN);
        for (int i = 0; i < kN; ++i)
            if (!std::strcmp(kKeys[i], "indexer_n_heads")) vals[(size_t)i] = "0";
    }
    RadModelMeta m = flash_next_meta();
    m.kv_val = vals.data();
    return m;
}

TEST(a_speed_straddle_downgraded_at_declare_is_said_once_at_startup) {
    for (bool indexer : {true, false})
        for (const char* mode : {"speed", "quality"}) {
            RadModelMeta meta = indexer ? flash_next_meta() : flash_next_meta_without_indexer();
            RadBuildCtx c = served_ctx();
            RadBuilder kva;
            served(kva);
            hold_kva(kva, {"kva.proj", "kva.st"});
            hold_score(kva, "kva.rowsel.score");
            Env env({{"RADIANCE_KVA", mode}});
            int st = RAD_OK;
            const std::string log = stderr_of([&] { st = qwen4exp_kva::declare(&kva, &meta, &c); });
            REQUIRE_EQ(st, RAD_OK);
            const bool speed = !std::strcmp(mode, "speed");
            CHECK_EQ(qwen4exp_kva::g_kva[0].straddle_layers, speed && indexer);
            CHECK_EQ(qwen4exp_kva::straddle_missing(qwen4exp_fp8::g_model[0].layers[7].attn) == nullptr, indexer);
            CHECK_EQ(count(log, "takes the masked path for straddle chunks"), speed && !indexer ? 1 : 0);
            if (speed && !indexer) CHECK(has(log, "late attention layer 7 lacks the indexer's selection (qsa_sel)"));
        }
}

/* R74 and R76's static half -- MEDIA STEPS RUN STOCK, AND THE TEXT AFTER AN IMAGE RESUMES (PLAN-FIX §6.3), on
 * the published container's geometry for pictures: interleaved M-RoPE 11/11/10 and a vision tower (cut to two
 * blocks, as radiance's arch_test cuts it), served with a patch budget. An encoder pass, a chunk holding media
 * rows (mixed rotary components or not) and a chunk with mixed components issue the in-tree step op for op in
 * speed and quality. A text-only chunk after the image (rope_pos set, its three components equal) is
 * approximated, and every in-tree op it issues takes the position operand the in-tree step gives that op --
 * the rotary planes, which run behind the index after an image, where the stock step reads them. */
RadModelMeta flash_next_vl_meta() {
    static std::vector<const char*> keys, vals;
    if (keys.empty()) {
        keys.assign(kKeys, kKeys + kN);
        vals.assign(kVals, kVals + kN);
        const char* k[] = {"rope_parameters.mrope_section", "rope_parameters.mrope_interleaved",
                           "vision_config.depth", "vision_config.hidden_size", "vision_config.num_heads",
                           "vision_config.intermediate_size", "vision_config.patch_size",
                           "vision_config.temporal_patch_size", "vision_config.in_channels",
                           "vision_config.spatial_merge_size", "vision_config.out_hidden_size",
                           "vision_config.num_position_embeddings", "vision_config.deepstack_visual_indexes",
                           "vision_config.hidden_act"};
        const char* v[] = {"11 11 10", "1", "2", "1152", "16", "4304", "16", "2", "3", "2", "2560", "2304", "",
                           "gelu_pytorch_tanh"};
        keys.insert(keys.end(), std::begin(k), std::end(k));
        vals.insert(vals.end(), std::begin(v), std::end(v));
    }
    RadModelMeta m = flash_next_meta();
    m.n_kv = (int)keys.size();
    m.kv_key = keys.data();
    m.kv_val = vals.data();
    return m;
}

/* The position-like operands of an issue, in order: 'P' this pass's index positions, 'R<rows>' its rotary
 * planes (plane 0 alone or all three). */
std::string position_tags(const RecIssue& r, const RadBatch& b) {
    std::string t;
    for (const RadOperand& o : r.opd) {
        if (o.raw && o.raw == (const void*)b.positions) t += "P";
        if (o.raw && o.raw == (const void*)b.rope_pos) t += "R" + std::to_string(o.rows);
    }
    return t;
}

TEST(media_steps_run_stock_and_the_text_after_an_image_approximates_at_its_rotary_positions) {
    static int32_t rp[3 * 4096], mm[8];
    static uint16_t embd[8 * 2560], pix[64 * 1536];
    static int32_t coord[4 * 64], ecu[2] = {0, 64};
    using qwen4exp_kva::PATH_STOCK;
    struct Edit { const char* why; std::function<void(RadBatch&)> f; };
    const std::vector<Edit> media = {
        {"encoder pass", [](RadBatch& b) { b.enc = 1; b.enc_n_patch = 64; b.enc_pixels = pix; b.enc_coord = coord;
                                           b.enc_n_seg = 1; b.enc_cu = ecu; b.enc_max_seg = 64; }},
        {"media rows, mixed components", [](RadBatch& b) { b.rope_pos = rp; b.rope_mixed = 1; b.n_mm_rows = 8;
                                                           b.mm_rows = mm; b.mm_embd = embd; }},
        {"media rows", [](RadBatch& b) { b.rope_pos = rp; b.n_mm_rows = 8; b.mm_rows = mm; b.mm_embd = embd; }},
        {"mixed components", [](RadBatch& b) { b.rope_pos = rp; b.rope_mixed = 1; }},
    };
    for (const char* mode : {"speed", "quality"}) {
        RadModelMeta meta = flash_next_vl_meta();
        RadBuildCtx c = served_ctx();
        c.max_enc_patches = 4096;
        RadBuilder kva;
        served(kva);
        hold_kva(kva, {"kva.proj", "kva.st"});
        hold_score(kva, "kva.rowsel.score");
        Env env({{"RADIANCE_KVA", mode}});
        REQUIRE_EQ(qwen4exp_kva::declare(&kva, &meta, &c), RAD_OK);
        const qwen4exp_kva::Kva& k = qwen4exp_kva::g_kva[0];
        REQUIRE(qwen4exp_fp8::g_model[0].media && qwen4exp_fp8::g_model[0].g.rope_mc && k.have_proj);
        for (const Edit& e : media) {
            Batch x = make_step(kva, {{2048}, 0, 2048});
            e.f(x.b);
            CHECK_EQ(qwen4exp_kva::derive(k, &x.b).path, PATH_STOCK);
            const Run got = run_step(qwen4exp_kva::step, x.b), want = run_step(qwen4exp_fp8::step, x.b);
            CHECK(want.issues.size() > (x.b.enc ? 10u : 100u));
            if (differ(got.all, want.issues) != 0) std::fprintf(stderr, "    %s / %s differs\n", mode, e.why);
            CHECK_EQ(differ(got.all, want.issues), 0);
            CHECK_EQ(got.device_calls, 0);
        }
        /* the text after it: rotary planes present, components equal -- approximated, at the rotary positions */
        Batch x = make_step(kva, {{2048}, 0, 2048});
        x.b.rope_pos = rp;
        const qwen4exp_kva::Pass p = qwen4exp_kva::derive(k, &x.b);
        CHECK(p.path != PATH_STOCK && p.b == 2048);
        const Run got = run_step(qwen4exp_kva::step, x.b), stock = run_step(qwen4exp_fp8::step, x.b);
        std::map<rad_op, std::set<std::string>> want;
        for (const RecIssue& r : stock.issues) want[r.op].insert(position_tags(r, x.b));
        int rotary = 0, bad = 0;
        for (const RecIssue& r : got.issues) {
            const std::string t = position_tags(r, x.b);
            rotary += t.find('R') != std::string::npos;
            const auto w = want.find(r.op);
            if (w == want.end() || w->second.count(t)) continue;
            if (bad++ < 3) std::fprintf(stderr, "    %s: op %u takes positions '%s'\n", mode, (unsigned)r.op, t.c_str());
        }
        CHECK_EQ(bad, 0);
        /* not vacuous: at least layer 3's q and k (in-tree) and layer 7's k and indexer work list (filled) */
        CHECK(rotary >= 4);
    }
}

/* The folder's files are mapped, hashed and their tensors named; what a partial or damaged download
 * looks like is refused naming the file. */
TEST(a_folder_is_read_and_a_damaged_one_refused_by_name) {
    TempDir dir;
    REQUIRE(!dir.path.empty());
    write_folder(dir.path);
    qwen4exp_kva::Folder f;
    f.place.dir = dir.path.string();
    std::string why;
    REQUIRE(qwen4exp_kva::read_folder(&f, &why));
    CHECK_EQ(f.tensors.size(), (size_t)3);
    CHECK(f.tensors["proj.4.weight"].dtype == RAD_BF16 && f.tensors["proj.4.weight"].shape == std::vector<int64_t>({2, 8}));
    CHECK_EQ(f.tensors["score"].bytes, 20);
    struct Case { std::function<void()> damage; const char* said; };
    const Case cases[] = {
        { [&] { std::filesystem::remove(dir.path / "rowsel.safetensors"); }, "lists rowsel.safetensors, which is missing" },
        { [&] { std::ofstream(dir.path / "README.md") << "hullo"; }, "README.md is corrupt" },
        { [&] { std::ofstream(dir.path / "kva.json") << "{\"format\": 2, \"files\": {}}"; }, "format-1" },
        { [&] { write_folder(dir.path, "{\"../x\": \"00\"}"); }, "lists ../x" },
        { [&] { const std::string a = safetensors({{"score", "F32", {5}, 4}});
                std::ofstream(dir.path / "proj.L4.safetensors", std::ios::binary) << a;
                write_folder(dir.path, "{\"proj.L4.safetensors\": \"" + sha_of(a) + "\", \"rowsel.safetensors\": \"" +
                                       sha_of(safetensors({{"score", "F32", {5}, 4}})) + "\"}");
                std::ofstream(dir.path / "proj.L4.safetensors", std::ios::binary) << a; }, "'score' is in two files" },
    };
    for (const Case& c : cases) {
        write_folder(dir.path);
        c.damage();
        qwen4exp_kva::Folder g;
        g.place.dir = dir.path.string();
        why.clear();
        CHECK(!qwen4exp_kva::read_folder(&g, &why));
        CHECK(has(why, c.said));
        if (!has(why, c.said)) std::fprintf(stderr, "    said: %s\n", why.c_str());
    }
}

/* Discovery: RADIANCE_KVA_PROJECTOR wins and names itself; otherwise projector/ beside the model
 * file this process has MAPPED (the test maps the tiny container, as the engine maps its model);
 * with neither there is no folder, and the places looked at are said. */
TEST(the_folder_is_found_by_the_env_then_beside_the_mapped_model) {
    TempDir dir;
    REQUIRE(!dir.path.empty());
    const std::filesystem::path model = dir.path / "model.rad", proj = dir.path / "projector";
    std::filesystem::copy_file(tiny_container(), model);
    {
        const qwen4exp_kva::FolderPlace none = qwen4exp_kva::find_folder();
        CHECK(none.dir.empty());
        CHECK(has(none.how, "no model file is mapped"));
    }
    size_t size = 0;
    const unsigned char* mapped = qwen4exp_kva::map_file(model.string(), &size);
    REQUIRE(mapped != nullptr);
    qwen4exp_kva::FolderPlace at = qwen4exp_kva::find_folder();
    CHECK_EQ(at.container, model.string());
    CHECK(at.dir.empty() && has(at.how, (proj).string().c_str()));
    std::filesystem::create_directory(proj);
    write_folder(proj);
    at = qwen4exp_kva::find_folder();
    CHECK_EQ(at.dir, proj.string());
    CHECK(has(at.how, "beside the resolved model file"));
    {
        TempDir other;
        write_folder(other.path);
        Env env({{"RADIANCE_KVA_PROJECTOR", other.path.c_str()}});
        at = qwen4exp_kva::find_folder();
        CHECK_EQ(at.dir, other.path.string());
        CHECK(has(at.how, "$RADIANCE_KVA_PROJECTOR="));
    }
    {
        Env env({{"RADIANCE_KVA_PROJECTOR", (dir.path / "nowhere").c_str()}});
        at = qwen4exp_kva::find_folder();
        CHECK(at.dir.empty() && has(at.how, "(no kva.json there)"));
    }
    munmap((void*)mapped, size);
}

/* The container is read through the public format: an entry's planes and the tokenizer's canonical
 * form hash to the values tools/kva_projector.py's test expects of the same bytes. */
TEST(the_container_hashes_are_the_builders) {
    qwen4exp_kva::Container c;
    REQUIRE(qwen4exp_kva::open_container(tiny_container(), &c));
    CHECK_EQ(qwen4exp_kva::vocab_sha256(c), std::string(kTinyVocab));
    CHECK_EQ(qwen4exp_kva::entry_sha256(c, "anchor.weight"), std::string(kTinyAnchor));
    CHECK(qwen4exp_kva::entry_sha256(c, "absent.weight").empty());
    munmap((void*)c.base, c.size);
    qwen4exp_kva::Container bad;
    CHECK(!qwen4exp_kva::open_container("/proc/self/cmdline", &bad));
}

/* The JSON reader: what the manifest and safetensors headers use, and what it refuses. */
TEST(the_json_reader_reads_the_manifest_forms) {
    qwen4exp_kva::Json j;
    const std::string ok = R"({"a": [1, -2.5e3, true, false, null], "s": "x\"\\\/\n\u00e9\ud83d\ude00", "a": {"b": 3}})";
    REQUIRE(qwen4exp_kva::json_parse(ok.data(), ok.size(), &j));
    CHECK_EQ(j.get("a")->integer("b", 0), 3);   /* a repeated key reads as its last */
    CHECK_EQ(j.text("s"), std::string("x\"\\/\n\xc3\xa9\xf0\x9f\x98\x80"));
    for (const char* bad : {"{", "{\"a\":}", "[1,]", "{\"a\":1} x", "\"\\q\"", "nul"})
        CHECK(!qwen4exp_kva::json_parse(bad, std::strlen(bad), &j));
}

/* Every case starts with no folder and no copies: a folder one case hands the plugin must not leak
 * into the next. (Static, after every registration in this file.) */
static const int g_reset_each_case = [] {
    for (auto& c : ::radtest::cases()) {
        auto fn = c.fn;
        c.fn = [fn] { reset_projector(); fn(); };
    }
    return 0;
}();

RAD_TEST_MAIN()
