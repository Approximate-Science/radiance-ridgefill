/* rad_fake.h -- the recording fakes behind the arch plugins' host tests (arch_static_test, adapter_core_test):
 * a RadBuilder that records every declaration, a RadCtx that records every issue, and the rest of the
 * plugin ABI's C row with the device faked (no GPU, no core). Modelled on radiance's tests/arch_test.cpp,
 * and the tests' own rather than core/build's: a plugin that only works against one builder has an
 * undeclared dependency. Include once per test binary, before the plugin source.
 */
#ifndef KVA_TEST_RAD_FAKE_H
#define KVA_TEST_RAD_FAKE_H

#include "rad_builder.h"
#include "rad_device.h"
#include "rad_runtime.h"

#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <set>
#include <string>
#include <utility>
#include <vector>

/* ==================================================================== the recording builder */
/* One test binary includes this once; the records live at namespace scope because RadBuilder and RadCtx,
 * the ABI's own opaque types, hold them. */
struct RecParam { std::string key; int kind = 0; long long ival = 0, ihi = 0; double dval = 0; std::string sval; };
struct RecOp {
    std::string op;
    std::vector<RecParam> p;
    std::vector<rad_weight> w;
    std::vector<rad_buf> reads, writes;
};
struct RecIssue { rad_op op = 0; std::vector<RadOperand> opd; int64_t n = 0; };

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

namespace {

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

}  /* namespace */

#endif /* KVA_TEST_RAD_FAKE_H */
