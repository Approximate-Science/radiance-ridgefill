/* kva_projector.h -- the projector folder for Qwen4-Exp: loaded once per process, checked against
 * the model, uploaded per rank at the rank's real declare, and handed to the issue sites as RAW
 * operands (PACKAGING.md §0, REFUTATION-3 §2.1, §2.4, §4).
 *
 * NOTHING COMES FROM THE CONTAINER. A container that still holds appended kva.* weights is served
 * exactly as a stock one: none of them is declared, so none is placed.
 *
 * MEMORY. Each rank's real declare runs on its own thread with its card bound (radiance
 * core/engine_bringup.cpp:477-488): the tensors this mode reads are copied there into one
 * rad_dev_alloc block, synchronously (a pageable copy at declare drains its bounce; nothing is
 * recording). plan() measures free VRAM after declare and subtracts what is already held
 * (core/mem/vram_budget.cpp:76-91), so the expert slab shrinks by exactly these bytes.
 * RADIANCE_KVA_PROJ_PLACE=host puts the projector maps in host-mapped memory instead (per rank:
 * the allocation is not portable, core/device/hip.cpp:300-306), outside the VRAM budget, read by
 * the GEMM over the link; the correction and the score table stay in VRAM (28 MiB a rank).
 * Sizing declares and tools allocate nothing. Freed at rad_plugin_close.
 *
 * THE INT8 VARIANT (a folder whose manifest says "projector": {"dtype": "i8"}, R79): codes and a
 * scale per 128 columns of a row instead of the bf16 map -- 0.51x the bytes, so about half the
 * resident experts the bf16 maps cost come back -- relaid out at load into the engine's own int8
 * GEMM's stored form (kva_int8.h) and read by kva_gemm_nt_q. VRAM placement only: host placement
 * and its staging ring move bf16 maps.
 *
 * WHAT CANNOT RUN IS REFUSED, AND THE ENGINE SERVES STOCK: no folder, a folder that is not this
 * model's (kva_match.h), or tensors whose shapes are not this model's. Nothing is declared then,
 * so the graph is the in-tree graph (R6/R7).
 */
#ifndef QWEN4EXP_KVA_PROJECTOR_H
#define QWEN4EXP_KVA_PROJECTOR_H

#include "kva_match.h"
#include "kva_int8.h"

#include <mutex>

namespace qwen4exp_kva {

/* The folder this process loaded, once, by the first real declare to get here. */
struct Loaded {
    bool    tried = false;
    bool    usable = false;
    Folder  folder;
    int64_t split = -1;
    bool    has_st = false;
    bool    int8 = false;      /* the maps are i8*bf16[1x128] (the manifest's projector dtype "i8") */
};
static Loaded     g_loaded;
static std::mutex g_load_mu;
/* Tests hand a folder in here instead of the disk (tests/arch_static_test.cpp); null in a build. */
static const Folder* g_folder_for_test = nullptr;

/* One rank's copies, made at its real declare. `key` is the configuration they were made for: a
 * process serves one, so they are made once (a test that declares several modes makes them again). */
struct Upload {
    bool  done = false, ok = false;
    int   key = -1;
    void* vram = nullptr;
    void* host = nullptr;
    int64_t vram_bytes = 0, host_bytes = 0;
    std::vector<RadOperand> proj_w, proj_b, proj_s, st, ring_src, ring_dst;
    RadOperand score = RAD_NONE;
};
static Upload g_upload[MAX_RANKS];

static const char* const kScoreNames[] = { "score", "score_none", "score_all" };

inline const FolderTensor* tensor(const Folder& f, const std::string& name) {
    auto it = f.tensors.find(name);
    return it == f.tensors.end() ? nullptr : &it->second;
}

inline bool shaped(const FolderTensor* t, uint32_t dtype, std::initializer_list<int64_t> shape) {
    return t && t->dtype == dtype && t->shape == std::vector<int64_t>(shape);
}

/* Layer li's map in the folder's dtype: bf16 `.weight`, or int8 `.codes` + `.scale` (i8*bf16[1x128]);
 * the bf16 `.bias` either way. Empty when it is all there and shaped. */
inline std::string check_map(const Folder& f, int64_t li, int64_t n, int64_t wide, bool int8) {
    const std::string p = "proj." + std::to_string(li), dims = "[" + std::to_string(n) + ", " + std::to_string(wide);
    const bool ok = int8 ? shaped(tensor(f, p + ".codes"), RAD_I8, {n, wide}) &&
                           shaped(tensor(f, p + ".scale"), RAD_BF16, {n, wide / kI8Group}) && wide % kI8Group == 0
                         : shaped(tensor(f, p + ".weight"), RAD_BF16, {n, wide});
    if (ok && shaped(tensor(f, p + ".bias"), RAD_BF16, {n})) return std::string();
    return int8 ? p + ".codes i8 " + dims + "] / .scale bf16 " + dims + " / 128] / .bias bf16 is missing or misshapen"
                : p + ".weight " + dims + "] / .bias bf16 is missing or misshapen";
}

/* What this model needs of the folder's tensors; the first one that cannot run, or empty. */
inline std::string check_tensors(const Folder& f, const qwen4exp_fp8::Model& m, Loaded* l) {
    const int64_t S = f.manifest.integer("split", -1), n = m.g.n_embd, wide = m.hccfg.hc * n;
    if (S <= m.ple_layer || S >= m.g.n_layer)
        return "its split " + std::to_string(S) + " is not a late layer of this model (n-gram layer " +
               std::to_string(m.ple_layer) + ", " + std::to_string(m.g.n_layer) + " layers)";
    const Json* proj = f.manifest.get("projector");
    const std::string dtype = proj ? proj->text("dtype", "bf16") : "bf16";
    if (dtype != "bf16" && dtype != "i8") return "its projector dtype '" + dtype + "' is neither bf16 nor i8";
    l->int8 = dtype == "i8";
    int held = 0, want = 0;
    for (int64_t li = S; li < m.g.n_layer; ++li) {
        if (const std::string why = check_map(f, li, n, wide, l->int8); !why.empty()) return why;
        if (m.layers[(size_t)li].full) continue;
        ++want;
        held += shaped(tensor(f, "st." + std::to_string(li)), RAD_F32,
                       {m.gcfg.n_head_v * m.g.world, m.gcfg.head_v, m.gcfg.head_k});
    }
    if (held && held != want)
        return "the correction covers " + std::to_string(held) + " of the " + std::to_string(want) +
               " delta-net layers from the split up; it is all of them or none";
    for (const char* s : kScoreNames)
        if (tensor(f, s) && !shaped(tensor(f, s), RAD_F32, {m.g.n_vocab_all}))
            return std::string("the row table '") + s + "' is not f32 [" + std::to_string(m.g.n_vocab_all) + "]";
    l->split = S;
    l->has_st = held > 0;
    return std::string();
}

/* Finds, reads and checks the folder once per process; every later declare reads the answer. */
inline const Loaded& load_folder(const RadModelMeta* meta, RadBuilder* b, const qwen4exp_fp8::Model& m) {
    std::lock_guard<std::mutex> lk(g_load_mu);
    Loaded& l = g_loaded;
    if (l.tried) return l;
    l.tried = true;
    std::string why;
    if (g_folder_for_test) l.folder = *g_folder_for_test;
    else l.folder.place = find_folder();
    const FolderPlace& at = l.folder.place;
    if (at.dir.empty()) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: KVA: no projector folder (looked at %s); "
                             "serving stock\n", at.how.c_str());
        return l;
    }
    if (!g_folder_for_test && !read_folder(&l.folder, &why)) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: KVA: projector %s REFUSED: %s; serving stock\n",
                     at.dir.c_str(), why.c_str());
        return l;
    }
    const Match mt = match_model(l.folder, meta, b, "qwen4exp");
    why = mt.refused.empty() ? check_tensors(l.folder, m, &l) : mt.refused;
    if (!why.empty()) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: KVA: projector %s (found %s) REFUSED, it cannot run "
                             "on this model: %s [%s]; serving stock\n", at.dir.c_str(), at.how.c_str(),
                     why.c_str(), mt.summary.c_str());
        return l;
    }
    for (const std::string& w : mt.warnings)
        std::fprintf(stderr, "radiance: qwen4exp_kva: KVA: WARNING: projector %s: %s -- it runs, but "
                             "was fitted on another variant\n", at.dir.c_str(), w.c_str());
    std::fprintf(stderr, "radiance: qwen4exp_kva: KVA: projector %s (found %s) matches %s: %s, "
                         "%zu warning(s); split %lld, %s, %.1f MiB in %zu files\n",
                 at.dir.c_str(), at.how.c_str(), meta->name ? meta->name : "(unnamed)", mt.summary.c_str(),
                 mt.warnings.size(), (long long)l.split, l.has_st ? "correction held" : "no correction",
                 (double)l.folder.file_bytes / (1 << 20), l.folder.manifest.get("files")->obj.size());
    l.usable = true;
    return l;
}

/* ---------------------------------------------------------------- one rank's copies */

/* A rank's copy plan: each tensor's source bytes and where in which block it lands. */
struct Piece { const unsigned char* src; int64_t bytes; bool host; int64_t at; };

/* The int8 maps' stored forms, made for one rank's upload and dropped after it. */
using Stored = std::vector<I8Stored>;

inline int64_t place_piece(std::vector<Piece>& plan, int64_t* end, const unsigned char* src,
                           int64_t bytes, bool host) {
    const int64_t at = (*end + 255) / 256 * 256;   /* the GEMM wants 16-byte rows; 256 is the plane rule */
    plan.push_back({ src, bytes, host, at });
    *end = at + bytes;
    return at;
}

/* Allocates and fills one block; null (and the reason on stderr) on failure. */
inline void* fill_block(const std::vector<Piece>& plan, bool host, int64_t bytes, int rank) {
    if (bytes == 0) return nullptr;
    void* p = rad_dev_alloc(bytes, host ? RAD_MEM_HOST_MAPPED : RAD_MEM_DEVICE);
    if (!p) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: KVA: rank %d could not allocate %.1f MiB of %s "
                             "for the projector: %s\n", rank, (double)bytes / (1 << 20),
                     host ? "host-mapped memory" : "VRAM", rad_dev_last_error());
        return nullptr;
    }
    if (host) {
        unsigned char* h = (unsigned char*)rad_dev_host_ptr(p);
        for (const Piece& x : plan) if (x.host && x.src && h) std::memcpy(h + x.at, x.src, (size_t)x.bytes);
        return h ? p : nullptr;
    }
    RadStream s = nullptr;
    bool ok = rad_stream_create(&s, 0) == RAD_OK;
    for (const Piece& x : plan)
        if (ok && !x.host && x.src) ok = rad_memcpy_async((unsigned char*)p + x.at, x.src, x.bytes, s) == RAD_OK;
    ok = ok && rad_stream_sync(s) == RAD_OK;
    if (s) rad_stream_destroy(s);
    if (ok) return p;
    std::fprintf(stderr, "radiance: qwen4exp_kva: KVA: rank %d: the projector upload failed: %s\n",
                 rank, rad_dev_last_error());
    rad_dev_free(p, RAD_MEM_DEVICE);
    return nullptr;
}

/* The device address of a block offset: host-mapped memory is read through its device view. */
inline unsigned char* dev_at(const Upload& u, bool host, int64_t at) {
    unsigned char* base = (unsigned char*)(host ? rad_dev_device_ptr(u.host) : u.vram);
    return base ? base + at : nullptr;
}

inline void free_upload(Upload& u) {
    if (u.vram) rad_dev_free(u.vram, RAD_MEM_DEVICE);
    if (u.host) rad_dev_free(u.host, RAD_MEM_HOST_MAPPED);
    u = Upload{};
}

/* Where each tensor of one rank's copy lands: a block offset per layer, -1 = not held. */
struct Layout {
    std::vector<int64_t> w, b, s, st, slot;
    int64_t score = -1, vend = 0, hend = 0;
};

/* THE STAGING RING (Dylan's DD-L, host placement only): each layer's map and bias sit in the host
 * block as ONE [n + 1, wide] row block (the bias in the first n elements of the last row), so one
 * strided copy moves a layer; two VRAM slots of that size take turns (kva_layer.h ring_*). */
inline void plan_maps(const Folder& f, const Loaded& l, const qwen4exp_fp8::Model& m, bool host,
                      bool ring, std::vector<Piece>& plan, Layout* x) {
    const int64_t n = m.g.n_embd, wide = m.hccfg.hc * n, block = (n + 1) * wide * 2;
    for (int64_t li = l.split; li < m.g.n_layer; ++li) {
        const FolderTensor* w = tensor(f, "proj." + std::to_string(li) + ".weight");
        const FolderTensor* b = tensor(f, "proj." + std::to_string(li) + ".bias");
        int64_t* end = host ? &x->hend : &x->vend;
        x->w[li] = place_piece(plan, end, w->data, w->bytes, host);
        x->b[li] = place_piece(plan, end, b->data, b->bytes, host);
        if (ring) *end = x->w[li] + block;   /* the bias row's tail: read by the copy, never used */
    }
    for (int s = 0; ring && s < 2; ++s) x->slot.push_back(place_piece(plan, &x->vend, nullptr, block, false));
}

/* The int8 maps (VRAM): each layer's codes and scale relaid out into the GEMM's stored form, then
 * its bias. False and `why` when the engine's int8 GEMM cannot take them. */
inline bool plan_maps_i8(const Folder& f, const Loaded& l, const qwen4exp_fp8::Model& m,
                         std::vector<Piece>& plan, Layout* x, Stored* stored, std::string* why) {
    const int64_t n = m.g.n_embd, wide = m.hccfg.hc * n;
    const std::vector<const RadKernelInfo*> rows = i8_rows();
    stored->assign((size_t)m.g.n_layer, I8Stored{});
    for (int64_t li = l.split; li < m.g.n_layer; ++li) {
        const std::string p = "proj." + std::to_string(li);
        I8Stored& st = (*stored)[(size_t)li];
        if (!relayout_i8(rows, n, wide, tensor(f, p + ".codes")->data, tensor(f, p + ".scale")->data, &st, why))
            return false;
        const FolderTensor* b = tensor(f, p + ".bias");
        x->w[li] = place_piece(plan, &x->vend, st.codes.data(), (int64_t)st.codes.size(), false);
        x->s[li] = place_piece(plan, &x->vend, st.scale.data(), (int64_t)st.scale.size(), false);
        x->b[li] = place_piece(plan, &x->vend, b->data, b->bytes, false);
    }
    return true;
}

/* This rank's copy plan for `mode`: the maps (vram or host), its value heads of the correction, the
 * selected row table. */
inline Layout plan_rank(const Loaded& l, const qwen4exp_fp8::Model& m, const Config& c, int rank,
                        std::vector<Piece>& plan, Stored* stored, std::string* why) {
    const Folder& f = l.folder;
    const GdnFP8::Config& g = m.gcfg;
    const int64_t heads = g.n_head_v * g.head_v * g.head_k * 4;   /* one rank's correction, bytes */
    Layout x;
    for (auto* v : { &x.w, &x.b, &x.s, &x.st }) v->assign(m.g.n_layer, -1);
    if (c.mode == MODE_PLUMB) return x;
    if (l.int8 && !plan_maps_i8(f, l, m, plan, &x, stored, why)) return x;
    if (!l.int8) plan_maps(f, l, m, c.place == PLACE_HOST, c.place == PLACE_HOST && c.ring, plan, &x);
    for (int64_t li = l.split; li < m.g.n_layer; ++li) {
        const FolderTensor* st = m.layers[(size_t)li].full ? nullptr : tensor(f, "st." + std::to_string(li));
        if (st) x.st[li] = place_piece(plan, &x.vend, st->data + rank * heads, heads, false);
    }
    const FolderTensor* sc = c.mode == MODE_QUALITY ? tensor(f, kScoreNames[c.rowsel_table]) : nullptr;
    if (sc) x.score = place_piece(plan, &x.vend, sc->data, sc->bytes, false);
    return x;
}

/* The issue sites' operands for a layout: the maps where the GEMM reads them (the slot a layer takes
 * turns on, under the ring), and the ring's copy source and destination. */
inline void take_operands(Upload& u, const Layout& x, const qwen4exp_fp8::Model& m, bool host) {
    const int64_t n = m.g.n_embd, wide = m.hccfg.hc * n, L = m.g.n_layer;
    const GdnFP8::Config& g = m.gcfg;
    for (auto* v : { &u.proj_w, &u.proj_b, &u.proj_s, &u.st, &u.ring_src, &u.ring_dst }) v->assign(L, RAD_NONE);
    for (int64_t li = 0; li < L; ++li) {
        const bool ring = !x.slot.empty() && x.w[li] >= 0;
        unsigned char* at = ring ? dev_at(u, false, x.slot[li % 2]) : x.w[li] >= 0 ? dev_at(u, host, x.w[li]) : nullptr;
        /* int8: the stored codes [n, wide] and scale [n, wide/128], shaped as the GEMM reads them */
        if (at) u.proj_w[li] = RAD_P_T2(at, x.s[li] >= 0 ? RAD_I8 : RAD_BF16, n, wide);
        if (x.s[li] >= 0) u.proj_s[li] = RAD_P_T2(dev_at(u, false, x.s[li]), RAD_BF16, n, wide / kI8Group);
        if (at) u.proj_b[li] = RAD_P_T2(ring ? at + n * wide * 2 : dev_at(u, host, x.b[li]), RAD_BF16, n, 0);
        if (ring) u.ring_src[li] = RAD_P_T2(dev_at(u, true, x.w[li]), RAD_BF16, n + 1, wide);
        if (ring) u.ring_dst[li] = RAD_P_T2(at, RAD_BF16, n + 1, wide);
        if (x.st[li] >= 0) u.st[li] = RAD_P_T2(dev_at(u, false, x.st[li]), RAD_F32, g.n_head_v, g.head_v * g.head_k);
    }
}

/* This rank's copies of what `mode` reads. Once per rank per process. */
inline bool upload_rank(const Loaded& l, const qwen4exp_fp8::Model& m, const Config& c, int rank) {
    Upload& u = g_upload[rank];
    const int key = ((c.mode * 2 + c.place) * 3 + c.rowsel_table) * 2 + c.ring;
    if (u.done && u.key == key) return u.ok;
    free_upload(u);
    u.done = true;
    u.key = key;
    std::vector<Piece> plan;
    Stored stored;
    std::string why;
    const Layout x = plan_rank(l, m, c, rank, plan, &stored, &why);
    if (!why.empty()) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: KVA: rank %d cannot load the int8 projector %s: %s\n",
                     rank, l.folder.place.dir.c_str(), why.c_str());
        return false;
    }
    u.vram = fill_block(plan, false, x.vend, rank);
    u.host = fill_block(plan, true, x.hend, rank);
    u.vram_bytes = x.vend;
    u.host_bytes = x.hend;
    if ((x.vend && !u.vram) || (x.hend && !u.host)) return false;
    take_operands(u, x, m, c.place == PLACE_HOST);
    if (x.score >= 0) u.score = RAD_P_T2(dev_at(u, false, x.score), RAD_F32, m.g.n_vocab_all, 0);
    std::fprintf(stderr, "radiance: qwen4exp_kva: KVA: rank %d holds the projector: %.1f MiB VRAM, %.1f MiB "
                         "host-mapped (RADIANCE_KVA_PROJ_PLACE=%s%s); %s maps\n", rank, (double)x.vend / (1 << 20),
                 (double)x.hend / (1 << 20), kPlaceNames[c.place], x.slot.empty() ? "" : ", staging ring",
                 l.int8 ? "int8" : "bf16");
    u.ok = true;
    return true;
}

/* Frees every rank's copies (rad_plugin_close). */
inline void free_uploads() {
    for (Upload& u : g_upload) free_upload(u);
}

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_KVA_PROJECTOR_H */
