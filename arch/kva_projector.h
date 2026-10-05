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
 * WHAT CANNOT RUN IS REFUSED, AND THE ENGINE SERVES STOCK: no folder, a folder that is not this
 * model's (kva_match.h), or tensors whose shapes are not this model's. Nothing is declared then,
 * so the graph is the in-tree graph (R6/R7).
 */
#ifndef QWEN4EXP_KVA_PROJECTOR_H
#define QWEN4EXP_KVA_PROJECTOR_H

#include "kva_match.h"

#include <mutex>

namespace qwen4exp_kva {

/* The folder this process loaded, once, by the first real declare to get here. */
struct Loaded {
    bool    tried = false;
    bool    usable = false;
    Folder  folder;
    int64_t split = -1;
    bool    has_st = false;
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
    std::vector<RadOperand> proj_w, proj_b, st;
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

/* What this model needs of the folder's tensors; the first one that cannot run, or empty. */
inline std::string check_tensors(const Folder& f, const qwen4exp_fp8::Model& m, Loaded* l) {
    const int64_t S = f.manifest.integer("split", -1), n = m.g.n_embd, wide = m.hccfg.hc * n;
    if (S <= m.ple_layer || S >= m.g.n_layer)
        return "its split " + std::to_string(S) + " is not a late layer of this model (n-gram layer " +
               std::to_string(m.ple_layer) + ", " + std::to_string(m.g.n_layer) + " layers)";
    int held = 0, want = 0;
    for (int64_t li = S; li < m.g.n_layer; ++li) {
        const std::string p = "proj." + std::to_string(li);
        if (!shaped(tensor(f, p + ".weight"), RAD_BF16, {n, wide}) || !shaped(tensor(f, p + ".bias"), RAD_BF16, {n}))
            return p + ".weight [" + std::to_string(n) + ", " + std::to_string(wide) + "] / .bias bf16 is missing or misshapen";
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
        for (const Piece& x : plan) if (x.host && h) std::memcpy(h + x.at, x.src, (size_t)x.bytes);
        return h ? p : nullptr;
    }
    RadStream s = nullptr;
    bool ok = rad_stream_create(&s, 0) == RAD_OK;
    for (const Piece& x : plan)
        if (ok && !x.host) ok = rad_memcpy_async((unsigned char*)p + x.at, x.src, x.bytes, s) == RAD_OK;
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

/* This rank's copies of what `mode` reads: the projector (vram or host), this rank's value heads of
 * the correction, the selected row table. Once per rank per process. */
inline void free_upload(Upload& u) {
    if (u.vram) rad_dev_free(u.vram, RAD_MEM_DEVICE);
    if (u.host) rad_dev_free(u.host, RAD_MEM_HOST_MAPPED);
    u = Upload{};
}

inline bool upload_rank(const Loaded& l, const qwen4exp_fp8::Model& m, const Config& c, int rank) {
    Upload& u = g_upload[rank];
    const int key = (c.mode * 2 + c.place) * 3 + c.rowsel_table;
    if (u.done && u.key == key) return u.ok;
    free_upload(u);
    u.done = true;
    u.key = key;
    const Folder& f = l.folder;
    const bool host = c.place == PLACE_HOST, project = c.mode != MODE_PLUMB;
    const int64_t n = m.g.n_embd, wide = m.hccfg.hc * n, L = m.g.n_layer;
    const GdnFP8::Config& g = m.gcfg;
    const int64_t heads = g.n_head_v * g.head_v * g.head_k * 4;   /* one rank's correction, bytes */
    std::vector<Piece> plan;
    int64_t vend = 0, hend = 0;
    std::vector<int64_t> w_at(L, -1), b_at(L, -1), st_at(L, -1);
    int64_t score_at = -1;
    for (int64_t li = l.split; project && li < L; ++li) {
        const FolderTensor* w = tensor(f, "proj." + std::to_string(li) + ".weight");
        const FolderTensor* b = tensor(f, "proj." + std::to_string(li) + ".bias");
        w_at[li] = place_piece(plan, host ? &hend : &vend, w->data, w->bytes, host);
        b_at[li] = place_piece(plan, host ? &hend : &vend, b->data, b->bytes, host);
        const FolderTensor* st = m.layers[(size_t)li].full ? nullptr : tensor(f, "st." + std::to_string(li));
        if (st) st_at[li] = place_piece(plan, &vend, st->data + rank * heads, heads, false);
    }
    const FolderTensor* sc = c.mode == MODE_QUALITY ? tensor(f, kScoreNames[c.rowsel_table]) : nullptr;
    if (sc) score_at = place_piece(plan, &vend, sc->data, sc->bytes, false);
    u.vram = fill_block(plan, false, vend, rank);
    u.host = fill_block(plan, true, hend, rank);
    u.vram_bytes = vend;
    u.host_bytes = hend;
    if ((vend && !u.vram) || (hend && !u.host)) return false;
    u.proj_w.assign(L, RAD_NONE); u.proj_b.assign(L, RAD_NONE); u.st.assign(L, RAD_NONE);
    for (int64_t li = 0; li < L; ++li) {
        if (w_at[li] >= 0) u.proj_w[li] = RAD_P_T2(dev_at(u, host, w_at[li]), RAD_BF16, n, wide);
        if (b_at[li] >= 0) u.proj_b[li] = RAD_P_T2(dev_at(u, host, b_at[li]), RAD_BF16, n, 0);
        if (st_at[li] >= 0) u.st[li] = RAD_P_T2(dev_at(u, false, st_at[li]), RAD_F32, g.n_head_v, g.head_v * g.head_k);
    }
    if (sc) u.score = RAD_P_T2(dev_at(u, false, score_at), RAD_F32, m.g.n_vocab_all, 0);
    std::fprintf(stderr, "radiance: qwen4exp_kva: KVA: rank %d holds the projector: %.1f MiB VRAM, %.1f MiB "
                         "host-mapped (RADIANCE_KVA_PROJ_PLACE=%s)\n", rank, (double)vend / (1 << 20),
                 (double)hend / (1 << 20), kPlaceNames[c.place]);
    u.ok = true;
    return true;
}

/* Frees every rank's copies (rad_plugin_close). */
inline void free_uploads() {
    for (Upload& u : g_upload) free_upload(u);
}

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_KVA_PROJECTOR_H */
