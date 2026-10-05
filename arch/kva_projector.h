/* kva_projector.h -- the projector folder for Qwen4-Exp: loaded once per process, checked against
 * the model, uploaded per rank at the rank's real declare, and handed to the issue sites as RAW
 * operands (PACKAGING.md §0, REFUTATION-3 §2.1, §2.4, §4).
 *
 * NOTHING COMES FROM THE CONTAINER. A container that still holds appended kva.* weights is served
 * exactly as a stock one: none of them is declared, so none is placed.
 *
 * MEMORY. Each rank's real declare runs on its own thread with its card bound (radiance
 * core/engine_bringup.cpp:477-488). THE MAPS LIVE IN HOST MEMORY (Dylan, Stage E): one host-mapped
 * block a rank (the allocation is not portable, core/device/hip.cpp:300-306), outside the VRAM
 * budget, holding each late layer's map as one row block, this rank's correction heads and the row
 * table; ONE VRAM slot of one block (50 MiB bf16, 25.4 MiB int8) takes each layer in turn, copied on
 * the second lane after the previous layer's GEMM (the staging ring, kva_layer.h). The slot is the
 * plugin's only rad_dev_alloc; plan() subtracts it as already held (core/mem/vram_budget.cpp:76-91), and
 * every request pays it in resident experts, so it is kept to one block (notes/stagee.md §12-§14).
 * Sizing declares and tools allocate nothing. Freed at rad_plugin_close.
 *
 * THE INT8 VARIANT (a folder whose manifest says "projector": {"dtype": "i8"}, R79): codes and a
 * scale per 128 columns of a row instead of the bf16 map -- 0.51x the bytes on the link and in the
 * slots -- relaid out at load into the engine's own int8 GEMM's stored form (kva_int8.h) and read by
 * kva_gemm_nt_q from the slot.
 *
 * WHAT CANNOT RUN IS REFUSED, AND THE ENGINE SERVES STOCK: no folder, a folder that is not this
 * model's (kva_match.h), or tensors whose shapes are not this model's. Nothing is declared then,
 * so the graph is the in-tree graph (R6/R7).
 */
#ifndef KVA_PROJECTOR_H
#define KVA_PROJECTOR_H

#include "kva_match.h"
#include "kva_int8.h"

#include <mutex>

namespace kva {

using namespace rad::arch;

/* The folder this process loaded, once, by the first real declare to get here. */
struct Loaded {
    bool    tried = false;
    bool    usable = false;
    Folder  folder;
    int64_t split = -1;
    bool    has_st = false;
    bool    int8 = false;      /* the maps are i8*bf16[1x128] (the manifest's projector dtype "i8") */
    bool    has_final = false; /* final.weight [w, w] + final.bias [w] bf16: the MTP map (kva_final.h) */
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
    std::vector<RadOperand> proj_w, proj_b, proj_s, st, ring_src, ring_dst, final_w, final_b;
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

/* What the adapter's model needs of the folder's tensors; the first one that cannot run, or empty. Every
 * shape is a fact (wide, the state, the vocabulary) and the maps' dtype is the manifest's, so a new
 * model needs no schema of its own here. */
inline std::string check_tensors(const Folder& f, const KvaAdapter& a, Loaded* l) {
    const int64_t S = f.manifest.integer("split", -1), n = a.n_embd, wide = a.wide;
    if (S < a.split_lo || S >= a.n_layer)
        return "its split " + std::to_string(S) + " is not a late layer of this model (n-gram layer " +
               std::to_string(a.split_lo - 1) + ", " + std::to_string(a.n_layer) + " layers)";
    const Json* proj = f.manifest.get("projector");
    const std::string dtype = proj ? proj->text("dtype", "bf16") : "bf16";
    if (dtype != "bf16" && dtype != "i8") return "its projector dtype '" + dtype + "' is neither bf16 nor i8";
    l->int8 = dtype == "i8";
    int held = 0, want = 0;
    for (int64_t li = S; li < a.n_layer; ++li) {
        if (const std::string why = check_map(f, li, n, wide, l->int8); !why.empty()) return why;
        if (a.full[(size_t)li]) continue;
        ++want;
        held += shaped(tensor(f, "st." + std::to_string(li)), RAD_F32,
                       {a.state.n_head * a.world, a.state.sd0, a.state.sd1});
    }
    if (held && held != want)
        return "the correction covers " + std::to_string(held) + " of the " + std::to_string(want) +
               " delta-net layers from the split up; it is all of them or none";
    if (f.manifest.get("final")) {
        if (!shaped(tensor(f, "final.weight"), RAD_BF16, {wide, wide}) || !shaped(tensor(f, "final.bias"), RAD_BF16, {wide}))
            return "its final map is not final.weight bf16 [" + std::to_string(wide) + ", " + std::to_string(wide) +
                   "] + final.bias [" + std::to_string(wide) + "]";
        l->has_final = true;
    }
    for (const char* s : kScoreNames)
        if (tensor(f, s) && !shaped(tensor(f, s), RAD_F32, {a.n_vocab_all}))
            return std::string("the row table '") + s + "' is not f32 [" + std::to_string(a.n_vocab_all) + "]";
    l->split = S;
    l->has_st = held > 0;
    return std::string();
}

/* Finds, reads and checks the folder once per process; every later declare reads the answer. */
inline const Loaded& load_folder(const RadModelMeta* meta, RadBuilder* b, const KvaAdapter& a) {
    std::lock_guard<std::mutex> lk(g_load_mu);
    Loaded& l = g_loaded;
    if (l.tried) return l;
    l.tried = true;
    std::string why;
    if (g_folder_for_test) l.folder = *g_folder_for_test;
    else l.folder.place = find_folder();
    const FolderPlace& at = l.folder.place;
    if (at.dir.empty()) {
        std::fprintf(stderr, "radiance: %s: KVA: no projector folder (looked at %s); "
                             "serving stock\n", g_log_name, at.how.c_str());
        return l;
    }
    if (!g_folder_for_test && !read_folder(&l.folder, &why)) {
        std::fprintf(stderr, "radiance: %s: KVA: projector %s REFUSED: %s; serving stock\n", g_log_name,
                     at.dir.c_str(), why.c_str());
        return l;
    }
    const Match mt = match_model(l.folder, meta, b, a.match_name);
    why = mt.refused.empty() ? check_tensors(l.folder, a, &l) : mt.refused;
    if (!why.empty()) {
        std::fprintf(stderr, "radiance: %s: KVA: projector %s (found %s) REFUSED, it cannot run "
                             "on this model: %s [%s]; serving stock\n", g_log_name, at.dir.c_str(), at.how.c_str(),
                     why.c_str(), mt.summary.c_str());
        return l;
    }
    for (const std::string& w : mt.warnings)
        std::fprintf(stderr, "radiance: %s: KVA: WARNING: projector %s: %s -- it runs, but "
                             "was fitted on another variant\n", g_log_name, at.dir.c_str(), w.c_str());
    std::fprintf(stderr, "radiance: %s: KVA: projector %s (found %s) matches %s: %s, "
                         "%zu warning(s); split %lld, %s, %.1f MiB in %zu files\n", g_log_name,
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
        std::fprintf(stderr, "radiance: %s: KVA: rank %d could not allocate %.1f MiB of %s "
                             "for the projector: %s\n", g_log_name, rank, (double)bytes / (1 << 20),
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
    std::fprintf(stderr, "radiance: %s: KVA: rank %d: the projector upload failed: %s\n", g_log_name,
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

/* ONE LATE LAYER'S ROW BLOCK, in the host block and in a ring slot alike: rows of `wide` bf16, because
 * the ring's copy moves whole rows (kva_declare.h decl_ring). bf16: the map's n rows, the bias in the
 * first n elements of row n. int8: the stored codes, then the stored scales, then the bias, each on a
 * 256-byte boundary -- at the sizes the GEMM's layout hook gave (`codes`, `scales`, bytes). Offsets in
 * bytes from the block's start. */
struct RowBlock { int64_t rows = 0, scale_at = -1, bias_at = 0; };

inline RowBlock row_block(const KvaAdapter& a, int64_t codes = -1, int64_t scales = 0) {
    const int64_t n = a.n_embd, row = a.wide * 2;
    RowBlock r;
    if (codes < 0) {
        r.rows = n + 1;
        r.bias_at = n * row;
        return r;
    }
    r.scale_at = (codes + 255) / 256 * 256;
    r.bias_at = (r.scale_at + scales + 255) / 256 * 256;
    r.rows = (r.bias_at + n * 2 + row - 1) / row;
    return r;
}

/* Where each tensor of one rank's copy lands: the host block offset of each late layer's row block
 * (-1 below S), of each layer's value heads of the correction and of the row table; the ring's one VRAM slot. */
struct Layout {
    std::vector<int64_t> w, st, slot, fw;   /* fw: the final map's hc row blocks (with MTP), else empty */
    RowBlock rb;
    int64_t score = -1, vend = 0, hend = 0;
};

/* A piece copied into `block` at offset `at` (the block's own placement is already reserved). */
inline void put_piece(std::vector<Piece>& plan, const unsigned char* src, int64_t bytes, int64_t at) {
    plan.push_back({ src, bytes, true, at });
}

/* THE STAGING RING (Dylan's DD-L; since Stage E the only placement): every late layer's row block in
 * the host block, ONE VRAM slot of one block's size taking each layer in turn (kva_layer.h ring_*). int8 maps are
 * relaid out into the engine's int8 GEMM's stored form first (kva_int8.h); `stored` keeps them alive
 * until the copy. False and `why` when that GEMM cannot take them. */
/* THE FINAL MAP'S BLOCKS (with MTP, kva_final.h): hc row blocks of [n + 1, wide] bf16 -- rows i*n .. i*n + n of
 * the map, then that slice of its bias -- the bf16 projector block's shape, so they ride the same ring after
 * the late layers. Each block's GEMM writes columns i*n .. i*n + n of the predicted final stream. */
inline void plan_final(const Folder& f, const KvaAdapter& a, std::vector<Piece>& plan, Layout* x) {
    const int64_t n = a.n_embd, row = a.wide * 2;
    const FolderTensor* w = tensor(f, "final.weight");
    const FolderTensor* b = tensor(f, "final.bias");
    for (int64_t i = 0; i < a.wide / n; ++i) {
        x->fw.push_back(place_piece(plan, &x->hend, nullptr, (n + 1) * row, true));
        put_piece(plan, w->data + i * n * row, n * row, x->fw.back());
        put_piece(plan, b->data + i * n * 2, n * 2, x->fw.back() + n * row);
    }
}

inline bool plan_maps(const Folder& f, const Loaded& l, const KvaAdapter& a, bool final,
                      std::vector<Piece>& plan, Layout* x, Stored* stored, std::string* why) {
    const int64_t n = a.n_embd, wide = a.wide;
    stored->assign((size_t)a.n_layer, I8Stored{});
    if (l.int8) {
        const std::vector<const RadKernelInfo*> rows = i8_rows();
        for (int64_t li = l.split; li < a.n_layer; ++li) {
            const std::string p = "proj." + std::to_string(li);
            if (!relayout_i8(rows, n, wide, tensor(f, p + ".codes")->data, tensor(f, p + ".scale")->data,
                             &(*stored)[(size_t)li], why))
                return false;
        }
    }
    const I8Stored& first = (*stored)[(size_t)l.split];
    x->rb = l.int8 ? row_block(a, (int64_t)first.codes.size(), (int64_t)first.scale.size()) : row_block(a);
    const int64_t block = x->rb.rows * wide * 2;
    for (int64_t li = l.split; li < a.n_layer; ++li) {
        const std::string p = "proj." + std::to_string(li);
        const FolderTensor* b = tensor(f, p + ".bias");
        x->w[li] = place_piece(plan, &x->hend, nullptr, block, true);
        if (!l.int8) {
            const FolderTensor* w = tensor(f, p + ".weight");
            put_piece(plan, w->data, w->bytes, x->w[li]);
        } else {
            const I8Stored& st = (*stored)[(size_t)li];
            put_piece(plan, st.codes.data(), (int64_t)st.codes.size(), x->w[li]);
            put_piece(plan, st.scale.data(), (int64_t)st.scale.size(), x->w[li] + x->rb.scale_at);
        }
        put_piece(plan, b->data, b->bytes, x->w[li] + x->rb.bias_at);
    }
    if (final) plan_final(f, a, plan, x);
    /* ONE slot (kva_layer.h ring_copy), the larger of a layer's block -- so int8 maps take half -- and (with
     * MTP) a final block */
    const int64_t slot = final ? std::max(block, (n + 1) * wide * 2) : block;
    x->slot.push_back(place_piece(plan, &x->vend, nullptr, slot, false));
    return true;
}

/* This rank's copy plan for `mode`: the maps (host block + ring slots), its value heads of the
 * correction, the selected row table. */
inline Layout plan_rank(const Loaded& l, const KvaAdapter& a, const Config& c, int rank, bool final,
                        std::vector<Piece>& plan, Stored* stored, std::string* why) {
    const Folder& f = l.folder;
    const int64_t heads = a.state.n_head * a.state.sd0 * a.state.sd1 * 4;   /* one rank's correction, bytes */
    Layout x;
    for (auto* v : { &x.w, &x.st }) v->assign(a.n_layer, -1);
    if (c.mode == MODE_PLUMB) return x;
    if (!plan_maps(f, l, a, final, plan, &x, stored, why)) return x;
    /* The correction and the row table live in the host block too (Stage E): read zero-copy by their kernels,
     * once an element a pass (the correction's undo/apply, M = 1; the mask's score gather), on approximate
     * passes only -- VRAM is what every request pays in resident experts. */
    for (int64_t li = l.split; li < a.n_layer; ++li) {
        const FolderTensor* st = a.full[(size_t)li] ? nullptr : tensor(f, "st." + std::to_string(li));
        if (st) x.st[li] = place_piece(plan, &x.hend, st->data + rank * heads, heads, true);
    }
    const FolderTensor* sc = c.mode == MODE_QUALITY ? tensor(f, kScoreNames[c.rowsel_table]) : nullptr;
    if (sc) x.score = place_piece(plan, &x.hend, sc->data, sc->bytes, true);
    return x;
}

/* The issue sites' operands for a layout: each late layer's GEMM reads the ring's one slot (codes /
 * scales / bias at the row block's offsets), and the ring copies its host row block there. */
inline void take_operands(Upload& u, const Layout& x, const KvaAdapter& a, bool int8) {
    const int64_t n = a.n_embd, wide = a.wide, L = a.n_layer, F = (int64_t)x.fw.size();
    for (auto* v : { &u.proj_w, &u.proj_b, &u.proj_s, &u.st }) v->assign(L, RAD_NONE);
    for (auto* v : { &u.ring_src, &u.ring_dst }) v->assign(L + F, RAD_NONE);
    u.final_w.assign(F, RAD_NONE);
    u.final_b.assign(F, RAD_NONE);
    for (int64_t i = 0; i < F; ++i) {   /* ring index L + i: after the last layer, the same slot */
        unsigned char* slot = dev_at(u, false, x.slot[0]);
        u.final_w[i] = RAD_P_T2(slot, RAD_BF16, n, wide);
        u.final_b[i] = RAD_P_T2(slot + n * wide * 2, RAD_BF16, n, 0);
        u.ring_src[L + i] = RAD_P_T2(dev_at(u, true, x.fw[(size_t)i]), RAD_BF16, n + 1, wide);
        u.ring_dst[L + i] = RAD_P_T2(slot, RAD_BF16, n + 1, wide);
    }
    for (int64_t li = 0; li < L; ++li) {
        if (x.st[li] >= 0) u.st[li] = RAD_P_T2(dev_at(u, true, x.st[li]), RAD_F32, a.state.n_head, a.state.sd0 * a.state.sd1);
        if (x.w[li] < 0 || x.slot.empty()) continue;
        unsigned char* slot = dev_at(u, false, x.slot[0]);
        u.proj_w[li] = RAD_P_T2(slot, int8 ? RAD_I8 : RAD_BF16, n, wide);
        if (int8) u.proj_s[li] = RAD_P_T2(slot + x.rb.scale_at, RAD_BF16, n, wide / kI8Group);
        u.proj_b[li] = RAD_P_T2(slot + x.rb.bias_at, RAD_BF16, n, 0);
        u.ring_src[li] = RAD_P_T2(dev_at(u, true, x.w[li]), RAD_BF16, x.rb.rows, wide);
        u.ring_dst[li] = RAD_P_T2(slot, RAD_BF16, x.rb.rows, wide);
    }
}

/* This rank's copies of what `mode` reads. Once per rank per process. */
inline bool upload_rank(const Loaded& l, const KvaAdapter& a, const Config& c, int rank, bool final) {
    Upload& u = g_upload[rank];
    const int key = (c.mode * 3 + c.rowsel_table) * 2 + final;
    if (u.done && u.key == key) return u.ok;
    free_upload(u);
    u.done = true;
    u.key = key;
    std::vector<Piece> plan;
    Stored stored;
    std::string why;
    const Layout x = plan_rank(l, a, c, rank, final, plan, &stored, &why);
    if (!why.empty()) {
        std::fprintf(stderr, "radiance: %s: KVA: rank %d cannot load the int8 projector %s: %s\n", g_log_name,
                     rank, l.folder.place.dir.c_str(), why.c_str());
        return false;
    }
    u.vram = fill_block(plan, false, x.vend, rank);
    u.host = fill_block(plan, true, x.hend, rank);
    u.vram_bytes = x.vend;
    u.host_bytes = x.hend;
    if ((x.vend && !u.vram) || (x.hend && !u.host)) return false;
    take_operands(u, x, a, l.int8);
    if (x.score >= 0) u.score = RAD_P_T2(dev_at(u, true, x.score), RAD_F32, a.n_vocab_all, 0);
    if (c.mode == MODE_PLUMB)   /* plumb reads no fitted tensor: say so rather than print a row of zeros */
        std::fprintf(stderr, "radiance: %s: KVA: rank %d holds nothing (plumb reads no fitted tensor)\n", g_log_name, rank);
    else std::fprintf(stderr, "radiance: %s: KVA: rank %d holds the projector: %.1f MiB host-mapped "
                         "(%s maps, correction, row table), %.1f MiB VRAM (the staging ring's one slot)\n", g_log_name,
                 rank, (double)x.hend / (1 << 20), l.int8 ? "int8" : "bf16", (double)x.vend / (1 << 20));
    u.ok = true;
    return true;
}

/* Frees every rank's copies (rad_plugin_close). */
inline void free_uploads() {
    for (Upload& u : g_upload) free_upload(u);
}

}  /* namespace kva */

#endif /* KVA_PROJECTOR_H */
