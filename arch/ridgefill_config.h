/* ridgefill_config.h -- what an operator can turn on the RidgeFill plugin, read ONCE, at declare.
 *
 * step() reads only the parsed Config and keyed batch fields, never the environment (HANDOVER
 * §2.1, R15/R99): a replayed pass must issue what a fresh one would.
 *
 * THE MODE COMES FROM THE ENVIRONMENT ONLY (PLAN-FIX §6.5, R81). A container's `ridgefill.mode` is read
 * only to say, once, that it is ignored: the default is `off`, so a container converted with a
 * mode in it can never switch the trade on behind an operator's back.
 *
 *   env RADIANCE_RIDGEFILL          off | plumb | speed | quality                      default off
 *                             exact tail T in tokens   env RADIANCE_RIDGEFILL_TAIL     default 2048
 *                             correction strength      env RADIANCE_RIDGEFILL_ALPHA    default 1
 *                             row selection rule       env RADIANCE_RIDGEFILL_ROWSEL   class|random|all
 *                             share of the window's class matches kept exact
 *                                                      env RADIANCE_RIDGEFILL_SHARE    default 0.25
 *                             seed of the random control                         default 0
 *   env RADIANCE_RIDGEFILL_STAGE    auto | stock: the stager lever (PLAN-FIX §6.1, notes/impl.md §2).
 *                             auto lets the late layers STREAM their routed experts on an
 *                             approximate pass whose exact rows are at most RADIANCE_RIDGEFILL_STAGE_ROWS;
 *                             stock leaves the stager as it is (the tested fallback, R96)  default auto
 *   env RADIANCE_RIDGEFILL_STAGE_ROWS  the exact rows a masked pass may carry and still stream; past it
 *                             the pass runs the stock step (speed straddles take the tail-only path).
 *                             Default: no limit, every masked pass streams. Measured on warmed servers
 *                             (notes/impl.md "Stage A.1 -- engine results", R96): streaming beats the
 *                             stock step by 220-240 ms a straddling chunk at 512 / 1,024 / 1,984 exact
 *                             rows inside an ON server, and A.1's 64-row guard cost 331 ms at 9,216
 *                             and ran every T 2560 chunk exact. A threshold stays available for a
 *                             machine where the link makes streaming many rows lose    default unlimited
 *   env RADIANCE_RIDGEFILL_CKPT_FLOOR  T_ck (DD-A): a chunk that writes a prefix-cache checkpoint keeps its
 *                             last T_ck rows exact (rounded up to the delta net's tile), so a request
 *                             that resumes from that checkpoint and branches has at least T_ck exact
 *                             rows before its resume point. Costs T_ck/2,048 of the bulk rows on
 *                             checkpoint chunks; 0 = off (Dylan decides the value after Stage C's
 *                             branch numbers, R65/R66)                                   default 0
 *   env RADIANCE_RIDGEFILL_MIN_BULK_ROWS  a pass approximates only when it has at least this many bulk rows
 *                             (b - s_lb), else it runs the stock step. The projector is always in host
 *                             memory, so EVERY approximate pass streams every late layer's map over the
 *                             link whatever its rows: measured
 *                             (notes/stageb.md session 2c), the 64-row checkpoint remainders a
 *                             decoder-shared prompt alternates with cost 208-222 ms masked vs 61-92 ms
 *                             stock, and the decoders riding them got 2.2-3.5x slower. The two measured
 *                             shapes put break-even near 640 rows; 1,024 because below the stager's
 *                             1,025-row arming the stock step does not stage and is cheaper than that
 *                             line (unmeasured in between)                         default 1024
 *   env RADIANCE_RIDGEFILL_FINAL    off | on: OPTIONAL, not in the release (Dylan, 2026-10-05). on, with MTP
 *                             (--num-speculative-tokens > 0) and a folder holding the `final` map: every
 *                             approximate pass writes the predicted final stream of its bulk rows into the
 *                             trunk's stream before the epilogue, which the MTP head reads (ridgefill_final.h; DD-D).
 *                             Measured (R70, notes/staged.md): +1.8% drafted tokens a step at T 2048, for a
 *                             about +105 MiB VRAM a rank with int8 maps (a bf16-sized ring slot, 50 instead
 *                             of 25.4 MiB, plus the copied layer-S stream h_S and the ridgefill_final buffer, 40 MiB
 *                             each at 2,048 rows; computed from the declarations), +200 MiB host and +210 MB a
 *                             pass over the link. R70 ran on radiance 1.0.8. off: nothing of the map is declared, held or
 *                             streamed, whatever the folder holds                         default off
 *   env RADIANCE_RIDGEFILL_SCORE_BULK  1: approximate in KL mode too, whose logits on bulk rows are then
 *                             not the model's -- score only the exact tail (PLAN-FIX §6.2, R73)
 *
 * DEBUG SWITCHES (gates only; each changes what an approximate pass computes, and says so):
 *   RADIANCE_RIDGEFILL_STRADDLE     split | end: the straddling chunk's correction between bulk and
 *                             tail in a split scan (default), or once at the chunk end (R50)
 *   RADIANCE_RIDGEFILL_FORCE_SPLIT  N: the bulk ends N rows before every approximate chunk's end (R47)
 *   RADIANCE_RIDGEFILL_SHIFT_B      +-N rows added to every bulk end (R51)
 *   RADIANCE_RIDGEFILL_FORCE_STREAM 1: the late layers stream on every masked pass (R94)
 *   RADIANCE_RIDGEFILL_TAIL_ONLY    0: speed straddles take the masked path instead of the tail-only one
 *                             (the oracle the tail-only path is compared with, A.1)
 *   RADIANCE_RIDGEFILL_MASK         all: a masked pass approximates EVERY row before its bulk end, the
 *                             decoders' and the other prompts' included -- R54's negative control,
 *                             which must change a decoder's text (speed and quality only)
 *
 * THE PROJECTOR FOLDER (PACKAGING.md; ridgefill_folder.h, ridgefill_projector.h). The fitted tensors come from
 * `<model dir>/projector/` and never from the container -- ridgefill.* weights and keys an earlier append
 * left in a container are ignored:
 *
 *   env RADIANCE_RIDGEFILL_PROJECTOR    the folder, ahead of projector/ beside the model file. A folder
 *                                 whose manifest says "projector": {"dtype": "i8"} (tools/ridgefill_projector.py
 *                                 int8) holds the int8 maps: half the bytes on the link, read by the
 *                                 engine's int8 GEMM (ridgefill_int8.h)
 *
 * THE PROJECTOR IS ALWAYS STREAMED FROM HOST MEMORY (Dylan, 2026-10-05): its maps live in host-mapped
 * memory and each late layer's map is copied into ONE VRAM slot on the second lane, after the previous
 * layer's GEMM has read it (ridgefill_projector.h, ridgefill_layer.h ring_*). There is no placement choice: maps kept in VRAM cost
 * ~1,100 resident expert slots a card and made a configuration stock serves refuse to start (the
 * pinned pool overflowed at --max-num-batched-tokens 8192 --max-num-seqs 10), and the plugin cannot
 * see the engine's budget at declare to choose safely (notes/stagee.md §8).
 *   env RADIANCE_RIDGEFILL_ROWSEL_TABLE class -> score   none -> score_none (R41)
 *                                 all   -> score_all (every id a match: R35')
 *
 * RETIRED with the container append (A'), refused by name so an old command line cannot silently
 * run something else: RADIANCE_RIDGEFILL_PROJ and RADIANCE_RIDGEFILL_ST (a variant is its own folder now:
 * point RADIANCE_RIDGEFILL_PROJECTOR at it) and RADIANCE_RIDGEFILL_DECLARE (tools/dev/README.md). RETIRED with
 * the vram placement (Stage E): RADIANCE_RIDGEFILL_PROJ_PLACE and RADIANCE_RIDGEFILL_PROJ_RING.
 */
#ifndef RIDGEFILL_CONFIG_H
#define RIDGEFILL_CONFIG_H

#include "ridgefill_log.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace ridgefill {

using namespace rad::arch;

enum Mode     { MODE_OFF = 0, MODE_PLUMB, MODE_SPEED, MODE_QUALITY };
enum Rowsel   { ROWSEL_CLASS = 0, ROWSEL_RANDOM, ROWSEL_ALL };
enum Stage    { STAGE_AUTO = 0, STAGE_STOCK };
enum Straddle { STRADDLE_SPLIT = 0, STRADDLE_END };

static const char* const kModeNames[]     = { "off", "plumb", "speed", "quality" };
static const char* const kRowselNames[]   = { "class", "random", "all" };
static const char* const kStageNames[]    = { "auto", "stock" };
static const char* const kStraddleNames[] = { "split", "end" };

/* RADIANCE_RIDGEFILL_MIN_BULK_ROWS's default with the projector in host memory (the doc block above). */
constexpr int64_t kHostMinBulkRows = 1024;

/* The shortest exact tail the method was ever run at (tcc's MIN_TAIL, KVA-FACTS §5) and the tail it was
 * measured at. Both are fit facts of a model family, so the adapter supplies them (RidgeFillAdapter
 * min_tail/default_tail); these are only read_config's defaults for a caller with no adapter. */
constexpr int64_t kMinTail = 512, kDefaultTail = 2048;

struct Config {
    int         mode        = MODE_OFF;
    /* T: the method's measured exact tail (PLAN D9; tcc's DYLUHN_KVA_TAIL default, KVA-FACTS §5) --
     * an operating choice, not a model or machine number. Changing it is Dylan's call (HANDOVER
     * §2.4.5); ridgefill.tail / RADIANCE_RIDGEFILL_TAIL override it. */
    int64_t     tail        = kDefaultTail;
    int64_t     min_tail    = kMinTail;    /* the adapter's floor under `tail`, refused below (check_mode) */
    double      alpha       = 1.0;
    int         rowsel      = ROWSEL_CLASS;
    double      share       = 0.25;
    int64_t     seed        = 0;
    int         rowsel_table = 0;          /* kScoreNames' index: score, score_none, score_all */
    int         stage       = STAGE_AUTO;
    int64_t     stage_rows  = INT64_MAX;   /* always stream: notes/impl.md A.1 R96, the guard's trade */
    int64_t     min_bulk_rows = -1;        /* -1: the default (kHostMinBulkRows), resolved in read_config */
    int64_t     ckpt_floor  = 0;           /* RADIANCE_RIDGEFILL_CKPT_FLOOR (T_ck) */
    bool        score_bulk  = false;
    bool        final_on    = false;       /* RADIANCE_RIDGEFILL_FINAL=on: the MTP `final` map, when held and MTP is on */
    int         straddle    = STRADDLE_SPLIT;
    int64_t     force_split = 0;
    int64_t     shift_b     = 0;
    bool        force_stream = false;
    bool        tail_only   = true;
    bool        mask_step   = false;       /* RADIANCE_RIDGEFILL_MASK=all */
    const char* meta_mode   = nullptr;     /* the container's ridgefill.mode, if it has one: ignored */
};

/* The position of `s` in `names`, or -1. */
inline int pick(const char* s, std::initializer_list<const char*> names) {
    int i = 0;
    for (const char* n : names) {
        if (std::strcmp(s, n) == 0) return i;
        ++i;
    }
    return -1;
}

/* Strict parses: the whole string or nothing. "2048x" or "1e3" for a row count is refused, never read as
 * a prefix -- a switch that silently means something else would mislabel every number taken under it. */
inline bool parse_int(const char* s, int64_t* out) {
    char* end = nullptr;
    errno = 0;
    const long long v = std::strtoll(s, &end, 10);
    if (errno || end == s || *end) return false;
    *out = v;
    return true;
}

inline bool parse_real(const char* s, double* out) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s, &end);
    if (errno || end == s || *end || !std::isfinite(v)) return false;
    *out = v;
    return true;
}

inline int bad_value(const char* what, const char* value, const char* allowed) {
    std::fprintf(stderr, "radiance: %s: %s is '%s'; it takes %s\n", g_log_name, what, value, allowed);
    return RAD_E_INVAL;
}

/* The environment's value if set and non-empty, else null. */
inline const char* env(const char* name) {
    const char* e = std::getenv(name);
    return e && *e ? e : nullptr;
}

/* An enumerated environment switch: its index in `names` (the first when unset), or a refusal
 * naming every allowed value. */
inline int read_choice(const char* name, std::initializer_list<const char*> names,
                       const char* allowed, int* out) {
    const char* v = env(name);
    if (!v) v = *names.begin();
    const int i = pick(v, names);
    if (i < 0) return bad_value(name, v, allowed);
    *out = i;
    return RAD_OK;
}

/* An integer environment switch within [lo, hi]; unset leaves `out` alone. */
inline int read_int(const char* name, int64_t lo, int64_t hi, const char* allowed, int64_t* out) {
    const char* v = env(name);
    if (!v) return RAD_OK;
    if (!parse_int(v, out) || *out < lo || *out > hi) return bad_value(name, v, allowed);
    return RAD_OK;
}

/* Switches from the retired container-append route: refused by name rather than ignored, so a script
 * written for it cannot run believing it selected a variant. */
inline int read_variants(Config* c) {
    for (const char* retired : { "RADIANCE_RIDGEFILL_PROJ", "RADIANCE_RIDGEFILL_ST", "RADIANCE_RIDGEFILL_DECLARE" })
        if (const char* v = env(retired)) {
            std::fprintf(stderr, "radiance: %s: %s=%s is retired with the container append: "
                                 "the fitted tensors come from the projector folder (a variant is a "
                                 "folder of its own, named by RADIANCE_RIDGEFILL_PROJECTOR); unset it\n", g_log_name,
                         retired, v);
            return RAD_E_INVAL;
        }
    for (const char* retired : { "RADIANCE_RIDGEFILL_PROJ_PLACE", "RADIANCE_RIDGEFILL_PROJ_RING" })
        if (const char* v = env(retired)) {
            std::fprintf(stderr, "radiance: %s: %s=%s is retired: the projector is always "
                                 "streamed from host memory through the staging ring (maps kept in VRAM "
                                 "could make a configuration stock serves refuse to start); unset it\n", g_log_name,
                         retired, v);
            return RAD_E_INVAL;
        }
    return read_choice("RADIANCE_RIDGEFILL_ROWSEL_TABLE", { "class", "none", "all" }, "class|none|all",
                       &c->rowsel_table);
}

/* The stager lever, the KL switch and the gate-only debug switches. */
inline int read_switches(Config* c) {
    int v = 0;
    RAD_ARCH_TRY(read_choice("RADIANCE_RIDGEFILL_STAGE", { "auto", "stock" }, "auto|stock", &c->stage));
    RAD_ARCH_TRY(read_int("RADIANCE_RIDGEFILL_STAGE_ROWS", 0, INT64_MAX, "a row count", &c->stage_rows));
    RAD_ARCH_TRY(read_int("RADIANCE_RIDGEFILL_MIN_BULK_ROWS", 0, INT64_MAX, "a row count", &c->min_bulk_rows));
    RAD_ARCH_TRY(read_int("RADIANCE_RIDGEFILL_CKPT_FLOOR", 0, INT64_MAX / 2, "a row count", &c->ckpt_floor));
    RAD_ARCH_TRY(read_choice("RADIANCE_RIDGEFILL_SCORE_BULK", { "0", "1" }, "1 or unset", &v));
    c->score_bulk = v == 1;
    RAD_ARCH_TRY(read_choice("RADIANCE_RIDGEFILL_FINAL", { "off", "on" }, "off|on", &v));
    c->final_on = v == 1;
    RAD_ARCH_TRY(read_choice("RADIANCE_RIDGEFILL_STRADDLE", { "split", "end" }, "split|end",
                             &c->straddle));
    RAD_ARCH_TRY(read_int("RADIANCE_RIDGEFILL_FORCE_SPLIT", 1, INT64_MAX, "a positive row count",
                          &c->force_split));
    RAD_ARCH_TRY(read_int("RADIANCE_RIDGEFILL_SHIFT_B", INT64_MIN / 2, INT64_MAX / 2, "a row count",
                          &c->shift_b));
    RAD_ARCH_TRY(read_choice("RADIANCE_RIDGEFILL_FORCE_STREAM", { "0", "1" }, "1 or unset", &v));
    c->force_stream = v == 1;
    RAD_ARCH_TRY(read_choice("RADIANCE_RIDGEFILL_TAIL_ONLY", { "1", "0" }, "0 or unset", &v));
    c->tail_only = v == 0;
    RAD_ARCH_TRY(read_choice("RADIANCE_RIDGEFILL_MASK", { "rule", "all" }, "all or unset", &v));
    c->mask_step = v == 1;
    if (c->mask_step && c->mode == MODE_PLUMB) {
        std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_MASK=all approximates rows and mode "
                             "plumb projects none; it takes speed or quality\n", g_log_name);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

inline int read_config(const RadModelMeta* meta, Config* c, int64_t min_tail = kMinTail,
                       int64_t default_tail = kDefaultTail) {
    *c = Config{};
    c->tail = default_tail;
    c->min_tail = min_tail;
    RAD_ARCH_TRY(read_choice("RADIANCE_RIDGEFILL", { "off", "plumb", "speed", "quality" },
                             "off|plumb|speed|quality", &c->mode));
    c->meta_mode = rad_meta_gets(meta, "ridgefill.mode", nullptr);
    if (const char* t = env("RADIANCE_RIDGEFILL_TAIL"); t && !parse_int(t, &c->tail))
        return bad_value("RADIANCE_RIDGEFILL_TAIL", t, "a token count");
    const char* a = env("RADIANCE_RIDGEFILL_ALPHA");
    if (a && !(parse_real(a, &c->alpha) && c->alpha >= 0.0 && c->alpha <= 1.0))
        return bad_value("RADIANCE_RIDGEFILL_ALPHA", a, "a strength in [0, 1]");
    RAD_ARCH_TRY(read_choice("RADIANCE_RIDGEFILL_ROWSEL", { "class", "random", "all" },
                             "class|random|all", &c->rowsel));
    if (const char* s = env("RADIANCE_RIDGEFILL_SHARE"); s && !parse_real(s, &c->share))
        return bad_value("RADIANCE_RIDGEFILL_SHARE", s, "(0, 1]");
    if (!(c->share > 0.0 && c->share <= 1.0)) {
        std::fprintf(stderr, "radiance: %s: the row share is %g; it takes (0, 1]\n", g_log_name,
                     c->share);
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(read_switches(c));
    RAD_ARCH_TRY(read_variants(c));
    if (c->min_bulk_rows < 0) c->min_bulk_rows = kHostMinBulkRows;
    return RAD_OK;
}

}  /* namespace ridgefill */

#endif /* RIDGEFILL_CONFIG_H */
