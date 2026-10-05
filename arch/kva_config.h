/* kva_config.h -- what an operator can turn on the KVA plugin, read ONCE, at declare.
 *
 * step() reads only the parsed Config and keyed batch fields, never the environment (HANDOVER
 * §2.1, R15/R99): a replayed pass must issue what a fresh one would.
 *
 * THE MODE COMES FROM THE ENVIRONMENT ONLY (PLAN-FIX §6.5, R81). A container's `kva.mode` is read
 * only to say, once, that it is ignored: the default is `off`, so a container converted with a
 * mode in it can never switch the trade on behind an operator's back.
 *
 *   env RADIANCE_KVA          off | plumb | speed | quality                      default off
 *   meta kva.tail             exact tail T in tokens   env RADIANCE_KVA_TAIL     default 2048
 *                             correction strength      env RADIANCE_KVA_ALPHA    default 1
 *                             row selection rule       env RADIANCE_KVA_ROWSEL   class|random|all
 *   meta kva.rowsel.share     share of the window's class matches kept exact
 *                                                      env RADIANCE_KVA_SHARE    default 0.25
 *   meta kva.rowsel.seed      seed of the random control                         default 0
 *   env RADIANCE_KVA_STAGE    auto | stock: the stager lever (PLAN-FIX §6.1, notes/impl.md §2).
 *                             auto lets the late layers STREAM their routed experts on an
 *                             approximate pass whose exact rows are at most RADIANCE_KVA_STAGE_ROWS;
 *                             stock leaves the stager as it is (the tested fallback, R96)  default auto
 *   env RADIANCE_KVA_STAGE_ROWS  the exact rows a masked pass may carry and still stream; past it
 *                             the pass runs exact (speed straddles take the tail-only path) --
 *                             streaming many rows reads every expert over the link (A.1)  default 64
 *   env RADIANCE_KVA_SCORE_BULK  1: approximate in KL mode too, whose logits on bulk rows are then
 *                             not the model's -- score only the exact tail (PLAN-FIX §6.2, R73)
 *
 * DEBUG SWITCHES (gates only; each changes what an approximate pass computes, and says so):
 *   RADIANCE_KVA_STRADDLE     split | end: the straddling chunk's correction between bulk and
 *                             tail in a split scan (default), or once at the chunk end (R50)
 *   RADIANCE_KVA_FORCE_SPLIT  N: the bulk ends N rows before every approximate chunk's end (R47)
 *   RADIANCE_KVA_SHIFT_B      +-N rows added to every bulk end (R51)
 *   RADIANCE_KVA_FORCE_STREAM 1: the late layers stream on every masked pass (R94)
 *   RADIANCE_KVA_TAIL_ONLY    0: speed straddles take the masked path instead of the tail-only one
 *                             (the oracle the tail-only path is compared with, A.1)
 *
 * WHICH COPY OF EACH FITTED TENSOR (the controls and the Stage 6 refit live in the same container
 * under suffixed names, because the disk has no room for a second 114 GiB container and
 * `rad-convert --reuse --in-place` refuses to change a weight it already holds):
 *
 *   env RADIANCE_KVA_PROJ         shipped -> kva.proj.L.{weight,bias}   refit -> kva.projr.L.*
 *   env RADIANCE_KVA_ST           shipped -> kva.st.L   swap -> kva.stswap.L (R24)   refit -> kva.str.L
 *   env RADIANCE_KVA_ROWSEL_TABLE class -> kva.rowsel.score   none -> kva.rowsel.score_none (R41)
 *                                 all   -> kva.rowsel.score_all (every id a match: R35')
 *   env RADIANCE_KVA_DECLARE      all: declare EVERY kva.* tensor the model holds, which is what
 *                                 rad-convert must see (it writes only declared weights, and an
 *                                 --in-place append drops any old entry the declare did not name).
 *                                 Unset when serving: only the selected set is declared, so the
 *                                 unselected copies are never placed in VRAM.
 */
#ifndef QWEN4EXP_KVA_CONFIG_H
#define QWEN4EXP_KVA_CONFIG_H

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace qwen4exp_kva {

enum Mode     { MODE_OFF = 0, MODE_PLUMB, MODE_SPEED, MODE_QUALITY };
enum Rowsel   { ROWSEL_CLASS = 0, ROWSEL_RANDOM, ROWSEL_ALL };
enum Stage    { STAGE_AUTO = 0, STAGE_STOCK };
enum Straddle { STRADDLE_SPLIT = 0, STRADDLE_END };

static const char* const kModeNames[]     = { "off", "plumb", "speed", "quality" };
static const char* const kRowselNames[]   = { "class", "random", "all" };
static const char* const kStageNames[]    = { "auto", "stock" };
static const char* const kStraddleNames[] = { "split", "end" };

/* The shortest exact tail the method was ever run at (tcc's MIN_TAIL, KVA-FACTS §5). */
constexpr int64_t kMinTail = 512;

struct Config {
    int         mode        = MODE_OFF;
    /* T: the method's measured exact tail (PLAN D9; tcc's DYLUHN_KVA_TAIL default, KVA-FACTS §5) --
     * an operating choice, not a model or machine number. Changing it is Dylan's call (HANDOVER
     * §2.4.5); kva.tail / RADIANCE_KVA_TAIL override it. */
    int64_t     tail        = 2048;
    double      alpha       = 1.0;
    int         rowsel      = ROWSEL_CLASS;
    double      share       = 0.25;
    int64_t     seed        = 0;
    const char* proj        = "kva.proj";
    const char* st          = "kva.st";
    const char* score       = "kva.rowsel.score";
    bool        declare_all = false;
    int         stage       = STAGE_AUTO;
    int64_t     stage_rows  = 64;          /* one tile: A.1's profile, notes/impl.md §6 */
    bool        score_bulk  = false;
    int         straddle    = STRADDLE_SPLIT;
    int64_t     force_split = 0;
    int64_t     shift_b     = 0;
    bool        force_stream = false;
    bool        tail_only   = true;
    const char* meta_mode   = nullptr;     /* the container's kva.mode, if it has one: ignored */
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
    std::fprintf(stderr, "radiance: qwen4exp_kva: %s is '%s'; it takes %s\n", what, value, allowed);
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

inline int read_variants(Config* c) {
    int v = 0;
    RAD_ARCH_TRY(read_choice("RADIANCE_KVA_PROJ", { "shipped", "refit" }, "shipped|refit", &v));
    c->proj = v == 0 ? "kva.proj" : "kva.projr";
    RAD_ARCH_TRY(read_choice("RADIANCE_KVA_ST", { "shipped", "swap", "refit" },
                             "shipped|swap|refit", &v));
    c->st = v == 0 ? "kva.st" : v == 1 ? "kva.stswap" : "kva.str";
    RAD_ARCH_TRY(read_choice("RADIANCE_KVA_ROWSEL_TABLE", { "class", "none", "all" },
                             "class|none|all", &v));
    c->score = v == 0 ? "kva.rowsel.score" : v == 1 ? "kva.rowsel.score_none"
                                                    : "kva.rowsel.score_all";
    RAD_ARCH_TRY(read_choice("RADIANCE_KVA_DECLARE", { "selected", "all" },
                             "all (rad-convert) or unset (serving)", &v));
    c->declare_all = v == 1;
    return RAD_OK;
}

/* The stager lever, the KL switch and the gate-only debug switches. */
inline int read_switches(Config* c) {
    int v = 0;
    RAD_ARCH_TRY(read_choice("RADIANCE_KVA_STAGE", { "auto", "stock" }, "auto|stock", &c->stage));
    RAD_ARCH_TRY(read_int("RADIANCE_KVA_STAGE_ROWS", 0, INT64_MAX, "a row count", &c->stage_rows));
    RAD_ARCH_TRY(read_choice("RADIANCE_KVA_SCORE_BULK", { "0", "1" }, "1 or unset", &v));
    c->score_bulk = v == 1;
    RAD_ARCH_TRY(read_choice("RADIANCE_KVA_STRADDLE", { "split", "end" }, "split|end",
                             &c->straddle));
    RAD_ARCH_TRY(read_int("RADIANCE_KVA_FORCE_SPLIT", 1, INT64_MAX, "a positive row count",
                          &c->force_split));
    RAD_ARCH_TRY(read_int("RADIANCE_KVA_SHIFT_B", INT64_MIN / 2, INT64_MAX / 2, "a row count",
                          &c->shift_b));
    RAD_ARCH_TRY(read_choice("RADIANCE_KVA_FORCE_STREAM", { "0", "1" }, "1 or unset", &v));
    c->force_stream = v == 1;
    RAD_ARCH_TRY(read_choice("RADIANCE_KVA_TAIL_ONLY", { "1", "0" }, "0 or unset", &v));
    c->tail_only = v == 0;
    return RAD_OK;
}

inline int read_config(const RadModelMeta* meta, Config* c) {
    *c = Config{};
    RAD_ARCH_TRY(read_choice("RADIANCE_KVA", { "off", "plumb", "speed", "quality" },
                             "off|plumb|speed|quality", &c->mode));
    c->meta_mode = rad_meta_gets(meta, "kva.mode", nullptr);
    c->tail = rad_meta_geti(meta, "kva.tail", c->tail);
    if (const char* t = env("RADIANCE_KVA_TAIL"); t && !parse_int(t, &c->tail))
        return bad_value("RADIANCE_KVA_TAIL", t, "a token count");
    const char* a = env("RADIANCE_KVA_ALPHA");
    if (a && !(parse_real(a, &c->alpha) && c->alpha >= 0.0 && c->alpha <= 1.0))
        return bad_value("RADIANCE_KVA_ALPHA", a, "a strength in [0, 1]");
    RAD_ARCH_TRY(read_choice("RADIANCE_KVA_ROWSEL", { "class", "random", "all" },
                             "class|random|all", &c->rowsel));
    c->share = rad_meta_getf(meta, "kva.rowsel.share", c->share);
    if (const char* s = env("RADIANCE_KVA_SHARE"); s && !parse_real(s, &c->share))
        return bad_value("RADIANCE_KVA_SHARE", s, "(0, 1]");
    if (!(c->share > 0.0 && c->share <= 1.0)) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: the row share is %g; it takes (0, 1]\n",
                     c->share);
        return RAD_E_INVAL;
    }
    c->seed = rad_meta_geti(meta, "kva.rowsel.seed", 0);
    RAD_ARCH_TRY(read_switches(c));
    return read_variants(c);
}

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_KVA_CONFIG_H */
