/* kva_config.h -- what an operator can turn on the KVA plugin, read ONCE, at declare.
 *
 * Container metadata first (set by `rad-convert --set`), then environment overrides. step() reads
 * only the parsed Config and keyed batch fields, never the environment (HANDOVER §2.1, R15): a
 * replayed pass must issue what a fresh one would.
 *
 *   meta kva.mode           off | plumb | speed | quality    env RADIANCE_KVA        default off
 *   meta kva.tail           exact tail T in tokens           env RADIANCE_KVA_TAIL   default 2048
 *                           correction strength alpha        env RADIANCE_KVA_ALPHA  default 1
 *                           row selection rule               env RADIANCE_KVA_ROWSEL class|random|all
 *   meta kva.rowsel.share   share of a chunk's class matches env RADIANCE_KVA_SHARE  default 0.25
 *   meta kva.rowsel.cap     compacted rows per chunk         (derived from share when unset or
 *                                                             when RADIANCE_KVA_SHARE is given)
 *   meta kva.rowsel.seed    seed of the random control                               default 0
 *
 * WHICH COPY OF EACH FITTED TENSOR (the controls and the Stage 6 refit live in the same container
 * under suffixed names, because the disk has no room for a second 114 GiB container and
 * `rad-convert --reuse --in-place` refuses to change a weight it already holds):
 *
 *   env RADIANCE_KVA_PROJ         shipped -> kva.proj.L.{weight,bias}   refit -> kva.projr.L.*
 *   env RADIANCE_KVA_ST           shipped -> kva.st.L   swap -> kva.stswap.L (R24)   refit -> kva.str.L
 *   env RADIANCE_KVA_ROWSEL_TABLE class -> kva.rowsel.score   none -> kva.rowsel.score_none (R41)
 *                                 all   -> kva.rowsel.score_all (every id a match: R35)
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace qwen4exp_kva {

enum Mode   { MODE_OFF = 0, MODE_PLUMB, MODE_SPEED, MODE_QUALITY };
enum Rowsel { ROWSEL_CLASS = 0, ROWSEL_RANDOM, ROWSEL_ALL };

static const char* const kModeNames[]   = { "off", "plumb", "speed", "quality" };
static const char* const kRowselNames[] = { "class", "random", "all" };

/* The shortest exact tail the method was ever run at (tcc's MIN_TAIL, KVA-FACTS §5). */
constexpr int64_t kMinTail = 512;
/* The compacted row count is padded to whole 64-row tiles (PLAN D12). */
constexpr int64_t kCapQuantum = 64;

struct Config {
    int         mode        = MODE_OFF;
    int64_t     tail        = 2048;
    double      alpha       = 1.0;
    int         rowsel      = ROWSEL_CLASS;
    double      share       = 0.25;
    int64_t     cap         = 0;
    int64_t     seed        = 0;
    const char* proj        = "kva.proj";
    const char* st          = "kva.st";
    const char* score       = "kva.rowsel.score";
    bool        declare_all = false;
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

/* A string setting: the environment's if set and non-empty, else the container's, else `dflt`. */
inline const char* setting(const RadModelMeta* meta, const char* key, const char* env,
                           const char* dflt) {
    const char* e = env ? std::getenv(env) : nullptr;
    if (e && *e) return e;
    return key ? rad_meta_gets(meta, key, dflt) : dflt;
}

/* An enumerated setting: its index in `names`, or a refusal naming every allowed value. */
inline int read_choice(const RadModelMeta* meta, const char* key, const char* env,
                       std::initializer_list<const char*> names, const char* allowed, int* out) {
    const char* v = setting(meta, key, env, *names.begin());
    const int i = pick(v, names);
    if (i < 0) return bad_value(env ? env : key, v, allowed);
    *out = i;
    return RAD_OK;
}

inline int read_variants(Config* c) {
    int v = 0;
    RAD_ARCH_TRY(read_choice(nullptr, nullptr, "RADIANCE_KVA_PROJ", { "shipped", "refit" },
                             "shipped|refit", &v));
    c->proj = v == 0 ? "kva.proj" : "kva.projr";
    RAD_ARCH_TRY(read_choice(nullptr, nullptr, "RADIANCE_KVA_ST", { "shipped", "swap", "refit" },
                             "shipped|swap|refit", &v));
    c->st = v == 0 ? "kva.st" : v == 1 ? "kva.stswap" : "kva.str";
    RAD_ARCH_TRY(read_choice(nullptr, nullptr, "RADIANCE_KVA_ROWSEL_TABLE",
                             { "class", "none", "all" }, "class|none|all", &v));
    c->score = v == 0 ? "kva.rowsel.score" : v == 1 ? "kva.rowsel.score_none"
                                                    : "kva.rowsel.score_all";
    RAD_ARCH_TRY(read_choice(nullptr, nullptr, "RADIANCE_KVA_DECLARE", { "selected", "all" },
                             "all (rad-convert) or unset (serving)", &v));
    c->declare_all = v == 1;
    return RAD_OK;
}

/* The row share and the compacted row count. A share from the environment re-derives the cap (the
 * Stage 6 share sweep); otherwise a cap the container states wins over the derived one. Never
 * more than max_tok: every compacted issue is at M = cap and the ops are declared to max_tok. */
inline int read_share(const RadModelMeta* meta, int64_t max_tok, Config* c) {
    const char* e = std::getenv("RADIANCE_KVA_SHARE");
    c->share = rad_meta_getf(meta, "kva.rowsel.share", c->share);
    if (e && *e && !parse_real(e, &c->share)) return bad_value("RADIANCE_KVA_SHARE", e, "(0, 1]");
    if (!(c->share > 0.0 && c->share <= 1.0)) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: the row share is %g; it takes (0, 1]\n",
                     c->share);
        return RAD_E_INVAL;
    }
    const int64_t meta_cap = rad_meta_geti(meta, "kva.rowsel.cap", 0);
    const double  rows = std::ceil(c->share * (double)max_tok / (double)kCapQuantum);
    c->cap = (e && *e) || meta_cap <= 0 ? (int64_t)rows * kCapQuantum : meta_cap;
    if (c->cap > max_tok) c->cap = max_tok;
    c->seed = rad_meta_geti(meta, "kva.rowsel.seed", 0);
    return RAD_OK;
}

inline int read_config(const RadModelMeta* meta, int64_t max_tok, Config* c) {
    *c = Config{};
    RAD_ARCH_TRY(read_choice(meta, "kva.mode", "RADIANCE_KVA",
                             { "off", "plumb", "speed", "quality" }, "off|plumb|speed|quality",
                             &c->mode));
    c->tail = rad_meta_geti(meta, "kva.tail", c->tail);
    const char* t = std::getenv("RADIANCE_KVA_TAIL");
    if (t && *t && !parse_int(t, &c->tail))
        return bad_value("RADIANCE_KVA_TAIL", t, "a token count");
    const char* a = std::getenv("RADIANCE_KVA_ALPHA");
    if (a && *a && !(parse_real(a, &c->alpha) && c->alpha >= 0.0 && c->alpha <= 1.0))
        return bad_value("RADIANCE_KVA_ALPHA", a, "a strength in [0, 1]");
    RAD_ARCH_TRY(read_choice(nullptr, nullptr, "RADIANCE_KVA_ROWSEL",
                             { "class", "random", "all" }, "class|random|all", &c->rowsel));
    RAD_ARCH_TRY(read_share(meta, max_tok, c));
    return read_variants(c);
}

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_KVA_CONFIG_H */
