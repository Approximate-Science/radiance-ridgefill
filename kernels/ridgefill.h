/* ridgefill.h -- what the RidgeFill ops share between the row table, the host rows and the device rows.
 *
 * The plugin is one .so with a host row (the oracle, plain C++) and a device row per op. The
 * operand parsing is host code shared by both rows, so a device row can never accept an operand
 * its oracle would refuse. The few formulas that DEFINE an op's answer -- the row hash of the
 * random selector, ridgefill_mask's window, the rho clamp, softplus, the bf16 decode -- are written once
 * here and compiled
 * for both sides, so "host == device" is a statement about the kernels and not about two spellings
 * of one definition.
 */
#pragma once

#include <rad_abi.h>
#include <rad_plugin.h>

#include <math.h>
#include <stdint.h>

#if defined(__HIPCC__)
#define RIDGEFILL_HD __host__ __device__
#else
#define RIDGEFILL_HD
#endif

/* ridgefill_mask's device row keeps one 4-byte key per window row in LDS: this many rows is 32 KiB of
 * the 64 KiB a dispatch may use (docs/PLUGIN.md, "What a launch may do"). The window is at most
 * the bulk end b rows (b' <= b), so the row is constrained to M (issued at b) <= this, which is
 * also the engine's default --max-num-batched-tokens; a larger band falls to no device row and is
 * refused at issue, by name. The mask itself (n rows) is not bounded by it. */
enum { RIDGEFILL_MASK_MAX_ROWS = 8192, RIDGEFILL_MASK_THREADS = 256 };

/* softplus' large-x cutoff: above it log1p(e^x) is x to every bit f32 holds. The value libref and
 * libr4d default `softplus_thr` to, and torch.nn.functional.softplus's threshold, which is what the
 * reference (ridgefill b0/worker_ext.py log_gate) used. */
#define RIDGEFILL_SOFTPLUS_THRESHOLD 20.0f

/* ridgefill_mask's `mode`, in the order rows.cpp's constraint and the parse spell them. `step` is a
 * debug control (R54's negative control): every row of the step before b' is approximated, the
 * other sequences' and the decoders' included -- the gate that proves a decoder's text can change. */
enum { RIDGEFILL_MODE_NONE = 0, RIDGEFILL_MODE_CLASS = 1, RIDGEFILL_MODE_RANDOM = 2, RIDGEFILL_MODE_ALL = 3, RIDGEFILL_MODE_STEP = 4 };

/* Where the mask's 1s may start: row 0 in step mode when the window is not empty, else s. The
 * bounds keep s whatever the mode (the delta net's split scan is the last sequence's). */
RIDGEFILL_HD inline int64_t ridgefill_mask_from(int mode, const int64_t* w) {
    return mode == RIDGEFILL_MODE_STEP && w[1] > w[0] ? 0 : w[0];
}

/* Operand positions, in schema order (rows.cpp holds the schemas). */
enum { MK_CU = 0, MK_TOKENS, MK_POS, MK_SCORE, MK_MASK, MK_BOUNDS, MK_ZEROS };
enum { RH_A = 0, RH_MASK, RH_ALOG, RH_DTBIAS, RH_ND, RH_SIDX, RH_BOUNDS };
enum { SC_STATE = 0, SC_STATE_IDX, SC_APPLIED, SC_APPLIED_IDX, SC_C, SC_ND, SC_ND_IDX, SC_BOUNDS };
enum { SR_STATE = 0, SR_STATE_IDX, SR_OUT };
enum { SL_MASK = 0, SL_X_SRC, SL_Q_SRC, SL_S_SRC, SL_X, SL_Q, SL_S };
enum { DR_MASK = 0, DR_IDS };
enum { HZ_CU = 0, HZ_POS, HZ_BOUNDS, HZ_SPAN, HZ_META, HZ_META_IDX, HZ_COUNT };

/* ridgefill_select's (source, destination) pairs: x, then the optional q codes and s scales. */
enum { RIDGEFILL_SELECT_PAIRS = 3 };

/* ---------------------------------------------------------------- the shared definitions */

/* Wellons' lowbias32 finaliser: a bijection on 32 bits with good avalanche, so distinct rows get
 * independent-looking ranks. */
RIDGEFILL_HD inline uint32_t ridgefill_mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/* The random selector's rank key for the token at absolute prompt position `position` under
 * `seed`: the k rows with the smallest key are kept, ties by lower row. Keyed on the ABSOLUTE
 * position, not the row in the chunk, so successive chunks of one prompt draw different offsets
 * and the control is not periodic in the chunk length. */
RIDGEFILL_HD inline uint32_t ridgefill_row_hash(long long seed, uint32_t position) {
    const unsigned long long s = (unsigned long long)seed;
    const uint32_t mixed_seed = ridgefill_mix32((uint32_t)s ^ ridgefill_mix32((uint32_t)(s >> 32)));
    return ridgefill_mix32(mixed_seed ^ (position * 0x9E3779B9U + 0x7F4A7C15U));
}

/* rho = clamp(N / D, 0, 1), and 1 while nothing has been accumulated (D == 0): the correction is
 * then the plain speed-mode one. Spelled with comparisons rather than fminf/fmaxf so a NaN ratio
 * lands on 0 identically on both sides. */
RIDGEFILL_HD inline float ridgefill_rho(float n, float d) {
    if (!(d > 0.0f)) return 1.0f;
    float r = n / d;
    r = r > 0.0f ? r : 0.0f;
    return r < 1.0f ? r : 1.0f;
}

/* softplus in the two-sided form libref and libr4d use (the GDN gate's own spelling). */
RIDGEFILL_HD inline float ridgefill_softplus(float x) {
    if (x > RIDGEFILL_SOFTPLUS_THRESHOLD) return x;
    return x > 0.0f ? x + log1pf(expf(-x)) : log1pf(expf(x));
}

RIDGEFILL_HD inline float ridgefill_bf16_to_f32(uint16_t h) {
    const uint32_t bits = (uint32_t)h << 16;
    float f;
    __builtin_memcpy(&f, &bits, sizeof f);
    return f;
}

/* ---------------------------------------------------------------- parsed operands
 *
 * Each op's operands, checked and reduced to pointers, extents and element strides. Passed BY
 * VALUE as the device kernel's argument, so every field is a plain scalar, pointer or array of
 * them (the largest, RidgeFillCorrect, is 248 bytes of the 768-byte argument limit). */

typedef struct RidgeFillMask {
    const int32_t* cu_last;                  /* [2] {s, e}; device memory on the device row */
    const int32_t* tokens;  int64_t b;       /* [b] ids of rows [0, b): b is the bulk end */
    const int32_t* positions; int64_t pos_stride;  /* row i's absolute position: positions[i * pos_stride] */
    const float*   score;   int64_t vocab;   /* [vocab]; null (absent) only in none / all mode */
    int32_t*       mask;    int64_t n;       /* [n], every row written: 1 = approximated */
    int32_t*       bounds;                   /* [4] {s, b', b', e} */
    int32_t*       zeros;   int64_t n_zeros; /* optional [n_zeros], every element written 0 */
    double         share;
    long long      seed;
    int            mode;                     /* RIDGEFILL_MODE_* */
} RidgeFillMask;

/* ridgefill_mask's window W = [s, b') of the step's last sequence -- s, e from cu_last, b the bulk end,
 * b' = min(max(b, s), e) -- into w[0..2] = {s, b', e}. False when cu_last is out of order (s < 0,
 * s > e or e > n): the host row refuses that. The device row cannot (the values are device memory
 * and a return code would need a synchronize), so it takes the window EMPTY inside the clamped
 * [s, e]: every row exact, the plain model, rather than approximating rows nobody chose. */
RIDGEFILL_HD inline bool ridgefill_mask_window(int64_t s, int64_t e, int64_t b, int64_t n, int64_t* w) {
    const bool valid = s >= 0 && s <= e && e <= n;
    const int64_t end = e < 0 ? 0 : (e > n ? n : e);
    const int64_t start = s < 0 ? 0 : (s > end ? end : s);
    const int64_t bulk = b < start ? start : (b > end ? end : b);
    w[0] = start;
    w[1] = valid ? bulk : start;
    w[2] = end;
    return valid;
}

/* Row i's class score, -inf when the row does not match: a non-finite score (NaN fails both
 * compares) or an id outside the table. */
RIDGEFILL_HD inline float ridgefill_mask_score(const RidgeFillMask* g, int64_t i) {
    const int32_t id = g->tokens[i];
    const float s = id >= 0 && id < g->vocab ? g->score[id] : -INFINITY;
    return s > -INFINITY && s < INFINITY ? s : -INFINITY;
}

typedef struct RidgeFillRho {
    const void*    a;       int64_t a_pitch, a_col; int a_bf16;   /* [n, n_head], any strides */
    const int32_t* mask;
    const float*   a_log;
    const float*   dt_bias;
    float*         nd;      int64_t nd_slot, nd_head, nd_inner, n_states;
    const int32_t* state_idx;
    const int32_t* bounds;                   /* [>=1] or null: ridgefill_mask's {s, ...} */
    int64_t        n, n_head;
} RidgeFillRho;

/* The recurrence's first row: bounds[0] (the step's last sequence's first row, s) clamped into
 * [0, n]; 0 when bounds is absent. Read where the row runs -- the bounds are device memory on the
 * device row, and reading them on the host would need a synchronize. */
RIDGEFILL_HD inline int64_t ridgefill_rho_first_row(const RidgeFillRho* g) {
    if (!g->bounds) return 0;
    const int64_t s = g->bounds[0];
    return s < 0 ? 0 : (s > g->n ? g->n : s);
}

/* Each KV operand comes with ITS OWN group's slot index: the KV manager hands every stateful group
 * the same slot per sequence today, but that is not a contract, so nothing here relies on it. */
typedef struct RidgeFillCorrect {
    float*         state;   int64_t st_slot, st_head, st_row, st_col;
    float*         applied; int64_t ap_slot, ap_head;
    const float*   c;       int64_t c_head, c_row, c_col;
    const float*   nd;      int64_t nd_slot, nd_head, nd_inner;   /* nd null: rho = 1 */
    const int32_t* st_idx;  const int32_t* ap_idx;  const int32_t* nd_idx;
    int64_t        st_pitch, ap_pitch, nd_pitch;      /* row pitch of each index; column 0 is used */
    int64_t        st_states, ap_states, nd_states;   /* each pool's slot count */
    int64_t        n_seq, n_head, sd0, sd1;
    const int32_t* bounds;                   /* [>=2] or null: ridgefill_mask's {s, b', ...} */
    float          alpha;
    int            apply;                    /* 1 apply, 0 undo */
} RidgeFillCorrect;

/* False when the step's last sequence had no bulk row (bounds[1] <= bounds[0]): the op then writes
 * nothing, so that sequence's correction stays as the previous chunk end left it (apply would add
 * alpha * C, rho being 1 while D is 0). Always true without bounds. Read where the row runs. */
RIDGEFILL_HD inline bool ridgefill_correct_has_bulk(const RidgeFillCorrect* g) {
    return !g->bounds || g->bounds[1] > g->bounds[0];
}

/* Sequence s's slot in each pool. False when any of them is outside its pool: the sequence is
 * skipped (a negative slot is how a sequence with no state is spelled). */
RIDGEFILL_HD inline bool ridgefill_correct_slots(const RidgeFillCorrect* g, int64_t s, int64_t* st, int64_t* ap,
                                     int64_t* nd) {
    *st = g->st_idx[s * g->st_pitch];
    *ap = g->ap_idx[s * g->ap_pitch];
    *nd = g->nd ? g->nd_idx[s * g->nd_pitch] : 0;
    return *st >= 0 && *st < g->st_states && *ap >= 0 && *ap < g->ap_states &&
           (!g->nd || (*nd >= 0 && *nd < g->nd_states));
}

/* ridgefill_state_read: each sequence's GDN state slot copied out densely (the Stage 6 correction refit
 * captures states this way; no ABI call returns a KV-pool pointer). */
typedef struct RidgeFillStateRead {
    const float*   state;   int64_t st_slot, st_head, st_row, st_col, n_states;
    const int32_t* idx;     int64_t idx_pitch;
    float*         out;                      /* dense [n_seq, n_head, sd0, sd1] */
    int64_t        n_seq, n_head, sd0, sd1;
} RidgeFillStateRead;

/* ridgefill_hazard (DD-A's exact instrument, PLAN-FIX §5.4). A sequence's meta slot (a LINEAR group, so it
 * is zeroed at admission and snapshotted with every checkpoint) holds {last approximated position +
 * 1, positions counted up to}. */
typedef struct RidgeFillHazard {
    const int32_t* cu_last;                  /* [2] {s, e} of the step's last sequence */
    const int32_t* positions; int64_t pos_stride;
    const int32_t* bounds;                   /* optional {s, b'}: record this pass's last bulk position */
    int64_t        span;                     /* T - n_ahead, the span operand's extent; 0 = absent */
    float*         meta;    int64_t meta_slot, meta_states;
    const int32_t* meta_idx;
    float*         count;                    /* [1], accumulated hazard rows */
} RidgeFillHazard;

/* THE HAZARD RULE. The pass's last sequence starts at position P with q rows and n_ahead more to
 * come, so its exact tail is [P + q + n_ahead - T, ...): `before` = span - q of those tail positions
 * lie BEFORE this pass, i.e. came from the prefix cache. Any of them at or below the slot's last
 * approximated position was approximated by the request that wrote the snapshot -- a hazard row
 * (counted once: the slot remembers how far it counted). Then, on an approximate pass, the slot
 * records this pass's last bulk position. A sequence's own earlier bulk always ends before its own
 * tail, so outside a branch the count is 0. */
RIDGEFILL_HD inline void ridgefill_hazard_step(const RidgeFillHazard* g) {
    const int32_t slot = g->meta_idx[0];
    if (slot < 0 || slot >= g->meta_states) return;
    float* m = g->meta + slot * g->meta_slot;
    const int64_t s = g->cu_last[0], e = g->cu_last[1];
    if (g->span > 0 && e > s) {
        const int64_t p0 = g->positions[s * g->pos_stride], before = g->span - (e - s);
        const int64_t counted = (int64_t)m[1], last = (int64_t)m[0];
        const int64_t lo = p0 - before > counted ? p0 - before : counted;
        const int64_t hi = p0 < last ? p0 : last;
        if (before > 0 && hi > lo) {
            g->count[0] += (float)(hi - lo);
            m[1] = (float)hi;
        }
    }
    if (g->bounds && g->bounds[1] > g->bounds[0])
        m[0] = (float)(g->positions[(int64_t)(g->bounds[1] - 1) * g->pos_stride] + 1);
}

/* One ridgefill_select pair, in BYTES (the op is dtype-agnostic). `word` is the widest load -- 16, 4
 * or 1 bytes -- that every address, pitch and row width of the pair is a multiple of; 0 = the pair
 * is absent. */
typedef struct RidgeFillCopy {
    const unsigned char* src;
    unsigned char*       dst;
    int64_t              src_pitch, dst_pitch, row_bytes;
    int                  word;
} RidgeFillCopy;

/* ridgefill_select: on every row i < n with mask[i] == 1, each present pair's destination row is
 * overwritten with its source row. */
typedef struct RidgeFillSelect {
    const int32_t* mask;    int64_t n;
    RidgeFillCopy        pair[RIDGEFILL_SELECT_PAIRS];
} RidgeFillSelect;

/* ridgefill_drop_rows: on every row i < n with mask[i] == 1, ids[i, 0 .. top_k) = -1. */
typedef struct RidgeFillDrop {
    const int32_t* mask;    int64_t n;
    int32_t*       ids;     int64_t pitch, top_k;
} RidgeFillDrop;

#ifdef __cplusplus
extern "C" {
#endif

/* host_ref.cpp: operand checks shared by both rows. RAD_OK or a named refusal. */
int ridgefill_mask_parse(const RadArgs* a, RidgeFillMask* out);
int ridgefill_rho_parse(const RadArgs* a, RidgeFillRho* out);
int ridgefill_correct_parse(const RadArgs* a, RidgeFillCorrect* out);
int ridgefill_state_read_parse(const RadArgs* a, RidgeFillStateRead* out);
int ridgefill_select_parse(const RadArgs* a, RidgeFillSelect* out);
int ridgefill_drop_parse(const RadArgs* a, RidgeFillDrop* out);
int ridgefill_hazard_parse(const RadArgs* a, RidgeFillHazard* out);

/* host_ref.cpp: the host rows (oracles). */
int ridgefill_mask_host(const RadArgs* a, RadStream s);
int ridgefill_rho_host(const RadArgs* a, RadStream s);
int ridgefill_correct_host(const RadArgs* a, RadStream s);
int ridgefill_state_read_host(const RadArgs* a, RadStream s);
int ridgefill_select_host(const RadArgs* a, RadStream s);
int ridgefill_drop_host(const RadArgs* a, RadStream s);
int ridgefill_hazard_host(const RadArgs* a, RadStream s);

/* forward.cpp: ridgefill_gemm_nt_bias's rows -- the engine's own gemm_nt_bias rows from the libraries
 * already loaded (libr4d device, libref host) -- and ridgefill_gemm_nt_q's (libr4d's int8 gemm_nt_q device
 * rows, layout hooks kept); none when no source is loaded. */
int ridgefill_forward_count(void);
const RadKernelInfo* ridgefill_forward_at(int i);
int ridgefill_forward_concurrent(int i);

/* mask.hip, rho.hip, state_correct.hip, select.hip: the device rows. */
int ridgefill_mask_device(const RadArgs* a, RadStream s);
int ridgefill_rho_device(const RadArgs* a, RadStream s);
int ridgefill_correct_device(const RadArgs* a, RadStream s);
int ridgefill_state_read_device(const RadArgs* a, RadStream s);
int ridgefill_select_device(const RadArgs* a, RadStream s);
int ridgefill_drop_device(const RadArgs* a, RadStream s);
int ridgefill_hazard_device(const RadArgs* a, RadStream s);

#ifdef __cplusplus
}
#endif
