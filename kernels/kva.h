/* kva.h -- what the three KVA ops share between the row table, the host rows and the device rows.
 *
 * The plugin is one .so with a host row (the oracle, plain C++) and a device row per op. The
 * operand parsing is host code shared by both rows, so a device row can never accept an operand
 * its oracle would refuse. The few formulas that DEFINE an op's answer -- the row hash of the
 * random selector, the rho clamp, softplus, the bf16 decode -- are written once here and compiled
 * for both sides, so "host == device" is a statement about the kernels and not about two spellings
 * of one definition.
 */
#pragma once

#include <rad_abi.h>
#include <rad_plugin.h>

#include <math.h>
#include <stdint.h>

#if defined(__HIPCC__)
#define KVA_HD __host__ __device__
#else
#define KVA_HD
#endif

/* The device selector keeps one 4-byte key per row in LDS: this many rows is 32 KiB of the 64 KiB
 * a dispatch may use (docs/PLUGIN.md, "What a launch may do"). It is also the engine's default
 * --max-num-batched-tokens, so the device row serves every chunk the scheduler cuts by default;
 * a larger step falls to no device row for that band and is refused at issue, by name. */
enum { KVA_ROWSEL_MAX_ROWS = 8192, KVA_ROWSEL_THREADS = 256 };

/* softplus' large-x cutoff: above it log1p(e^x) is x to every bit f32 holds. The value libref and
 * libr4d default `softplus_thr` to, and torch.nn.functional.softplus's threshold, which is what the
 * reference (kva b0/worker_ext.py log_gate) used. */
#define KVA_SOFTPLUS_THRESHOLD 20.0f

enum { KVA_MODE_CLASS = 0, KVA_MODE_RANDOM = 1, KVA_MODE_ALL = 2 };

/* Operand positions, in schema order (rows.cpp holds the schemas). */
enum { RS_TOKENS = 0, RS_POS, RS_SCORE, RS_ROWS, RS_MASK };
enum { RH_A = 0, RH_MASK, RH_ALOG, RH_DTBIAS, RH_ND, RH_SIDX };
enum { SC_STATE = 0, SC_STATE_IDX, SC_APPLIED, SC_APPLIED_IDX, SC_C, SC_ND, SC_ND_IDX };

/* ---------------------------------------------------------------- the shared definitions */

/* Wellons' lowbias32 finaliser: a bijection on 32 bits with good avalanche, so distinct rows get
 * independent-looking ranks. */
KVA_HD inline uint32_t kva_mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/* The random selector's rank key for the token at absolute prompt position `position` under
 * `seed`: the k rows with the smallest key are kept, ties by lower row. Keyed on the ABSOLUTE
 * position, not the row in the chunk, so successive chunks of one prompt draw different offsets
 * and the control is not periodic in the chunk length. */
KVA_HD inline uint32_t kva_row_hash(long long seed, uint32_t position) {
    const unsigned long long s = (unsigned long long)seed;
    const uint32_t mixed_seed = kva_mix32((uint32_t)s ^ kva_mix32((uint32_t)(s >> 32)));
    return kva_mix32(mixed_seed ^ (position * 0x9E3779B9U + 0x7F4A7C15U));
}

/* rho = clamp(N / D, 0, 1), and 1 while nothing has been accumulated (D == 0): the correction is
 * then the plain speed-mode one. Spelled with comparisons rather than fminf/fmaxf so a NaN ratio
 * lands on 0 identically on both sides. */
KVA_HD inline float kva_rho(float n, float d) {
    if (!(d > 0.0f)) return 1.0f;
    float r = n / d;
    r = r > 0.0f ? r : 0.0f;
    return r < 1.0f ? r : 1.0f;
}

/* softplus in the two-sided form libref and libr4d use (the GDN gate's own spelling). */
KVA_HD inline float kva_softplus(float x) {
    if (x > KVA_SOFTPLUS_THRESHOLD) return x;
    return x > 0.0f ? x + log1pf(expf(-x)) : log1pf(expf(x));
}

KVA_HD inline float kva_bf16_to_f32(uint16_t h) {
    const uint32_t bits = (uint32_t)h << 16;
    float f;
    __builtin_memcpy(&f, &bits, sizeof f);
    return f;
}

/* ---------------------------------------------------------------- parsed operands
 *
 * Each op's operands, checked and reduced to pointers, extents and element strides. Passed BY
 * VALUE as the device kernel's argument, so every field is a plain scalar or pointer (all three
 * structs are well under the 768-byte argument limit). */

typedef struct KvaRowsel {
    const int32_t* tokens;  int64_t n;       /* [n] token ids of the chunk */
    const int32_t* positions; int64_t pos_stride;  /* token t's absolute position: positions[t * pos_stride] */
    const float*   score;   int64_t vocab;   /* [vocab] per-id score; non-finite = not a match */
    int32_t*       rows;    int64_t cap;     /* [cap] selected rows, ascending, -1 padded */
    int32_t*       mask;                     /* [n] 1 = approximated (not selected) */
    double         share;
    long long      seed;
    int            mode;                     /* KVA_MODE_* */
} KvaRowsel;

typedef struct KvaRho {
    const void*    a;       int64_t a_pitch, a_col; int a_bf16;   /* [n, n_head], any strides */
    const int32_t* mask;
    const float*   a_log;
    const float*   dt_bias;
    float*         nd;      int64_t nd_slot, nd_head, nd_inner, n_states;
    const int32_t* state_idx;
    int64_t        n, n_head;
} KvaRho;

/* Each KV operand comes with ITS OWN group's slot index: the KV manager hands every stateful group
 * the same slot per sequence today, but that is not a contract, so nothing here relies on it. */
typedef struct KvaCorrect {
    float*         state;   int64_t st_slot, st_head, st_row, st_col;
    float*         applied; int64_t ap_slot, ap_head;
    const float*   c;       int64_t c_head, c_row, c_col;
    const float*   nd;      int64_t nd_slot, nd_head, nd_inner;   /* nd null: rho = 1 */
    const int32_t* st_idx;  const int32_t* ap_idx;  const int32_t* nd_idx;
    int64_t        st_pitch, ap_pitch, nd_pitch;      /* row pitch of each index; column 0 is used */
    int64_t        st_states, ap_states, nd_states;   /* each pool's slot count */
    int64_t        n_seq, n_head, sd0, sd1;
    float          alpha;
    int            apply;                    /* 1 apply, 0 undo */
} KvaCorrect;

/* Sequence s's slot in each pool. False when any of them is outside its pool: the sequence is
 * skipped (a negative slot is how a sequence with no state is spelled). */
KVA_HD inline bool kva_correct_slots(const KvaCorrect* g, int64_t s, int64_t* st, int64_t* ap,
                                     int64_t* nd) {
    *st = g->st_idx[s * g->st_pitch];
    *ap = g->ap_idx[s * g->ap_pitch];
    *nd = g->nd ? g->nd_idx[s * g->nd_pitch] : 0;
    return *st >= 0 && *st < g->st_states && *ap >= 0 && *ap < g->ap_states &&
           (!g->nd || (*nd >= 0 && *nd < g->nd_states));
}

#ifdef __cplusplus
extern "C" {
#endif

/* host_ref.cpp: operand checks shared by both rows. RAD_OK or a named refusal. */
int kva_rowsel_parse(const RadArgs* a, KvaRowsel* out);
int kva_rho_parse(const RadArgs* a, KvaRho* out);
int kva_correct_parse(const RadArgs* a, KvaCorrect* out);

/* host_ref.cpp: the host rows (oracles). */
int kva_rowsel_host(const RadArgs* a, RadStream s);
int kva_rho_host(const RadArgs* a, RadStream s);
int kva_correct_host(const RadArgs* a, RadStream s);

/* rowsel.hip, rho.hip, state_correct.hip: the device rows. */
int kva_rowsel_device(const RadArgs* a, RadStream s);
int kva_rho_device(const RadArgs* a, RadStream s);
int kva_correct_device(const RadArgs* a, RadStream s);

#ifdef __cplusplus
}
#endif
