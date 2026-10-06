/* ridgefill_guard.h -- the engine-release guard (PLAN-FIX §6.6, DD-E, R83).
 *
 * WHY. An adapter compiles its in-tree architecture source of ONE radiance release and shadows the
 * installed file of that name (its `so` stem), so an engine upgraded under it would run last release's
 * model against this release's core -- and the RidgeFill paths lean on core behaviour no ABI number covers (the
 * stager's release rule, stager.cpp:143-144). No ABI call says which release the engine is, so at
 * rad_plugin_open the plugin finds the object that defines rad_issue (dladdr) and requires exactly
 * one NUL-delimited copy of the release string it was built against (RAD_VERSION is compiled into
 * the core as one such string, core/CMakeLists.txt:25). On a mismatch it FORWARDS every export to
 * the engine's own in-tree <so>, found on $RADIANCE_HOME after this plugin's own home:
 * the engine then serves its current architecture with RidgeFill off, which is stock -- loud, never
 * silent. With no in-tree file to forward to (e.g. the home was given only as --radiance-home,
 * which a plugin cannot see) the plugin DECLINES, and startup fails by name for want of a claimant
 * for the architecture (loader.cpp:425-436). Residual risk, named: a patched build that keeps the release
 * string; the engine's sha256 is logged so such a build is identifiable.
 */
#ifndef RIDGEFILL_GUARD_H
#define RIDGEFILL_GUARD_H

#include "ridgefill_log.h"

#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef RIDGEFILL_RADIANCE_VERSION
#error "RIDGEFILL_RADIANCE_VERSION (the radiance release this plugin is built against) is not defined; arch/CMakeLists.txt sets it from RADIANCE_SRC"
#endif

namespace ridgefill {

using namespace rad::arch;

/* How many times "\0<version>\0" occurs in the file, or -1 when it cannot be read. Streams the
 * file in 1 MiB blocks, keeping the pattern's length minus one bytes across each boundary. The guard
 * wants exactly one: zero is another release, and two would be a build that carries a second release
 * string for some other reason, which it will not guess about. Cost: one read of the engine object at
 * open (tens of MiB, page cache after the engine's own load). */
inline int count_version(const char* path, const char* version) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return -1;
    std::string want(1, '\0');
    want += version;
    want += '\0';
    std::string buf;
    std::vector<char> block(1 << 20);
    int hits = 0;
    for (size_t n; (n = std::fread(block.data(), 1, block.size(), f)) > 0;) {
        buf.append(block.data(), n);
        for (size_t at = buf.find(want); at != std::string::npos; at = buf.find(want, at + 1)) ++hits;
        buf.erase(0, buf.size() > want.size() - 1 ? buf.size() - (want.size() - 1) : 0);
    }
    std::fclose(f);
    return hits;
}

/* Every NUL-delimited "d.d.d" string in the file, comma-separated -- the releases it carries, for the
 * log (the same strings CMake's release check reads, CMakeLists.txt). */
inline std::string releases_in(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return "unreadable";
    std::string out, run;
    auto flush = [&] {
        int dots = 0;
        bool ok = !run.empty() && std::isdigit((unsigned char)run.front()) && std::isdigit((unsigned char)run.back());
        for (size_t i = 0; ok && i < run.size(); ++i) {
            if (run[i] == '.') ok = ++dots <= 2 && run[i - 1] != '.';
            else ok = std::isdigit((unsigned char)run[i]) != 0;
        }
        if (ok && dots == 2 && out.find(run) == std::string::npos) out += (out.empty() ? "" : ", ") + run;
        run.clear();
    };
    for (int c; (c = std::fgetc(f)) != EOF;) {
        if (c == 0) flush();
        else if (run.size() < 16) run += (char)c;
        else run = "x";
    }
    std::fclose(f);
    return out.empty() ? "none" : out;
}

/* ---------------------------------------------------------------- SHA-256 (FIPS 180-4) */
/* Its own, because the core may lean on nothing the engine image might not ship: the guard logs the
 * engine's hash so a patched build is identifiable, and ridgefill_folder.h checks every projector file. */

inline void sha256_block(uint32_t h[8], const unsigned char* p) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    uint32_t w[64], v[8];
    for (int i = 0; i < 16; ++i)
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; ++i)
        w[i] = w[i - 16] + (rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] +
               (rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10));
    std::memcpy(v, h, sizeof v);
    for (int i = 0; i < 64; ++i) {
        const uint32_t t1 = v[7] + (rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25)) +
                            ((v[4] & v[5]) ^ (~v[4] & v[6])) + K[i] + w[i];
        const uint32_t t2 = (rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22)) +
                            ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
        std::memmove(v + 1, v, 7 * sizeof v[0]);
        v[4] += t1;
        v[0] = t1 + t2;
    }
    for (int i = 0; i < 8; ++i) h[i] += v[i];
}

/* The file's SHA-256 in hex, or "unreadable". */
inline std::string sha256_file(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return "unreadable";
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    unsigned char blk[64];
    uint64_t total = 0;
    size_t n;
    while ((n = std::fread(blk, 1, 64, f)) == 64) { sha256_block(h, blk); total += 64; }
    std::fclose(f);
    total += n;
    unsigned char tail[128] = {};
    std::memcpy(tail, blk, n);
    tail[n] = 0x80;
    const size_t len = n < 56 ? 64 : 128;
    for (int i = 0; i < 8; ++i) tail[len - 1 - i] = (unsigned char)((total * 8) >> (8 * i));
    sha256_block(h, tail);
    if (len == 128) sha256_block(h, tail + 64);
    char hex[65];
    for (int i = 0; i < 8; ++i) std::snprintf(hex + 8 * i, 9, "%08x", h[i]);
    return hex;
}

/* ---------------------------------------------------------------- finding the files */

inline std::string real_path(const std::string& p) {
    char out[PATH_MAX];
    return realpath(p.c_str(), out) ? std::string(out) : std::string();
}

/* The object that defines rad_issue -- the engine binary, or the library the core lives in. The
 * main program's name as dladdr reports it is argv[0], which may be bare; /proc/self/exe is it. */
inline std::string engine_object() {
    Dl_info i{};
    if (!dladdr((void*)&rad_issue, &i) || !i.dli_fname || !std::strchr(i.dli_fname, '/')) {
        const std::string exe = real_path("/proc/self/exe");
        return exe.empty() ? std::string("/proc/self/exe") : exe;
    }
    return i.dli_fname;
}


/* The in-tree architecture this plugin shadows: architectures/<so> in a $RADIANCE_HOME entry (split
 * as the engine splits it, startup.cpp:96-108) that is not this file itself. `so` is the adapter's
 * stem (it shadows that file by name, arch/CMakeLists.txt), so the core names no model. */
inline std::string find_shadowed(const std::string& self, const char* so) {
    const char* home = std::getenv("RADIANCE_HOME");
    const std::string h = home ? home : "";
    for (size_t at = 0; at <= h.size();) {
        size_t end = h.find(':', at);
        if (end == std::string::npos) end = h.size();
        const std::string cand = real_path(h.substr(at, end - at) + "/architectures/" + so);
        if (!cand.empty() && cand != self) return cand;
        at = end + 1;
    }
    return std::string();
}

/* ---------------------------------------------------------------- the forward */

struct Forward {
    int  (*probe)(const RadModelMeta*, RadArchProbe*) = nullptr;
    int  (*declare)(RadBuilder*, const RadModelMeta*, const RadBuildCtx*) = nullptr;
    void (*step)(RadCtx*, const RadBatch*) = nullptr;
};
static Forward g_forward;   /* set once at open on a mismatch; null = this plugin serves */

/* Loads the shadowed file and takes its exports; false (and the reason on stderr) if it cannot. */
inline bool take_forward(const std::string& path) {
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::fprintf(stderr, "radiance: %s: cannot load %s: %s\n", g_log_name, path.c_str(), dlerror());
        return false;
    }
    Forward f;
    f.probe   = (decltype(f.probe))dlsym(h, "rad_arch_probe");
    f.declare = (decltype(f.declare))dlsym(h, "rad_arch_declare");
    f.step    = (decltype(f.step))dlsym(h, "rad_arch_step");
    if (!f.declare || !f.step) {
        std::fprintf(stderr, "radiance: %s: %s exports no rad_arch_declare/step\n", g_log_name, path.c_str());
        return false;
    }
    g_forward = f;
    return true;
}

/* rad_plugin_open's body: RAD_OK to serve (as RidgeFill, or forwarded to the in-tree `so`), RAD_E_UNSUPPORTED
 * to decline. */
inline int open_guard(const char* so) {
    const std::string engine = engine_object();
    const int hits = count_version(engine.c_str(), RIDGEFILL_RADIANCE_VERSION);
    if (hits == 1) return RAD_OK;
    Dl_info me{};
    const std::string self = dladdr((void*)&open_guard, &me) && me.dli_fname
                                 ? real_path(me.dli_fname) : std::string();
    const std::string sha = sha256_file(engine.c_str()), found = releases_in(engine.c_str());
    const std::string shadow = find_shadowed(self, so);
    if (!shadow.empty() && take_forward(shadow)) {
        std::fprintf(stderr, "radiance: %s: WARNING: built against radiance %s, and the "
                             "engine %s (sha256 %s) carries release string(s) '%s' (%s %d times); "
                             "forwarding to the engine's own architecture %s, RidgeFill off\n", g_log_name,
                     RIDGEFILL_RADIANCE_VERSION, engine.c_str(), sha.c_str(), found.c_str(),
                     RIDGEFILL_RADIANCE_VERSION, hits, shadow.c_str());
        return RAD_OK;
    }
    std::fprintf(stderr, "radiance: %s: built against radiance %s, and the engine %s "
                         "(sha256 %s) carries release string(s) '%s' (%s %d times); no in-tree "
                         "architectures/%s on $RADIANCE_HOME to forward to, so this "
                         "plugin declines (a home given only as --radiance-home is not visible to "
                         "it)\n", g_log_name, RIDGEFILL_RADIANCE_VERSION, engine.c_str(), sha.c_str(), found.c_str(),
                 RIDGEFILL_RADIANCE_VERSION, hits, so);
    return RAD_E_UNSUPPORTED;
}

}  /* namespace ridgefill */

#endif /* RIDGEFILL_GUARD_H */
