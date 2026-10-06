/* ridgefill_folder.h -- the projector folder: where it is, and what it holds (PACKAGING.md §0, §2, §3;
 * REFUTATION-3 §1). Model-agnostic: no model type appears here.
 *
 * WHERE. The user runs the stock model file and puts the projector in `<model dir>/projector/`. No
 * ABI call names the model's path, so it is read from the process (REFUTATION-3 §1): the engine
 * maps the container before any plugin loads (core/format/radfile.cpp:102-116, engine_bringup.cpp
 * :2467 then :2472), so /proc/self/maps names its RESOLVED file -- the mapping whose bytes start
 * with the container magic (abi/rad_format.h:28) -- and /proc/self/cmdline holds `--model PATH` as
 * typed (the one spelling, core/config.cpp:32, 338), accepted only when it is the same file
 * (device and inode). Order: $RADIANCE_RIDGEFILL_PROJECTOR, else projector/ beside the typed path (a
 * Hugging Face cache's snapshots/<rev>/model.rad symlink keeps its own directory), else beside the
 * resolved file. Anything that fails is "no projector", never an error.
 *
 * WHAT. ridgefill.json (format 1) lists every file with its sha256; each .safetensors file's tensors are
 * mapped read-only and named by their own names (proj.24.weight, st.24, score, ...). A partial
 * download, a corrupt file or a duplicate tensor name refuses the folder by name. The hashes run
 * one thread a file, once per process (~1.3 GiB: about a second, mostly the first read).
 */
#ifndef RIDGEFILL_FOLDER_H
#define RIDGEFILL_FOLDER_H

#include "ridgefill_guard.h"   /* sha256_block, real_path */
#include "ridgefill_json.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <rad_format.h>

#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace ridgefill {

using namespace rad::arch;

/* ---------------------------------------------------------------- where it is */

struct FolderPlace {
    std::string dir;        /* the folder, empty when none was found */
    std::string how;        /* the rule that found it, or every place looked at */
    std::string container;  /* the resolved model file, empty when none is mapped */
};

/* The first mapped file that starts with the container magic, or empty. Opens each distinct mapped
 * file once to read 4 bytes -- a few hundred opens, once per process at the first declare. */
inline std::string mapped_container() {
    std::ifstream maps("/proc/self/maps");
    std::string line, last;
    while (std::getline(maps, line)) {
        const size_t slash = line.find('/');
        if (slash == std::string::npos) continue;
        const std::string path = line.substr(slash);
        if (path == last) continue;
        last = path;
        uint32_t magic = 0;
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) continue;
        const bool read = std::fread(&magic, 4, 1, f) == 1;
        std::fclose(f);
        if (read && magic == RAD_MAGIC) return path;
    }
    return std::string();
}

/* The argument after `--model` on the command line, as typed; empty when there is none. A plugin is
 * handed no argv, so it reads the process's own. */
inline std::string typed_model() {
    std::ifstream f("/proc/self/cmdline", std::ios::binary);
    const std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<std::string> argv;
    for (size_t at = 0; at < all.size();) {
        const size_t end = all.find('\0', at);
        argv.push_back(all.substr(at, end == std::string::npos ? std::string::npos : end - at));
        if (end == std::string::npos) break;
        at = end + 1;
    }
    for (size_t i = 0; i + 1 < argv.size(); ++i)
        if (argv[i] == "--model") return argv[i + 1];
    return std::string();
}

/* By device and inode, not by name: the typed path may be a symlink (a Hugging Face snapshot) or a bind
 * mount of the file the engine resolved and mapped. */
inline bool same_file(const std::string& a, const std::string& b) {
    struct stat x {}, y {};
    return stat(a.c_str(), &x) == 0 && stat(b.c_str(), &y) == 0 && x.st_dev == y.st_dev &&
           x.st_ino == y.st_ino;
}

inline std::string dir_of(const std::string& p) {
    const size_t slash = p.rfind('/');
    return slash == std::string::npos ? "." : slash == 0 ? "/" : p.substr(0, slash);
}

inline bool has_manifest(const std::string& dir) {
    struct stat st {};
    return stat((dir + "/ridgefill.json").c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

inline FolderPlace find_folder() {
    FolderPlace f;
    f.container = mapped_container();
    if (const char* e = std::getenv("RADIANCE_RIDGEFILL_PROJECTOR"); e && *e) {
        f.how = std::string("$RADIANCE_RIDGEFILL_PROJECTOR=") + e;
        if (has_manifest(e)) f.dir = e;
        else f.how += " (no ridgefill.json there)";
        return f;
    }
    const std::string typed = typed_model();
    if (!typed.empty() && !f.container.empty() && same_file(typed, f.container)) {
        const std::string cand = dir_of(typed) + "/projector";
        f.how = "beside --model " + typed + " (same device and inode as the mapped " + f.container + ")";
        if (has_manifest(cand)) { f.dir = cand; return f; }
        f.how = cand;
    } else if (!typed.empty()) {
        f.how = "--model " + typed + " is not the mapped model file";
    }
    if (!f.container.empty()) {
        const std::string cand = dir_of(f.container) + "/projector";
        if (has_manifest(cand)) {
            f.dir = cand;
            f.how = "beside the resolved model file " + f.container;
            return f;
        }
        if (f.how != cand) f.how += (f.how.empty() ? "" : ", ") + cand;   /* typed dir == resolved dir */
    }
    if (f.container.empty()) f.how += (f.how.empty() ? "" : "; ") + std::string("no model file is mapped");
    return f;
}

/* ---------------------------------------------------------------- what it holds */

struct FolderTensor {
    const unsigned char* data = nullptr;   /* into the file's read-only mapping */
    uint32_t             dtype = 0;
    std::vector<int64_t> shape;
    int64_t              bytes = 0;
};

/* The folder as read: tensors point into the files' read-only mappings and are never copied here --
 * the one copy is each rank's upload (ridgefill_projector.h). */
struct Folder {
    FolderPlace place;
    Json        manifest;
    std::map<std::string, FolderTensor> tensors;
    int64_t     file_bytes = 0;
};

inline std::string hex_digest(const uint32_t h[8]) {
    char hex[65];
    for (int i = 0; i < 8; ++i) std::snprintf(hex + 8 * i, 9, "%08x", h[i]);
    return hex;
}

/* SHA-256 of n bytes in memory (FIPS 180-4; ridgefill_guard.h's block function). */
inline std::string sha256_bytes(const unsigned char* p, size_t n) {
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    size_t at = 0;
    for (; at + 64 <= n; at += 64) sha256_block(h, p + at);
    unsigned char tail[128] = {};
    const size_t rest = n - at;
    if (rest) std::memcpy(tail, p + at, rest);
    tail[rest] = 0x80;
    const size_t len = rest < 56 ? 64 : 128;
    for (int i = 0; i < 8; ++i) tail[len - 1 - i] = (unsigned char)(((uint64_t)n * 8) >> (8 * i));
    sha256_block(h, tail);
    if (len == 128) sha256_block(h, tail + 64);
    return hex_digest(h);
}

/* The whole of `path` mapped read-only, or null. The mapping lives as long as the process: the
 * tensors point into it, and it is page cache, not anonymous memory. */
inline const unsigned char* map_file(const std::string& path, size_t* size) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return nullptr;
    struct stat st {};
    void* p = MAP_FAILED;
    if (fstat(fd, &st) == 0 && st.st_size > 0)
        p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return nullptr;
    *size = (size_t)st.st_size;
    return (const unsigned char*)p;
}

/* The dtypes tools/ridgefill_projector.py writes; any other tensor refuses the folder (RAD_DT_INVALID). */
inline uint32_t safetensors_dtype(const std::string& s) {
    if (s == "BF16") return RAD_BF16;
    if (s == "F32")  return RAD_F32;
    if (s == "F16")  return RAD_F16;
    if (s == "I8")   return RAD_I8;
    if (s == "I32")  return RAD_I32;
    return RAD_DT_INVALID;
}

/* One safetensors file's tensors into f->tensors; false and *why on any malformation. Each tensor's byte
 * span must equal its shape times its dtype and lie inside the file: a truncated or hand-edited file is
 * refused here, before a GEMM could read past a mapping. */
inline bool read_safetensors(const std::string& name, const unsigned char* p, size_t size,
                             Folder* f, std::string* why) {
    uint64_t hlen = 0;
    if (size < 8) { *why = name + " is shorter than a safetensors header"; return false; }
    std::memcpy(&hlen, p, 8);
    Json h;
    if (hlen > size - 8 || !json_parse((const char*)p + 8, (size_t)hlen, &h) || h.kind != Json::OBJ) {
        *why = name + "'s safetensors header does not parse";
        return false;
    }
    for (const auto& [tname, t] : h.obj) {
        if (tname == "__metadata__") continue;
        FolderTensor x;
        x.dtype = safetensors_dtype(t.text("dtype"));
        const Json* shape = t.get("shape");
        const Json* off = t.get("data_offsets");
        int64_t numel = 1;
        for (size_t i = 0; shape && shape->kind == Json::ARR && i < shape->arr.size(); ++i) {
            x.shape.push_back((int64_t)shape->arr[i].num);
            numel *= x.shape.back();
        }
        const bool offsets = off && off->kind == Json::ARR && off->arr.size() == 2;
        const uint64_t lo = offsets ? (uint64_t)off->arr[0].num : 1, hi = offsets ? (uint64_t)off->arr[1].num : 0;
        x.bytes = x.dtype ? rad_dtype_bytes(x.dtype, numel) : -1;
        if (!x.dtype || !shape || lo > hi || hi > size - 8 - hlen || (int64_t)(hi - lo) != x.bytes) {
            *why = name + ": tensor '" + tname + "' has an unknown dtype or its bytes do not match its shape";
            return false;
        }
        x.data = p + 8 + hlen + lo;
        if (!f->tensors.emplace(tname, x).second) {
            *why = "tensor '" + tname + "' is in two files";
            return false;
        }
    }
    return true;
}

/* Reads and verifies the folder at f->place.dir: the manifest, every file it lists (sha256), and
 * the tensors of the safetensors files. False and *why (naming the file) on any failure. */
inline bool read_folder(Folder* f, std::string* why) {
    const std::string& dir = f->place.dir;
    std::ifstream in(dir + "/ridgefill.json", std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (!json_parse(text.data(), text.size(), &f->manifest) || f->manifest.integer("format", 0) != 1) {
        *why = dir + "/ridgefill.json does not parse as a format-1 projector manifest";
        return false;
    }
    const Json* files = f->manifest.get("files");
    if (!files || files->kind != Json::OBJ || files->obj.empty()) {
        *why = dir + "/ridgefill.json lists no files";
        return false;
    }
    const size_t n = files->obj.size();
    std::vector<const unsigned char*> data(n, nullptr);
    std::vector<size_t> size(n, 0);
    std::vector<std::string> got(n);
    for (size_t i = 0; i < n; ++i) {
        const std::string& name = files->obj[i].first;
        /* a bare file name only: a manifest must not reach outside its folder */
        if (name.find('/') != std::string::npos || !(data[i] = map_file(dir + "/" + name, &size[i]))) {
            *why = "the manifest lists " + name + ", which is missing or unreadable in " + dir;
            return false;
        }
        f->file_bytes += (int64_t)size[i];
    }
    /* One thread a file: the hashes are read-bound on a cold page cache, and ~30 files hash in about the
     * time of the largest. */
    std::vector<std::thread> pool;
    for (size_t i = 0; i < n; ++i)
        pool.emplace_back([&, i] { got[i] = sha256_bytes(data[i], size[i]); });
    for (std::thread& t : pool) t.join();
    for (size_t i = 0; i < n; ++i) {
        const std::string& name = files->obj[i].first;
        if (got[i] != files->obj[i].second.str) {
            *why = name + " is corrupt or not the file the manifest names (sha256 " + got[i] +
                   ", the manifest says " + files->obj[i].second.str + ")";
            return false;
        }
        const bool st = name.size() > 12 && name.compare(name.size() - 12, 12, ".safetensors") == 0;
        if (st && !read_safetensors(name, data[i], size[i], f, why)) return false;
    }
    return true;
}

}  /* namespace ridgefill */

#endif /* RIDGEFILL_FOLDER_H */
