/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* ridgefill_match.h -- does the projector folder belong to the model being served (PACKAGING.md §3,
 * REFUTATION-3 §6, Dylan's DD-K split)? Model-agnostic: it reads the manifest's `model` block and
 * answers from RadModelMeta, the builder's encoding query and the container file itself.
 *
 * TWO CLASSES OF DIFFERENCE (DD-K, binding):
 *   CANNOT RUN -- another architecture, other dimensions or layer layout (the metadata keys the
 *     manifest names), another tokenizer (the score table is indexed by token id). The folder is
 *     REFUSED by name and the engine serves stock.
 *   FITTED ON ANOTHER VARIANT -- another quantisation or recipe of the trunk (the encodings of the
 *     named late-layer weights), other base weights (sha256 of a few small named tensors), another
 *     model name. One projector serves its model's quant variants, so this is a WARNING naming
 *     both sides, and RidgeFill runs.
 *
 * The container is read through the public format (abi/rad_format.h): the string blob, the
 * directory, the planes of a few KiB of named entries, and the vocab section. The canonical vocab
 * hash covers what the token ids mean -- each token's text and type in id order, then the merge
 * table -- and not the section's bytes, which hold string-blob offsets an in-place append moves.
 * tools/ridgefill_projector.py computes the same three things the same way.
 */
#ifndef RIDGEFILL_MATCH_H
#define RIDGEFILL_MATCH_H

#include "ridgefill_folder.h"

#include <rad_encoding.h>

namespace ridgefill {

using namespace rad::arch;

static_assert(sizeof(RadFileHeader) == 248 && sizeof(RadFileEntry) == 136 &&
              sizeof(RadFilePlane) == 16 && sizeof(RadFileKV) == 24 && sizeof(RadVocabHeader) == 128,
              "tools/ridgefill_projector.py reads the container with these record sizes");

/* The container file, mapped read-only (its bytes are touched only where a check reads). */
struct Container {
    const unsigned char* base = nullptr;
    size_t size = 0;
    const RadFileHeader* h = nullptr;
};

inline bool open_container(const std::string& path, Container* c) {
    c->base = path.empty() ? nullptr : map_file(path, &c->size);
    if (!c->base || c->size < sizeof(RadFileHeader)) return false;
    c->h = (const RadFileHeader*)c->base;
    const RadFileHeader& h = *c->h;
    const auto fits = [&](uint64_t off, uint64_t n, uint64_t each) {
        return off <= c->size && n <= (c->size - off) / (each ? each : 1);
    };
    return h.magic == RAD_MAGIC && h.version == RAD_FORMAT_VER && fits(h.str_off, h.str_bytes, 1) &&
           h.str_bytes > 0 && c->base[h.str_off + h.str_bytes - 1] == 0 &&
           fits(h.dir_off, h.dir_count, sizeof(RadFileEntry)) &&
           fits(h.plane_off, h.plane_count, sizeof(RadFilePlane)) &&
           fits(h.vocab_off, h.vocab_bytes, 1) && h.vocab_bytes >= sizeof(RadVocabHeader);
}

/* A string of the blob; "" when the offset is outside it. */
inline const char* blob_str(const Container& c, rad_stroff o) {
    return o < c.h->str_bytes ? (const char*)c.base + c.h->str_off + o : "";
}

/* sha256 of the named entry's planes, in order; empty when the entry is not in the file. */
inline std::string entry_sha256(const Container& c, const std::string& name) {
    const RadFileEntry* dir = (const RadFileEntry*)(c.base + c.h->dir_off);
    for (uint64_t i = 0; i < c.h->dir_count; ++i) {
        if (name != blob_str(c, dir[i].name)) continue;
        const RadFilePlane* pl = (const RadFilePlane*)(c.base + c.h->plane_off);
        std::string all;
        for (uint64_t k = dir[i].plane_first; k < dir[i].plane_first + dir[i].n_planes; ++k) {
            if (k >= c.h->plane_count || pl[k].offset > c.size || pl[k].bytes > c.size - pl[k].offset)
                return std::string();
            all.append((const char*)c.base + pl[k].offset, (size_t)pl[k].bytes);
        }
        return sha256_bytes((const unsigned char*)all.data(), all.size());
    }
    return std::string();
}

/* What the token ids mean: every token's text, a NUL and its type byte, in id order, then the merge
 * table's id pairs as stored. Empty when the section does not hold together. */
inline std::string vocab_sha256(const Container& c) {
    const RadVocabHeader& v = *(const RadVocabHeader*)(c.base + c.h->vocab_off);
    const auto inside = [&](uint64_t off, uint64_t bytes) {
        return off >= c.h->vocab_off && off - c.h->vocab_off <= c.h->vocab_bytes &&
               bytes <= c.h->vocab_bytes - (off - c.h->vocab_off);
    };
    if (!inside(v.tok_text_off, (uint64_t)v.n_tokens * 8) || !inside(v.tok_type_off, v.n_tokens) ||
        !inside(v.merge_off, (uint64_t)v.n_merges * 8))
        return std::string();
    const rad_stroff* text = (const rad_stroff*)(c.base + v.tok_text_off);
    std::string all;
    for (uint32_t i = 0; i < v.n_tokens; ++i) {
        all += blob_str(c, text[i]);
        all += '\0';
        all += (char)c.base[v.tok_type_off + i];
    }
    all.append((const char*)c.base + v.merge_off, (size_t)v.n_merges * 8);
    return sha256_bytes((const unsigned char*)all.data(), all.size());
}

struct Match {
    std::string refused;                  /* why the folder cannot run here; empty = it can */
    std::vector<std::string> warnings;    /* fitted on another variant: named, and RidgeFill runs */
    std::string summary;                  /* the five checks' results, for the log */
};

/* Check 2: every metadata key the manifest names has the same value here. */
inline void match_meta(const Json& model, const RadModelMeta* meta, Match* m, int* ok, int* n) {
    const Json* keys = model.get("meta");
    for (size_t i = 0; keys && keys->kind == Json::OBJ && i < keys->obj.size(); ++i, ++*n) {
        const auto& [key, want] = keys->obj[i];
        const char* have = rad_meta_gets(meta, key.c_str(), nullptr);
        if (have && want.str == have) { ++*ok; continue; }
        if (m->refused.empty())
            m->refused = "metadata '" + key + "' is '" + (have ? have : "(absent)") +
                         "' in this model and '" + want.str + "' in the projector's";
    }
}

/* Check 4: the named trunk weights' encodings (another quant or recipe = a warning). */
inline void match_encodings(const Json& model, RadBuilder* b, Match* m, int* ok, int* n) {
    const Json* enc = model.get("encodings");
    for (size_t i = 0; enc && enc->kind == Json::OBJ && i < enc->obj.size(); ++i, ++*n) {
        const auto& [name, want] = enc->obj[i];
        RadEncoding e{};
        char have[160] = "(absent)";
        if (rad_weight_encoding(b, name.c_str(), &e, nullptr, nullptr) == RAD_OK)
            rad_enc_format(&e, have, sizeof have);
        if (want.str == have) { ++*ok; continue; }
        m->warnings.push_back("weight " + name + " is " + have + " here; the projector was fitted on " + want.str);
    }
}

/* Check 5: the content anchors (other base weights = a warning). */
inline void match_anchors(const Json& model, const Container& c, Match* m, int* ok, int* n) {
    const Json* anchors = model.get("anchors");
    for (size_t i = 0; anchors && anchors->kind == Json::OBJ && i < anchors->obj.size(); ++i, ++*n) {
        const auto& [name, want] = anchors->obj[i];
        const std::string have = c.base ? entry_sha256(c, name) : std::string();
        if (have == want.str) { ++*ok; continue; }
        m->warnings.push_back("tensor " + name + " hashes " + (have.empty() ? "(unreadable)" : have) +
                              " here and " + want.str + " in the model the projector was fitted on");
    }
}

inline Match match_model(const Folder& f, const RadModelMeta* meta, RadBuilder* b, const char* adapter) {
    Match m;
    const Json* model = f.manifest.get("model");
    static const Json kEmpty;
    const Json& mo = model ? *model : kEmpty;
    /* 1. the adapter and the architecture */
    if (f.manifest.text("adapter") != adapter || mo.text("arch_id") != (meta->arch_id ? meta->arch_id : ""))
        m.refused = "the projector is for adapter '" + f.manifest.text("adapter") + "' / arch '" +
                    mo.text("arch_id") + "' and this plugin serves '" + adapter + "' / '" +
                    (meta->arch_id ? meta->arch_id : "") + "'";
    int meta_ok = 0, meta_n = 0, enc_ok = 0, enc_n = 0, anc_ok = 0, anc_n = 0;
    match_meta(mo, meta, &m, &meta_ok, &meta_n);
    /* 3. the tokenizer */
    Container c;
    const bool readable = open_container(f.place.container, &c);
    const std::string vocab = readable ? vocab_sha256(c) : std::string();
    const bool vocab_ok = !vocab.empty() && vocab == mo.text("vocab_sha256");
    if (!vocab_ok && m.refused.empty())
        m.refused = readable ? "the tokenizer differs (vocab sha256 " + vocab + " here, " +
                                   mo.text("vocab_sha256") + " in the projector's model)"
                             : "the model file is not a readable container, so its tokenizer cannot be checked";
    match_encodings(mo, b, &m, &enc_ok, &enc_n);
    match_anchors(mo, c, &m, &anc_ok, &anc_n);
    const std::string name = meta->name ? meta->name : "";
    if (mo.text("name") != name)
        m.warnings.push_back("this model is named '" + name + "' and the projector was fitted on '" +
                             mo.text("name") + "'");
    if (c.base) munmap((void*)c.base, c.size);
    char line[320];
    std::snprintf(line, sizeof line, "arch %s, metadata %d/%d, tokenizer %s, encodings %d/%d, anchors %d/%d",
                  m.refused.rfind("the projector is for", 0) == 0 ? "DIFFERS" : "ok", meta_ok, meta_n,
                  vocab_ok ? "ok" : "DIFFERS", enc_ok, enc_n, anc_ok, anc_n);
    m.summary = line;
    return m;
}

}  /* namespace ridgefill */

#endif /* RIDGEFILL_MATCH_H */
