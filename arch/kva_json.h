/* kva_json.h -- the JSON the projector folder carries: its manifest (kva.json) and each
 * safetensors file's header. A reader, nothing more: objects, arrays, strings (with \u escapes),
 * numbers, true/false/null. Malformed text is refused, never guessed at -- a folder whose manifest
 * does not parse is "no projector" (PACKAGING.md §0).
 */
#ifndef QWEN4EXP_KVA_JSON_H
#define QWEN4EXP_KVA_JSON_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace kva {

using namespace rad::arch;

struct Json {
    enum Kind { NUL, BOOL, NUM, STR, ARR, OBJ };
    Kind kind = NUL;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    /* The member `key` of an object -- the last, when the text repeats a key, as Python's json
     * reads it -- or null (also when this is not an object). */
    const Json* get(const char* key) const {
        const Json* found = nullptr;
        for (size_t i = 0; kind == OBJ && i < obj.size(); ++i)
            if (obj[i].first == key) found = &obj[i].second;
        return found;
    }
    /* A string member, or `dflt` when absent or not a string. */
    std::string text(const char* key, const char* dflt = "") const {
        const Json* v = get(key);
        return v && v->kind == STR ? v->str : std::string(dflt);
    }
    /* A numeric member as an integer, or `dflt`. */
    int64_t integer(const char* key, int64_t dflt) const {
        const Json* v = get(key);
        return v && v->kind == NUM ? (int64_t)v->num : dflt;
    }
};

struct JsonReader {
    const char* p;
    const char* end;
    int depth = 0;

    void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    bool lit(const char* s) {
        const size_t n = std::strlen(s);
        if ((size_t)(end - p) < n || std::strncmp(p, s, n) != 0) return false;
        p += n;
        return true;
    }
    static void utf8(std::string& out, uint32_t c) {
        if (c < 0x80) { out += (char)c; return; }
        if (c < 0x800) { out += (char)(0xC0 | c >> 6); }
        else if (c < 0x10000) { out += (char)(0xE0 | c >> 12); out += (char)(0x80 | ((c >> 6) & 0x3F)); }
        else { out += (char)(0xF0 | c >> 18); out += (char)(0x80 | ((c >> 12) & 0x3F));
               out += (char)(0x80 | ((c >> 6) & 0x3F)); }
        out += (char)(0x80 | (c & 0x3F));
    }
    bool hex4(uint32_t* c) {
        if (end - p < 4) return false;
        *c = 0;
        for (int i = 0; i < 4; ++i, ++p) {
            const char h = *p;
            const int v = h >= '0' && h <= '9' ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10
                        : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
            if (v < 0) return false;
            *c = *c << 4 | (uint32_t)v;
        }
        return true;
    }
    bool string(std::string& out) {
        if (p >= end || *p++ != '"') return false;
        while (p < end && *p != '"') {
            if ((unsigned char)*p < 0x20) return false;
            if (*p != '\\') { out += *p++; continue; }
            if (++p >= end) return false;
            const char e = *p++;
            uint32_t c = 0;
            switch (e) {
                case '"': case '\\': case '/': out += e; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u':
                    if (!hex4(&c)) return false;
                    if (c >= 0xD800 && c < 0xDC00) {   /* a surrogate pair */
                        uint32_t lo = 0;
                        if (!lit("\\u") || !hex4(&lo) || lo < 0xDC00 || lo > 0xDFFF) return false;
                        c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(out, c);
                    break;
                default: return false;
            }
        }
        return p++ < end;
    }
    bool value(Json& v) {
        ws();
        if (p >= end || ++depth > 64) return false;
        bool ok = true;
        if (*p == '{') {
            v.kind = Json::OBJ;
            ++p; ws();
            if (p < end && *p == '}') ++p;
            else for (;;) {
                std::pair<std::string, Json> kv;
                ws();
                if (!string(kv.first)) { ok = false; break; }
                ws();
                if (p >= end || *p++ != ':' || !value(kv.second)) { ok = false; break; }
                v.obj.push_back(std::move(kv));
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                ok = p < end && *p++ == '}';
                break;
            }
        } else if (*p == '[') {
            v.kind = Json::ARR;
            ++p; ws();
            if (p < end && *p == ']') ++p;
            else for (;;) {
                v.arr.emplace_back();
                if (!value(v.arr.back())) { ok = false; break; }
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                ok = p < end && *p++ == ']';
                break;
            }
        } else if (*p == '"') {
            v.kind = Json::STR;
            ok = string(v.str);
        } else if (lit("true")) { v.kind = Json::BOOL; v.b = true; }
        else if (lit("false")) { v.kind = Json::BOOL; }
        else if (lit("null")) { v.kind = Json::NUL; }
        else {
            const std::string num(p, (size_t)std::min<ptrdiff_t>(end - p, 64));
            char* stop = nullptr;
            v.kind = Json::NUM;
            v.num = std::strtod(num.c_str(), &stop);
            ok = stop != num.c_str();
            p += stop - num.c_str();
        }
        --depth;
        return ok;
    }
};

/* `text` as one JSON value with nothing but whitespace after it. */
inline bool json_parse(const char* text, size_t n, Json* out) {
    JsonReader r{ text, text + n };
    *out = Json{};
    if (!r.value(*out)) return false;
    r.ws();
    return r.p == r.end;
}

}  /* namespace kva */

/* The adapter and its static test still name these qwen4exp_kva:: (the split moves them, the names
 * stay). */
namespace qwen4exp_kva { using namespace kva; }

#endif /* QWEN4EXP_KVA_JSON_H */
