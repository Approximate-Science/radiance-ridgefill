/* kva_log.h -- the plugin's name in the core's stderr lines ("radiance: <name>: ...").
 *
 * The adapter's log_name fact, mirrored in one global because the core speaks before any Kva exists
 * (the release guard at open, read_config's refusals) and from helpers that take none (the upload).
 * The adapter's rad_plugin_open and declare set it before anything is said; "kva" is only ever seen by
 * a test that calls the core directly.
 */
#ifndef KVA_LOG_H
#define KVA_LOG_H

namespace kva {

inline const char* g_log_name = "kva";

}  /* namespace kva */

#endif /* KVA_LOG_H */
