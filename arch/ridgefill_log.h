/* ridgefill_log.h -- the plugin's name in the core's stderr lines ("radiance: <name>: ...").
 *
 * The adapter's log_name fact, mirrored in one global because the core speaks before any RidgeFill exists
 * (the release guard at open, read_config's refusals) and from helpers that take none (the upload).
 * The adapter's rad_plugin_open and declare set it before anything is said; "ridgefill" is only ever seen by
 * a test that calls the core directly.
 */
#ifndef RIDGEFILL_LOG_H
#define RIDGEFILL_LOG_H

namespace ridgefill {

inline const char* g_log_name = "ridgefill";

}  /* namespace ridgefill */

#endif /* RIDGEFILL_LOG_H */
