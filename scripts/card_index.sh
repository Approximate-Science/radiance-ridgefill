#!/bin/sh
# card_index.sh -- the ROCR_VISIBLE_DEVICES index of a card, from its PCI address, as ROCm numbers it.
#
# USAGE: scripts/card_index.sh [<pci address, default 0000:13:00.0>]     prints e.g. 1
#
# ROCR_VISIBLE_DEVICES takes the card's index among rocminfo's GPU agents, not a PCI address, and that
# order is the runtime's (on this box: 03:00.0 -> 0, 13:00.0 -> 1, the iGPU -> 2; notes/kernels.md). So it
# is read, not assumed: rocminfo in the build image ($RK_BUILD_IMAGE, which carries ROCm), each agent's
# Device Type then BDFID, compared with the address's (bus << 8 | device << 3 | function). A query only:
# nothing runs on the card. Exits non-zero naming what it saw when no GPU agent has that address.
set -eu

. "$(dirname "$0")/common.sh"

pci=${1:-0000:13:00.0}
bus=$(printf '%s' "$pci" | cut -d: -f2) dev=$(printf '%s' "$pci" | cut -d: -f3 | cut -d. -f1)
fn=$(printf '%s' "$pci" | cut -d. -f2)
want=$(( (0x$bus << 8) | (0x$dev << 3) | 0x$fn ))
img=$RK_BUILD_IMAGE
info=$(docker run --rm --security-opt label=disable --device /dev/kfd --device /dev/dri "$img" rocminfo 2>/dev/null) ||
    rk_die "rocminfo failed in $img"
printf '%s\n' "$info" | awk -v want="$want" '
    /Device Type:/ { gpu = ($3 == "GPU") }
    /BDFID:/       { if (gpu) { if ($2 == want) { print idx + 0; found = 1; exit } idx++ } }
    END            { exit !found }' ||
    rk_die "no GPU agent at $pci (BDFID $want); rocminfo has: $(printf '%s\n' "$info" | grep -E 'Device Type|BDFID' | tr -s ' ' | tr '\n' ' ')"
