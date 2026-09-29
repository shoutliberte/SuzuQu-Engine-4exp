# q4preflight.sh — sourced by serve.sh and scripts/q4safe.sh before q4 starts.
#
# q4_preflight refuses to start q4 when
#   1. another q4 process is running,
#   2. an earlier q4 scope ended in oom-kill and nobody acknowledged it
#      (relaunching a ~51 GiB resident fill right after an OOM kill, while
#      the kernel is still reaping the last one, is what takes the desktop
#      session down), or
#   3. MemAvailable is below Q4_NEED_GIB (default 54 resident / 16 not).
# Q4_PREFLIGHT=0 skips all checks.
#
# q4_oom_first marks the calling shell (and so the exec'd q4) as the
# preferred OOM victim, so a global OOM takes q4 instead of the desktop.

q4_oom_first() {
    { echo 1000 > /proc/self/oom_score_adj; } 2>/dev/null || true
}

# GTT usage of the discrete GPU. Card numbering is not stable, so resolve by
# PCI id (RX 7900 XTX = 1002:744C) each call; Q4_GTT_PATH overrides. Falls
# back to card1, then prints 0 when nothing matches.
q4_gtt_used() {
    p=${Q4_GTT_PATH:-}
    if [ -z "$p" ]; then
        for d in /sys/class/drm/card*/device; do
            [ -f "$d/uevent" ] || continue
            if grep -q 'PCI_ID=1002:744C' "$d/uevent" 2>/dev/null; then
                p="$d/mem_info_gtt_used"; break
            fi
        done
        [ -z "$p" ] && p=/sys/class/drm/card1/device/mem_info_gtt_used
    fi
    cat "$p" 2>/dev/null || echo 0
}

q4_preflight() {
    [ "${Q4_PREFLIGHT:-1}" = "0" ] && return 0
    if pgrep -x q4 >/dev/null 2>&1; then
        echo "q4 preflight: another q4 process is running:" >&2
        pgrep -ax q4 >&2
        return 1
    fi
    if command -v systemctl >/dev/null 2>&1; then
        for u in $(systemctl --user list-units --type=scope --state=failed \
                       --plain --no-legend 2>/dev/null | awk '{print $1}'); do
            r=$(systemctl --user show -p Result --value "$u" 2>/dev/null)
            d=$(systemctl --user show -p Description --value "$u" 2>/dev/null)
            case "$r:$d" in
            oom-kill:*q4*)
                echo "q4 preflight: $u was OOM-killed ($d)." >&2
                echo "  Find out why before relaunching (journalctl -b | grep -i oom)," >&2
                echo "  then acknowledge with: systemctl --user reset-failed" >&2
                return 1
                ;;
            esac
        done
    fi
    need=${Q4_NEED_GIB:-}
    if [ -z "$need" ]; then
        # Resident image ~45.3 GiB + ~7 GiB driver-pinned/GTT the cgroup
        # cannot see + a few GiB for the desktop. The MTP head defaults to
        # page cache now; only Q4_MTP_RESIDENT=1 needs headroom for +3 GiB.
        if [ "${Q4_EXPERT_RESIDENT:-0}" = "1" ]; then need=55; else need=16; fi
        if [ "${Q4_MTP_RESIDENT:-0}" = "1" ]; then need=$((need + 4)); fi
    fi
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$avail" -lt "$need" ]; then
        echo "q4 preflight: MemAvailable ${avail} GiB < ${need} GiB needed." >&2
        echo "  Close heavy apps / pause large downloads first (or set Q4_NEED_GIB)." >&2
        return 1
    fi
    gtt=$(q4_gtt_used)
    echo "q4 preflight: ok (MemAvailable ${avail} GiB, gtt $((gtt/1048576)) MiB)" >&2
    return 0
}
