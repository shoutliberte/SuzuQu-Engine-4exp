#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL=${Q4_MODEL:-"/home/shoutliberte/Projects/models/IQ3E-Q8D-MTP/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"}
# Train length. QSA attends a fixed 2051 positions, so prefill stays near
# 60 tok/s as the cache grows (measured through 90K). OpenCode compaction
# is the practical window inside this allocation. See DESIGN.md.
CTX=${Q4_CTX:-262144}
HOST=${Q4_HOST:-127.0.0.1}
PORT=${Q4_PORT:-8090}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export HIP_VISIBLE_DEVICES

# IQ3E on this PC (RX 7900 XTX 24 GiB / 60 GiB RAM / 990 PRO).
# Routed experts are one 45.29 GiB DRAM image. PLE stays on SSD.
# Do not run two q4 processes. Do not browse heavy tabs while it runs.
# 4 GiB stays free for the desktop session. Do not lower it.
# A 10 GiB floor refused residency whenever available RAM was under 55 GiB.
Q4_EXPERT_RESIDENT=${Q4_EXPERT_RESIDENT:-1}
Q4_L2_GIB=${Q4_L2_GIB:-0}              # unused when resident alloc succeeds
Q4_DRAM_RESERVE_GIB=${Q4_DRAM_RESERVE_GIB:-4}
Q4_MLOCK_L2=${Q4_MLOCK_L2:-0}          # resident image is not mlocked (rlimit 8 MB)
Q4_KV=${Q4_KV:-q8}                     # f16 or f32 to widen. MTP wants f32
Q4_L1_SLOTS=${Q4_L1_SLOTS:-8192}       # planner sizes L1; alloc halves on OOM
Q4_IO_THREADS=${Q4_IO_THREADS:-8}
Q4_PIN_ARENA=${Q4_PIN_ARENA:-0}        # no staging arena in resident mode
# Unallocated VRAM after dense, KV, the QSA indexer and expert L1.
# Idle display is ~0.9 GiB and is already outside the 23 GiB budget.
# 1 GiB is the floor. 0 is raised to 1. Do not spend this on weights.
Q4_VRAM_RESERVE_GIB=${Q4_VRAM_RESERVE_GIB:-2}
export Q4_EXPERT_RESIDENT Q4_L2_GIB Q4_DRAM_RESERVE_GIB Q4_MLOCK_L2 Q4_KV
export Q4_L1_SLOTS Q4_IO_THREADS Q4_PIN_ARENA Q4_VRAM_RESERVE_GIB

# Pinned set drives partial-resident fill order. The route profile
# (collected over 9000 tok of mixed ja/en/code evals) matches this
# model's actual routing; the old ja-experts.warm was a UD-Q4 set and
# does not match.
Q4_WARM_FILE=${Q4_WARM_FILE:-$ROOT/../models/IQ3E-Q8D-MTP/route-prof-iq3e.warm}
Q4_SESS_CACHE=${Q4_SESS_CACHE:-$ROOT/session.cache}
# CPU expert pool width. 9950X3D has 16 cores; the pool idles between
# tokens (bounded spin -> futex), so 16 does not starve the desktop.
# ~40.5 -> ~42.5 t/s median at 16K ctx.
Q4_CPU_THREADS=${Q4_CPU_THREADS:-12}   # 12 == 16 for decode; keeps CCD1 free
# Device-side routing keeps whole layers in one graph but adds ~60us/layer
# of in-graph nodes (route kernel, mailbox burst, x D2H copy node); the host
# hybrid path overlaps its seam with gA's tail and wins ~11% at 16K ctx.
Q4_DEVROUTE=${Q4_DEVROUTE:-0}
export Q4_CPU_THREADS Q4_DEVROUTE
# int8-activation kernels are on by default and measured faster or neutral
# on this box: Q4_CPXQ8 drives the CPU-expert i8 dot (AVX2 maddubs/madd,
# ~1.4-1.6x on the GSQ formats), Q4_WGQ8 drives the GPU i8-dot MoE kernels
# (v_dot4_i32_iu8, iq2_gu ~1.6x). Both are bit-different from the float path,
# so =0 restores 9ffd59c-identical output for A/B. Q4_WGQ8_IQ4YA=1 opts the
# iq4_nl/iq4_xs down-proj into i8 — measured slower in-model, keep it off.
Q4_THINK=${Q4_THINK:-1}
Q4_SHOW_THINKING=${Q4_SHOW_THINKING:-1}
Q4_MAX_NEW=${Q4_MAX_NEW:-32768}
# MTP stays off. The embedded draft block works via a head GGUF extracted
# with scripts/mtp-extract.py (e.g. ~/Projects/models/IQ3E-Q8D-MTP/mtp-head.gguf):
#   Q4_MTP=<head.gguf> Q4_KV=f32 [Q4_MTP_SLOTS=128..512] [Q4_MTP_N=1..3]
# Measured: draft acceptance ~85-89%, mean len ~3.6 — but decode lands at
# ~18 t/s vs ~43 plain. The verify batch re-reads routed experts for every
# token, so on this DRAM-resident model the traffic scales with tokens and
# speculation does not amortize. Left as an opt-in.
Q4_MTP=${Q4_MTP:-0}
if [ "$Q4_MTP" = "0" ]; then
    Q4_MTP=""
elif [ -n "$Q4_MTP" ] && [ ! -e "$Q4_MTP" ]; then
    echo "MTP head not found: $Q4_MTP (Q4_MTP=0 disables)" >&2
    Q4_MTP=""
fi
export Q4_WARM_FILE Q4_SESS_CACHE Q4_THINK Q4_SHOW_THINKING Q4_MAX_NEW Q4_MTP
Q4_MTP_N=${Q4_MTP_N:-1}
export Q4_MTP_N

if [ ! -x "$ROOT/q4" ]; then
    echo "q4 is not built. Run make first." >&2
    exit 1
fi
if [ ! -e "$MODEL" ]; then
    echo "Model not found: $MODEL" >&2
    echo "Download with: ./download_q4.sh" >&2
    exit 1
fi

echo "q4 serve  $MODEL"
echo "  ctx=$CTX  http://$HOST:$PORT/  (OpenCode baseURL http://$HOST:$PORT/v1)"
# Own cgroup, hard RAM cap, no swap. Swap is zram and the expert image does
# not compress, so swapping it only thrashes. A global OOM kill inside the
# desktop's unit also stops the session (OOMPolicy=stop). Over the cap only
# q4 dies. Q4_MEM_MAX=0 skips.
# Preflight (scripts/q4preflight.sh): no second q4, no relaunch after an
# unacknowledged OOM kill, enough MemAvailable. q4 is also made the first
# OOM victim so a global OOM does not take the desktop.
#
# 2026-09-29 OOM lesson: the cgroup cap does NOT see amdgpu-pinned/GTT
# pages (~7 GiB unaccounted in the kernel OOM dump), so q4 can push the box
# into a global OOM while under MemoryMax. The kernel oom_reaper could not
# reap the pinned image fast enough and killed the desktop instead.
# Two extra layers now:
#   1. the MTP head keeps its experts in page cache, not resident DRAM
#      (Q4_MTP_RESIDENT=0 default), and
#   2. a 1 s watchdog below kills q4 -9 when MemAvailable < Q4_SAFE_FLOOR
#      (default 3 GiB) or swap grows >1 GiB — BEFORE the kernel OOM. A normal
#      kill lets the driver release pinned pages cleanly; a kernel OOM does
#      not. No stall check here: an idle server legitimately logs nothing.
. "$ROOT/scripts/q4preflight.sh"
q4_preflight || exit 1
q4_oom_first
ulimit -c unlimited 2>/dev/null || true   # keep the amdhip64 abort's core
Q4_MEM_MAX=${Q4_MEM_MAX:-52G}
FLOOR=${Q4_SAFE_FLOOR:-3145728}
MON=${Q4_SERVE_MON:-$ROOT/serve.mon}
if [ "$Q4_MEM_MAX" != "0" ] && command -v systemd-run >/dev/null 2>&1; then
    (exec systemd-run --user --scope --quiet -p MemoryMax="$Q4_MEM_MAX" \
        -p MemorySwapMax=0 "$ROOT/q4" serve "$MODEL" --ctx "$CTX" \
        --host "$HOST" --port "$PORT" "$@") &
    runner=$!
else
    ("$ROOT/q4" serve "$MODEL" --ctx "$CTX" --host "$HOST" --port "$PORT" \
        "$@") &
    runner=$!
fi
: > "$MON"
swap0=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print t-f}' /proc/meminfo)
qpid=""
killreason=""
trap 'kill -TERM $qpid $runner 2>/dev/null; exit 130' INT TERM
while kill -0 "$runner" 2>/dev/null; do
    sleep 1
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    sw=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print t-f}' /proc/meminfo)
    if [ -z "$qpid" ]; then qpid=$(pgrep -n -x q4 2>/dev/null || true); fi
    rss=0
    if [ -n "$qpid" ] && [ -r "/proc/$qpid/status" ]; then
        rss=$(awk '/VmRSS/{print $2}' "/proc/$qpid/status" 2>/dev/null || echo 0)
    fi
    gtt=$(q4_gtt_used)
    printf '%s avail=%sG swap=+%sM rss=%sG gtt=%sB\n' \
        "$(date +%T)" "$((avail/1048576))" "$(( (sw-swap0)/1024 ))" \
        "$((rss/1048576))" "$gtt" >> "$MON"
    if [ -n "$qpid" ] && [ -z "$killreason" ]; then
        [ "$avail" -lt "$FLOOR" ] && killreason="MemAvailable<floor"
        [ $((sw - swap0)) -gt 1048576 ] && killreason="swap-grew>1GiB"
        if [ -n "$killreason" ]; then
            echo "q4 watchdog: $killreason — killing q4 ($qpid)" >&2
            echo "=== WATCHDOG KILL $qpid: $killreason $(date) ===" >> "$MON"
            kill -9 "$qpid" 2>/dev/null
            qpid=""
        fi
    fi
done
wait "$runner"; rc=$?
printf '=== serve done rc=%s reason=%s ===\n' "$rc" "${killreason:-none}" >> "$MON"
exit "$rc"
