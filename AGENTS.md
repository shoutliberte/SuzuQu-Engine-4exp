# SuzuQu-Engine-4exp (q4) agent notes

## Resource safety

- Only one q4 process at a time. The resident expert image is tens of GiB.
- Run q4 under a memory cap. `serve.sh` does it via
  `systemd-run --user --scope -p MemoryMax=$Q4_MEM_MAX -p MemorySwapMax=0`.
  For probes/benches use `scripts/q4safe.sh <logprefix> <q4 args...>`
  (same cap + a watchdog that logs MemAvailable/swap/RSS/VRAM and kills q4
  on floor/swap-growth/stall). Quantized weights don't compress, so letting
  the image swap only thrashes.
- `serve.sh` and `q4safe.sh` source `scripts/q4preflight.sh`: they refuse to
  start with another q4 alive, after an unacknowledged q4 `oom-kill` scope, or
  with MemAvailable under `Q4_NEED_GIB`, and they set `oom_score_adj` so a
  global OOM takes q4 rather than the rest of the system.
- After any OOM kill of q4: stop, read `journalctl -b | grep -i oom`, and
  understand the cause before relaunching — the kernel is still reaping the
  resident image. Acknowledge with `systemctl --user reset-failed` only after
  the cause is understood.
- No q4 runs while a large download or other heavy memory use competes for RAM.

## Build / verify

- Build: `make` (hipcc, ROCm; `--offload-arch` via `ROCM_ARCH`, default gfx1100).
- Prefill bench: `Q4_EXPERT_RESIDENT=1 Q4_L1_SLOTS=8192 Q4_IO_THREADS=8 Q4_DRAM_RESERVE_GIB=4 HIP_VISIBLE_DEVICES=0 ./q4 prefill-probe <model> --tokens N --ctx 8192`
  (profiling: `Q4_PROFILE=1`, `Q4_GEMV_PROF=2`; `Q4_PREF_CHUNK` 512..4096, default 2048;
  `Q4_PF=0` disables next-layer expert prefetch; `Q4_LOGIT_DUMP=1` prints top logits).
- Prefill fallbacks (each `=0` restores the previous kernels bit-identically, for bisecting):
  `Q4_WMMA`, `Q4_QSA_PF`, `Q4_PLE_BATCH`, `Q4_GDN_SCAN`.
- Decode is not bit-reproducible across cache histories: hybrid decode picks GPU vs
  CPU expert kernels by L1 residency. For exact A/B use `Q4_CPU_EXPERTS=0`.
- Correctness: compare the probe's generated token ids against a baseline binary
  built from the previous commit in a `git worktree`.
