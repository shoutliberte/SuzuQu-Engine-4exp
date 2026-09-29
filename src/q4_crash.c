#define _GNU_SOURCE
#include "q4.h"

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

/* Crash capture for the intermittent amdhip64/libstdc++ vector abort.
 *
 * The assert fires inside libamdhip64, the process dies with SIGABRT, and
 * a bare core dump is useless on this box: q4's core would include the
 * ~45 GiB resident expert image, so systemd-coredump drops it ("missing"
 * in coredumpctl).  Two layers here:
 *
 *   1. This handler prints the faulting backtrace straight to stderr
 *      (the q4safe/serve .log captures it) BEFORE the core write — so the
 *      stack survives even when the core does not.
 *   2. The huge weight arenas are marked MADV_DONTDUMP (q4_expert.c) so a
 *      real core stays small enough to store; `coredumpctl debug q4`
 *      then opens it in gdb.
 *
 * backtrace_symbols_fd writes unsymbolized frames but needs no allocation
 * in the fast path; backtrace() itself may resolve through libgcc on first
 * use, so install() pre-warms it. */

static volatile sig_atomic_t q4_in_crash;

static void q4_crash_bt(int sig, siginfo_t *si, void *uc) {
    (void)uc;
    if (q4_in_crash) _exit(128 + sig); /* nested fault: die quietly */
    q4_in_crash = 1;
    char hdr[192];
    int n = snprintf(hdr, sizeof hdr,
                     "\n=== q4 fatal signal %d si_addr=%p pid=%ld ===\n",
                     sig, si ? si->si_addr : NULL, (long)getpid());
    if (n > 0) (void)!write(STDERR_FILENO, hdr, (size_t)n);
    void *fr[64];
    int nf = backtrace(fr, 64);
    backtrace_symbols_fd(fr, nf, STDERR_FILENO);
    /* Re-raise under the default disposition: the kernel then produces a
     * real core through systemd-coredump (the arenas are DONTDUMP).  The
     * signal is blocked inside this handler, so raise() alone would only
     * pend it — unblock first or _exit would win the race and dump
     * nothing. */
    signal(sig, SIG_DFL);
    sigset_t s;
    sigemptyset(&s);
    sigaddset(&s, sig);
    sigprocmask(SIG_UNBLOCK, &s, NULL);
    raise(sig);
    _exit(128 + sig);
}

void q4_install_crashdump(void) {
    /* Best effort: make sure a core can be written even if the caller's
     * ulimit was 0 (systemd-run inherits the caller's rlimits, but probes
     * launched by hand often run under `ulimit -c 0`). */
    struct rlimit rl = {RLIM_INFINITY, RLIM_INFINITY};
    setrlimit(RLIMIT_CORE, &rl);
    /* Prewarm the unwinder so the handler never takes its first malloc
     * inside a signal frame. */
    void *fr[8];
    int nf = backtrace(fr, 8);
    char **syms = backtrace_symbols(fr, nf);
    free(syms);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = q4_crash_bt;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    static const int sigs[] = {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE};
    for (size_t i = 0; i < sizeof sigs / sizeof sigs[0]; i++)
        sigaction(sigs[i], &sa, NULL);
}
