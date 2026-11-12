/* Keep the kernel's default unbound workers at SCHED_FIFO so KMS page flips
 * land on time with every CPU busy.
 *
 * On vc4 (Raspberry Pi 4) each page flip's commit runs in an unbound kworker
 * and makes firmware clock calls (~1.7 ms each, two or three before the
 * display list is updated). With all four cores busy, the normal-priority
 * kworker waited up to 6.7 ms for a CPU after those calls and missed the
 * vblank: test_08_raycast_orbit repeated 44-62 of ~600 refreshes. With these
 * workers at SCHED_FIFO 1, the same load repeated 0, and pacing matched an
 * idle system (tests/evidence-measure-20261114/vc4fix/kworker-fifo).
 *
 * The kernel offers no priority setting for this workqueue, and its pool
 * creates and retires worker threads over time, so this re-applies the
 * policy every interval. Each pass reads ~170 /proc entries: 1.6% of one core
 * at 500 ms. No new unbound worker appeared in 180 s of rendering under full
 * load, so the default is 5 s; a worker created in between runs at normal
 * priority until the next pass. (This kernel has no CONFIG_PROC_EVENTS, so
 * new threads cannot be caught by event.) Targets: threads named "kworker/u..." at nice 0 (the
 * default unbound pool; the high-priority pool runs at nice -20 and is left
 * alone). Page flips stay non-blocking, so other programs reading display
 * state are not held up. On SIGTERM/SIGINT, or with --restore, the workers are
 * returned to SCHED_OTHER.
 *
 * Usage: sudo tools/neogpu_kworker_rt [interval_ms=5000] [priority=1]
 *        sudo tools/neogpu_kworker_rt --once | --restore
 * Install as a service: sudo make install-kworker-rt (remove: uninstall-kworker-rt).
 */
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t stopping;

static void on_signal(int sig) {
    (void)sig;
    stopping = 1;
}

/* nice and scheduling policy of an unbound kworker; false for other threads. */
static bool unbound_worker(pid_t pid, long* nice, long* policy) {
    char path[64], comm[64];
    snprintf(path, sizeof(path), "/proc/%d/comm", (int)pid);
    FILE* f = fopen(path, "r");
    if (!f) return false;
    bool named = fgets(comm, sizeof(comm), f) && strncmp(comm, "kworker/u", 9) == 0;
    fclose(f);
    if (!named) return false;
    char stat[1024];
    snprintf(path, sizeof(path), "/proc/%d/stat", (int)pid);
    f = fopen(path, "r");
    if (!f) return false;
    size_t n = fread(stat, 1, sizeof(stat) - 1, f);
    fclose(f);
    stat[n] = '\0';
    /* Fields after the parenthesised name: state is field 3; nice is 19,
     * policy is 41. */
    char* p = strrchr(stat, ')');
    if (!p) return false;
    p++;
    long field = 2;
    *nice = *policy = -1;
    for (char* tok = strtok(p, " "); tok; tok = strtok(NULL, " ")) {
        field++;
        if (field == 19) *nice = strtol(tok, NULL, 10);
        if (field == 41) { *policy = strtol(tok, NULL, 10); break; }
    }
    return *nice != -1 && *policy != -1;
}

/* One pass over /proc. restore: back to SCHED_OTHER. Returns workers changed. */
static int apply(int priority, bool restore, int* seen) {
    DIR* d = opendir("/proc");
    if (!d) return -1;
    int changed = 0;
    *seen = 0;
    struct dirent* e;
    while ((e = readdir(d))) {
        if (!isdigit((unsigned char)e->d_name[0])) continue;
        pid_t pid = (pid_t)strtol(e->d_name, NULL, 10);
        long nice, policy;
        if (!unbound_worker(pid, &nice, &policy) || nice != 0) continue;
        (*seen)++;
        int want = restore ? SCHED_OTHER : SCHED_FIFO;
        struct sched_param sp;
        if (policy == want && (restore || (sched_getparam(pid, &sp) == 0 && sp.sched_priority == priority)))
            continue; /* already as wanted */
        sp.sched_priority = restore ? 0 : priority;
        if (sched_setscheduler(pid, want, &sp) == 0) changed++;
        else if (errno == EPERM) { closedir(d); return -1; } /* ESRCH: the worker exited */
    }
    closedir(d);
    return changed;
}

int main(int argc, char** argv) {
    int seen = 0;
    if (argc > 1 && (strcmp(argv[1], "--restore") == 0 || strcmp(argv[1], "--once") == 0)) {
        bool restore = argv[1][2] == 'r';
        int changed = apply(1, restore, &seen);
        if (changed < 0) { perror("sched_setscheduler (needs root)"); return 1; }
        printf("%d of %d unbound workers %s\n", changed, seen, restore ? "returned to SCHED_OTHER" : "set to SCHED_FIFO 1");
        return 0;
    }
    long interval_ms = argc > 1 ? strtol(argv[1], NULL, 10) : 5000;
    int priority = argc > 2 ? (int)strtol(argv[2], NULL, 10) : 1;
    if (interval_ms < 10 || priority < 1 || priority > 99) {
        fprintf(stderr, "usage: %s [interval_ms>=10] [priority 1-99] | --once | --restore\n", argv[0]);
        return 2;
    }
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
    if (apply(priority, false, &seen) < 0) { perror("sched_setscheduler (needs root)"); return 1; }
    struct timespec ts = {interval_ms / 1000, (interval_ms % 1000) * 1000000L};
    while (!stopping) {
        nanosleep(&ts, NULL);
        if (!stopping) apply(priority, false, &seen);
    }
    apply(priority, true, &seen);
    return 0;
}
