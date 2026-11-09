#define _GNU_SOURCE
#include "bedrock_support.h"
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static uint64_t ns(void) {
    struct timespec time;
    REQUIRE(clock_gettime(CLOCK_MONOTONIC, &time) == 0);
    return (uint64_t)time.tv_sec * 1000000000ull + time.tv_nsec;
}
int main(int argc, char** argv) {
    REQUIRE(argc == 2);
    unsigned mode = !strcmp(argv[1], "sched-yield") ? 1 : !strcmp(argv[1], "getpid") ? 2 :
                    !strcmp(argv[1], "cpu-yield") ? 3 : !strcmp(argv[1], "loop") ? 4 : 0;
    REQUIRE(mode);
    alarm(30);
    cpu_set_t set;
    CPU_ZERO(&set); CPU_SET(1, &set);
    REQUIRE(pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0);
    bedrock_thread(0);
    const unsigned iterations = 1000000;
    pid_t pid = getpid();
    bedrock_control("enable");
    uint64_t start = ns();
    if (mode == 1) for (unsigned i = 0; i < iterations; i++) REQUIRE(sched_yield() == 0);
    if (mode == 2) for (unsigned i = 0; i < iterations; i++) REQUIRE(getpid() == pid);
    if (mode == 3) for (unsigned i = 0; i < iterations; i++) __asm__ volatile("yield" ::: "memory");
    if (mode == 4) for (unsigned i = 0; i < iterations; i++) __asm__ volatile("" ::: "memory");
    uint64_t elapsed = ns() - start;
    bedrock_control("disable");
    printf("{\"mode\":\"%s\",\"iterations\":%u,\"elapsed_ns\":%" PRIu64 ",\"pid\":%d}\n", argv[1], iterations, elapsed, (int)pid);
    return 0;
}
