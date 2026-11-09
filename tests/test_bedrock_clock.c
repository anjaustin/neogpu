#define _GNU_SOURCE
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static void unsupported(int signal_number) {
    (void)signal_number;
    const char message[] = "{\"architectural_counter_read\":\"SIGILL\"}\n";
    ssize_t result = write(STDOUT_FILENO,message,sizeof(message)-1);
    (void)result;
    _Exit(77);
}
static uint64_t counter(void) {
    uint64_t value;
    __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(value) :: "memory");
    return value;
}
static uint64_t ns(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC,&time)) abort();
    return (uint64_t)time.tv_sec * 1000000000ull + time.tv_nsec;
}
int main(void) {
    struct sigaction action = {.sa_handler=unsupported};
    if (sigaction(SIGILL,&action,NULL)) return 1;
    cpu_set_t set;
    CPU_ZERO(&set);CPU_SET(0,&set);
    if (pthread_setaffinity_np(pthread_self(),sizeof(set),&set)) return 1;
    uint64_t frequency;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(frequency));
    uint64_t before = counter(), start = ns();
    struct timespec pause = {.tv_sec=0,.tv_nsec=10000000};
    if (nanosleep(&pause,NULL)) return 1;
    uint64_t end = ns(), after = counter();
    struct timespec resolution;
    if (clock_getres(CLOCK_MONOTONIC,&resolution)) return 1;
    printf("{\"cntfrq_hz\":%" PRIu64 ",\"cntvct_before\":%" PRIu64 ",\"cntvct_after\":%" PRIu64
           ",\"monotonic_elapsed_ns\":%" PRIu64 ",\"clock_getres_ns\":%ld}\n",
           frequency,before,after,end-start,resolution.tv_sec*1000000000L+resolution.tv_nsec);
    return 0;
}
