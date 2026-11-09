/* Count-only linker instrumentation. Do not use this binary for rate claims. */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

static pthread_t main_thread;
static pthread_mutex_t* first_mutex;
static uint64_t active_attempts, other_attempts, failures, unlocks;
extern int __real_pthread_mutex_trylock(pthread_mutex_t* mutex);
extern int __real_pthread_mutex_unlock(pthread_mutex_t* mutex);

int __wrap_pthread_mutex_trylock(pthread_mutex_t* mutex) {
    int result = __real_pthread_mutex_trylock(mutex);
    if (pthread_equal(pthread_self(), main_thread)) {
        if (!first_mutex) first_mutex = mutex; /* native join's slot-zero mutex */
        if (mutex == first_mutex) active_attempts++; else other_attempts++;
        if (result != 0) failures++;
    }
    return result;
}
int __wrap_pthread_mutex_unlock(pthread_mutex_t* mutex) {
    if (pthread_equal(pthread_self(), main_thread)) unlocks++;
    return __real_pthread_mutex_unlock(mutex);
}
static void report(void) {
    fprintf(stderr, "{\"record\":\"main_thread_lock_calls\",\"first_mutex_attempts\":%" PRIu64
            ",\"other_mutex_attempts\":%" PRIu64 ",\"failed_attempts\":%" PRIu64
            ",\"unlocks\":%" PRIu64 "}\n", active_attempts, other_attempts, failures, unlocks);
}
__attribute__((constructor)) static void setup(void) {
    main_thread = pthread_self();
    if (atexit(report) != 0) abort();
}
