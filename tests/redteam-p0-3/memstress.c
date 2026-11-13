/* memstress <threads> <secs>: each thread memcpy's 32 MiB back and forth; prints GB/s. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double secs; static double tot[8];
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static void* run(void* a) {
    long i = (long)a; size_t n = 32u << 20; char* s = malloc(n); char* d = malloc(n);
    memset(s, 1, n); memset(d, 2, n);
    double t0 = now(); double b = 0;
    while (now() - t0 < secs) { memcpy(d, s, n); memcpy(s, d, n); b += 2.0 * n * 2; }
    tot[i] = b / (now() - t0) / 1e9; return NULL;
}
int main(int c, char** v) {
    int n = c > 1 ? atoi(v[1]) : 4; secs = c > 2 ? atof(v[2]) : 5; pthread_t t[8];
    for (long i = 0; i < n; i++) pthread_create(&t[i], 0, run, (void*)i);
    double s = 0; for (int i = 0; i < n; i++) { pthread_join(t[i], 0); s += tot[i]; }
    printf("memstress threads=%d GB/s(read+write)=%.2f\n", n, s); return 0;
}
