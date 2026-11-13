/* test_v3d_runtime: hs_v3d error paths and resource hygiene on the hardware.
 * Wrong device nodes refused; no fd or mapping leaks across many open/close
 * and BO create/free cycles; double free harmless; invalid sizes refused.
 * Needs the render group. Exit 0 only if every check passes. */
#include "hs_v3d.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

static int failures;
static void check(int ok, const char* what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}
static int count_fds(void) {
    int n = 0;
    DIR* d = opendir("/proc/self/fd");
    if (!d) return -1;
    while (readdir(d)) n++;
    closedir(d);
    return n;
}
static int count_maps(void) {
    FILE* f = fopen("/proc/self/maps", "r");
    int n = 0, c;
    if (!f) return -1;
    while ((c = fgetc(f)) != EOF) n += c == '\n';
    fclose(f);
    return n;
}

int main(void) {
    hs_v3d dev;
    check(hs_v3d_open(&dev, "/dev/null") == -ENODEV && dev.fd == -1, "a non-DRM node is refused (-ENODEV), fd left -1");
    check(hs_v3d_open(&dev, "/nonexistent/renderD999") == -ENOENT, "a missing node gives -ENOENT");
    /* on this Pi card0 is v3d's primary node and card1 is vc4 (display) */
    /* -ENODEV only: -EACCES would mean the open failed before the driver
     * name check this is meant to exercise */
    int raw = open("/dev/dri/card1", O_RDWR | O_CLOEXEC); /* it must open as a file, so the name check runs */
    if (raw >= 0) close(raw);
    int e = hs_v3d_open(&dev, "/dev/dri/card1");
    printf("     (card1 opens as a file: %s; hs_v3d_open returned %d)\n", raw >= 0 ? "yes" : "no", e);
    check(raw >= 0 && e == -ENODEV, "card1 (vc4, the display driver) opens but is refused by the driver-name check");
    if (e == 0) hs_v3d_close(&dev);

    int fd0 = count_fds();
    for (int i = 0; i < 200; i++) {
        if (hs_v3d_open(&dev, NULL)) {
            check(0, "open in loop");
            return 1;
        }
        hs_v3d_close(&dev);
    }
    check(count_fds() == fd0, "200 open/close cycles leak no fd");
    hs_v3d_close(&dev); /* double close: harmless */
    check(dev.fd == -1, "double close is harmless");

    if (hs_v3d_open(&dev, NULL)) return 1;
    int fd1 = count_fds(), m0 = count_maps();
    for (int i = 0; i < 300; i++) {
        hs_v3d_bo bo;
        if (hs_v3d_bo_new(&dev, &bo, (size_t)(1 + i % 7) * 4096 + (i % 3) * 100)) {
            check(0, "bo_new in loop");
            return 1;
        }
        bo.map[0] = (uint8_t)i;
        bo.map[bo.size - 1] = (uint8_t)i;
        hs_v3d_bo_free(&dev, &bo);
        hs_v3d_bo_free(&dev, &bo); /* double free: harmless */
    }
    check(count_fds() == fd1 && count_maps() == m0, "300 BO cycles (odd sizes, double free) leak no fd or mapping");
    int m1 = count_maps();
    for (int i = 0; i < 100; i++) {
        hs_v3d_bo bo;
        if (hs_v3d_bo_new(&dev, &bo, 65536)) return 1;
        hs_v3d_bo_free(&dev, &bo);
    }
    check(count_maps() == m1, "100 BO cycles leak no mapping");
    hs_v3d_bo bo;
    check(hs_v3d_bo_new(&dev, &bo, 0) == -EINVAL && bo.handle == 0 && bo.map == NULL, "size 0 is refused, BO left empty");
    check(hs_v3d_bo_new(&dev, &bo, (size_t)1 << 33) == -EINVAL, "size above 4 GiB is refused");
    /* 2 GiB + 2.5 GiB exceeds the GPU's 4 GiB address space: the second must
     * fail cleanly and leave its BO empty */
    hs_v3d_bo big;
    e = hs_v3d_bo_new(&dev, &big, (size_t)2 << 30);
    check(e == 0, "a 2 GiB BO can be created");
    int e2 = hs_v3d_bo_new(&dev, &bo, (size_t)5 << 29);
    check(e2 == -ENOSPC && bo.handle == 0 && bo.map == NULL,
          "a BO beyond the GPU's 4 GiB address space fails cleanly with -ENOSPC");
    printf("     (the refused allocation returned %d: %s)\n", e2, strerror(e2 < 0 ? -e2 : e2));
    if (!e2) hs_v3d_bo_free(&dev, &bo);
    if (!e) hs_v3d_bo_free(&dev, &big);
    check(count_fds() == fd1, "no fd leaked by the failure paths");
    hs_v3d_close(&dev);
    printf("%s\n", failures ? "FAILED" : "all runtime hygiene checks passed");
    return failures ? 1 : 0;
}
