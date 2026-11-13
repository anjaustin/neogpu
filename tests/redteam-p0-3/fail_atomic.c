/* LD_PRELOAD: drmModeAtomicCommit fails with EINVAL (counts calls). */
#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
typedef struct _drmModeAtomicReq drmModeAtomicReq;
int drmModeAtomicCommit(int fd, drmModeAtomicReq* req, uint32_t flags, void* user) {
    (void)fd; (void)req; (void)flags; (void)user;
    fprintf(stderr, "fail_atomic: forcing EINVAL\n");
    errno = EINVAL;
    return -EINVAL;
}
