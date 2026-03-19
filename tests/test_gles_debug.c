#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <xf86drm.h>
#include <gbm.h>
#include <EGL/egl.h>

int main() {
    printf("=== Debug ===\n");
    
    int fd = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    }
    if (fd < 0) {
        printf("No DRM: %s\n", strerror(errno));
        return 1;
    }
    printf("DRM fd: %d\n", fd);
    
    int ret = drmSetMaster(fd);
    printf("drmSetMaster: %d (%s)\n", ret, strerror(errno));
    
    struct gbm_device* gbm = gbm_create_device(fd);
    if (!gbm) {
        printf("GBM create failed: %s\n", strerror(errno));
        return 1;
    }
    printf("GBM created\n");
    
    EGLDisplay egl = eglGetDisplay((EGLNativeDisplayType)gbm);
    if (egl == EGL_NO_DISPLAY) {
        printf("EGL no display: %x\n", eglGetError());
        return 1;
    }
    printf("EGL display: %p\n", egl);
    
    EGLint major, minor;
    if (!eglInitialize(egl, &major, &minor)) {
        printf("EGL init failed: %x\n", eglGetError());
        return 1;
    }
    printf("EGL: %d.%d\n", major, minor);
    
    return 0;
}
