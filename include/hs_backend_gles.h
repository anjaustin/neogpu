#ifndef HS_BACKEND_GLES_H
#define HS_BACKEND_GLES_H

#include "hs_backend.h"

/* Create a GLES backend that owns its HSGraphics instance. Attach it once; its
 * context is freed by the backend's shutdown (hs_gpu_shutdown, or a failed
 * hs_gpu_attach_backend), after which the HSBackend value must not be reused.
 * Only one backend can own the display at a time (DRM master is exclusive). */
HSBackend hs_backend_gles_create(void);

/* Read back the next executed frame, just before it is presented, into a
 * binary PPM at path (after a page flip the back buffer is undefined, so a
 * glReadPixels after presenting does not show the frame). */
bool hs_backend_gles_capture_next(HSBackend* backend, const char* path);
/* True once the requested capture has been written successfully. */
bool hs_backend_gles_capture_ok(const HSBackend* backend);

#endif
