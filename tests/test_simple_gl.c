#include <stdio.h>
#include <fcntl.h>
#include <xf86drm.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <time.h>

static uint64_t ns_now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}

static GLuint prog(const char*v,const char*f){GLuint p=glCreateProgram();
GLuint V=glCreateShader(GL_VERTEX_SHADER);glShaderSource(V,1,&v,NULL);glCompileShader(V);
GLuint F=glCreateShader(GL_FRAGMENT_SHADER);glShaderSource(F,1,&f,NULL);glCompileShader(F);
glAttachShader(p,V);glAttachShader(p,F);glLinkProgram(p);return p;}

int main(){
    int fd=open("/dev/dri/card1",0);if(fd<0)fd=open("/dev/dri/card0",0);
    struct gbm_device*gbm=gbm_create_device(fd);
    EGLDisplay egl=eglGetDisplay((EGLNativeDisplayType)gbm);
    eglInitialize(egl,(EGLint[]){0},(EGLint[]){0});
    EGLConfig cfg;EGLint n;eglChooseConfig(egl,(EGLint[]){EGL_SURFACE_TYPE,EGL_WINDOW_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES2_BIT,EGL_NONE},&cfg,1,&n);
    EGLContext ctx=eglCreateContext(egl,cfg,EGL_NO_CONTEXT,(EGLint[]){EGL_CONTEXT_CLIENT_VERSION,2,EGL_NONE});
    eglMakeCurrent(egl,EGL_NO_SURFACE,EGL_NO_SURFACE,ctx);
    printf("GL: %s\n",glGetString(GL_VERSION));
    
    const char*vs="attribute vec2 a_pos;void main(){gl_Position=vec4(a_pos,0,1);}";
    const char*fs="precision mediump float;void main(){gl_FragColor=vec4(1,0,0,1);}";
    GLuint p=prog(vs,fs);glUseProgram(p);
    
    float in[16];for(int i=0;i<16;i++)in[i]=i;
    
    GLuint tex,fbo;
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D,tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_LUMINANCE,4,4,0,GL_LUMINANCE,GL_FLOAT,in);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    
    glGenFramebuffers(1,&fbo);
    glBindFramebuffer(GL_FRAMEBUFFER,fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0);
    glViewport(0,0,4,4);
    
    glClearColor(0,1,0,1);
    glClear(GL_COLOR_BUFFER_BIT);
    
    uint64_t s=ns_now();
    glDrawArrays(GL_TRIANGLE_STRIP,0,4);
    uint64_t e=ns_now();
    printf("GPU: %.3fms\n",(e-s)/1e6);
    
    float out[16];
    glReadPixels(0,0,4,4,GL_LUMINANCE,GL_FLOAT,out);
    for(int i=0;i<16;i++)printf("%d: %f\n",i,out[i]);
    return 0;
}
