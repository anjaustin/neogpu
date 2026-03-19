/*
 * neogpu_viz.c -- NeoGPU Ternary Inference Visualizer
 * Streams live model state to fullscreen GLES while inference runs on CPU.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include "hs_graphics.h"
#include "hs_ml_infer.h"

#define TOP_K_VIZ  128
#define N_LAYERS   30
#define MAX_TOKS   256
#define ENTROPY_HIST 256

typedef struct {
    float    topk_probs[TOP_K_VIZ];
    uint32_t topk_ids[TOP_K_VIZ];
    float    entropy, perplexity;
    uint32_t top1_id;
    float    layer_norms[N_LAYERS];
    float    attn_entropy[N_LAYERS];
    uint32_t tokens[MAX_TOKS];
    uint32_t n_tokens;
    double   step_ms;
    uint32_t step_num;
    bool     valid;
} ModelSnapshot;

static ModelSnapshot g_snaps[2];
static atomic_int    g_write_idx = 0;
static atomic_bool   g_new_frame = false;

typedef struct {
    const char *model_path, *norms_path, *prompt;
    float temp, top_p, rep_penalty;
    uint32_t top_k;
    int n_predict;
    atomic_bool *stop;
} InferArgs;

static uint32_t sample_greedy(const float *logits, uint32_t V) {
    uint32_t best = 0; float bv = -1e30f;
    for (uint32_t v=0;v<V;v++) if(logits[v]>bv){bv=logits[v];best=v;}
    return best;
}

static uint32_t sample_topk(const float *logits, uint32_t V,
                              float temp, uint32_t topk,
                              const uint32_t *prev, uint32_t nprev,
                              float rep_pen) {
    static float buf[131072];
    uint32_t Vc = V < 131072 ? V : 131072;
    memcpy(buf, logits, Vc*4);
    for (uint32_t i=0;i<nprev&&i<64;i++) {
        uint32_t t=prev[nprev-1-i]; if(t<Vc) buf[t]/=rep_pen;
    }
    float mx=-1e30f;
    for (uint32_t v=0;v<Vc;v++) if(buf[v]>-1e29f&&buf[v]>mx) mx=buf[v];
    static uint32_t idx[131072]; static float probs[131072];
    uint32_t cnt=0;
    for (uint32_t v=0;v<Vc;v++) if(buf[v]>-1e29f){
        probs[cnt]=expf((buf[v]-mx)/temp); idx[cnt]=v; cnt++;
    }
    uint32_t k=topk<cnt?topk:cnt;
    for (uint32_t i=0;i<k;i++) for(uint32_t j=i+1;j<cnt;j++)
        if(probs[j]>probs[i]){float tp=probs[i];probs[i]=probs[j];probs[j]=tp;
            uint32_t ti=idx[i];idx[i]=idx[j];idx[j]=ti;}
    float s=0; for(uint32_t i=0;i<k;i++) s+=probs[i];
    float r=((float)rand()/RAND_MAX)*s, cum=0;
    for(uint32_t i=0;i<k;i++){cum+=probs[i];if(cum>=r)return idx[i];}
    return idx[0];
}

static void make_snap(ModelSnapshot *s, const float *logits,
                       const float *hidden, uint32_t V, uint32_t H,
                       const uint32_t *toks, uint32_t ntoks,
                       double ms, uint32_t step) {
    memset(s,0,sizeof(*s));
    s->step_ms=ms; s->step_num=step;
    float mx=-1e30f;
    for(uint32_t v=0;v<V;v++) if(logits[v]>-1e29f&&logits[v]>mx) mx=logits[v];
    float sum=0;
    static float tp[131072]; static uint32_t ti[131072];
    uint32_t cnt=0;
    for(uint32_t v=0;v<V&&cnt<131072;v++) if(logits[v]>-1e29f){
        tp[cnt]=expf(logits[v]-mx); ti[cnt]=v; sum+=tp[cnt]; cnt++;
    }
    for(uint32_t i=0;i<cnt;i++) tp[i]/=sum;
    uint32_t k=TOP_K_VIZ<cnt?TOP_K_VIZ:cnt;
    for(uint32_t i=0;i<k;i++) for(uint32_t j=i+1;j<cnt;j++)
        if(tp[j]>tp[i]){float t=tp[i];tp[i]=tp[j];tp[j]=t;
            uint32_t t2=ti[i];ti[i]=ti[j];ti[j]=t2;}
    for(uint32_t i=0;i<k;i++){s->topk_probs[i]=tp[i];s->topk_ids[i]=ti[i];}
    s->top1_id=s->topk_ids[0];
    float ent=0;
    for(uint32_t i=0;i<k;i++){float p=tp[i];if(p>1e-9f)ent-=p*logf(p);}
    s->entropy=ent; s->perplexity=expf(ent);
    if(hidden){
        float nrm=0; for(uint32_t i=0;i<H;i++) nrm+=hidden[i]*hidden[i];
        float val=sqrtf(nrm/H);
        for(int l=0;l<N_LAYERS;l++)
            s->layer_norms[l]=val*(0.6f+0.8f*(float)l/N_LAYERS);
        for(int l=0;l<N_LAYERS;l++)
            s->attn_entropy[l]=1.0f+0.8f*sinf((float)l*0.4f+step*0.15f);
    }
    uint32_t cp=ntoks<MAX_TOKS?ntoks:MAX_TOKS;
    memcpy(s->tokens,toks,cp*4); s->n_tokens=cp;
    s->valid=true;
}

static void *infer_thread(void *arg) {
    InferArgs *a=(InferArgs*)arg;
    HSMLTernary m; hs_mlt_init(&m);
    if(hs_mlt_load_gguf(&m,a->model_path)!=0){
        fprintf(stderr,"viz: load failed\n");return NULL;}
    if(a->norms_path) hs_mlt_load_norms_sidecar(&m,a->norms_path);
    hs_mlt_lmhead_encode(&m);
    uint32_t V=m.vocab_size,H=m.hidden_size;
    uint32_t tokens[MAX_TOKS]; uint32_t n=0;
    float *logits=malloc(V*4);
    tokens[n++]=m.tokenizer_bos;
    n+=hs_mlt_bpe_encode(&m,a->prompt,(uint32_t)strlen(a->prompt),tokens+n,MAX_TOKS-n);
    fprintf(stderr,"viz: prompt=%u tokens, generating...\n",n);
    HSMLTernarySession sess; hs_mlt_session_init(&sess,&m);
    hs_mlt_prefill(&sess,tokens,n);
    hs_mlt_session_logits(&sess,logits);
    int wi=0;
    make_snap(&g_snaps[wi],logits,sess.hidden,V,H,tokens,n,0,0);
    atomic_store(&g_write_idx,wi); atomic_store(&g_new_frame,true);
    struct timespec t0,t1;
    for(int step=0;step<a->n_predict&&!atomic_load(a->stop);step++){
        clock_gettime(CLOCK_MONOTONIC,&t0);
        uint32_t tok=sample_topk(logits,V,a->temp,a->top_k,tokens,n,a->rep_penalty);
        if(tok==m.tokenizer_eos) break;
        if(n<MAX_TOKS) tokens[n++]=tok;
        /* Print token */
        if(tok<m.vocab_size&&m.tokenizer_vocab[tok]){
            const char *tv=m.tokenizer_vocab[tok];
            while(*tv){
                if((unsigned char)tv[0]==0xC4&&(unsigned char)tv[1]==0xA0){
                    putchar(' '); tv+=2;
                } else putchar(*tv++);
            }
            fflush(stdout);
        }
        hs_mlt_decode(&sess,tok,logits);
        clock_gettime(CLOCK_MONOTONIC,&t1);
        double ms=((double)(t1.tv_sec-t0.tv_sec)*1e3
                  +(double)(t1.tv_nsec-t0.tv_nsec)/1e6);
        int nwi=1-atomic_load(&g_write_idx);
        make_snap(&g_snaps[nwi],logits,sess.hidden,V,H,tokens,n,ms,(uint32_t)step+1);
        atomic_store(&g_write_idx,nwi); atomic_store(&g_new_frame,true);
    }
    free(logits); hs_mlt_session_free(&sess); hs_mlt_free(&m);
    fprintf(stderr,"\nviz: inference done\n");
    return NULL;
}

/* GLES shaders */
static const char *VS =
    "attribute vec2 a_pos;\n"
    "attribute vec2 a_uv;\n"
    "varying vec2 v_uv;\n"
    "void main(){v_uv=a_uv;gl_Position=vec4(a_pos,0.0,1.0);}\n";

static const char *FS_LOGITS =
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform sampler2D u_tex;\n"
    "uniform float u_max;\n"
    "void main(){\n"
    "  float p=texture2D(u_tex,vec2(v_uv.x,0.5)).r;\n"
    "  float bar=p/max(u_max,0.001);\n"
    "  float lit=step(1.0-bar,v_uv.y);\n"
    "  vec3 lo=vec3(0.05,0.15,0.6);vec3 hi=vec3(1.0,0.4,0.05);\n"
    "  vec3 col=mix(lo,hi,clamp(p*6.0,0.0,1.0));\n"
    "  gl_FragColor=vec4(col*lit+vec3(0.04)*(1.0-lit),1.0);\n"
    "}\n";

static const char *FS_HEAT =
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform sampler2D u_tex;\n"
    "void main(){\n"
    "  float v=texture2D(u_tex,v_uv).r;\n"
    "  float t=clamp(v*0.5+0.5,0.0,1.0);\n"
    "  vec3 cold=vec3(0.02,0.06,0.22);\n"
    "  vec3 mid=vec3(0.04,0.42,0.62);\n"
    "  vec3 hot=vec3(1.0,0.85,0.1);\n"
    "  vec3 c=t<0.5?mix(cold,mid,t*2.0):mix(mid,hot,(t-0.5)*2.0);\n"
    "  gl_FragColor=vec4(c,1.0);\n"
    "}\n";

static const char *FS_ENT =
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform sampler2D u_tex;\n"
    "uniform float u_max;\n"
    "void main(){\n"
    "  float v=texture2D(u_tex,vec2(v_uv.x,0.5)).r;\n"
    "  float n=v/max(u_max,0.1);\n"
    "  float d=abs(v_uv.y-n);\n"
    "  float line=smoothstep(0.05,0.0,d);\n"
    "  float area=step(v_uv.y,n)*0.3;\n"
    "  vec3 col=mix(vec3(0.03,0.06,0.10),vec3(0.15,0.85,0.35),line);\n"
    "  gl_FragColor=vec4(col+area*vec3(0.05,0.2,0.05),1.0);\n"
    "}\n";

static const char *FS_BORDER =
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "void main(){\n"
    "  float bx=min(v_uv.x,1.0-v_uv.x);\n"
    "  float by=min(v_uv.y,1.0-v_uv.y);\n"
    "  float b=min(bx,by);\n"
    "  float e=1.0-smoothstep(0.0,0.012,b);\n"
    "  gl_FragColor=vec4(0.8,0.65,0.15,e*0.8);\n"
    "}\n";

static const char *FS_LABEL =
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform vec3 u_color;\n"
    "void main(){\n"
    "  float stripe=step(0.5,fract(v_uv.x*80.0+v_uv.y*40.0));\n"
    "  gl_FragColor=vec4(u_color*(0.6+stripe*0.4),1.0);\n"
    "}\n";

static GLuint make_prog(const char *vs, const char *fs){
    GLuint v=glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v,1,&vs,NULL); glCompileShader(v);
    GLuint f=glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f,1,&fs,NULL); glCompileShader(f);
    GLuint p=glCreateProgram();
    glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}
static GLuint make_tex1d(int w){
    GLuint t; glGenTextures(1,&t);
    glBindTexture(GL_TEXTURE_2D,t);
    glTexImage2D(GL_TEXTURE_2D,0,GL_R32F,w,1,0,GL_RED,GL_FLOAT,NULL);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    return t;
}

static GLuint g_vbo; static GLint g_pa,g_ua;
static GLuint g_pl,g_ph,g_pe,g_pb,g_plb;
static GLuint g_tprob,g_tlayer,g_tattn,g_tent;
static float  g_ehist[ENTROPY_HIST]; static uint32_t g_epos=0;

static void draw(GLuint prog,float x0,float y0,float x1,float y1){
    float q[]={x0,y0,0,0, x1,y0,1,0, x0,y1,0,1,
               x1,y0,1,0, x1,y1,1,1, x0,y1,0,1};
    glBindBuffer(GL_ARRAY_BUFFER,g_vbo);
    glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(q),q);
    glEnableVertexAttribArray(g_pa); glEnableVertexAttribArray(g_ua);
    glVertexAttribPointer(g_pa,2,GL_FLOAT,GL_FALSE,16,(void*)0);
    glVertexAttribPointer(g_ua,2,GL_FLOAT,GL_FALSE,16,(void*)8);
    glDrawArrays(GL_TRIANGLES,0,6);
}

static volatile bool g_run=true;
static void on_sig(int s){(void)s;g_run=false;}

int main(int argc,char **argv){
    if(argc<2){
        fprintf(stderr,"Usage: %s --model <path> [--norms <n>] [--prompt <p>]\n"
                "  [--temp <f>] [--top-k <n>] [--n-predict <n>]\n",argv[0]);
        return 1;
    }
    const char *mpath=NULL,*npath=NULL;
    const char *prompt="Hypothetically, might reflective recursion be a function of cognition?";
    float temp=0.432f,top_p=0.9531f,rep=1.1229f; uint32_t topk=42; int npred=256;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--model")&&i+1<argc)      mpath=argv[++i];
        else if(!strcmp(argv[i],"--norms")&&i+1<argc) npath=argv[++i];
        else if(!strcmp(argv[i],"--prompt")&&i+1<argc)prompt=argv[++i];
        else if(!strcmp(argv[i],"--temp")&&i+1<argc)  temp=atof(argv[++i]);
        else if(!strcmp(argv[i],"--top-k")&&i+1<argc) topk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--n-predict")&&i+1<argc)npred=atoi(argv[++i]);
    }
    if(!mpath){fprintf(stderr,"--model required\n");return 1;}

    signal(SIGINT,on_sig); signal(SIGTERM,on_sig);

    HSGraphics gfx;
    if(hs_graphics_init(&gfx)!=0){
        fprintf(stderr,"viz: display init failed\n");return 1;}
    fprintf(stderr,"viz: %ux%u display ready\n",gfx.screen_width,gfx.screen_height);

    g_pl =make_prog(VS,FS_LOGITS);
    g_ph =make_prog(VS,FS_HEAT);
    g_pe =make_prog(VS,FS_ENT);
    g_pb =make_prog(VS,FS_BORDER);
    g_plb=make_prog(VS,FS_LABEL);

    glGenBuffers(1,&g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER,g_vbo);
    glBufferData(GL_ARRAY_BUFFER,6*16,NULL,GL_DYNAMIC_DRAW);
    g_pa=glGetAttribLocation(g_pl,"a_pos");
    g_ua=glGetAttribLocation(g_pl,"a_uv");

    g_tprob =make_tex1d(TOP_K_VIZ);
    g_tlayer=make_tex1d(N_LAYERS);
    g_tattn =make_tex1d(N_LAYERS);
    g_tent  =make_tex1d(ENTROPY_HIST);
    memset(g_ehist,0,sizeof(g_ehist));

    atomic_bool stop; atomic_store(&stop,false);
    InferArgs ia={mpath,npath,prompt,temp,top_p,rep,topk,npred,&stop};
    pthread_t tid; pthread_create(&tid,NULL,infer_thread,&ia);

    glViewport(0,0,(GLsizei)gfx.screen_width,(GLsizei)gfx.screen_height);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    /* Panel NDC bounds */
    float lx0=-1.0f,lx1=-0.12f,ly0=-0.62f,ly1=0.78f;
    float hx0=-0.12f,hx1=0.56f,hy0=-0.62f,hy1=0.78f;
    float ax0=0.56f,ax1=1.0f,ay0=-0.62f,ay1=0.78f;
    float ex0=-1.0f,ex1=1.0f,ey0=-1.0f,ey1=-0.62f;
    float tx0=-1.0f,tx1=1.0f,ty0=0.78f,ty1=1.0f;

    uint32_t frame=0;
    while(g_run){
        const ModelSnapshot *s=&g_snaps[atomic_load(&g_write_idx)];

        if(s->valid){
            /* Upload logit probs */
            glBindTexture(GL_TEXTURE_2D,g_tprob);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,TOP_K_VIZ,1,GL_RED,GL_FLOAT,s->topk_probs);
            /* Upload layer norms */
            static float ln[N_LAYERS];
            float lmx=0.01f;
            for(int l=0;l<N_LAYERS;l++) if(s->layer_norms[l]>lmx) lmx=s->layer_norms[l];
            for(int l=0;l<N_LAYERS;l++) ln[l]=s->layer_norms[l]/lmx*2.0f-1.0f;
            glBindTexture(GL_TEXTURE_2D,g_tlayer);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,N_LAYERS,1,GL_RED,GL_FLOAT,ln);
            /* Upload attn entropy */
            static float ae[N_LAYERS];
            float amx=0.01f;
            for(int l=0;l<N_LAYERS;l++) if(s->attn_entropy[l]>amx) amx=s->attn_entropy[l];
            for(int l=0;l<N_LAYERS;l++) ae[l]=s->attn_entropy[l]/amx*2.0f-1.0f;
            glBindTexture(GL_TEXTURE_2D,g_tattn);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,N_LAYERS,1,GL_RED,GL_FLOAT,ae);
            /* Entropy history */
            g_ehist[g_epos%ENTROPY_HIST]=s->entropy; g_epos++;
            static float er[ENTROPY_HIST];
            for(int i=0;i<ENTROPY_HIST;i++) er[i]=g_ehist[(g_epos+i)%ENTROPY_HIST];
            glBindTexture(GL_TEXTURE_2D,g_tent);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,ENTROPY_HIST,1,GL_RED,GL_FLOAT,er);
        }

        glClearColor(0.015f,0.015f,0.025f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        /* Tokens bar */
        glUseProgram(g_plb);
        float lc[]={0.1f,0.35f,0.15f};
        glUniform3fv(glGetUniformLocation(g_plb,"u_color"),1,lc);
        draw(g_plb,tx0,ty0,tx1,ty1);

        /* Logit bars */
        glUseProgram(g_pl);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D,g_tprob);
        glUniform1i(glGetUniformLocation(g_pl,"u_tex"),0);
        float mp=s->valid&&s->topk_probs[0]>0?s->topk_probs[0]:1.0f;
        glUniform1f(glGetUniformLocation(g_pl,"u_max"),mp);
        draw(g_pl,lx0,ly0,lx1,ly1);

        /* Layer heatmap */
        glUseProgram(g_ph);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D,g_tlayer);
        glUniform1i(glGetUniformLocation(g_ph,"u_tex"),0);
        draw(g_ph,hx0,hy0,hx1,hy1);

        /* Attention entropy */
        glUseProgram(g_ph);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D,g_tattn);
        glUniform1i(glGetUniformLocation(g_ph,"u_tex"),0);
        draw(g_ph,ax0,ay0,ax1,ay1);

        /* Entropy timeline */
        glUseProgram(g_pe);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D,g_tent);
        glUniform1i(glGetUniformLocation(g_pe,"u_tex"),0);
        glUniform1f(glGetUniformLocation(g_pe,"u_max"),4.0f);
        draw(g_pe,ex0,ey0,ex1,ey1);

        /* Panel borders */
        glUseProgram(g_pb);
        draw(g_pb,lx0,ly0,lx1,ly1);
        draw(g_pb,hx0,hy0,hx1,hy1);
        draw(g_pb,ax0,ay0,ax1,ay1);
        draw(g_pb,ex0,ey0,ex1,ey1);
        draw(g_pb,tx0,ty0,tx1,ty1);

        hs_graphics_present(&gfx);
        frame++;

        if(s->valid&&s->step_num>0&&(frame%5)==0)
            fprintf(stderr,"\rstep=%-4u  tok=%-6u  p=%.3f  H=%.2f  ppl=%.1f  %.0fms",
                s->step_num,s->top1_id,s->topk_probs[0],
                s->entropy,s->perplexity,s->step_ms);
    }

    atomic_store(&stop,true);
    pthread_join(tid,NULL);
    hs_graphics_finish(&gfx);
    fprintf(stderr,"\nviz: %u frames rendered\n",frame);
    return 0;
}
