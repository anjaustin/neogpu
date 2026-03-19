/*
 * hs_procmesh.h - Procedural Mesh Generation
 * 
 * Vertex compression:
 * - Position: 7+7+6 = 20 bits (range -64 to 63)
 * - UV: 4+4 = 8 bits (16x16 grid)
 * - Bone index: 4 bits
 * - Total: 4 bytes per vertex
 */

#ifndef HS_PROCMESH_H
#define HS_PROCMESH_H

#include "hs_core.h"
#include <stdint.h>

#ifndef i16
#define i16 int16_t
#endif

#define HS_PROC_MAX_VERTICES 256
#define HS_PROC_MAX_INDICES 512

typedef struct {
    i16 x, y, z;      // 7+7+6 bits, range -64 to 63 (scaled)
    u8 u, v;           // 4+4 bits, 0-15
    u8 bone;            // 4 bits, 0-15
} ProcVertex;

typedef struct {
    ProcVertex vertices[HS_PROC_MAX_VERTICES];
    u32 num_vertices;
    
    u16 indices[HS_PROC_MAX_INDICES];
    u32 num_indices;
    
    u8 bone_count;
} ProcMesh;

static void proc_mesh_init(ProcMesh* m) {
    memset(m, 0, sizeof(*m));
}

static void proc_mesh_add_vertex(ProcMesh* m, f32 x, f32 y, f32 z, f32 u, f32 v, u8 bone) {
    if (m->num_vertices >= HS_PROC_MAX_VERTICES) return;
    
    ProcVertex* vtx = &m->vertices[m->num_vertices++];
    
    // Quantize position to 7+7+6 bits
    vtx->x = (i16)(x * 64.0f);
    vtx->y = (i16)(y * 64.0f);
    vtx->z = (i16)(z * 32.0f);
    
    // Quantize UV to 4 bits each
    vtx->u = (u8)(u * 15.99f);
    vtx->v = (u8)(v * 15.99f);
    
    vtx->bone = bone & 0xF;
    
    if (bone >= m->bone_count) m->bone_count = bone + 1;
}

static void proc_mesh_add_quad(ProcMesh* m, f32 x, f32 y, f32 z, f32 w, f32 h, f32 u, f32 v, u8 bone) {
    u16 base = (u16)m->num_vertices;
    
    proc_mesh_add_vertex(m, x,     y,     z, u,       v,       bone);
    proc_mesh_add_vertex(m, x + w, y,     z, u + 0.99f, v,       bone);
    proc_mesh_add_vertex(m, x,     y + h, z, u,       v + 0.99f, bone);
    proc_mesh_add_vertex(m, x + w, y + h, z, u + 0.99f, v + 0.99f, bone);
    
    m->indices[m->num_indices++] = base + 0;
    m->indices[m->num_indices++] = base + 1;
    m->indices[m->num_indices++] = base + 2;
    m->indices[m->num_indices++] = base + 1;
    m->indices[m->num_indices++] = base + 3;
    m->indices[m->num_indices++] = base + 2;
}

static void proc_mesh_add_box(ProcMesh* m, f32 x, f32 y, f32 z, f32 w, f32 h, f32 d, u8 bone) {
    // Front
    proc_mesh_add_quad(m, x, y, z+d, w, h, 0, 0, bone);
    // Back
    proc_mesh_add_quad(m, x+w, y, z, -w, h, 1, 0, bone);
    // Top
    proc_mesh_add_quad(m, x, y+h, z+d, w, -d, 0, 0, bone);
    // Bottom
    proc_mesh_add_quad(m, x, y, z, w, d, 0, 0, bone);
    // Right
    proc_mesh_add_quad(m, x+w, y, z+d, -d, h, 0, 0, bone);
    // Left
    proc_mesh_add_quad(m, x, y, z, d, h, 0, 0, bone);
}

static void proc_mesh_add_sphere(ProcMesh* m, f32 cx, f32 cy, f32 cz, f32 r, int segs, u8 bone) {
    u16 base = (u16)m->num_vertices;
    
    for (int i = 0; i <= segs; i++) {
        f32 theta = (f32)i / segs * 3.14159f;
        for (int j = 0; j <= segs; j++) {
            f32 phi = (f32)j / segs * 3.14159f * 2.0f;
            
            f32 x = cx + r * sinf(theta) * cosf(phi);
            f32 y = cy + r * cosf(theta);
            f32 z = cz + r * sinf(theta) * sinf(phi);
            
            f32 u = (f32)i / segs;
            f32 v = (f32)j / segs;
            
            proc_mesh_add_vertex(m, x, y, z, u, v, bone);
        }
    }
    
    for (int i = 0; i < segs; i++) {
        for (int j = 0; j < segs; j++) {
            u16 a = base + i * (segs + 1) + j;
            u16 b = a + segs + 1;
            
            m->indices[m->num_indices++] = a;
            m->indices[m->num_indices++] = b;
            m->indices[m->num_indices++] = a + 1;
            m->indices[m->num_indices++] = b;
            m->indices[m->num_indices++] = b + 1;
            m->indices[m->num_indices++] = a + 1;
        }
    }
}

static void proc_mesh_mirror_x(ProcMesh* m) {
    u16 base = (u16)m->num_vertices;
    u32 orig_count = m->num_vertices;
    
    for (u32 i = 0; i < orig_count; i++) {
        ProcVertex v = m->vertices[i];
        v.x = -v.x;
        v.bone = (v.bone | 8); // Mirror bone
        m->vertices[m->num_vertices++] = v;
    }
    
    // Mirror indices
    for (u32 i = 0; i < m->num_indices; i += 3) {
        u16 a = m->indices[i];
        u16 b = m->indices[i + 1];
        u16 c = m->indices[i + 2];
        
        m->indices[m->num_indices++] = base + (a - base);
        m->indices[m->num_indices++] = base + (c - base);
        m->indices[m->num_indices++] = base + (b - base);
    }
}

static u32 proc_mesh_size_bytes(const ProcMesh* m) {
    return m->num_vertices * 4 + m->num_indices * 2;
}

#endif
