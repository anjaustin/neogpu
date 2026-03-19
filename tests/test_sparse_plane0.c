/*
 * hs_ml_ternary_sparse.c - Sparse plane-0 encoding for lm_head
 * 
 * Plane-0 is 96.5% zeros. Store as:
 *   - Bitmask: 128256 bits = 16KB (one bit per vocab row)
 *   - Payload: Only non-zero rows, each row is K/4 bytes (640 bytes)
 * 
 * This gives ~2.7MB vs 78MB = 28x reduction for plane-0.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifndef HS_ML_INFER_H
#include "hs_ml_infer.h"
#endif

/*
 * Sparse plane-0 structure
 * 
 * Format:
 *   - num_nonzero: uint32_t (4 bytes)
 *   - bitmask: num_vocab bits (padded to byte boundary)
 *   - payload: num_nonzero * row_bytes bytes
 */
typedef struct {
    uint32_t num_vocab;        /* Total vocab size */
    uint32_t row_bytes;        /* Bytes per row (K/4) */
    uint32_t num_nonzero;      /* Number of non-zero rows */
    uint8_t *bitmask;         /* num_vocab bits, 1=nonzero */
    uint8_t *payload;         /* num_nonzero * row_bytes */
} HSSparsePlane0;

/*
 * Encode plane-0 to sparse format
 * 
 * Input: P0 - packed I2_S plane (num_vocab * row_bytes)
 * Output: sparse - allocated sparse structure
 */
static int encode_sparse_plane0(HSSparsePlane0 *sparse,
                               const uint8_t *P0,
                               uint32_t num_vocab,
                               uint32_t row_bytes) {
    sparse->num_vocab = num_vocab;
    sparse->row_bytes = row_bytes;
    
    /* First pass: count non-zero rows */
    uint32_t nonzero_count = 0;
    for (uint32_t v = 0; v < num_vocab; v++) {
        const uint8_t *row = P0 + (size_t)v * row_bytes;
        int is_nonzero = 0;
        for (uint32_t b = 0; b < row_bytes; b++) {
            if (row[b] != 0x00) {  /* 0x00 = all 4 values are 0 in I2_S */
                is_nonzero = 1;
                break;
            }
        }
        if (is_nonzero) nonzero_count++;
    }
    sparse->num_nonzero = nonzero_count;
    
    /* Allocate */
    uint32_t bitmask_bytes = (num_vocab + 7) / 8;
    size_t bitmask_size = bitmask_bytes * sizeof(uint8_t);
    size_t payload_size = (size_t)nonzero_count * row_bytes * sizeof(uint8_t);
    
    sparse->bitmask = malloc(bitmask_size);
    sparse->payload = malloc(payload_size);
    if (!sparse->bitmask || !sparse->payload) {
        free(sparse->bitmask);
        free(sparse->payload);
        return -1;
    }
    memset(sparse->bitmask, 0, bitmask_size);
    
    /* Second pass: encode */
    uint32_t payload_idx = 0;
    for (uint32_t v = 0; v < num_vocab; v++) {
        const uint8_t *row = P0 + (size_t)v * row_bytes;
        int is_nonzero = 0;
        for (uint32_t b = 0; b < row_bytes; b++) {
            if (row[b] != 0x00) {
                is_nonzero = 1;
                break;
            }
        }
        
        /* Set bitmask */
        if (is_nonzero) {
            sparse->bitmask[v / 8] |= (1 << (v % 8));
            /* Copy payload */
            memcpy(sparse->payload + (size_t)payload_idx * row_bytes, row, row_bytes);
            payload_idx++;
        }
    }
    
    printf("sparse_plane0: %u/%u rows non-zero (%.1f%%)\n",
           nonzero_count, num_vocab, 100.0 * nonzero_count / num_vocab);
    printf("sparse_plane0: %u bytes (vs %u bytes) = %.1fx reduction\n",
           bitmask_bytes + nonzero_count * row_bytes,
           num_vocab * row_bytes,
           (float)(num_vocab * row_bytes) / (bitmask_bytes + nonzero_count * row_bytes));
    
    return 0;
}

static void free_sparse_plane0(HSSparsePlane0 *sparse) {
    free(sparse->bitmask);
    free(sparse->payload);
    sparse->bitmask = NULL;
    sparse->payload = NULL;
}

/*
 * Test sparse encoding on real model
 */
int main(int argc, char **argv) {
    const char *model_path = "models/bitnet-2b4t-i2s.gguf";
    const char *norms_path = "models/norms_v2.bin";
    
    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) norms_path = argv[2];
    
    printf("Loading model...\n");
    HSMLTernary m;
    hs_mlt_init(&m);
    
    if (hs_mlt_load_gguf(&m, model_path) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    if (norms_path) hs_mlt_load_norms_sidecar(&m, norms_path);
    
    /* Encode lm_head to ternary planes */
    hs_mlt_lmhead_encode(&m);
    
    if (!m.use_trit_lmhead || !m.lm_head_planes[0]) {
        fprintf(stderr, "lm_head not encoded to planes\n");
        return 1;
    }
    
    uint32_t V = m.vocab_size;
    uint32_t H = m.hidden_size;
    uint32_t row_bytes = H / 4;  /* I2_S packs 4 values per byte */
    
    printf("\n=== Encoding plane-0 to sparse format ===\n");
    printf("V=%u, H=%u, row_bytes=%u\n", V, H, row_bytes);
    
    /* Check per-byte sparsity */
    const uint8_t *P0 = m.lm_head_planes[0];
    uint64_t total_bytes = (uint64_t)V * row_bytes;
    uint64_t zero_bytes = 0;
    for (uint32_t v = 0; v < V; v++) {
        const uint8_t *row = P0 + (size_t)v * row_bytes;
        for (uint32_t b = 0; b < row_bytes; b++) {
            if (row[b] == 0x00) zero_bytes++;
        }
    }
    printf("Plane-0 byte sparsity: %lu/%lu = %.1f%% zero\n",
           zero_bytes, total_bytes, 100.0 * zero_bytes / total_bytes);
    
    HSSparsePlane0 sparse;
    if (encode_sparse_plane0(&sparse, m.lm_head_planes[0], V, row_bytes) != 0) {
        fprintf(stderr, "Failed to encode sparse plane\n");
        return 1;
    }
    
    /* Verify by reconstructing and comparing */
    printf("\n=== Verifying reconstruction ===\n");
    uint32_t max_diff_row = 0;
    uint32_t max_diff_byte = 0;
    uint8_t max_diff_orig = 0;
    uint8_t max_diff_recon = 0;
    
    for (uint32_t v = 0; v < V; v++) {
        int is_nonzero = sparse.bitmask[v / 8] & (1 << (v % 8));
        const uint8_t *orig = m.lm_head_planes[0] + (size_t)v * row_bytes;
        
        if (is_nonzero) {
            /* Find this row's position in payload */
            uint32_t payload_pos = 0;
            for (uint32_t vv = 0; vv < v; vv++) {
                if (sparse.bitmask[vv / 8] & (1 << (vv % 8))) {
                    payload_pos++;
                }
            }
            const uint8_t *recon = sparse.payload + (size_t)payload_pos * row_bytes;
            
            for (uint32_t b = 0; b < row_bytes; b++) {
                if (orig[b] != recon[b]) {
                    if (v < max_diff_row || (v == max_diff_row && b < max_diff_byte)) {
                        max_diff_row = v;
                        max_diff_byte = b;
                        max_diff_orig = orig[b];
                        max_diff_recon = recon[b];
                    }
                }
            }
        } else {
            /* Should be all 0x00 */
            for (uint32_t b = 0; b < row_bytes; b++) {
                if (orig[b] != 0x00) {
                    printf("WARNING: zero row %u byte %u has value 0x%02x\n", v, b, orig[b]);
                }
            }
        }
    }
    
    if (max_diff_row == 0 && max_diff_byte == 0) {
        printf("Verification: PASS (perfect reconstruction)\n");
    } else {
        printf("Verification: diff at row %u byte %u: orig=0x%02x recon=0x%02x\n",
               max_diff_row, max_diff_byte, max_diff_orig, max_diff_recon);
    }
    
    free_sparse_plane0(&sparse);
    hs_mlt_free(&m);
    
    printf("\nDone.\n");
    return 0;
}
