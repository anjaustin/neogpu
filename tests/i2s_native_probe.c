#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define I2S_QK 64u

typedef struct {
    char name[256];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
} TI;

static int rd_u32(FILE *f, uint32_t *v) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return -1;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}

static int rd_u64(FILE *f, uint64_t *v) {
    uint8_t b[8];
    if (fread(b, 1, 8, f) != 8) return -1;
    *v = (uint64_t)b[0] | ((uint64_t)b[1] << 8) | ((uint64_t)b[2] << 16) | ((uint64_t)b[3] << 24) |
         ((uint64_t)b[4] << 32) | ((uint64_t)b[5] << 40) | ((uint64_t)b[6] << 48) | ((uint64_t)b[7] << 56);
    return 0;
}

static char *rd_str(FILE *f) {
    uint64_t n;
    if (rd_u64(f, &n)) return NULL;
    char *s = malloc(n + 1);
    if (!s) return NULL;
    if (fread(s, 1, n, f) != n) {
        free(s);
        return NULL;
    }
    s[n] = '\0';
    return s;
}

static int skip(FILE *f, uint32_t t) {
    uint64_t n;
    uint32_t et;
    char *s;
    switch (t) {
        case 0: case 1: case 7: return fseek(f, 1, SEEK_CUR);
        case 2: case 3: return fseek(f, 2, SEEK_CUR);
        case 4: case 5: case 6: return fseek(f, 4, SEEK_CUR);
        case 10: case 11: case 12: return fseek(f, 8, SEEK_CUR);
        case 8:
            s = rd_str(f);
            free(s);
            return s ? 0 : -1;
        case 9:
            if (rd_u32(f, &et) || rd_u64(f, &n)) return -1;
            for (uint64_t i = 0; i < n; i++) if (skip(f, et)) return -1;
            return 0;
        default: return -1;
    }
}

static int find_ti(TI *tis, int n, const char *name) {
    for (int i = 0; i < n; i++) if (strcmp(tis[i].name, name) == 0) return i;
    return -1;
}

static int32_t dot_i2s_raw_scalar(const uint8_t *row, const int8_t *x, uint32_t cols) {
    int32_t acc = 0;
    uint32_t nblk = (cols + I2S_QK - 1) / I2S_QK;
    for (uint32_t bi = 0; bi < nblk; bi++) {
        const uint8_t *block = row + bi * (I2S_QK / 4);
        for (uint32_t j = 0; j < I2S_QK && bi * I2S_QK + j < cols; j++) {
            uint32_t k = bi * I2S_QK + j;
            uint32_t gi = j / 16;
            uint32_t gp = j % 16;
            uint8_t raw = (block[gp] >> (6 - 2 * gi)) & 0x3;
            acc += (int32_t) raw * (int32_t) x[k];
        }
    }
    return acc;
}

static int32_t dot_i2s_centered_scalar(const uint8_t *row, const int8_t *x, uint32_t cols) {
    int32_t acc = 0;
    uint32_t nblk = (cols + I2S_QK - 1) / I2S_QK;
    for (uint32_t bi = 0; bi < nblk; bi++) {
        const uint8_t *block = row + bi * (I2S_QK / 4);
        for (uint32_t j = 0; j < I2S_QK && bi * I2S_QK + j < cols; j++) {
            uint32_t k = bi * I2S_QK + j;
            uint32_t gi = j / 16;
            uint32_t gp = j % 16;
            uint8_t raw = (block[gp] >> (6 - 2 * gi)) & 0x3;
            int8_t w = (raw == 0) ? -1 : (raw == 1) ? 0 : +1;
            acc += (int32_t) w * (int32_t) x[k];
        }
    }
    return acc;
}

int main(void) {
    const char *path = "/home/ztflynn/001/neogpu/models/ggml-model-i2_s.gguf";
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("open");
        return 1;
    }

    uint32_t magic, ver;
    uint64_t tc, kvc;
    rd_u32(f, &magic);
    rd_u32(f, &ver);
    rd_u64(f, &tc);
    rd_u64(f, &kvc);
    for (uint64_t i = 0; i < kvc; i++) {
        char *k = rd_str(f);
        uint32_t t;
        rd_u32(f, &t);
        skip(f, t);
        free(k);
    }

    TI *tis = calloc(tc, sizeof(TI));
    for (uint64_t i = 0; i < tc; i++) {
        char *n = rd_str(f);
        strncpy(tis[i].name, n, 255);
        free(n);
        rd_u32(f, &tis[i].n_dims);
        for (uint32_t d = 0; d < tis[i].n_dims; d++) rd_u64(f, &tis[i].dims[d]);
        rd_u32(f, &tis[i].type);
        rd_u64(f, &tis[i].offset);
    }

    long data_start = ftell(f);
    data_start = (long) (((uint64_t) data_start + 31) & ~31ULL);

    int idx = find_ti(tis, (int) tc, "blk.0.attn_q.weight");
    if (idx < 0) {
        puts("tensor not found");
        return 2;
    }

    TI *ti = &tis[idx];
    uint32_t cols = (uint32_t) ti->dims[0];
    size_t row_bytes = cols / 4;
    uint8_t *row = malloc(row_bytes);
    fseek(f, data_start + (long) ti->offset, SEEK_SET);
    fread(row, 1, row_bytes, f);

    int8_t *x = malloc(cols);
    srand(1);
    for (uint32_t i = 0; i < cols; i++) x[i] = (rand() % 255) - 127;

    int32_t raw = dot_i2s_raw_scalar(row, x, cols);
    int32_t ctr = dot_i2s_centered_scalar(row, x, cols);

    printf("dot_raw=%d\n", raw);
    printf("dot_centered=%d\n", ctr);
    printf("ratio=%.6f\n", ctr != 0 ? (double) raw / (double) ctr : 0.0);
    return 0;
}
