// b3p_vr3_test.c (R59 B3+): CPU test of the loader's VR3 code in llama-vr3.h (vrn3_rows, vrn3_expand), the exact functions
// llama-model-loader.cpp calls. Input: VR3 file tensor bytes and the expected v2 in-memory bytes (r51_transcode.transcode of the
// natural-order blob), both written by b3p_native.py dump. PASS = expansion byte-identical, every output byte written (dst pre-filled 0xA5).
// build: cc -std=c11 -O2 -Wall -Wextra -I research/kernels/llama_f44/src/ggml/include -I research/kernels/llama_f44/src/src research/detail/b3p_vr3_test.c -o work/b3p/vr3_test
// usage: vr3_test K file.vr3 file.v2 [K file.vr3 file.v2 ...]
#include <stdio.h>
#include <stdlib.h>
#include "llama-vr3.h"

static uint8_t * rd(const char * p, size_t * n) {
    FILE * f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
    fseek(f, 0, SEEK_END); *n = (size_t) ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t * b = (uint8_t *) malloc(*n); if (fread(b, 1, *n, f) != *n) { perror(p); exit(2); } fclose(f); return b;
}

int main(int argc, char ** argv) {
    int fails = 0;
    for (int a = 1; a + 2 < argc; a += 3) {
        size_t K = strtoull(argv[a], 0, 10), fb, eb; uint8_t * f = rd(argv[a + 1], &fb), * e = rd(argv[a + 2], &eb);
        uint32_t w515; memcpy(&w515, f + 2060, 4); size_t kb = 0; const size_t rows = vrn3_rows(fb, K, w515, &kb);
        int ok = rows > 0 && kb == eb;
        if (ok) {
            uint8_t * d = (uint8_t *) malloc(kb); memset(d, 0xA5, kb); vrn3_expand(f, fb, rows, K, d);
            ok = memcmp(d, e, kb) == 0; free(d);
        }
        // negative cases: the size must match the class exactly, the tag must be VR3, K must divide the tile count
        const size_t n1 = vrn3_rows(fb - 4, K, w515, 0), n2 = vrn3_rows(fb, K, w515 ^ 0x100u, 0), n3 = vrn3_rows(fb, K, w515 & ~7u, 0);
        printf("VR3TEST %s %s K %zu rows %zu class %u fbytes %zu kbytes %zu (expected %zu); reject size-4 %s, bad tag %s, class 0 %s\n",
               ok ? "PASS" : "FAIL", argv[a + 1], K, rows, w515 & 7u, fb, kb, eb, n1 ? "NO" : "yes", n2 ? "NO" : "yes", n3 ? "NO" : "yes");
        fails += !ok || n1 || n2 || n3; free(f); free(e);
    }
    printf("VR3TEST SUMMARY %s (%d failing)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
