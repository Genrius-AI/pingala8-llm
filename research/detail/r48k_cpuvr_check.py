# R48k: independent scalar transliteration of r49_vr.cu cpu_vr (TR=1 NS=116 / TR=0) applied to a real blob's symbols, vs the
# build decoder (r48k.unpack_blob). Checks rows' int accumulations sum_t value_t . x_t on random int8 activations, exact.
import sys, os, struct, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r48k as K, r48_a1gpu as A1, torch
bd, name, nrow = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 64
T, tr = K.tables_from_blob(open(os.path.join(bd, "__vrk_tables__.blob"), "rb").read()); b = open(os.path.join(bd, name + ".blob"), "rb").read()
if b[:4] == b"VRK1": b = K.relayout(b)                         # check the kernel (tile-block-major) layout
h, Yq = K.unpack_blob(b, T)
# parse raw symbols again (independent of unpack's value path)
mg, rows, n, trh, pb, wm = struct.unpack_from(K.HDR, b, 0); assert mg == b"VRK2"; o = struct.calcsize(K.HDR) + 2 * n + (rows * pb + 7) // 8 + 2 * rows * (n // 256)
nbk = n // 256; tiles = rows // 32; meta = np.frombuffer(b[o:o + 4 * tiles * nbk], np.uint32).reshape(tiles, nbk); o += 4 * tiles * nbk; stream = np.frombuffer(b[o:], np.uint8)
Ai = T["A"].numpy().astype(np.int64); Bi = (T["A"] if tr else T["B"]).numpy().astype(np.int64); ci = T["c"].numpy().astype(np.int64)
Cc = Ai + ci                                                     # coset table cC = A + c (all stages)
rng = np.random.default_rng(0); x = rng.integers(-127, 128, size=n).astype(np.int64); bad = 0
prm = h["prm"].numpy(); Ss = h["Sb"].numpy()[prm]               # sorted-row order
for rs in range(0, rows, max(1, rows // nrow)):
    t_ = rs // 32; acc_ref = 0.0; acc_dec = 0.0; l1 = 0.0
    for bk in range(nbk):
        c = int(meta[t_, bk] >> 29); wo = int(meta[t_, bk] & 0x1FFFFFFF) * 4
        if c == 0: continue
        W1, W2, W3, S1, S2 = K.W1[c], K.W2[c], K.W3[c], K.S1[c], K.S2[c]; nw = W1 + W2 + W3; rowbytes = K.BPB[c]
        start = wo + (rs % 32) * rowbytes                         # rows of a tile have equal byte counts (classes per tile)
        bits = np.unpackbits(stream[start:start + rowbytes]); acc = 0
        for s in range(2):                                        # two 128-col slices per block; trellis state reset per slice
            ust = 0; bit = []
            for t in range(32):
                g = 32 * s + t; sy = int("".join(map(str, bits[g * nw:(g + 1) * nw])), 2)
                f1 = sy & ((1 << W1) - 1); f2 = (sy >> W1) & ((1 << W2) - 1); f3 = (sy >> (W1 + W2)) & ((1 << W3) - 1)
                bit.append((f1 >> (W1 - 1)) & 1); U1 = Ai; U2 = Bi
                if tr:
                    hh = bin(ust & 0x2).count("1") & 1; uu = bit[t] ^ (bin(ust & 0x9).count("1") & 1); ust = ((ust << 1) | uu) & 15
                    U1 = U2 = (Cc if hh else Ai)
                xs = x[bk * 256 + 4 * g: bk * 256 + 4 * g + 4]
                v = int(U1[f1] @ xs) << S1
                if W2: v += int(U2[f2] @ xs) << S2
                if W3: v += int(U2[f3] @ xs)
                acc += v
        acc_ref += acc * float(Ss[rs, bk]); l1 += abs(acc * float(Ss[rs, bk]))
    r_nat = prm[rs]; acc_dec = float((Yq[r_nat].double() * torch.as_tensor(x, dtype=torch.float64)).sum())
    if abs(acc_ref - acc_dec) > 1e-6 * max(1e-30, l1): bad += 1; print("DIFF row", rs, acc_ref, acc_dec)
print(f"CPUVR_CHECK {name} tr {tr} rows checked {len(range(0, rows, max(1, rows // nrow)))} bad {bad}")
