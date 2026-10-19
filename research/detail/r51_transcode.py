# R51 item 3: VRK2 blob (r48k.py pack_blob + relayout) -> fast-kernel tensor bytes ("D layout") + side tensors.
# Kernel layout = research/kernels/llama_f44/src/ggml/src/ggml-cuda/vr49.cuh:8-15 (in-tensor offsets), :405-440 (vr_prepare),
# encoder research/kernels/r49_vr.cu:523-551 (putb, encode_vr TRL 2), reader vr49.cuh:207-257 (vr_bits, vr_f1, vr_wint, vr_wscale).
#   words 0-255 A (int8x4) | 256-511 B (= A, VRii-C) | 512 c | 513 NS (108 = 8 gana states, Ungerboeck ff labels) | 514 TRL (2) | 515 loader 0
#   M  @2064: u32 per tile-block, index g*nb + b = D word offset | class << 29     (identical to the VRK2 meta words)
#   S  @2064 + 4nt: fp16 [b][r], r = sorted row
#   D  @align16(2064 + 68 nt): per (g, b): [hs][word pair i>>1][lane j][i&1]; lane j step k holds chunk t = (j + k) & 31.
# Side tensors (not in the kernel tensor): prm (sorted -> original row), c16 (fp16 per input column), wmax.
# Equivalence: numpy port of r49_vr.cu cpu_vr (563-591) over the transcoded bytes vs r48k.unpack_blob Yq*S, bit-equal fp32, all rows.
# usage: r51_transcode.py test BLOBDIR name... | dump BLOBDIR name OUTDIR   (dump writes kernel bytes for the C++ host check)
import sys, os, json, struct, math, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r48k as K, r48_a1gpu as A1
W1 = [0, 4, 6, 8, 8, 8, 8, 8]; W2 = [0, 0, 0, 0, 2, 4, 8, 8]; W3 = [0, 0, 0, 0, 0, 0, 0, 8]     # vr49.cuh:17-19
SH1 = [0, 0, 0, 0, 2, 3, 2, 4]; SH2 = [0, 0, 0, 0, 0, 0, 0, 2]                                   # vr49.cuh:20-21 (R51 shift fix)
NW = [W1[c] + W2[c] + W3[c] for c in range(8)]
HDR = "<4sIIBBe"; NS_PINGALA8 = 108; TRL = 2

def parse_vrk2(b):
    mg, rows, n, tr, pb, wm = struct.unpack_from(HDR, b, 0); assert mg == b"VRK2", mg
    o = struct.calcsize(HDR); tiles, nbk = rows // 32, n // 256
    c16 = np.frombuffer(b, np.float16, n, o); o += 2 * n
    prm = A1._unpack(b[o:o + (rows * pb + 7) // 8], rows, pb).astype(np.int64); o += (rows * pb + 7) // 8
    S = np.frombuffer(b, np.float16, rows * nbk, o).reshape(rows, nbk); o += 2 * rows * nbk
    meta = np.frombuffer(b, np.uint32, tiles * nbk, o).reshape(tiles, nbk); o += 4 * tiles * nbk
    return dict(rows=rows, n=n, tr=tr, pb=pb, wmax=wm, c16=c16, prm=prm, S=S, meta=meta, stream=np.frombuffer(b, np.uint8, len(b) - o, o))

def pack_lsb(F, W):
    """F (..., 32) fields of width W (step order) -> (..., W) u32 words, field k at bit W*k (LSB-first, may straddle words) = putb."""
    bits = ((F[..., None] >> np.arange(W, dtype=np.uint64)) & 1).astype(np.uint8).reshape(*F.shape[:-1], 32 * W)
    return np.packbits(bits, axis=-1, bitorder="little").view("<u4")

def transcode(blob, tab):
    """blob: VRK2 bytes; tab: __vrk_tables__ bytes (tr 2). Returns (kernel bytes, side dict)."""
    assert tab[:4] == b"VRKT" and tab[4] == 2, "need the 8-state (Pingala) table blob"
    Atab = tab[5:5 + 1024]; Btab = tab[5 + 1024:5 + 2048]; cs = tab[5 + 2048:5 + 2052]
    assert list(tab[5 + 2052:5 + 2060]) == SH1 and list(tab[5 + 2060:5 + 2068]) == SH2, "table shifts != kernel shifts"
    p = parse_vrk2(blob); rows, n = p["rows"], p["n"]; tiles, nbk = rows // 32, n // 256; nt = tiles * nbk
    cs_t = (p["meta"] >> 29).astype(np.int64); woff = (p["meta"] & 0x1FFFFFFF).astype(np.int64)
    ndw = int(sum(64 * NW[c] for c in cs_t.ravel()))
    assert 4 * ndw == len(p["stream"]), (4 * ndw, len(p["stream"]))
    D = np.zeros(ndw, np.uint32); st = p["stream"]
    j = np.arange(32); kk = np.arange(32); tsel = (j[:, None] + kk[None, :]) & 31                     # (lane, step) -> chunk
    for c in range(1, 8):
        tt, bb = np.nonzero(cs_t == c)
        if len(tt) == 0: continue
        m = len(tt); BPB = 8 * NW[c]                                                                   # bytes per (row, block) = 64 NW / 8
        src = (4 * woff[tt, bb])[:, None, None] + (np.arange(32)[None, :, None] * BPB) + np.arange(BPB)[None, None, :]
        sy = A1._unpack(st[src.ravel()].tobytes(), m * 32 * 64, NW[c]).reshape(m, 32, 2, 32)        # (m, lane j, hs, chunk t) MSB-first symbols
        sy = np.take_along_axis(sy, np.broadcast_to(tsel[None, :, None, :], (m, 32, 2, 32)), axis=3)  # -> (m, j, hs, step k)
        f1 = sy & np.uint64((1 << W1[c]) - 1); f2 = (sy >> np.uint64(W1[c])) & np.uint64((1 << W2[c]) - 1); f3 = (sy >> np.uint64(W1[c] + W2[c])) & np.uint64((1 << W3[c]) - 1)
        if W1[c] < 8:                                                                                  # TRL 2, W1 < 8: MSB word + (W1-1)-bit fields
            msb = (f1 >> np.uint64(W1[c] - 1)) & np.uint64(1)
            w0 = (msb << np.arange(32, dtype=np.uint64)).sum(-1).astype(np.uint32)[..., None]
            parts = [w0, pack_lsb(f1 & np.uint64((1 << (W1[c] - 1)) - 1), W1[c] - 1)]
        else: parts = [pack_lsb(f1, 8)]
        if W2[c]: parts.append(pack_lsb(f2, W2[c]))
        if W3[c]: parts.append(pack_lsb(f3, W3[c]))
        wl = np.concatenate(parts, -1); assert wl.shape == (m, 32, 2, NW[c])                          # (m, j, hs, i)
        reg = wl.reshape(m, 32, 2, NW[c] // 2, 2).transpose(0, 2, 3, 1, 4).reshape(m, 64 * NW[c])      # [hs][i>>1][j][i&1]
        D[(woff[tt, bb][:, None] + np.arange(64 * NW[c])[None, :]).ravel()] = reg.ravel()
    hdr = bytearray(2064); hdr[0:1024] = Atab; hdr[1024:2048] = Btab; hdr[2048:2052] = cs
    struct.pack_into("<III", hdr, 2052, NS_PINGALA8, TRL, 0)
    offS = 2064 + 4 * nt; offD = (2064 + 68 * nt + 15) & ~15
    Sk = np.ascontiguousarray(p["S"].T)                                                                # [b][r]
    out = bytes(hdr) + p["meta"].astype("<u4").tobytes() + Sk.tobytes() + bytes(offD - (offS + 2 * rows * nbk)) + D.tobytes()
    assert len(out) == offD + 4 * ndw
    side = dict(rows=rows, n=n, prm=p["prm"], c16=p["c16"], wmax=float(p["wmax"]), offS=offS, offD=offD, nt=nt)
    return out, side

def kbytes_size(rows, n, payload_words):
    nt = rows // 32 * (n // 256); return ((2064 + 68 * nt + 15) & ~15) + 4 * payload_words

# ---------------------------------------------------------------- numpy port of r49_vr.cu cpu_vr (563-591) on kernel bytes
def cpu_vr_port(kb, rows, n):
    """Per-weight value v * S (fp32), SORTED rows, read from the kernel bytes only (vr_wint addressing vr49.cuh:207-253,
    value/trellis of cpu_vr r49_vr.cu:563-591: U = h ? C : A with C = int8(A + c) (mk_coset r49_vr.cu:624-625),
    hh = par(ust & m0), u = z1 ^ par(ust & m1), ust = (ust << 1 | u) & 15, NS 108: m1 0x5, m0 0x2, reset per 128-col slice)."""
    w32 = np.frombuffer(kb, "<u4"); A = w32[0:256]; cs = int(w32[512]); ns = int(w32[513]); trl = int(w32[514]); assert ns == 108 and trl == 2
    m1, m0 = 0x5, 0x2
    A8 = A.view(np.int8).reshape(256, 4).astype(np.int64); c8 = np.frombuffer(struct.pack("<I", cs), np.int8).astype(np.int64)
    C8 = ((A8 + c8[None, :] + 128) % 256) - 128                                                       # int8 wrap as g_cC
    nb = n // 256; ng = rows // 32; nt = ng * nb
    M = np.frombuffer(kb, "<u4", nt, 2064).reshape(ng, nb); S = np.frombuffer(kb, np.float16, nb * rows, 2064 + 4 * nt).reshape(nb, rows)
    D = np.frombuffer(kb, "<u4", offset=(2064 + 68 * nt + 15) & ~15)
    out = np.zeros((rows, n), np.float32); cl = (M >> 29).astype(np.int64)
    jj = np.arange(32); tk = (np.arange(32)[None, :] - jj[:, None]) & 31                              # (lane j, chunk t) -> step k
    for c in range(1, 8):
        gg, bb = np.nonzero(cl == c)
        if len(gg) == 0: continue
        m = len(gg); nw = NW[c]; base = (M[gg, bb] & 0x1FFFFFFF).astype(np.int64)
        wi = np.arange(nw)                                                                             # word wi of lane j, half hs:
        addr = base[:, None, None, None] + np.arange(2)[None, :, None, None] * nw * 32 + (wi[None, None, None, :] >> 1) * 64 + jj[None, None, :, None] * 2 + (wi[None, None, None, :] & 1)
        wd = D[addr].astype(np.uint64)                                                                 # (m, hs, j, nw)
        bits = ((wd[..., None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).reshape(m, 2, 32, 32 * nw)
        def field(wb, W):                                                                              # bits W*k.. from word wb, (m,2,32 lanes,32 steps)
            k = np.arange(32); idx = 32 * wb + W * k[:, None] + np.arange(W)[None, :]                   # (32, W)
            return (bits[..., idx] << np.arange(W, dtype=np.uint64)).sum(-1)
        if W1[c] < 8: f1 = field(1, W1[c] - 1) | (field(0, 1) << np.uint64(W1[c] - 1))
        else: f1 = field(0, 8)
        rot = lambda f: np.take_along_axis(f, np.broadcast_to(tk, f.shape), 3).astype(np.int64)        # step order -> chunk order
        F1 = rot(f1); F2 = rot(field(W1[c], W2[c])) if W2[c] else None; F3 = rot(field(W1[c] + W2[c], W3[c])) if W3[c] else None
        z = (F1 >> (W1[c] - 1)) & 1; ust = np.zeros((m, 2, 32), np.int64); val = np.zeros((m, 2, 32, 32, 4), np.int64)
        for t in range(32):
            hh = ((ust & m0) > 0)                                                                      # popcount(ust & 0x2) & 1
            p1 = (ust & 1) ^ ((ust >> 2) & 1)                                                          # popcount(ust & 0x5) & 1
            ust = ((ust << 1) | (z[..., t] ^ p1)) & 15
            U = lambda F: np.where(hh[..., None], C8[F[..., t]], A8[F[..., t]])
            v = U(F1) << SH1[c]
            if W2[c]: v = v + (U(F2) << SH2[c])
            if W3[c]: v = v + U(F3)
            val[:, :, :, t] = v
        vf = val.reshape(m, 2, 32, 128).transpose(0, 2, 1, 3).reshape(m, 32, 256).astype(np.float32)   # (m, lane, 256 cols)
        rr = 32 * gg[:, None] + jj[None, :]                                                            # sorted rows (m, 32)
        sc = S[bb[:, None], rr].astype(np.float32)
        out[rr[:, :, None], (256 * bb)[:, None, None] + np.arange(256)[None, None, :]] = vf * sc[:, :, None]
    return out

def scalar_wint(kb, rows, n, r, col):
    """Literal per-weight transliteration of vr49.cuh vr_wint (V 4, ns >= 100 branch) + vr_wscale; returns fp32 v*S."""
    w = np.frombuffer(kb, "<u4"); nb = n // 256; nt = (rows >> 5) * nb; cs = int(w[512]); ns = int(w[513]); trl = int(w[514])
    M = np.frombuffer(kb, "<u4", nt, 2064); D = np.frombuffer(kb, "<u4", offset=(2064 + nt * 68 + 15) & ~15)
    g, j, b, hs, t, e = r >> 5, r & 31, col >> 8, (col >> 7) & 1, (col & 127) >> 2, col & 3
    mt = int(M[g * nb + b]); c = mt >> 29
    if not c: return np.float32(0)
    nw = NW[c]; k = (t - j) & 31; base = (mt & 0x1FFFFFFF) + hs * nw * 32
    def vb(wb, pos, W):
        v = 0
        for q in range(W):
            ps = pos + q; wi = wb + ps // 32; v |= ((int(D[base + (wi >> 1) * 64 + j * 2 + (wi & 1)]) >> (ps % 32)) & 1) << q
        return v
    def f1_of(tq):
        kq = (tq - j) & 31
        if trl == 1 or W1[c] < 8: return vb(1, (W1[c] - 1) * kq, W1[c] - 1) | (vb(0, kq, 1) << (W1[c] - 1))
        return vb(0, W1[c] * kq, W1[c])
    f1 = f1_of(t); f2 = vb(W1[c], W2[c] * k, W2[c]) if W2[c] else 0; f3 = vb(W1[c] + W2[c], W3[c] * k, W3[c]) if W3[c] else 0
    u = []; ug = lambda i: u[i] if i >= 0 else 0
    for q in range(t):
        z = (f1_of(q) >> (W1[c] - 1)) & 1; u.append(z ^ ug(q - 1) ^ ug(q - 3) if ns == 108 else None)
    h = u[t - 2] if t - 2 >= 0 else 0
    ce = int(np.int8(np.uint8((cs >> (8 * e)) & 0xFF)))
    U1 = lambda f: int(np.int8(np.uint8((int(w[f]) >> (8 * e)) & 0xFF))) + (ce if h else 0)
    v = U1(f1) * (1 << SH1[c])
    if W2[c]: v += U1(f2) * (1 << SH2[c])
    if W3[c]: v += U1(f3)
    Sv = np.frombuffer(kb, np.float16, 1, 2064 + nt * 4 + 2 * (b * rows + r))[0]
    return np.float32(v) * np.float32(Sv)

def ref_sorted(blob, T):
    h, Yq = K.unpack_blob(blob, T); return h, Yq[h["prm"]].numpy()                                     # r48k Yq*S, sorted rows

def cmd_test(bd, names):
    tab = open(os.path.join(bd, "__vrk_tables__.blob"), "rb").read(); T, tr = K.tables_from_blob(tab); assert tr == 2 and K.NSC[0] == 8
    st = {s["name"]: s for s in json.load(open(os.path.join(os.path.dirname(bd.rstrip("/")), "stats_vrk.json"))) if s["v"] == "vrc"}
    allok = True; rng = np.random.default_rng(51)
    for nm in names:
        t0 = time.time(); blob = open(os.path.join(bd, nm + ".blob"), "rb").read()
        kb, side = transcode(blob, tab); h, ref = ref_sorted(blob, T)
        got = cpu_vr_port(kb, side["rows"], side["n"])
        eq = got.view(np.uint32) == ref.view(np.uint32); nbad = int((~eq).sum()); badrows = int((~eq).any(1).sum())
        # literal scalar vr_wint on 3000 random weights
        R = rng.integers(0, side["rows"], 3000); Cc = rng.integers(0, side["n"], 3000)
        sbad = sum(scalar_wint(kb, side["rows"], side["n"], int(r), int(c_)).view(np.uint32) != ref[r, c_].view(np.uint32) for r, c_ in zip(R, Cc))
        meta_same = np.array_equal(np.frombuffer(kb, "<u4", side["nt"], 2064), parse_vrk2(blob)["meta"].ravel())
        hist = np.bincount((np.frombuffer(kb, "<u4", side["nt"], 2064) >> 29).astype(np.int64), minlength=8).tolist()
        ok = nbad == 0 and sbad == 0; allok &= ok
        print(f"EQ {nm} slot {st[nm]['slot']} rows {side['rows']} n {side['n']} kbytes {len(kb)} blob {len(blob)} cls_hist {hist} "
              f"port_vs_r48k: bad_weights {nbad} bad_rows {badrows}/{side['rows']} | scalar vr_wint 3000 samples bad {sbad} | M==VRK2 meta {meta_same} | "
              f"{'PASS' if ok else 'FAIL'} t {time.time() - t0:.0f}s", flush=True)
        if nbad:
            r_, c_ = np.argwhere(~eq)[0]; print(f"  first bad sorted row {r_} col {c_} got {got[r_, c_]!r} ref {ref[r_, c_]!r} class {int(np.frombuffer(kb, '<u4', side['nt'], 2064)[(r_ // 32) * (side['n'] // 256) + c_ // 256] >> 29)}")
    print(f"RESULT {'ALL PASS' if allok else 'FAIL'} ({len(names)} tensors)")

def cmd_dump(bd, nm, od):
    tab = open(os.path.join(bd, "__vrk_tables__.blob"), "rb").read(); T, _ = K.tables_from_blob(tab)
    blob = open(os.path.join(bd, nm + ".blob"), "rb").read(); kb, side = transcode(blob, tab); h, ref = ref_sorted(blob, T)
    if os.environ.get("SHARD_STEM"):                                                                   # take the kernel bytes from the native shards instead
        import glob; sys.path.insert(0, "~/repo/llama.cpp/gguf-py"); from gguf import GGUFReader
        for sp in sorted(glob.glob(os.environ["SHARD_STEM"] + "-0000*-of-*.gguf")):
            t = {x.name: x for x in GGUFReader(sp).tensors}.get(nm)
            if t is not None: kb = bytes(np.asarray(t.data).view(np.uint8)); print(f"DUMP source shard {os.path.basename(sp)} type {t.tensor_type.name}"); break
        else: raise SystemExit("tensor not in shards")
    os.makedirs(od, exist_ok=True); open(os.path.join(od, nm + ".vr"), "wb").write(kb); ref.astype("<f4").tofile(os.path.join(od, nm + ".ref"))
    print(f"DUMP {nm} rows {side['rows']} n {side['n']} kbytes {len(kb)}")

if __name__ == "__main__":
    if sys.argv[1] == "test": cmd_test(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "dump": cmd_dump(*sys.argv[2:5])

# ---------------------------------------------------------------- R51 delta check: host-decoded kernel bytes -> un-permute -> ifwht -> /c -> + Delta == r48k clamped Wn
def cmd_deltaeq(bd, nm, dec_path, npz):
    tab = open(os.path.join(bd, "__vrk_tables__.blob"), "rb").read(); T, _ = K.tables_from_blob(tab)
    blob = open(os.path.join(bd, nm + ".blob"), "rb").read(); p = parse_vrk2(blob); rows, n = p["rows"], p["n"]
    dec = np.fromfile(dec_path, "<f4").reshape(rows, n)                                               # vr49.cuh vr_wint*vr_wscale, sorted rows
    Wk = torch.empty(rows, n); Wk[torch.as_tensor(p["prm"])] = torch.as_tensor(dec)                    # un-permute: sorted row r -> prm[r]
    _, ifw = A1.make_fw("cpu"); c = torch.as_tensor(p["c16"].astype(np.float32)); wm = float(np.float16(p["wmax"]))
    Wk = (ifw(Wk) / c).numpy()
    h, Yq = K.unpack_blob(blob, T); Wref = (ifw(Yq) / h["c"]).clamp(-h["wmax"], h["wmax"]).numpy()  # r48k clamped Wn (= carrier source)
    z = np.load(npz); r_ = z[nm + "/rows"].astype(np.int64); c_ = z[nm + "/cols"].astype(np.int64)
    print(f"DELTAEQ {nm} rows {rows} n {n} delta entries {len(r_)}; unclamped kernel W vs ref: max abs diff {np.abs(Wk - Wref).max():.4e}")
    for key in ("d32", "d16"):
        W2 = Wk.copy(); W2[r_, c_] += z[nm + "/" + key].astype(np.float32); d = np.abs(W2 - Wref)
        print(f"DELTAEQ {nm} +Delta({key}): max abs diff {d.max():.4e}  bit-equal {int((W2.view(np.uint32) == Wref.view(np.uint32)).sum())}/{W2.size}  "
              f"max rel to wmax {d.max() / wm:.3e}  {'PASS (exact)' if d.max() == 0 else 'within fp32 order' if d.max() <= 4 * np.finfo(np.float32).eps * wm else 'FAIL'}")
if __name__ == "__main__" and sys.argv[1] == "deltaeq": cmd_deltaeq(*sys.argv[2:6])
