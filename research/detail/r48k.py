# R48k: KERNEL-EXACT VR47/R49 formats (research/kernels/r49_vr.cu, cpu_vr reference) for the full-model build.
#   Tile = 32 sorted rows x 256 cols, one class c in 0..7 per tile (u32 meta: word offset | class << 29); fp16 scale per 256 per row;
#   per 4 weights fields f1 (W1 bits), f2 (W2), f3 (W3): W1 = 0,4,6,8,8,8,8,8; W2 = 0,0,0,0,2,4,8,8; W3 = 0,...,0,8 (class 7);
#   S1 = 0,0,0,0,2,4,4,8; S2 = 0,...,0,4 (class 7). b/w = 0,1,1.5,2,2.5,3,4,6.
#   VRi  (TR=0): value = A[f1] << S1 + B[f2] << S2 + B[f3]            (A, B: 256 x int8x4, |entry| <= 64)
#   VRii-C (TR=1, coset, R49 Ungerboeck 16 states, NS=116): bit_t = MSB(f1); state ust (4 bits, bit i-1 = u_{t-i}, reset 0 per 128-col
#     slice); h = par(ust & 0x2) (= u_{t-2}); u = bit ^ par(ust & 0x9); ust = (ust << 1 | u) & 15;
#     value = (A[f1] + h c) << S1 + (A[f2] + h c) << S2 + (A[f3] + h c)     (c = one int8x4 coset vector; kernel MC term)
#   Weight block = fp16 S * value. Encoder: per-survivor LDLQ Viterbi (16 states, 2 branches) or 1-state VQ, H-metric per 4-dim step.
import os, math, json, struct, numpy as np, torch
import r48_a1gpu as A1
W1 = [0, 4, 6, 8, 8, 8, 8, 8]; W2 = [0, 0, 0, 0, 2, 4, 8, 8]; W3 = [0, 0, 0, 0, 0, 0, 0, 8]
S1 = [0, 0, 0, 0, 2, 4, 4, 8]; S2 = [0, 0, 0, 0, 0, 0, 0, 4]; NC = 8          # r49_vr.cu vSH1/vSH2 (as-is kernel)
if os.environ.get("R48K_SHIFTS"):                                                 # R48k shift fix: e.g. "0,0,0,0,2,3,2,4/0,0,0,0,0,0,0,2"
    a_, b_ = os.environ["R48K_SHIFTS"].split("/"); S1[:] = [int(v) for v in a_.split(",")]; S2[:] = [int(v) for v in b_.split(",")]
NW = [W1[k] + W2[k] + W3[k] for k in range(NC)]; BPB = [64 * NW[k] // 8 for k in range(NC)]
CR = torch.tensor([0, 1, 1.5, 2, 2.5, 3, 4, 6], dtype=torch.float64)
par = lambda v: bin(v).count("1") & 1
NSC = [16]; M1, M0 = 0x9, 0x2; Z0 = UX = None
def set_ns(ns):                                     # 16: Ungerboeck z1 = u^s0^s3 (m1 9), z0 = s1; 8 (Pingala 8 gana states): z1 = u^s0^s2 (m1 5), z0 = s1
    global M1, M0, Z0, UX
    NSC[0] = ns; M1, M0 = {16: (0x9, 0x2), 8: (0x5, 0x2)}[ns]
    Z0 = torch.tensor([par(s_ & M0) for s_ in range(ns)]); UX = torch.tensor([par(s_ & M1) for s_ in range(ns)])
set_ns(int(os.environ.get("R48K_NS", "16")))

# ------------------------------------------------------------------ one 4-dim step: best codeword per (state, branch)
def stage_tabs(T, k, tr):
    """(T1 (K1,4), T2 (K2,4) or None, T3 (K3,4) or None) in int units, already shifted. tr: T = dict(A, B, c)."""
    A = T["A"]; B = T["A"] if tr else T["B"]
    t1 = A[:1 << W1[k]] * (1 << S1[k]); t2 = B[:1 << W2[k]] * (1 << S2[k]) if W2[k] else None; t3 = B[:1 << W3[k]] if W3[k] else None
    return t1, t2, t3

def best_cw(xa, Am, t1, t2, t3, halves, TOP=4):
    """xa (..., 4) metric-space targets; tables in int units; returns cost (..., H), idx (..., H, 3), cw (..., H, 4) int units
    with H = 2 halves (stage-1 MSB = 0 / 1) if halves else H = 1."""
    T1A = t1 @ Am; K1 = t1.shape[0]; sh = xa.shape[:-1]
    d1 = (xa[..., None, :] - T1A).pow(2).sum(-1)                        # (..., K1)
    H = 2 if halves else 1; d1h = d1.view(*sh, H, K1 // H)
    if t2 is None:
        c, i = d1h.min(-1); i = i + (torch.arange(H, device=xa.device) * (K1 // H)); z = torch.zeros_like(i)
        return c, torch.stack([i, z, z], -1), t1[i]
    T = min(TOP, K1 // H); _, ti = d1h.topk(T, -1, largest=False); ti = ti + (torch.arange(H, device=xa.device) * (K1 // H))[:, None]   # (...,H,T)
    r1 = xa[..., None, None, :] - T1A[ti]                                # (...,H,T,4)
    T2A = t2 @ Am; d2 = (r1[..., None, :] - T2A).pow(2).sum(-1)         # (...,H,T,K2)
    if t3 is None:
        c2, i2 = d2.min(-1); c, j = c2.min(-1); i1 = ti.gather(-1, j[..., None])[..., 0]; i2 = i2.gather(-1, j[..., None])[..., 0]
        z = torch.zeros_like(i1); return c, torch.stack([i1, i2, z], -1), t1[i1] + t2[i2]
    T2 = TOP; _, ti2 = d2.topk(T2, -1, largest=False)                    # (...,H,T,T2)
    r2 = r1[..., None, :] - T2A[ti2]                                     # (...,H,T,T2,4)
    T3A = t3 @ Am; d3 = (r2[..., None, :] - T3A).pow(2).sum(-1)          # (...,H,T,T2,K3)
    c3, i3 = d3.min(-1); c3f = c3.flatten(-2); c, j = c3f.min(-1)
    i1 = ti.gather(-1, (j // T2)[..., None])[..., 0]; i2 = ti2.flatten(-2).gather(-1, j[..., None])[..., 0]; i3 = i3.flatten(-2).gather(-1, j[..., None])[..., 0]
    return c, torch.stack([i1, i2, i3], -1), t1[i1] + t2[i2] + t3[i3]

def MC(k): return (1 << S1[k]) + ((1 << S2[k]) if W2[k] else 0) + (1 if W3[k] else 0)

def vit_k(X, Hb, A4, T, S, k, tr):
    """X (R,128) slice, Hb (128,128) block of U, A4 (32,4,4) inverses of diagonal 4x4 blocks, S (R,1), class k>=1.
    Returns Q (R,128), E (R,128), IDX (R,32,3) long."""
    R = X.shape[0]; dev = X.device; NS = NSC[0]; St = NS if tr else 1; L = 32
    t1, t2, t3 = stage_tabs(T, k, tr); rr = torch.arange(R, device=dev)
    Xs = X.unsqueeze(0).expand(St, R, 128).contiguous(); PM = torch.full((St, R), float("inf"), device=dev); PM[0] = 0
    SRC = torch.empty(L, St, R, dtype=torch.uint8, device=dev); ID = torch.empty(L, St, R, 3, dtype=torch.uint8, device=dev)
    XN = torch.empty(L, St, R, 4, device=dev); CW = torch.empty(L, St, R, 4, device=dev)
    if tr:
        off = Z0.to(dev)[:, None].float() * (MC(k) * T["c"])[None, :]                          # (16,4) coset offset per state (h = z0(s))
        sp = torch.arange(NS, device=dev); ux = UX.to(dev)
        nxt = ((sp[:, None] << 1) | (torch.arange(2, device=dev)[None, :] ^ ux[:, None])) & (NS - 1)   # half bit b -> u = b ^ par(s & M1)
    for t in range(L):
        j = 4 * t; Am = A4[t]; x = Xs[:, :, 0:4]
        if tr:
            xa = ((x / S) - off[:, None, :]) @ Am
            cost, idx, cw = best_cw(xa, Am, t1, t2, t3, True)                                 # (16,R,2) per stored half bit b
            cw = cw + off[:, None, None, :]
            cand = (PM[:, :, None] + cost)                                                      # (16,R,2) from state s, bit b -> nxt[s,b]
            cf = cand.permute(0, 2, 1).reshape(2 * NS, R); dst = nxt.reshape(2 * NS)                      # source index q = 2 s + b
            # each destination has exactly 2 sources
            order = torch.argsort(dst); ds = dst[order]; q2 = order.view(NS, 2)
            c2 = cf[q2]                                                                         # (16 dest, 2, R)
            mn, w = c2.min(1); qsel = q2.gather(1, w.t()).t() if False else torch.gather(q2[:, :, None].expand(NS, 2, R), 1, w[:, None, :])[:, 0]
            src = qsel // 2; b = qsel % 2
            xn = x[src, rr]; idx_s = idx[src, rr, b]; cw_s = cw[src, rr, b]
        else:
            xa = (x / S) @ Am
            cost, idx, cw = best_cw(xa, Am, t1, t2, t3, False)
            mn = PM + cost[:, :, 0]; src = torch.zeros(1, R, dtype=torch.long, device=dev); xn = x; idx_s = idx[:, :, 0]; cw_s = cw[:, :, 0]
        if t + 1 < L:
            e = (xn - cw_s * S) @ Am
            Xs = (Xs[src, rr, 4:] if tr else Xs[:, :, 4:]) - e @ Hb[j:j + 4, j + 4:]
        PM = mn; SRC[t] = src.to(torch.uint8); ID[t] = idx_s.to(torch.uint8); XN[t] = xn; CW[t] = cw_s
    s = PM.argmin(0); Q = torch.empty(R, 128, device=dev); E = torch.empty(R, 128, device=dev); IDX = torch.empty(R, L, 3, dtype=torch.long, device=dev)
    for t in reversed(range(L)):
        j = 4 * t; qv = CW[t][s, rr] * S; Q[:, j:j + 4] = qv; E[:, j:j + 4] = (XN[t][s, rr] - qv) @ A4[t]; IDX[:, t] = ID[t][s, rr].long()
        s = SRC[t][s, rr].long()
    return Q, E, IDX

def zero_ldlq(X, Hb, A4):
    R = X.shape[0]; X = X.clone(); E = torch.empty(R, 128, device=X.device)
    for t in range(32):
        j = 4 * t; e = X[:, j:j + 4] @ A4[t]; E[:, j:j + 4] = e
        if j + 4 < 128: X[:, j + 4:] -= e @ Hb[j:j + 4, j + 4:]
    return torch.zeros(R, 128, device=X.device), E

CHUNK = [16384, 8192, 8192, 4096, 2048, 2048, 1024, 256]
def encode(Y, Uc, cls, Sb, T, tr):
    rows, n = Y.shape; dev = Y.device; W = Y.clone(); Q = torch.zeros_like(Y); IDX = torch.zeros(rows, n // 4, 3, dtype=torch.uint8, device=dev)
    for i0 in range(0, n, 128):
        i1 = i0 + 128; Hb = Uc[i0:i1, i0:i1].contiguous(); cb = cls[:, i0 // 256]; Sr = Sb[:, i0 // 256][:, None]
        A4 = torch.linalg.inv(torch.stack([Hb[j:j + 4, j:j + 4] for j in range(0, 128, 4)])); Eb = torch.zeros(rows, 128, device=dev)
        for k in range(NC):
            m = (cb == k).nonzero().flatten()
            if len(m) == 0: continue
            ch = CHUNK[k] // (4 if tr else 1)
            for c0 in range(0, len(m), max(ch, 16)):
                mm = m[c0:c0 + max(ch, 16)]; X = W[mm, i0:i1]
                if k == 0: q, e = zero_ldlq(X, Hb, A4); Q[mm, i0:i1] = q; Eb[mm] = e; continue
                q, e, idx = vit_k(X, Hb, A4, T, Sr[mm], k, tr); Q[mm, i0:i1] = q; Eb[mm] = e; IDX[mm, i0 // 4:i1 // 4] = idx.to(torch.uint8)
        if i1 < n: W[:, i1:] -= Eb @ Uc[i0:i1, i1:]
    return Q, IDX

# ------------------------------------------------------------------ allocation (r47_a1 alloc + forced floor), scales
def alloc(E, budget, CD, floor=None):
    E = E.double(); rows, nb = E.shape; Dm = E[:, :, None] * CD; Rm = 256 * CR
    if floor is not None: Dm = torch.where(floor[:, :, None] & (torch.arange(NC)[None, None, :] < 6), torch.full_like(Dm, 1e300), Dm)
    pick = lambda lam: (Dm + lam * Rm).argmin(2)
    lo, hi = 1e-12, 1e6
    for _ in range(80):
        mid = math.sqrt(lo * hi); c = pick(mid)
        if float(Rm[c].sum()) > budget: lo = mid
        else: hi = mid
    c = pick(hi).clone()
    for _ in range(400):
        slack = budget - float(Rm[c].sum()); best = 0.0; arg = None
        for k in range(1, NC):
            ok = c < k; dR = Rm[k] - Rm[c]; dD = Dm.gather(2, c[..., None])[..., 0] - Dm[:, :, k]
            fit = ok & (dR <= slack) & (dR > 0); gain = torch.where(fit, dD / dR, torch.zeros_like(dD))
            g = float(gain.max())
            if g > best: best = g; arg = (int(gain.argmax()) // nb, int(gain.argmax()) % nb, k)
        if arg is None: break
        c[arg[0], arg[1]] = arg[2]
    return c

def side_bytes(rows, n):
    tiles = rows // 32; nbk = n // 256; pb = max(1, math.ceil(math.log2(rows)))
    return struct.calcsize(HDR) + 2 * n + (rows * pb + 7) // 8 + 2 * rows * nbk + 4 * tiles * nbk
HDR = "<4sIIBBe"

# ------------------------------------------------------------------ codec
def pack_blob(c16, prm, S16, cs, IDX, wmax, tr):
    """c16 (n,) fp16; prm (rows,) sorted->orig; S16 (rows,nbk) fp16 numpy (natural rows); cs (tiles,nbk) classes; IDX (rows,n/4,3) uint8 numpy."""
    rows = prm.numel(); n = c16.shape[0]; tiles, nbk = cs.shape; pb = max(1, math.ceil(math.log2(rows))); csn = cs.numpy().astype(np.int64)
    clsr = np.repeat(csn, 32, 0); bpb = np.array(BPB)[clsr]; off = np.concatenate([[0], np.cumsum(bpb.ravel())])[:-1].reshape(rows, nbk)
    stream = np.zeros(int(bpb.sum()), np.uint8); Is = IDX[prm.numpy()].reshape(rows, nbk, 64, 3).astype(np.uint64)
    for k in range(1, NC):
        r_, b_ = np.nonzero(clsr == k)
        if len(r_) == 0: continue
        f = Is[r_, b_]; sy = f[..., 0] | (f[..., 1] << np.uint64(W1[k])) | (f[..., 2] << np.uint64(W1[k] + W2[k]))
        pk = A1._pack(sy, NW[k]).reshape(len(r_), BPB[k]); stream[(off[r_, b_][:, None] + np.arange(BPB[k])[None, :]).ravel()] = pk.ravel()
    meta = ((off[::32] // 4).astype(np.uint64) & np.uint64(0x1FFFFFFF)) | (csn.astype(np.uint64) << np.uint64(29))
    parts = [struct.pack(HDR, b"VRK1", rows, n, int(tr), pb, np.float16(wmax)), np.asarray(c16, np.float16).tobytes(), A1._pack(prm.numpy(), pb).tobytes(),
             np.asarray(S16, np.float16)[prm.numpy()].tobytes(), meta.astype(np.uint32).tobytes(), stream.tobytes()]
    b = b"".join(parts); assert len(b) == side_bytes(rows, n) + len(stream); return b

def unpack_blob(b, T):
    """T = dict(A, B, c) int-valued float tensors (CPU). Returns header dict and Yq (rows,n) float32 CPU (kernel cpu_vr semantics)."""
    mg, rows, n, tr, pb, wm = struct.unpack_from(HDR, b, 0); assert mg == b"VRK1"; o = struct.calcsize(HDR); tiles = rows // 32; nbk = n // 256
    def take(k_):
        nonlocal o; r = b[o:o + k_]; o += k_; return r
    c = torch.as_tensor(np.frombuffer(take(2 * n), np.float16).astype(np.float32))
    prm = torch.as_tensor(A1._unpack(take((rows * pb + 7) // 8), rows, pb).astype(np.int64))
    Ss = torch.as_tensor(np.frombuffer(take(2 * rows * nbk), np.float16).astype(np.float32).reshape(rows, nbk))
    meta = np.frombuffer(take(4 * tiles * nbk), np.uint32).reshape(tiles, nbk); cs = (meta >> 29).astype(np.int64); stream = np.frombuffer(b[o:], np.uint8)
    clsr = np.repeat(cs, 32, 0); bpb = np.array(BPB)[clsr]; off = np.concatenate([[0], np.cumsum(bpb.ravel())])[:-1].reshape(rows, nbk)
    assert np.array_equal((off[::32] // 4).astype(np.uint32) & 0x1FFFFFFF, meta & 0x1FFFFFFF) and int(bpb.sum()) == len(stream)
    Ys = torch.zeros(rows, nbk, 256)
    for k in range(1, NC):
        r_, b_ = np.nonzero(clsr == k)
        if len(r_) == 0: continue
        sy = A1._unpack(stream[(off[r_, b_][:, None] + np.arange(BPB[k])[None, :]).ravel()].tobytes(), len(r_) * 64, NW[k]).reshape(len(r_), 64)
        f1 = sy & ((1 << W1[k]) - 1); f2 = (sy >> W1[k]) & ((1 << W2[k]) - 1); f3 = (sy >> (W1[k] + W2[k])) & ((1 << W3[k]) - 1)
        f1, f2, f3 = (torch.as_tensor(v.astype(np.int64)) for v in (f1, f2, f3))
        Bt = T["A"] if tr else T["B"]
        v = T["A"][f1] * (1 << S1[k])
        if W2[k]: v = v + Bt[f2] * (1 << S2[k])
        if W3[k]: v = v + Bt[f3]
        if tr:
            h = torch.zeros(len(r_), 64); ust = torch.zeros(len(r_), dtype=torch.long)
            bit = f1 >> (W1[k] - 1)
            for t in range(64):
                if t % 32 == 0: ust = torch.zeros(len(r_), dtype=torch.long)
                h[:, t] = ((ust & M0).bitwise_and(2) > 0).float()                                  # par(ust & 0x2) = bit 1
                u = bit[:, t] ^ (((ust & 1) ^ ((ust >> (3 if NSC[0] == 16 else 2)) & 1)))                                     # par(ust & 0x9) = s0 ^ s3
                ust = ((ust << 1) | u) & (NSC[0] - 1)
            v = v + h[:, :, None] * (MC(k) * T["c"])[None, None, :]
        Ys[torch.as_tensor(r_), torch.as_tensor(b_)] = (v * Ss[torch.as_tensor(r_), torch.as_tensor(b_)][:, None, None]).reshape(len(r_), 256)
    Yq = torch.empty(rows, n); Yq[prm] = Ys.reshape(rows, n)
    Sb = torch.empty(rows, nbk); Sb[prm] = Ss
    return dict(rows=rows, n=n, tr=tr, wmax=float(np.float16(wm)), c=c, prm=prm, cs=cs, Sb=Sb), Yq

def tables_blob(T, tr):
    b = b"VRKT" + struct.pack("<B", (2 if NSC[0] == 8 else 1) if tr else 0) + T["A"].to(torch.int8).numpy().tobytes() + T["B"].to(torch.int8).numpy().tobytes() + T["c"].to(torch.int8).numpy().tobytes()
    return b + bytes(S1) + bytes(S2)                                               # class shifts (8 + 8 bytes)
def tables_from_blob(b):
    assert b[:4] == b"VRKT"; tr = b[4]; o = 5
    if tr == 2: set_ns(8)
    elif tr == 1: set_ns(16)
    A = torch.as_tensor(np.frombuffer(b[o:o + 1024], np.int8).astype(np.float32)).view(256, 4); B = torch.as_tensor(np.frombuffer(b[o + 1024:o + 2048], np.int8).astype(np.float32)).view(256, 4)
    c = torch.as_tensor(np.frombuffer(b[o + 2048:o + 2052], np.int8).astype(np.float32))
    if len(b) >= o + 2052 + 16: S1[:] = list(b[o + 2052:o + 2060]); S2[:] = list(b[o + 2060:o + 2068])
    return dict(A=A, B=B, c=c), tr

# ------------------------------------------------------------------ one tensor
def k_tensor(W, c, Hf, sw, budget_fn, T, CD, G, tr, alpha=1.05, dev="cuda", damp=0.1, nsal=4):
    fw, ifw = A1.make_fw(dev); rows, n = W.shape; nbk = n // 256
    Y = fw(W * c); rmsb = Y.view(rows, -1, 256).pow(2).mean(2).sqrt().clamp_min(1e-12).cpu()
    Hc = Hf / c[:, None] / c[None, :]; Hp = fw(fw(Hc).T.contiguous()); del Hc
    gch = n <= int(os.environ.get("GPUCHOL_MAXN", "4096"))
    Hq = Hp.double() if gch else Hp.double().cpu(); Hq.diagonal().add_(damp * float(Hq.diag().mean())); Lc = torch.linalg.cholesky(Hq); del Hq
    Hi = torch.cholesky_inverse(Lc); del Lc; Uc = torch.linalg.cholesky(Hi, upper=True).cpu(); del Hi
    wb = Uc.diagonal().pow(-2).view(nbk, 256).mean(1); wb = (wb / wb.mean()).double(); Ucg = Uc.float().to(dev); del Uc, Hp
    budget = budget_fn(rows, n)
    Ebk = (Y.view(rows, nbk, 256) ** 2).sum(2).double().cpu()
    prm = torch.argsort(Ebk.clamp_min(1e-30).log().mean(1)); Es = Ebk[prm] * wb[None]
    floor = None
    if nsal:                                                                            # salient fix: tile-blocks holding the top-|w| weights >= class 6
        top = torch.topk(W.abs().flatten(), nsal).indices.cpu(); inv = torch.argsort(prm)
        floor = torch.zeros(rows // 32, nbk, dtype=torch.bool); floor[inv[top // n] // 32, (top % n) // 256] = True
    cs = alloc(Es.view(rows // 32, 32, nbk).sum(1), budget / 32, CD, floor)
    clsS = cs.repeat_interleave(32, 0); cls = torch.empty_like(clsS); cls[prm] = clsS
    Gk = torch.tensor([1.0] + [G[k] for k in range(1, NC)], dtype=torch.float32)
    S16 = (alpha * rmsb / Gk[cls]).to(torch.float16); S16[cls == 0] = 0; Sb = S16.float()
    old = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = False
    TG = {k_: v.to(dev) for k_, v in T.items()}
    try: Yq, IDX = encode(Y, Ucg, cls.to(dev), Sb.to(dev), TG, tr)
    finally: torch.backends.cuda.matmul.allow_tf32 = old
    wmax = float(np.float16(float(W.abs().max()) * 1.001)); Wq = (ifw(Yq) / c).clamp(-wmax, wmax)
    del Y, Ucg
    return dict(Yq=Yq, Wq=Wq, IDX=IDX, cls=cls, cs=cs, prm=prm, S16=S16, Sb=Sb, wmax=wmax, idx_bits=float((256 * CR[cls]).sum()), budget=budget,
                hist=torch.bincount(cls.flatten(), minlength=NC).tolist(), nfloor=int(floor.sum()) if floor is not None else 0)


# ------------------------------------------------------------------ R48k layout fix (09:55): kernel payload order
# VRK1 (as built) stores the index stream row-major (sorted row, block). The kernel (r49_vr.cu) reads a 32x256 tile-block as one contiguous
# region addressed by its u32 meta word offset, rows interleaved inside it. VRK2 = same bytes per (row, block), reordered tile-block-major
# ([tile][block][row in tile]); meta = word offset of the tile-block | class << 29. relayout() converts VRK1 -> VRK2 losslessly.
def _hdr_sizes(b):
    mg, rows, n, tr, pb, wm = struct.unpack_from(HDR, b, 0); o = struct.calcsize(HDR) + 2 * n + (rows * pb + 7) // 8 + 2 * rows * (n // 256)
    return mg, rows, n, o
def relayout(b):
    mg, rows, n, o = _hdr_sizes(b); assert mg == b"VRK1"; tiles = rows // 32; nbk = n // 256
    meta = np.frombuffer(b[o:o + 4 * tiles * nbk], np.uint32).reshape(tiles, nbk); cs = (meta >> 29).astype(np.int64); st = np.frombuffer(b[o + 4 * tiles * nbk:], np.uint8)
    clsr = np.repeat(cs, 32, 0); bpb = np.array(BPB)[clsr]; off = np.concatenate([[0], np.cumsum(bpb.ravel())])[:-1].reshape(rows, nbk)
    bt = np.array(BPB)[cs]; toff = np.concatenate([[0], np.cumsum((32 * bt).ravel())])[:-1].reshape(tiles, nbk)   # tile-block-major offsets
    out = np.zeros_like(st)
    for k in range(1, NC):
        t_, b_ = np.nonzero(cs == k)
        if len(t_) == 0: continue
        r = (32 * t_[:, None] + np.arange(32)[None, :])                                       # (m, 32) sorted rows
        src = off[r, b_[:, None]][:, :, None] + np.arange(BPB[k])[None, None, :]
        dst = toff[t_, b_][:, None, None] + (np.arange(32)[None, :, None] * BPB[k]) + np.arange(BPB[k])[None, None, :]
        out[dst.ravel()] = st[src.ravel()]
    meta2 = ((toff // 4).astype(np.uint64) & np.uint64(0x1FFFFFFF)) | (cs.astype(np.uint64) << np.uint64(29))
    return b"VRK2" + b[4:o] + meta2.astype(np.uint32).tobytes() + out.tobytes()
def to_vrk1(b):
    mg, rows, n, o = _hdr_sizes(b)
    if mg == b"VRK1": return b
    assert mg == b"VRK2"; tiles = rows // 32; nbk = n // 256
    meta = np.frombuffer(b[o:o + 4 * tiles * nbk], np.uint32).reshape(tiles, nbk); cs = (meta >> 29).astype(np.int64); st = np.frombuffer(b[o + 4 * tiles * nbk:], np.uint8)
    clsr = np.repeat(cs, 32, 0); bpb = np.array(BPB)[clsr]; off = np.concatenate([[0], np.cumsum(bpb.ravel())])[:-1].reshape(rows, nbk)
    bt = np.array(BPB)[cs]; toff = np.concatenate([[0], np.cumsum((32 * bt).ravel())])[:-1].reshape(tiles, nbk)
    assert np.array_equal((toff // 4).astype(np.uint32) & 0x1FFFFFFF, meta & 0x1FFFFFFF)
    out = np.zeros_like(st)
    for k in range(1, NC):
        t_, b_ = np.nonzero(cs == k)
        if len(t_) == 0: continue
        r = (32 * t_[:, None] + np.arange(32)[None, :])
        dstv = off[r, b_[:, None]][:, :, None] + np.arange(BPB[k])[None, None, :]
        srcv = toff[t_, b_][:, None, None] + (np.arange(32)[None, :, None] * BPB[k]) + np.arange(BPB[k])[None, None, :]
        out[dstv.ravel()] = st[srcv.ravel()]
    meta1 = ((np.concatenate([[0], np.cumsum(bpb.ravel())])[:-1].reshape(rows, nbk)[::32] // 4).astype(np.uint64) & np.uint64(0x1FFFFFFF)) | (cs.astype(np.uint64) << np.uint64(29))
    return b"VRK1" + b[4:o] + meta1.astype(np.uint32).tobytes() + out.tobytes()
_unpack_vrk1 = unpack_blob
def unpack_blob(b, T): return _unpack_vrk1(to_vrk1(b), T)
