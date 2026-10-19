# R48: GPU (torch CUDA) port of the A1 encoder VRtile-B2-c8-Hw of research/detail/r47_a1.py (CODES=b2, HW=1), plus the
# real A1 blob codec (every side-info byte written) and its decoder.
# Algorithm = r47_a1.py: Had256 RHT, sqrt(imatrix) column prescale c, natural-order block LDLQ (U = chol(H^-1) upper,
# damp 0.1), rows sorted by mean log block energy, class per 32x256 tile (Lagrangian+greedy on w_b*E), fp16 tile scale +
# DB-bit per-block delta on a 1/8-octave grid, Viterbi per 128-w slice (state 0 at slice start) with per-survivor LDLQ.
# Classes: c0 = 0; c1..c5 = bitshift trellis L=8, k=2..6 bits per 2-dim step, one 256x2 int8 table each (QTIP-type);
# c6 = VQ2-256 (4 b/w), c7 = scalar 64 (6 b/w), c8 = int8 escape (8 b/w, absmax scale).
# GPU change vs the CPU reference: label-centric bitshift Viterbi (256 branch costs per row per step, exact same search),
# explicit squared differences instead of cdist, batched 2x2/4x4 inverses. Same decisions up to float32 rounding/ties.
import os, math, json, struct, numpy as np, torch
R47 = "~/repo/research/work/r47"
CR = torch.tensor([0, 1, 1.5, 2, 2.5, 3, 4, 6, 8], dtype=torch.float64); NC = 9; TILE = 32; CB = 4
BN = {1: "B2r1", 2: "B2k3", 3: "B2", 4: "B2k5", 5: "B2k6"}
SYM = {1: (128, 2), 2: (128, 3), 3: (128, 4), 4: (128, 5), 5: (128, 6), 6: (128, 8), 7: (256, 6), 8: (256, 8)}   # per 256-block: (symbols, bits)
SYM_T = dict(SYM); SYM_V = {1: (64, 4), 2: (64, 6), 3: (64, 8), 4: (128, 5), 5: (128, 6), 6: (128, 8), 7: (256, 6), 8: (256, 8)}
DIM_T = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1}; DIM_V = {1: 4, 2: 4, 3: 4, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1}; DIM = dict(DIM_T)
MODE = ["trellis"]   # "trellis" (A1: c1..c5 bitshift L=8) or "vq" (VRi = solver V0: c1..c5 1-state VQ 16x4/64x4/256x4/32x2/64x2)
def set_mode(m):
    MODE[0] = m; SYM.clear(); SYM.update(SYM_V if m == "vq" else SYM_T); DIM.clear(); DIM.update(DIM_V if m == "vq" else DIM_T)
    BPB.clear(); BPB.update({k: s_ * w // 8 for k, (s_, w) in SYM.items()}); BPB[0] = 0
BPB = {}; set_mode("trellis")

def i8q(tab):
    tab = tab.double(); s = 127.0 / float(tab.abs().max()); return torch.round(tab * s).clamp(-127, 127).to(torch.int8), s
def i8deq(q, s): return (q.double() / s).float()                      # == r47 i8(): round(tab*s).clamp/s in float64, then float32

def esc_D():                                            # r47_a1.esc_D verbatim (Gaussian D/dim of c8)
    g = torch.Generator().manual_seed(7); X = torch.randn(4000, 256, generator=g, dtype=torch.float64); s = X.pow(2).mean(1, keepdim=True).sqrt()
    st = 2.0 ** (torch.ceil(8 * torch.log2(X.abs().amax(1, keepdim=True) / 127)) / 8)
    return float(((torch.clamp(torch.round(X / st), -127, 127) * st - X) ** 2 / s ** 2).mean())

JOINT_T = "~/repo/research/work/r48p/joint_table.pt"; GJ = "~/repo/research/work/r48p/gauss.json"
def load_codes(joint=False, vq=False):
    """int8 tables + scales for c1..c7 and the class D table CDA (float64). joint=False: r47_a1 CODES=b2 (5 per-class tables);
    joint=True: r48p_a1 CODES=joint (ONE 256x2 table for c1..c5, as the k_b2 kernel; Gaussian D from r48p gauss.json)."""
    TK = torch.load(f"{R47}/tables_onek.pt"); DK = json.load(open(f"{R47}/dclass_onek.json"))
    BJ = {**json.load(open(f"{R47}/r47m_bitshift.json")), **json.load(open(f"{R47}/r47m_bitshift2.json"))}
    codes = {}; DCL = [1.0]; set_mode("vq" if vq else "trellis")
    for c in range(1, 6):
        if vq: codes[c] = i8q(TK[f"c{c}"].float().view(TK[f"c{c}"].shape[1], TK[f"c{c}"].shape[2])); DCL.append(DK[f"c{c}"])
        elif joint: codes[c] = i8q(torch.load(JOINT_T).float().view(256, 2)); DCL.append(json.load(open(GJ))["joint"][str(c + 1)])
        else: codes[c] = i8q(torch.load(f"{R47}/r47m_{BN[c]}_table.pt").float().view(256, 2)); DCL.append(BJ[BN[c]]["D"])
    codes[6] = i8q(TK["c6"].float().view(256, 2)); DCL.append(DK["c6"])
    codes[7] = i8q(TK["c7"].float().view(64, 1)); DCL.append(DK["c7"])
    DCL.append(esc_D())
    return codes, torch.tensor(DCL, dtype=torch.float64)

def tables_blob(codes):
    joint = MODE[0] == "trellis" and all(torch.equal(codes[c][0], codes[1][0]) and codes[c][1] == codes[1][1] for c in range(2, 6))
    b = b"A1TV" if MODE[0] == "vq" else (b"A1TJ" if joint else b"A1TB")
    for c in ((1, 6, 7) if joint else range(1, 8)): q, s = codes[c]; b += struct.pack("<HBd", q.shape[0], q.shape[1], s) + q.numpy().tobytes()
    return b

def tables_from_blob(b):
    assert b[:4] in (b"A1TB", b"A1TJ", b"A1TV"); o = 4; codes = {}; joint = b[:4] == b"A1TJ"; set_mode("vq" if b[:4] == b"A1TV" else "trellis")
    for c in ((1, 6, 7) if joint else range(1, 8)):
        K, d, s = struct.unpack_from("<HBd", b, o); o += struct.calcsize("<HBd")
        codes[c] = (torch.from_numpy(np.frombuffer(b[o:o + K * d], np.int8).copy()).view(K, d), s); o += K * d
    if joint:
        for c in range(2, 6): codes[c] = codes[1]
    return codes

def gpu_tabs(codes, dev):
    T = {c: i8deq(*codes[c]).to(dev) for c in range(1, 8)}; T[8] = torch.arange(-127, 128, dtype=torch.float32, device=dev).view(255, 1); return T

# ------------------------------------------------------------------ Viterbi / VQ with per-survivor LDLQ (one 128-w slice)
def vit_bs(X, Hb, A2, T, S, k):
    """bitshift trellis L=8, k bits per 2-dim step, state 0 at slice start. X (R,128), Hb (128,128) block of U, A2 (64,2,2)
    inverses of the 2x2 diagonal blocks, T (256,2) table, S (R,1). Returns Q (R,128), E (R,128), labels (R,64) long."""
    R = X.shape[0]; dev = X.device; St = 1 << (8 - k); nb = 1 << k; L = 64
    rr = torch.arange(R, device=dev); sl = torch.arange(256, device=dev) >> k; sidx = torch.arange(St, device=dev)[:, None]
    Xs = X.unsqueeze(0).expand(St, R, 128).contiguous()                  # remaining columns of every survivor
    PM = torch.full((St, R), float("inf"), device=dev); PM[0] = 0
    SRC = torch.empty(L, St, R, dtype=torch.uint8, device=dev); QI = torch.empty(L, St, R, dtype=torch.uint8, device=dev)
    XN = torch.empty(L, St, R, 2, device=dev); UA = torch.matmul(T, A2)  # (64,256,2)
    for t in range(L):
        j = 2 * t; A = A2[t]; x = Xs[:, :, 0:2]
        xa = (x / S) @ A
        cost = (xa[sl] - UA[t][:, None, :]).pow(2).sum(-1)               # (256,R): label l from state l>>k
        mn, h = (PM[sl] + cost).view(nb, St, R).min(0)                   # new state s' = l & (St-1), l = h*St + s'
        l = h * St + sidx; src = l >> k
        xn = x[src, rr]
        if t + 1 < L:
            e = (xn - T[l] * S) @ A
            Xs = Xs[src, rr, 2:] - e @ Hb[j:j + 2, j + 2:]
        PM = mn; SRC[t] = src.to(torch.uint8); QI[t] = l.to(torch.uint8); XN[t] = xn
    s = PM.argmin(0); Q = torch.empty(R, 128, device=dev); E = torch.empty(R, 128, device=dev); LB = torch.empty(R, L, dtype=torch.long, device=dev)
    for t in reversed(range(L)):
        j = 2 * t; l = QI[t][s, rr].long(); qv = T[l] * S; Q[:, j:j + 2] = qv; E[:, j:j + 2] = (XN[t][s, rr] - qv) @ A2[t]; LB[:, t] = l
        s = SRC[t][s, rr].long()
    return Q, E, LB

def vq_ldlq(X, Hb, Ad, T, S, d):
    """1-state VQ (T (K,d)) or zero class (T None, d=4) with LDLQ inside the slice. Returns Q, E, labels (R,128/d)."""
    R = X.shape[0]; dev = X.device; X = X.clone(); Q = torch.zeros(R, 128, device=dev); E = torch.empty(R, 128, device=dev)
    LB = torch.zeros(R, 128 // d, dtype=torch.long, device=dev)
    for t in range(128 // d):
        j = t * d; A = Ad[t]; x = X[:, j:j + d]
        if T is None: qv = torch.zeros_like(x)
        else:
            xa = (x / S) @ A; UA = T @ A
            l = (xa[:, None, :] - UA[None]).pow(2).sum(-1).argmin(1); qv = T[l] * S; LB[:, t] = l
        e = (x - qv) @ A; Q[:, j:j + d] = qv; E[:, j:j + d] = e
        if j + d < 128: X[:, j + d:] -= e @ Hb[j:j + d, j + d:]
    return Q, E, LB

def encode(Y, Uc, cls, Sb, TG, chunk=10240):
    """Y (rows,n), Uc (n,n) upper chol of H^-1, cls (rows,nbk) long, Sb (rows,nbk) all on the GPU. Returns Yq, LBL (rows,n) uint8
    (label of the step starting at each column; d=2 classes use even columns)."""
    rows, n = Y.shape; dev = Y.device; W = Y.clone(); Q = torch.zeros_like(Y); LBL = torch.zeros(rows, n, dtype=torch.uint8, device=dev)
    for i0 in range(0, n, 128):
        i1 = i0 + 128; Hb = Uc[i0:i1, i0:i1].contiguous(); cb = cls[:, i0 // 256]; Sr = Sb[:, i0 // 256][:, None]
        Ad = {d: torch.linalg.inv(torch.stack([Hb[j:j + d, j:j + d] for j in range(0, 128, d)])) for d in (1, 2, 4)}
        Eb = torch.zeros(rows, 128, device=dev)
        for k in range(NC):
            m = (cb == k).nonzero().flatten()
            if len(m) == 0: continue
            for c0 in range(0, len(m), chunk):
                mm = m[c0:c0 + chunk]; X = W[mm, i0:i1]; S = Sr[mm]
                if k == 0: q, e, lb = vq_ldlq(X, Hb, Ad[4], None, S, 4)
                elif k <= 5 and MODE[0] == "trellis": q, e, lb = vit_bs(X, Hb, Ad[2], TG[k], S, k + 1)
                else: d = TG[k].shape[1]; q, e, lb = vq_ldlq(X, Hb, Ad[d], TG[k], S, d)
                Q[mm, i0:i1] = q; Eb[mm] = e
                if k > 0:
                    d = DIM[k]; LBL[mm[:, None], (i0 + torch.arange(0, 128, d, device=dev))[None, :]] = lb.to(torch.uint8)
        if i1 < n: W[:, i1:] -= Eb @ Uc[i0:i1, i1:]
    return Q, LBL

# ------------------------------------------------------------------ allocation and scales (r47_a1 verbatim, CPU float64)
def alloc(E, budget, CD):
    E = E.double(); rows, nb = E.shape; Dm = E[:, :, None] * CD; Rm = 256 * CR
    pick = lambda lam: (Dm + lam * Rm).argmin(2)
    lo, hi = 1e-12, 1e6
    for _ in range(80):
        mid = math.sqrt(lo * hi); c = pick(mid)
        if float(Rm[c].sum()) > budget: lo = mid
        else: hi = mid
    c = pick(hi).clone()
    for _ in range(int(os.environ.get("GREEDY", "400"))):
        slack = budget - float(Rm[c].sum()); best = 0.0; arg = None
        for k in range(1, NC):
            ok = c < k; dR = Rm[k] - Rm[c]; dD = Dm.gather(2, c[..., None])[..., 0] - Dm[:, :, k]
            fit = ok & (dR <= slack) & (dR > 0); gain = torch.where(fit, dD / dR, torch.zeros_like(dD))
            g = float(gain.max())
            if g > best: best = g; arg = (int(gain.argmax()) // nb, int(gain.argmax()) % nb, k)
        if arg is None: break
        c[arg[0], arg[1]] = arg[2]
    return c

def scales(cls, prm, rmsb, amaxb, al, DB):
    """r47_a1 scale path: fp16 tile scale (rounded up) + DB-bit delta on 1/8 octave. All CPU float32. Returns Sb (rows,nbk),
    Tt (tiles,nbk) fp32 (fp16-exact), dl (tiles,32,nbk) int, clip fraction."""
    rows, nbk = cls.shape
    ideal = torch.where(cls == 8, amaxb / 127, al * rmsb)
    idS = ideal[prm].view(rows // TILE, TILE, nbk)
    mx = idS.amax(1).numpy(); t16 = mx.astype(np.float16); t16 = np.where(t16.astype(np.float32) < mx, np.nextafter(t16, np.float16(np.inf)), t16)
    Tt = torch.as_tensor(t16.astype(np.float32))
    dl = 8 * torch.log2(Tt[:, None, :] / idS)
    dl = torch.where((cls[prm] == 8).view(rows // TILE, TILE, nbk), torch.floor(dl), torch.round(dl)).clamp(0, 2 ** DB - 1)
    Sb = sb_from(Tt, dl.long(), prm)
    return Sb, Tt, dl.long(), float((dl == 2 ** DB - 1).float().mean())

def sb_from(Tt, dl, prm):
    rows = prm.numel(); nbk = Tt.shape[1]
    Sq = (Tt[:, None, :] * 2.0 ** (-dl.float() / 8)).reshape(rows, nbk); Sb = torch.empty_like(Sq); Sb[prm] = Sq; return Sb

# ------------------------------------------------------------------ blob codec
def _pack(v, w):
    v = np.asarray(v, np.uint64).ravel(); bits = ((v[:, None] >> np.arange(w - 1, -1, -1, dtype=np.uint64)) & 1).astype(np.uint8)
    return np.packbits(bits.ravel())
def _unpack(buf, cnt, w):
    bits = np.unpackbits(np.frombuffer(buf, np.uint8))[:cnt * w].reshape(cnt, w).astype(np.uint64)
    return (bits << np.arange(w - 1, -1, -1, dtype=np.uint64)).sum(1)
HDR = "<4sIIBBHe"
def side_bytes(rows, n, DB):
    """exact bytes of everything but the index stream (header, c, perm, tile scales, classes, deltas, tile + slice offsets)."""
    tiles = rows // TILE; nbk = n // 256; pb = max(1, math.ceil(math.log2(rows)))
    return (struct.calcsize(HDR) + 2 * n + (rows * pb + 7) // 8 + 2 * tiles * nbk + (tiles * nbk * CB + 7) // 8 + (rows * nbk * DB + 7) // 8
            + 4 * tiles + 2 * tiles * math.ceil(n / 1024))

def symbols(LBL_rows, k):
    """LBL_rows (m,256) uint8 labels of one 256-block (natural column order) -> (m, nsym) symbols of class k."""
    if k <= 5 and MODE[0] == "trellis": return (LBL_rows[:, 0::2] & ((1 << (k + 1)) - 1)).astype(np.uint64)
    return LBL_rows[:, 0::DIM[k]].astype(np.uint64)

def pack_blob(c16, prm, Tt, cs, dl, LBL, wmax, DB):
    """c16 (n,) fp16, prm (rows,) sorted->orig row, Tt (tiles,nbk) fp32 (fp16-exact), cs (tiles,nbk) classes, dl (tiles,32,nbk) int,
    LBL (rows,n) uint8 numpy (natural rows). Index stream: sorted rows, blocks in order, BPB[k] bytes each."""
    rows, n = LBL.shape; tiles, nbk = cs.shape; pb = max(1, math.ceil(math.log2(rows)))
    csn = cs.numpy().astype(np.int64); clsr = np.repeat(csn, TILE, 0)                       # (rows sorted, nbk)
    bpbv = np.vectorize(BPB.get)(clsr).astype(np.int64); off = np.concatenate([[0], np.cumsum(bpbv.ravel())])[:-1].reshape(rows, nbk)
    stream = np.zeros(int(bpbv.sum()), np.uint8); Ls = LBL[prm.numpy()]
    for k in range(1, NC):
        r_, b_ = np.nonzero(clsr == k)
        if len(r_) == 0: continue
        blk = Ls.reshape(rows, nbk, 256)[r_, b_]; sy = symbols(blk, k); ns, w = SYM[k]
        pk = _pack(sy, w).reshape(len(r_), BPB[k]); stream[(off[r_, b_][:, None] + np.arange(BPB[k])[None, :]).ravel()] = pk.ravel()
    rowbytes = bpbv.reshape(tiles, TILE, nbk)[:, 0, :]                                           # same for all rows of a tile
    toff = np.concatenate([[0], np.cumsum(rowbytes.sum(1) * TILE)])[:-1].astype(np.uint32)
    ch = math.ceil(n / 1024); soff = np.zeros((tiles, ch), np.uint16)
    for q in range(ch): soff[:, q] = rowbytes[:, :q * 4].sum(1)
    parts = [struct.pack(HDR, b"A1B1", rows, n, DB, pb, 0, np.float16(wmax)), np.asarray(c16, np.float16).tobytes(), _pack(prm.numpy(), pb).tobytes(),
             Tt.numpy().astype(np.float16).tobytes(), _pack(csn, CB).tobytes(), _pack(dl.numpy().astype(np.int64), DB).tobytes(),
             toff.tobytes(), soff.tobytes(), stream.tobytes()]
    b = b"".join(parts); assert len(b) == side_bytes(rows, n, DB) + len(stream), (len(b), side_bytes(rows, n, DB), len(stream)); return b

def unpack_blob(b, codes):
    """-> dict(rows, n, DB, wmax, c (n,) f32, prm, Tt, cs, dl, Sb (rows,nbk), labels per class) and Yq (rows,n) float32 CPU."""
    mg, rows, n, DB, pb, _, wm = struct.unpack_from(HDR, b, 0); assert mg == b"A1B1"; o = struct.calcsize(HDR)
    tiles = rows // TILE; nbk = n // 256
    def take(nb_):
        nonlocal o; r = b[o:o + nb_]; o += nb_; return r
    c = np.frombuffer(take(2 * n), np.float16).astype(np.float32)
    prm = torch.as_tensor(_unpack(take((rows * pb + 7) // 8), rows, pb).astype(np.int64))
    Tt = torch.as_tensor(np.frombuffer(take(2 * tiles * nbk), np.float16).astype(np.float32).reshape(tiles, nbk))
    cs = _unpack(take((tiles * nbk * CB + 7) // 8), tiles * nbk, CB).astype(np.int64).reshape(tiles, nbk)
    dl = torch.as_tensor(_unpack(take((rows * nbk * DB + 7) // 8), rows * nbk, DB).astype(np.int64).reshape(tiles, TILE, nbk))
    take(4 * tiles); take(2 * tiles * math.ceil(n / 1024)); stream = np.frombuffer(b[o:], np.uint8)
    Sb = sb_from(Tt, dl, prm)
    clsr = np.repeat(cs, TILE, 0); bpbv = np.vectorize(BPB.get)(clsr).astype(np.int64)
    off = np.concatenate([[0], np.cumsum(bpbv.ravel())])[:-1].reshape(rows, nbk); assert int(bpbv.sum()) == len(stream)
    T = {cc: i8deq(*codes[cc]) for cc in range(1, 8)}; T[8] = torch.arange(-127, 128, dtype=torch.float32).view(255, 1)
    Ys = torch.zeros(rows, nbk, 256); Ss = Sb[prm]                                              # sorted-row order
    for k in range(1, NC):
        r_, b_ = np.nonzero(clsr == k)
        if len(r_) == 0: continue
        ns, w = SYM[k]; sy = _unpack(stream[(off[r_, b_][:, None] + np.arange(BPB[k])[None, :]).ravel()].tobytes(), len(r_) * ns, w).reshape(len(r_), ns).astype(np.int64)
        if k <= 5 and MODE[0] == "trellis":
            kk = k + 1; lab = np.zeros_like(sy)
            for t in range(ns):
                prev = lab[:, t - 1] if t % 64 else 0                                              # state reset every 128-w slice
                lab[:, t] = ((prev << kk) | sy[:, t]) & 255
        else: lab = sy
        lab = torch.as_tensor(lab); qv = T[k][lab] * Ss[r_, b_][:, None, None]                     # (m, ns, d)
        Ys[torch.as_tensor(r_), torch.as_tensor(b_)] = qv.reshape(len(r_), 256)
    Yq = torch.empty(rows, n); Yq[prm] = Ys.reshape(rows, n)
    return dict(rows=rows, n=n, DB=DB, wmax=float(np.float16(wm)), c=torch.as_tensor(c), prm=prm, Tt=Tt, cs=cs, dl=dl, Sb=Sb), Yq

# ------------------------------------------------------------------ RHT
def make_fw(dev):
    g = torch.Generator().manual_seed(0); SG = (torch.randint(0, 2, (256,), generator=g) * 2 - 1).float().to(dev)
    def fwht(x):
        sh = x.shape; x = x.reshape(-1, 256); h = 1
        while h < 256:
            x = x.view(-1, 256 // (2 * h), 2, h); x = torch.cat([x[:, :, 0] + x[:, :, 1], x[:, :, 0] - x[:, :, 1]], 2).view(-1, 256); h *= 2
        return (x / 16).view(sh)
    fw = lambda Z: fwht(Z.reshape(Z.shape[0], -1, 256) * SG).reshape(Z.shape)
    ifw = lambda Z: (fwht(Z.reshape(Z.shape[0], -1, 256)) * SG).reshape(Z.shape)
    return fw, ifw

# ------------------------------------------------------------------ one tensor
def a1_tensor(W, c, Hf, sw, DB, budget_fn, codes, CDA, alpha=1.05, dev="cuda", damp=0.1):
    """W (rows,n) fp32 GPU; c (n,) fp32 GPU (column prescale, the stored value); Hf (n,n) fp32 fit H in ORIGINAL space (GPU);
    budget_fn(rows, n, DB) -> index-bit budget for the whole tensor. Returns dict with Yq, Wq, LBL, side info and wall times."""
    fw, ifw = make_fw(dev); rows, n = W.shape; nbk = n // 256
    Y = fw(W * c); rmsb = Y.view(rows, -1, 256).pow(2).mean(2).sqrt().clamp_min(1e-12).cpu(); amaxb = Y.view(rows, -1, 256).abs().amax(2).clamp_min(1e-12).cpu()
    Hc = Hf / c[:, None] / c[None, :]; Hp = fw(fw(Hc).T.contiguous()); del Hc
    gch = n <= int(os.environ.get("GPUCHOL_MAXN", "4096")) and str(dev).startswith("cuda")   # R48 VRi speed-up: float64 chain on the GPU for n <= 4096
    Hq = Hp.double() if gch else Hp.double().cpu(); Hq.diagonal().add_(damp * float(Hq.diag().mean())); Lc = torch.linalg.cholesky(Hq); del Hq
    Hi = torch.cholesky_inverse(Lc); del Lc; Uc = torch.linalg.cholesky(Hi, upper=True); del Hi; Uc = Uc.cpu()
    wb = Uc.diagonal().pow(-2).view(nbk, 256).mean(1); wb = (wb / wb.mean()).double(); Ucg = Uc.float().to(dev); del Uc
    budget = budget_fn(rows, n, DB)
    Ebk = (Y.view(rows, nbk, 256) ** 2).sum(2).double().cpu()
    prm = torch.argsort(Ebk.clamp_min(1e-30).log().mean(1)); Es = Ebk[prm] * wb[None]
    cs = alloc(Es.view(rows // TILE, TILE, nbk).sum(1), budget / TILE, CDA)
    clsS = cs.repeat_interleave(TILE, 0); cls = torch.empty_like(clsS); cls[prm] = clsS
    Sb, Tt, dl, clip = scales(cls, prm, rmsb, amaxb, alpha, DB)
    TG = gpu_tabs(codes, dev)
    old = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = False
    try: Yq, LBL = encode(Y, Ucg, cls.to(dev), Sb.to(dev), TG)
    finally: torch.backends.cuda.matmul.allow_tf32 = old
    wmax = float(np.float16(float(W.abs().max()) * 1.001)); Wq = (ifw(Yq) / c).clamp(-wmax, wmax)
    idx_bits = float((256 * CR[cls]).sum())
    del Y, Hp, Ucg
    return dict(Yq=Yq, Wq=Wq, LBL=LBL, cls=cls, cs=cs, prm=prm, Tt=Tt, dl=dl, Sb=Sb, wmax=wmax, clip=clip, idx_bits=idx_bits, budget=budget,
                hist=torch.bincount(cls.flatten(), minlength=NC).tolist())

def dmetric(W, Wq, H):
    dW = Wq - W; return float(((dW @ H) * dW).sum() / ((W @ H) * W).sum())
