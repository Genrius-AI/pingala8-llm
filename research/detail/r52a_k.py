# R52a ablations: r48k (kernel-exact VRii-C) generalised to 4/8/16/32 trellis states + rate-classes-off allocation.
# Imports r48k unchanged and patches module globals. Trellis forms (feedforward Ungerboeck, r45_ectcq.tr_ff):
#   z1 = u ^ par(s & M1), z0 = par(s & M0) (coset half bit h = z0), state bit i-1 = u_{t-i}, reset per 128-col slice.
#   4: (h0,h1) = (5,2)    -> M1 0x2,  M0 0x1      8: (13,04) -> M1 0x5, M0 0x2 (= r48k "Pingala 8")
#   16: (23,04)           -> M1 0x9,  M0 0x2      32: (45,10) -> M1 0x12, M0 0x4
import os, sys, math, struct, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r48k as K, r48_a1gpu as A1
MASKS = {4: (0x2, 0x1), 8: (0x5, 0x2), 16: (0x9, 0x2), 32: (0x12, 0x4)}
CHUNK0 = list(K.CHUNK)
def set_ns(ns):
    K.NSC[0] = ns; K.M1, K.M0 = MASKS[ns]
    K.Z0 = torch.tensor([K.par(s_ & K.M0) for s_ in range(ns)]); K.UX = torch.tensor([K.par(s_ & K.M1) for s_ in range(ns)])
    f = max(1, ns // 16); K.CHUNK[:] = [max(16, c // f) for c in CHUNK0]          # 32 states: half the rows per Viterbi chunk

def unpack_vrk1(b, T):
    """r48k._unpack_vrk1 with the trellis half bit generalised (h = par(ust & M0), u = bit ^ par(ust & M1))."""
    mg, rows, n, tr, pb, wm = struct.unpack_from(K.HDR, b, 0); assert mg == b"VRK1"; o = struct.calcsize(K.HDR); tiles = rows // 32; nbk = n // 256
    def take(k_):
        nonlocal o; r = b[o:o + k_]; o += k_; return r
    c = torch.as_tensor(np.frombuffer(take(2 * n), np.float16).astype(np.float32))
    prm = torch.as_tensor(A1._unpack(take((rows * pb + 7) // 8), rows, pb).astype(np.int64))
    Ss = torch.as_tensor(np.frombuffer(take(2 * rows * nbk), np.float16).astype(np.float32).reshape(rows, nbk))
    meta = np.frombuffer(take(4 * tiles * nbk), np.uint32).reshape(tiles, nbk); cs = (meta >> 29).astype(np.int64); stream = np.frombuffer(b[o:], np.uint8)
    clsr = np.repeat(cs, 32, 0); bpb = np.array(K.BPB)[clsr]; off = np.concatenate([[0], np.cumsum(bpb.ravel())])[:-1].reshape(rows, nbk)
    assert int(bpb.sum()) == len(stream)
    Ys = torch.zeros(rows, nbk, 256); NS = K.NSC[0]
    for k in range(1, K.NC):
        r_, b_ = np.nonzero(clsr == k)
        if len(r_) == 0: continue
        sy = A1._unpack(stream[(off[r_, b_][:, None] + np.arange(K.BPB[k])[None, :]).ravel()].tobytes(), len(r_) * 64, K.NW[k]).reshape(len(r_), 64)
        f1 = sy & ((1 << K.W1[k]) - 1); f2 = (sy >> K.W1[k]) & ((1 << K.W2[k]) - 1); f3 = (sy >> (K.W1[k] + K.W2[k])) & ((1 << K.W3[k]) - 1)
        f1, f2, f3 = (torch.as_tensor(v.astype(np.int64)) for v in (f1, f2, f3))
        Bt = T["A"] if tr else T["B"]
        v = T["A"][f1] * (1 << K.S1[k])
        if K.W2[k]: v = v + Bt[f2] * (1 << K.S2[k])
        if K.W3[k]: v = v + Bt[f3]
        if tr:
            h = torch.zeros(len(r_), 64); ust = torch.zeros(len(r_), dtype=torch.long); bit = f1 >> (K.W1[k] - 1)
            pz = torch.tensor([K.par(s_ & K.M0) for s_ in range(NS)]); pu = torch.tensor([K.par(s_ & K.M1) for s_ in range(NS)])
            for t in range(64):
                if t % 32 == 0: ust = torch.zeros(len(r_), dtype=torch.long)
                h[:, t] = pz[ust].float(); u = bit[:, t] ^ pu[ust]; ust = ((ust << 1) | u) & (NS - 1)
            v = v + h[:, :, None] * (K.MC(k) * T["c"])[None, None, :]
        Ys[torch.as_tensor(r_), torch.as_tensor(b_)] = (v * Ss[torch.as_tensor(r_), torch.as_tensor(b_)][:, None, None]).reshape(len(r_), 256)
    Yq = torch.empty(rows, n); Yq[prm] = Ys.reshape(rows, n)
    Sb = torch.empty(rows, nbk); Sb[prm] = Ss
    return dict(rows=rows, n=n, tr=tr, wmax=float(np.float16(wm)), c=c, prm=prm, cs=cs, Sb=Sb), Yq
def unpack_blob(b, T): return unpack_vrk1(K.to_vrk1(b), T)
K.unpack_blob = unpack_blob

def tables_blob(T, tr):
    code = {4: 3, 8: 2, 16: 1, 32: 4}[K.NSC[0]] if tr else 0
    return b"VRKT" + struct.pack("<B", code) + T["A"].to(torch.int8).numpy().tobytes() + T["B"].to(torch.int8).numpy().tobytes() + T["c"].to(torch.int8).numpy().tobytes() + bytes(K.S1) + bytes(K.S2)

# rate classes OFF: one class per tensor = the class whose b/w is nearest the tensor's index budget; no salient floor.
_alloc0 = K.alloc
RATEOFF = [False]
def alloc(E, budget, CD, floor=None):
    if not RATEOFF[0]: return _alloc0(E, budget, CD, floor)
    tl, nb = E.shape; r = budget / (tl * nb * 256)
    k = int(torch.argmin((K.CR[1:] - r).abs())) + 1
    return torch.full((tl, nb), k, dtype=torch.long)
K.alloc = alloc
