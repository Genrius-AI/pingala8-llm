# R48k: train the KERNEL-EXACT tables (r49_vr.cu layout): A (256 x int8x4, |v| <= 64, prefixes 16/64/256 = classes 1/2/3,
# stage 1 of classes 4-7), B (VRi residual table, prefixes 4/16/256), c (VRii-C coset int8x4), per-class gain G_k (S = alpha*rms/G_k)
# and the Gaussian class D table used by the allocation. A comes from the solver's joint tables (research/work/r49/joint_{one,ung16}.pt,
# r49_train.py joint mode). Gains and D are measured with the build's own encoder (r48k.vit_k, identity metric, no feedback).
# usage: r48k_train.py   -> research/work/r48k/tables_{vri,vrc}.pt + .json
import os, sys, json, math, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r48k as K
dev = os.environ.get("DEV", "cuda"); OUT = os.environ.get("OUT", "~/repo/research/work/r48k"); os.makedirs(OUT, exist_ok=True)
g = torch.Generator().manual_seed(11); XG = torch.randn(2048, 128, generator=g).to(dev); XT = torch.randn(4096, 128, generator=g).to(dev)
I128 = torch.eye(128, device=dev); A4 = torch.eye(4, device=dev).expand(32, 4, 4).contiguous()
def run(T, k, G, tr, X=XG):
    S = torch.full((X.shape[0], 1), 1.0 / G, device=dev); Q, IDX = [], []
    for r0 in range(0, X.shape[0], 256):
        q, _, idx = K.vit_k(X[r0:r0 + 256], I128, A4, T, S[r0:r0 + 256], k, tr); Q.append(q); IDX.append(idx)
    Q = torch.cat(Q); return float((Q - X).pow(2).mean()), torch.cat(IDX)
def gain(T, k, tr, qa):
    G0 = qa * (1 << K.S1[k]); best = None
    for j in range(-8, 9, 2):
        G = G0 * 2 ** (j / 16); D, _ = run(T, k, G, tr)
        if best is None or D < best[1]: best = (G, D)
    G0 = best[0]
    for j in (-1, 1):
        G = G0 * 2 ** (j / 16); D, _ = run(T, k, G, tr)
        if D < best[1]: best = (G, D)
    D, _ = run(T, k, best[0], tr, XT); return best[0], D
def quantA(A, extra=None):
    m = float(A.abs().max()) if extra is None else max(float(A.abs().max()), float((A.abs() + extra.abs()).max()))
    qa = 64.0 / float(A.abs().max()); return torch.round(A * qa).clamp(-64, 64), qa
for mode in os.environ.get("MODES", "vri,vrc").split(","):
    t0 = time.time(); tr = mode == "vrc"
    J = torch.load(f"~/repo/research/work/r49/joint_{os.environ.get('JTR', 'ung16') if tr else 'one'}.pt")
    Aint, qa = quantA(J["A"].float()); c = torch.round(J["c"].float() * qa) if tr else torch.zeros(4)
    T = dict(A=Aint.to(dev), B=Aint.clone().to(dev), c=c.to(dev)); G = {}; D = {0: 1.0}
    if not tr:                                                   # B: Lloyd on the stage-2/3 targets of classes 4-7 (nested prefixes 4/16/256)
        Gc = {k: qa * (1 << K.S1[k]) for k in range(4, 8)}
        for it in range(6):
            Sm = torch.zeros(256, 4, dtype=torch.float64, device=dev); Cn = torch.zeros(256, dtype=torch.float64, device=dev); tot = []
            for k in range(4, 8):
                Dk, IDX = run(T, k, Gc[k], False); tot.append(round(Dk, 5)); X4 = (XG * Gc[k]).view(-1, 32, 4); i1, i2, i3 = IDX[..., 0], IDX[..., 1], IDX[..., 2]
                base = X4 - T["A"][i1] * (1 << K.S1[k])
                if k < 7: tg = [(i2, base / (1 << K.S2[k]))]
                else: tg = [(i2, (base - T["B"][i3]) / (1 << K.S2[k])), (i3, base - (1 << K.S2[k]) * T["B"][i2])]
                for ii, tt in tg: Sm.index_add_(0, ii.flatten(), tt.reshape(-1, 4).double()); Cn.index_add_(0, ii.flatten(), torch.ones(ii.numel(), dtype=torch.float64, device=dev))
            m = Cn > 0; Bn = T["B"].clone(); Bn[m] = torch.round(Sm[m] / Cn[m, None]).float().clamp(-64, 64); T["B"] = Bn
            print(f"{mode} B it {it} D/dim c4..c7 {tot} [{time.time() - t0:.0f}s]", flush=True)
            for k in range(4, 8): Gc[k], _ = gain(T, k, False, qa) if it in (2, 5) else (Gc[k], 0)
    for k in range(1, 8):
        G[k], D[k] = gain(T, k, tr, qa); print(f"{mode} c{k} G {G[k]:.3f} D/dim {D[k]:.5f} [{time.time() - t0:.0f}s]", flush=True)
    torch.save({k_: v.cpu() for k_, v in T.items()}, f"{OUT}/tables_{mode}.pt"); json.dump(dict(G=G, D=D, qa=qa), open(f"{OUT}/tables_{mode}.json", "w"))
print("TRAIN_DONE shifts", K.S1, K.S2, flush=True)
