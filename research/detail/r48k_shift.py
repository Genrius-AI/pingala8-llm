# R48k: Gaussian D/dim of kernel residual classes vs the stage-1 shift S1 (and S2), B trained per class (upper bound), 1-state, int7 A/B.
# Purpose: find shifts that make classes 5-7 useful (kernel r49_vr.cu uses S1 = 2,4,4,8 and S2 = 4 for class 7).
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r48k as K
torch.set_num_threads(4); dev = "cpu"; g = torch.Generator().manual_seed(11); XG = torch.randn(512, 128, generator=g); XT = torch.randn(1024, 128, generator=g)
I128 = torch.eye(128); A4 = torch.eye(4).expand(32, 4, 4).contiguous()
T0 = torch.load("~/repo/research/work/r48k/tables_vri.pt"); qa = json.load(open("~/repo/research/work/r48k/tables_vri.json"))["qa"]
def run(T, k, G, X):
    S = torch.full((X.shape[0], 1), 1.0 / G); Q, I = [], []
    for r0 in range(0, X.shape[0], 256): q, _, idx = K.vit_k(X[r0:r0 + 256], I128, A4, T, S[r0:r0 + 256], k, False); Q.append(q); I.append(idx)
    Q = torch.cat(Q); return float((Q - X).pow(2).mean()), torch.cat(I)
def best(T, k, X):
    G0 = qa * (1 << K.S1[k]); b = None
    for j in range(-12, 13, 3):
        D, _ = run(T, k, G0 * 2 ** (j / 16), X)
        if b is None or D < b[1]: b = (G0 * 2 ** (j / 16), D)
    return b
for k, s1s in ((5, (1, 2, 3, 4)), (6, (2, 3, 4)), (7, ((4, 2), (5, 2), (6, 3), (8, 4)))):
    for s in s1s:
        if k == 7: K.S1[7], K.S2[7] = s
        else: K.S1[k] = s
        T = {a: b.clone() for a, b in T0.items()}
        for it in range(4):
            G, D = best(T, k, XG); _, IDX = run(T, k, G, XG); X4 = (XG * G).view(-1, 32, 4); i1, i2, i3 = IDX[..., 0], IDX[..., 1], IDX[..., 2]
            base = X4 - T["A"][i1] * (1 << K.S1[k]); Sm = torch.zeros(256, 4, dtype=torch.float64); Cn = torch.zeros(256, dtype=torch.float64)
            tg = [(i2, base / (1 << K.S2[k]))] if k < 7 else [(i2, (base - T["B"][i3]) / (1 << K.S2[k])), (i3, base - (1 << K.S2[k]) * T["B"][i2])]
            for ii, tt in tg: Sm.index_add_(0, ii.flatten(), tt.reshape(-1, 4).double()); Cn.index_add_(0, ii.flatten(), torch.ones(ii.numel(), dtype=torch.float64))
            m = Cn > 0; T["B"][m] = torch.round(Sm[m] / Cn[m, None]).float().clamp(-64, 64)
        G, _ = best(T, k, XG); D, _ = run(T, k, G, XT)
        print(f"SHIFT class {k} S1/S2 {s} D/dim {D:.5f} (kernel shifts: c5 .043, c6 .033, c7 .030)", flush=True)
    K.S1[:] = [0, 0, 0, 0, 2, 4, 4, 8]; K.S2[:] = [0, 0, 0, 0, 0, 0, 0, 4]
