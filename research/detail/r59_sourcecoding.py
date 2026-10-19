# R59: how far is the coset-trellis code from the Gaussian distortion-rate bound at 2 bpw, versus llama.cpp IQ2_XXS / Q2_K?
# Stages (run each as: CPU=1 ~/repo/bin/job python r59_sourcecoding.py <stage>):
#   gauss : iid N(0,1) source, 4 seeds x 2^20 4-dim symbols. Plain-MSE (H = identity) Viterbi with the shipped table A and coset c
#           (research/work/r48k3/tables_vrc.pt) for S = 4/8/16/32 states, VRi (no trellis), Lloyd references, retrained-table variant.
#   check : bit-exact comparison of this file's Viterbi against the shipped encoder r48k.vit_k (via r52a_k.set_ns) on identity metric.
#   iq    : same Gaussian tensors written to a GGUF, quantized by llama.cpp's llama-quantize (CPU, uniform imatrix), dequantized by gguf-py.
#   real  : 3 real Hadamard-rotated, imatrix-weighted Qwen3-4B tensors (as r23_bound.py): our code, IQ2_XXS (llama.cpp, real imatrix), bound.
# Rate accounting (every bit): our code 8 index bits / 4 weights = 2.0 bpw + fp16 scale / 256 weights = 0.0625 bpw -> 2.0625 bpw. IQ2_XXS 2.0625 (66 B / 256 w).
# Bound: Gaussian D(R) = 2^(-2R) at the same total rate, SNR_bound = 6.0206 R dB.
import os, sys, math, time, json, struct, subprocess, shutil
import numpy as np, torch

ROOT = os.path.expanduser("~/repo")
sys.path.insert(0, ROOT + "/research/detail"); sys.path.insert(0, ROOT + "/llama.cpp/gguf-py")
TB = ROOT + "/research/work/r48k3/"; WD = ROOT + "/work/r59sc/"; LOG = ROOT + "/logs/r59_sourcecoding.log"
os.makedirs(WD, exist_ok=True)
G3 = json.load(open(TB + "tables_vrc.json"))["G"]["3"]           # class-3 (2 bpw) gain: scale S = alpha * rms / G3  (r48k.k_tensor)
ALPHA_SHIPPED = 1.05                                              # k_tensor default alpha
MASKS = {4: (0x2, 0x1), 8: (0x5, 0x2), 16: (0x9, 0x2), 32: (0x12, 0x4)}   # (M1, M0), r52a_k.py: z1 = u ^ par(s & M1), z0 = par(s & M0)
RIDX = 2.0; RTOT = 2.0 + 16 / 256                                  # index bpw; total bpw with one fp16 scale per 256 weights
NSYM_LOG = 20; EVAL_SEEDS = [0, 1, 2, 3]; TUNE_SEED = 1000
torch.set_num_threads(int(os.environ.get("R59_THREADS", "8")))

def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    with open(LOG, "a") as f: f.write(s + "\n")
par = lambda v: bin(v).count("1") & 1
def snr(mse): return -10 * math.log10(mse)
def bound_snr(R): return 20 * math.log10(2) * R               # -10 log10 2^(-2R)

# ------------------------------------------------------------------ data and tables
def gauss(seed, n=1 << (NSYM_LOG + 2)):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)        # float32, same values as the GGUF tensors
def load_tables():
    T = torch.load(TB + "tables_vrc.pt"); T0 = torch.load(TB + "tables_vri.pt")
    return dict(A=T["A"].double(), c=T["c"].double(), A_vri=T0["A"].double())

# ------------------------------------------------------------------ plain-MSE costs per (coset h, stage-1 MSB b)
def half_costs(X, tabs, chunk=1 << 15):
    """X (N,4) float64 targets (table units). tabs: list of (256,4) codebooks. Returns cost (N,H,2) float32 = min squared distance over
    the 128 entries whose index MSB is b, idx (N,H,2) int16 = that entry's full 8-bit index (MSB 0 <-> entries 0..127, as r48k.best_cw)."""
    N = X.shape[0]; H = len(tabs); cost = torch.empty(N, H, 2); idx = torch.empty(N, H, 2, dtype=torch.int16); x2 = (X * X).sum(1)
    off = torch.tensor([0, 128])
    for h, Tb in enumerate(tabs):
        t2 = (Tb * Tb).sum(1)
        for i0 in range(0, N, chunk):
            x = X[i0:i0 + chunk]; d = (x2[i0:i0 + chunk, None] - 2 * (x @ Tb.T) + t2[None]).clamp_min(0).view(-1, 2, 128)
            m, a = d.min(-1); cost[i0:i0 + chunk, h] = m.float(); idx[i0:i0 + chunk, h] = (a + off).short()
    return cost, idx

def viterbi(cost, idx, S, masks=None):
    """cost (R,L,2,2) [seq, t, h, b], idx same (int16). Shift-register trellis of r48k/r52a_k with S states, start state 0, free end.
    Branch (state s, half bit b) -> next state ((s<<1)|(b ^ par(s&M1))) & (S-1); coset h = par(s&M0). Returns best cost (R,), idx path (R,L), h path (R,L)."""
    M1, M0 = masks or MASKS[S]; R, L = cost.shape[:2]; sa = torch.arange(S)
    z0 = torch.tensor([par(int(s) & M0) for s in sa]); ux = torch.tensor([par(int(s) & M1) for s in sa])
    d = sa; src0 = d >> 1; src1 = src0 | (S >> 1); b0 = (d & 1) ^ ux[src0]; b1 = (d & 1) ^ ux[src1]; h0 = z0[src0]; h1 = z0[src1]
    PM = torch.full((R, S), float("inf")); PM[:, 0] = 0; BP = torch.empty(L, R, S, dtype=torch.bool)
    for t in range(L):
        c = cost[:, t]; c0 = PM[:, src0] + c[:, h0, b0]; c1 = PM[:, src1] + c[:, h1, b1]
        w = c1 < c0; PM = torch.where(w, c1, c0); BP[t] = w
    best, s = PM.min(1); rr = torch.arange(R); IP = torch.empty(R, L, dtype=torch.int16); HP = torch.empty(R, L, dtype=torch.long)
    sh = S.bit_length() - 2                                              # log2(S) - 1
    for t in reversed(range(L)):
        w = BP[t][rr, s].long(); src = (s >> 1) | (w << sh); h = z0[src]; b = (s & 1) ^ ux[src]
        IP[:, t] = idx[rr, t, h, b]; HP[:, t] = h; s = src
    return best, IP, HP

def recon(IP, HP, A, c):                                                 # encoder-side reconstruction (table units)
    return A[IP.long()] + HP[..., None].double() * c

def decode(IP, S, A, c, masks=None):
    """Decoder from the stored 8-bit indices ONLY (the stored MSB is the label z1): u = msb ^ par(ust & M1), h = par(ust & M0),
    ust = ((ust << 1) | u) & (S-1), state reset every 32 symbols. Mirrors r52a_k.unpack_vrk1 and r49_vr.cu cpu_vr."""
    M1, M0 = masks or MASKS[S]; R, L = IP.shape; ust = torch.zeros(R, dtype=torch.long); out = torch.empty(R, L, 4, dtype=torch.float64)
    pz = torch.tensor([par(s & M0) for s in range(S)]); pu = torch.tensor([par(s & M1) for s in range(S)])
    for t in range(L):
        ix = IP[:, t].long(); bit = ix >> 7; h = pz[ust]; u = bit ^ pu[ust]; ust = ((ust << 1) | u) & (S - 1)
        out[:, t] = A[ix] + h[:, None].double() * c
    return out

# ------------------------------------------------------------------ scale: S_b = fp16(alpha * rms_b / G3), one per 256 weights (k_tensor)
def targets(w, alpha):
    B = w.reshape(-1, 256).double(); rms = B.pow(2).mean(1).sqrt().clamp_min(1e-12)
    sc = (alpha * rms / G3).to(torch.float16).double()
    return (B / sc[:, None]).reshape(-1, 4), sc                          # (Nsym,4) targets in table units, (Nblk,) scales

def code_mse(w, alpha, T, Ss, A=None, c=None, check_decode=False, masks=None):
    """MSE per weight (absolute, source variance 1) of the trellis code for each S in Ss, plus VRi variants. Costs are S independent."""
    A = T["A"] if A is None else A; c = T["c"] if c is None else c
    X, sc = targets(w, alpha); R = X.shape[0] // 32; cost, idx = half_costs(X, [A, A + c]); cost = cost.view(R, 32, 2, 2); idx = idx.view(R, 32, 2, 2)
    scs = sc.repeat_interleave(2); n = w.numel(); out = {}                # 2 sequences (128 weights) per 256-block
    for S in Ss:
        best, IP, HP = viterbi(cost, idx, S, masks); out[S] = float((best.double() * scs ** 2).sum()) / n
        if check_decode:
            d = decode(IP, S, A, c, masks); r = recon(IP, HP, A, c)
            assert torch.equal(d, r), "decoder from stored indices differs from encoder reconstruction"
            e = float((((X.view(R, 32, 4) - d) ** 2).sum((1, 2)) * scs ** 2).sum()) / n
            assert abs(e - out[S]) < 1e-6 * out[S], (e, out[S]); out[("dec", S)] = e
    return out

def vri_mse(w, alpha, At):                                               # no trellis: nearest entry of the 256-entry table, 8 bits / 4 weights
    X, sc = targets(w, alpha); cost, _ = half_costs(X, [At]); m = cost[:, 0].min(1).values.double()
    return float((m.view(-1, 64).sum(1) * sc ** 2).sum()) / w.numel()

def tune(fn, grid, refine=True):
    """grid search on a scalar alpha; fn(alpha) -> mse (or dict of mse per key -> tuned per key). Returns best alpha(s) and mse."""
    res = {}
    for a in grid: res[a] = fn(a)
    keys = list(res[grid[0]].keys()) if isinstance(res[grid[0]], dict) else None
    def best(key):
        return min(res, key=lambda a: res[a][key] if key is not None else res[a])
    if refine:
        extra = sorted({round(best(k) + d, 4) for k in (keys if keys else [None]) for d in (-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04)} - set(res))
        for a in extra: res[a] = fn(a)
    return res, {k: best(k) for k in (keys if keys else [None])}

# ------------------------------------------------------------------ references: Lloyd-Max scalar (2 bit) and 4-dim 8-bit LBG on the tuning seed
def lloyd(Xn, K, iters, seed=0):
    g = torch.Generator().manual_seed(seed); C = Xn[torch.randperm(Xn.shape[0], generator=g)[:K]].clone()
    for _ in range(iters):
        a = torch.empty(Xn.shape[0], dtype=torch.long)
        for i0 in range(0, Xn.shape[0], 1 << 15):
            x = Xn[i0:i0 + (1 << 15)]; a[i0:i0 + (1 << 15)] = (-2 * x @ C.T + (C * C).sum(1)[None]).argmin(1)
        num = torch.zeros_like(C).index_add_(0, a, Xn); cnt = torch.zeros(K, dtype=torch.double).index_add_(0, a, torch.ones(Xn.shape[0], dtype=torch.double))
        C = torch.where(cnt[:, None] > 0, num / cnt.clamp_min(1)[:, None], C)
    return C
def ref_mse(w, C, dim):                                                  # exact fp16 per-block rms scale, codebook in unit-rms units
    B = w.reshape(-1, 256).double(); rms = B.pow(2).mean(1).sqrt().to(torch.float16).double(); X = (B / rms[:, None]).reshape(-1, dim)
    m = torch.empty(X.shape[0], dtype=torch.double)
    for i0 in range(0, X.shape[0], 1 << 15):
        x = X[i0:i0 + (1 << 15)]; d = (x * x).sum(1, keepdim=True) - 2 * x @ C.T + (C * C).sum(1)[None]; m[i0:i0 + (1 << 15)] = d.min(1).values
    return float((m.view(-1, 256 // dim).sum(1) * rms ** 2).sum()) / w.numel()

# ------------------------------------------------------------------ joint (A, c) retraining for one S (Lloyd on the trellis; r49_train 'joint' mode in spirit)
def retrain(T, S, w_tune, alpha, iters=12, masks=None):
    A = T["A"].clone(); c = T["c"].clone(); X, sc = targets(w_tune, alpha); R = X.shape[0] // 32
    for it in range(iters):
        cost, idx = half_costs(X, [A, A + c]); best, IP, HP = viterbi(cost.view(R, 32, 2, 2), idx.view(R, 32, 2, 2), S, masks)
        ix = IP.long().flatten(); h = HP.flatten().double(); xx = X.reshape(R * 32, 4)
        num = torch.zeros(256, 4, dtype=torch.double).index_add_(0, ix, xx - h[:, None] * c); cnt = torch.zeros(256, dtype=torch.double).index_add_(0, ix, torch.ones_like(h))
        A = torch.where(cnt[:, None] > 0, num / cnt.clamp_min(1)[:, None], A); m = h > 0.5
        c = (xx[m] - A[ix[m]]).mean(0)
    return A.round().clamp(-64, 64), c.round().clamp(-128, 127)

# ------------------------------------------------------------------ stage: gauss
def stage_gauss():
    T = load_tables(); Ss = [4, 8, 16, 32]; t0 = time.time()
    P(f"== R59 gauss {time.strftime('%F %T')}  G3={G3:.6f} nsym=2^{NSYM_LOG} per seed, eval seeds {EVAL_SEEDS}, tune seed {TUNE_SEED}")
    P(f"bound: D(R)=2^(-2R). SNR_bound(2.0625)={bound_snr(RTOT):.3f} dB, SNR_bound(2.0)={bound_snr(RIDX):.3f} dB")
    wt = torch.as_tensor(gauss(TUNE_SEED)); grid = [round(0.80 + 0.05 * i, 2) for i in range(11)]
    # --- 1. tune alpha on the tuning seed, per S and for VRi
    def fn(a):
        r = code_mse(wt, a, T, Ss); r = {f"S{s}": v for s, v in r.items()}
        r["VRi_vrcA"] = vri_mse(wt, a, T["A"]); r["VRi_vriA"] = vri_mse(wt, a, T["A_vri"]); return r
    res, ab = tune(fn, grid); P("-- alpha tuning on the tuning seed (SNR dB), shipped alpha = 1.05")
    for k, a in ab.items(): P(f"   {k:10s} alpha*={a:.2f} SNR={snr(res[a][k]):.4f}  SNR@1.05={snr(res[ALPHA_SHIPPED][k]) if ALPHA_SHIPPED in res else float('nan'):.4f}")
    # --- 2. evaluate on the fresh seeds (decoder check on seed 0)
    R = {k: [] for k in list(ab) + [k + "@1.05" for k in ab]}
    for sd in EVAL_SEEDS:
        w = torch.as_tensor(gauss(sd)); P(f"-- eval seed {sd}  mean(w^2)={float(w.double().pow(2).mean()):.6f}")
        for tag, alpha_of in (("", lambda k: ab[k]), ("@1.05", lambda k: ALPHA_SHIPPED)):
            for S in Ss:
                k = f"S{S}"; r = code_mse(w, alpha_of(k), T, [S], check_decode=(sd == 0 and tag == ""))
                R[k + tag].append(r[S]); P(f"   {k}{tag:6s} alpha={alpha_of(k):.2f} SNR={snr(r[S]):.4f}" + ("  [decoder-from-indices check passed]" if sd == 0 and tag == "" else ""))
            for k, At in (("VRi_vrcA", T["A"]), ("VRi_vriA", T["A_vri"])):
                v = vri_mse(w, alpha_of(k), At); R[k + tag].append(v); P(f"   {k}{tag:6s} alpha={alpha_of(k):.2f} SNR={snr(v):.4f}")
    # --- 3. Lloyd references (trained on the tuning seed, exact fp16 rms scale per 256)
    B = wt.reshape(-1, 256).double(); rmsb = B.pow(2).mean(1).sqrt().to(torch.float16).double(); Xn = (B / rmsb[:, None])
    C1 = lloyd(Xn.reshape(-1, 1), 4, 60); C4 = lloyd(Xn.reshape(-1, 4), 256, 40); refs = {"Lloyd-Max scalar 2b": [], "LBG 4-dim 256 cw": []}
    for sd in EVAL_SEEDS:
        w = torch.as_tensor(gauss(sd)); refs["Lloyd-Max scalar 2b"].append(ref_mse(w, C1, 1)); refs["LBG 4-dim 256 cw"].append(ref_mse(w, C4, 4))
    for k, v in refs.items(): P(f"-- reference {k}: SNR per seed {[round(snr(x), 4) for x in v]}")
    # --- 4. retrained (A, c) per S
    RT = {}
    for S in Ss:
        a0 = ab[f"S{S}"]; Ar, cr = retrain(T, S, wt, a0); Tr = dict(A=Ar, c=cr)
        resr, abr = tune(lambda a: code_mse(wt, a, Tr, [S])[S], [round(a0 + d, 2) for d in (-0.1, -0.05, 0, 0.05, 0.1)], refine=True); a1 = abr[None]
        v = [code_mse(torch.as_tensor(gauss(sd)), a1, Tr, [S])[S] for sd in EVAL_SEEDS]; RT[S] = v
        P(f"-- retrained (A,c) for S={S}: alpha*={a1:.2f} SNR per seed {[round(snr(x), 4) for x in v]}  c={cr.tolist()}")
    # --- 5. summary
    out = dict(alpha=ab, mse=R, refs=refs, retrained={str(k): v for k, v in RT.items()}, G3=G3, seeds=EVAL_SEEDS)
    json.dump(out, open(WD + "gauss_results.json", "w")); P("-- SUMMARY (SNR dB mean +- std over seeds; gap to D(2.0625) and to D(2.0))")
    def line(name, v, rate):
        s = np.array([snr(x) for x in v]); P(f"   {name:34s} rate {rate:.4f} SNR {s.mean():.3f} +- {s.std(ddof=1):.3f}  gap_total {bound_snr(RTOT) - s.mean() if rate >= RTOT - 1e-9 else float('nan'):.3f}  gap_idx {bound_snr(RIDX) - s.mean():.3f}")
    for k in R:
        if not k.startswith("VRi_"): line(k, R[k], RTOT)
    for k in R:
        if k.startswith("VRi_"): line(k, R[k], RTOT)
    for k, v in refs.items(): line(k, v, RTOT)
    for S, v in RT.items(): line(f"retrained S{S}", v, RTOT)
    P("-- paired differences of per-seed SNR (dB), mean +- std:")
    for a, b in ((8, 4), (16, 8), (32, 16), (32, 8)):
        d = np.array([snr(x) for x in R[f"S{a}"]]) - np.array([snr(x) for x in R[f"S{b}"]]); P(f"   S{a}-S{b}: {d.mean():+.4f} +- {d.std(ddof=1):.4f}   (shipped table)")
        d = np.array([snr(x) for x in RT[a]]) - np.array([snr(x) for x in RT[b]]); P(f"   S{a}-S{b}: {d.mean():+.4f} +- {d.std(ddof=1):.4f}   (retrained tables)")
    P(f"== gauss done in {time.time() - t0:.0f} s")

# ------------------------------------------------------------------ stage: check against the shipped encoder r48k.vit_k
def stage_check():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); import r52a_k as R52, r48k as K
    T = load_tables(); Tf = dict(A=T["A"].float(), B=T["A"].float(), c=T["c"].float()); P(f"== R59 check vs r48k.vit_k {time.strftime('%F %T')}")
    w = torch.as_tensor(gauss(7)[:512 * 128]); alpha = 1.05; rows = 512
    X = w.view(rows, 128); sc = (alpha * w.view(-1, 256).double().pow(2).mean(1).sqrt() / G3).to(torch.float16).double().repeat_interleave(2).view(rows, 1)
    I128 = torch.eye(128); A4 = torch.eye(4).expand(32, 4, 4).contiguous()
    for S in (4, 8, 16, 32):
        R52.set_ns(S); Q, E, IDX = K.vit_k(X.clone(), I128, A4, Tf, sc.float(), 3, True)
        Xt = (X.double() / sc).reshape(-1, 4); Rr = rows; cost, idx = half_costs(Xt, [T["A"], T["A"] + T["c"]])
        best, IP, HP = viterbi(cost.view(Rr, 32, 2, 2), idx.view(Rr, 32, 2, 2), S); Qm = (recon(IP, HP, T["A"], T["c"]).reshape(rows, 128) * sc)
        m_ship = float(((Q.double() - X.double()) ** 2).mean()); m_mine = float(((Qm - X.double()) ** 2).mean())
        same_idx = float((IDX[:, :, 0].long() == IP.long()).double().mean())
        P(f"   S={S:2d}: MSE shipped vit_k {m_ship:.8f}  this file {m_mine:.8f}  rel diff {abs(m_ship - m_mine) / m_ship:.2e}  identical index fraction {same_idx:.6f}  max|dQ| {float((Q.double() - Qm).abs().max()):.3e}")
    P("== check done")

# ------------------------------------------------------------------ stage: llama.cpp quantizers on the same Gaussian tensors
def stage_iq():
    import gguf
    from gguf import GGUFWriter, GGUFReader
    from gguf.quants import dequantize
    NR, NC = 4096, 1024; nt = len(EVAL_SEEDS); t0 = time.time(); P(f"== R59 iq {time.strftime('%F %T')}  llama.cpp: {ROOT}/llama.cpp/build-cuda/bin/llama-quantize --pure --imatrix <uniform ones>")
    f32 = WD + "gauss_f32.gguf"; w_ = GGUFWriter(f32, "llama"); w_.add_name("r59-gauss"); w_.add_context_length(2048); w_.add_embedding_length(NC); w_.add_block_count(nt)
    w_.add_feed_forward_length(NR); w_.add_head_count(8); w_.add_head_count_kv(8); w_.add_layer_norm_rms_eps(1e-5); w_.add_rope_dimension_count(128); w_.add_file_type(0)
    W = {}
    for i, sd in enumerate(EVAL_SEEDS): W[i] = gauss(sd).reshape(NR, NC); w_.add_tensor(f"blk.{i}.ffn_up.weight", W[i])
    w_.write_header_to_file(); w_.write_kv_data_to_file(); w_.write_tensors_to_file(); w_.close()
    im = WD + "uniform.imatrix"; g = GGUFWriter(im, "imatrix"); g.add_string("general.type", "imatrix"); g.add_array("imatrix.datasets", ["uniform-ones"])
    g.add_uint32("imatrix.chunk_count", 1); g.add_uint32("imatrix.chunk_size", 512)
    for i in range(nt):
        g.add_tensor(f"blk.{i}.ffn_up.weight.in_sum2", np.ones(NC, dtype=np.float32)); g.add_tensor(f"blk.{i}.ffn_up.weight.counts", np.ones(1, dtype=np.float32))
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()
    out = {}
    for ty in ("IQ2_XXS", "IQ2_XS", "IQ2_M", "Q2_K"):                # NB: ftype IQ2_S means tensor type IQ2_XS in llama.cpp; ftype IQ2_M --pure gives tensor type IQ2_S
        q = WD + f"gauss_{ty.lower()}.gguf"; cmd = [ROOT + "/llama.cpp/build-cuda/bin/llama-quantize", "--pure", "--imatrix", im, f32, q, ty, "8"]
        r = subprocess.run(cmd, capture_output=True, text=True); P(f"-- {' '.join(cmd[1:])} -> rc={r.returncode}");
        if r.returncode: P(r.stderr[-1500:]); continue
        rd = GGUFReader(q); mses = []; bits = 0; nw = 0
        for t in rd.tensors:
            i = int(t.name.split(".")[1]); Q = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float64).reshape(NR, NC)
            mses.append(float(((W[i].astype(np.float64) - Q) ** 2).mean())); bits += int(t.data.nbytes) * 8; nw += NR * NC; tname = t.tensor_type.name
        bpw = bits / nw; s = np.array([snr(m) for m in mses]); out[ty] = dict(tensor_type=tname, bpw=bpw, snr=s.tolist(), mse=mses)
        P(f"   ftype {ty} -> tensor type {tname}: bpw (from file bytes) {bpw:.4f}  SNR per tensor {np.round(s, 4).tolist()}  mean {s.mean():.3f} +- {s.std(ddof=1):.3f}  D(R) bound {bound_snr(bpw):.3f}  gap {bound_snr(bpw) - s.mean():.3f}")
        os.remove(q)
    json.dump(out, open(WD + "iq_results.json", "w")); os.remove(f32); os.remove(im); P(f"== iq done in {time.time() - t0:.0f} s (temporary GGUFs deleted)")

# ------------------------------------------------------------------ stage: real tensors (as r23_bound.py)
def fwht(x):
    x = x.copy(); h = 1
    while h < x.shape[-1]:
        a = x.reshape(*x.shape[:-1], -1, 2, h); a0, a1 = a[..., 0, :].copy(), a[..., 1, :].copy(); a[..., 0, :], a[..., 1, :] = a0 + a1, a0 - a1; h *= 2
    return x / math.sqrt(x.shape[-1])
def wf_bound(v, R):                                                        # reverse water-filling (r23_bound.bound)
    lo, hi = math.log(v.min() * 1e-8), math.log(v.max())
    for _ in range(100):
        th = math.exp((lo + hi) / 2); r = np.maximum(0, 0.5 * np.log2(v / th)).mean(); lo, hi = ((lo + hi) / 2, hi) if r > R else (lo, (lo + hi) / 2)
    return np.minimum(th, v).sum() / v.sum()

def real_iq(ref, im, M, NAMES):
    from gguf import GGUFReader, GGUFWriter
    from gguf.quants import dequantize
    # llama.cpp IQ2_XXS with the REAL imatrix on exactly these three tensors
    f32 = WD + "real_f32.gguf"; w_ = GGUFWriter(f32, "llama"); w_.add_name("r59-real"); w_.add_context_length(2048); w_.add_embedding_length(2560); w_.add_block_count(12)
    w_.add_feed_forward_length(9728); w_.add_head_count(20); w_.add_head_count_kv(4); w_.add_layer_norm_rms_eps(1e-5); w_.add_rope_dimension_count(128); w_.add_file_type(0)
    for n in NAMES: w_.add_tensor(n, ref[n])
    w_.write_header_to_file(); w_.write_kv_data_to_file(); w_.write_tensors_to_file(); w_.close()
    imf = WD + "real.imatrix"; g = GGUFWriter(imf, "imatrix"); g.add_string("general.type", "imatrix"); g.add_array("imatrix.datasets", ["qwen3-4B.imatrix subset"])
    g.add_uint32("imatrix.chunk_count", 200); g.add_uint32("imatrix.chunk_size", 512)
    src = {t.name: t for t in GGUFReader(M + "qwen3-4B.imatrix").tensors}
    for n in NAMES:
        g.add_tensor(n + ".in_sum2", np.asarray(src[n + ".in_sum2"].data, dtype=np.float32).ravel()); g.add_tensor(n + ".counts", np.asarray(src[n + ".counts"].data, dtype=np.float32).ravel())
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()
    IQ = {}
    for ty in ("IQ2_XXS", "IQ2_XS", "Q2_K"):
        q = WD + f"real_{ty.lower()}.gguf"; r = subprocess.run([ROOT + "/llama.cpp/build-cuda/bin/llama-quantize", "--pure", "--imatrix", imf, f32, q, ty, "8"], capture_output=True, text=True)
        P(f"-- llama-quantize {ty} on the 3 real tensors rc={r.returncode}")
        if r.returncode: P(r.stderr[-1500:]); continue
        for t in GGUFReader(q).tensors:
            Q = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float64).reshape(ref[t.name].shape); W = ref[t.name].astype(np.float64); s = im[t.name] / im[t.name].mean()
            IQ[(ty, t.name)] = float((s[None, :] * (W - Q) ** 2).sum() / (s[None, :] * W ** 2).sum())
        os.remove(q)
    os.remove(f32); os.remove(imf); return IQ

def stage_realiq():
    from gguf import GGUFReader
    M = ROOT + "/models/"; NAMES = ["blk.10.attn_q.weight", "blk.10.ffn_up.weight", "blk.10.ffn_down.weight"]
    P(f"== R59 realiq {time.strftime('%F %T')}: llama.cpp quantizers, real imatrix, weighted rel MSE in the native domain: sum s_j (W-Wq)^2 / sum s_j W^2")
    ref = {t.name: np.asarray(t.data, dtype=np.float32).reshape(t.shape[::-1]) for t in GGUFReader(M + "qwen3-4B-f16.gguf").tensors if t.name in NAMES}
    im = {t.name[:-8]: np.asarray(t.data, dtype=np.float64).ravel() for t in GGUFReader(M + "qwen3-4B.imatrix").tensors if t.name.endswith(".in_sum2")}
    IQ = real_iq(ref, im, M, NAMES); out = {f"{ty}|{n}": v for (ty, n), v in IQ.items()}
    for (ty, n), v in IQ.items(): P(f"   llama.cpp {ty:8s} {n:24s} weighted relMSE {v:.5f}  SNR {snr(v):.3f} dB")
    json.dump(out, open(WD + "real_iq_results.json", "w")); P("== realiq done")

def stage_real():
    from gguf import GGUFReader, GGUFWriter
    from gguf.quants import dequantize
    M = ROOT + "/models/"; NAMES = ["blk.10.attn_q.weight", "blk.10.ffn_up.weight", "blk.10.ffn_down.weight"]; T = load_tables(); Ss = [4, 8, 16, 32]; t0 = time.time()
    P(f"== R59 real {time.strftime('%F %T')}  tensors {NAMES}  (imatrix-weighted, random-sign Hadamard per 256, as r23_bound.py)")
    rd = GGUFReader(M + "qwen3-4B-f16.gguf"); ref = {t.name: np.asarray(t.data, dtype=np.float32).reshape(t.shape[::-1]) for t in rd.tensors if t.name in NAMES}
    im = {t.name[:-8]: np.asarray(t.data, dtype=np.float64).ravel() for t in GGUFReader(M + "qwen3-4B.imatrix").tensors if t.name.endswith(".in_sum2")}
    IQ = real_iq(ref, im, M, NAMES)
    rng = np.random.default_rng(0); res = {}
    for name in NAMES:
        W = ref[name].astype(np.float64); s = im[name] / im[name].mean(); Y = W * np.sqrt(s); sg = np.where(rng.random(256) < 0.5, -1.0, 1.0)
        Yr = fwht(Y.reshape(Y.shape[0], -1, 256) * sg).reshape(Y.shape); v = (Yr.reshape(-1, 256) ** 2).mean(1); y = torch.as_tensor(Yr.astype(np.float32).ravel()); den = float((Yr ** 2).sum()); n = y.numel()
        bnd = wf_bound(v, RIDX); P(f"== {name} shape {W.shape}  water-filling bound (index rate 2.0, scale side-info 0.0625): rel MSE {bnd:.5f} = {snr(bnd):.3f} dB;  flat Gaussian D(2.0625) = {bound_snr(RTOT):.3f} dB")
        sub = y[: 1 << 22]                                                  # alpha tuned on the first 2^20 symbols of the same tensor
        rr = {}
        def fn(a):
            r = code_mse(sub, a, T, Ss); r = {f"S{s_}": x for s_, x in r.items()}; r["VRi_vriA"] = vri_mse(sub, a, T["A_vri"]); return r
        resa, ab = tune(fn, [0.9, 1.0, 1.1, 1.2, 1.3])
        for k, a in ab.items():
            if k.startswith("S"): m = code_mse(y, a, T, [int(k[1:])])[int(k[1:])]
            else: m = vri_mse(y, a, T["A_vri"])
            rr[k] = (m * n / den, a); P(f"   {k:9s} alpha={a:.2f} weighted relMSE {m * n / den:.5f}  SNR {snr(m * n / den):.3f} dB   gap to bound {snr(bnd) - snr(m * n / den):.3f} dB")
        for ty in ("IQ2_XXS", "IQ2_XS", "Q2_K"):
            if (ty, name) in IQ: P(f"   llama.cpp {ty:8s} weighted relMSE (native domain, real imatrix) {IQ[(ty, name)]:.5f}  SNR {snr(IQ[(ty, name)]):.3f} dB")
        res[name] = dict(bound=bnd, ours={k: v for k, v in rr.items()}, iq={ty: IQ.get((ty, name)) for ty in ("IQ2_XXS", "IQ2_XS", "Q2_K")})
    json.dump(res, open(WD + "real_results.json", "w")); P(f"== real done in {time.time() - t0:.0f} s")

# ------------------------------------------------------------------ stage: generator search (is the flat S-curve an artifact of the r52a_k generators?)
def stage_gensearch():
    T = load_tables(); P(f"== R59 gensearch {time.strftime('%F %T')}: all (M1, M0) in [0,S)^2 for S=4/8/16/32 with the SHIPPED table, alpha fixed 0.995; selection on tuning seed (2^19 symbols), confirmation on eval seeds")
    wt = torch.as_tensor(gauss(TUNE_SEED)[: 1 << 21]); alpha = 0.995; X, sc = targets(wt, alpha); R = X.shape[0] // 32; cost, idx = half_costs(X, [T["A"], T["A"] + T["c"]])
    cost = cost.view(R, 32, 2, 2); idx = idx.view(R, 32, 2, 2); scs = sc.repeat_interleave(2); n = wt.numel(); out = {}
    ev = []
    for sd in EVAL_SEEDS:
        w = torch.as_tensor(gauss(sd)); X2, sc2 = targets(w, alpha); R2 = X2.shape[0] // 32; c2, i2 = half_costs(X2, [T["A"], T["A"] + T["c"]]); ev.append((c2.view(R2, 32, 2, 2), i2.view(R2, 32, 2, 2), sc2.repeat_interleave(2), w.numel()))
    for S in (4, 8, 16, 32):
        sel = {}
        for M1 in range(S):
            for M0 in range(S):
                best, _, _ = viterbi(cost, idx, S, (M1, M0)); sel[(M1, M0)] = snr(float((best.double() * scs ** 2).sum()) / n)
        rk = sorted(sel, key=lambda k: -sel[k]); r52 = MASKS[S]; P(f"-- S={S}: {len(sel)} generator pairs; r52a_k pair {r52} SNR(select seed) {sel[r52]:.4f} rank {rk.index(r52) + 1}; best {rk[0]} {sel[rk[0]]:.4f}; median {np.median(list(sel.values())):.4f}; M0=0 (no coset) {sel[(1, 0)]:.4f}")
        out[S] = {}
        for k in [r52] + [q for q in rk[:3] if q != r52]:
            v = []
            for c2, i2, scs2, n2 in ev:
                best, _, _ = viterbi(c2, i2, S, k); v.append(snr(float((best.double() * scs2 ** 2).sum()) / n2))
            out[S][str(k)] = v; P(f"   S={S} (M1,M0)={k}  eval SNR {np.mean(v):.4f} +- {np.std(v, ddof=1):.4f}" + ("   <- r52a_k generator" if k == r52 else ""))
    json.dump(out, open(WD + "gensearch_results.json", "w")); P("== gensearch done")

# ------------------------------------------------------------------ stage: best generator per S (from gensearch) + retrained (A, c): the intrinsic gap-vs-states curve
def stage_bestret():
    T = load_tables(); gs = json.load(open(WD + "gensearch_results.json")); Ss = [4, 8, 16, 32]; wt = torch.as_tensor(gauss(TUNE_SEED)); t0 = time.time()
    P(f"== R59 bestret {time.strftime('%F %T')}: best generator per S from gensearch (top-1 by selection seed) + (A,c) retrained on the tuning seed for that generator (12 Lloyd-Viterbi iterations from the shipped table, int8-rounded)")
    out = {}
    for S in Ss:
        keys = list(gs[str(S)].keys()); k = keys[1] if len(keys) > 1 else keys[0]; masks = tuple(int(x) for x in k.strip("()").split(","))
        Ar, cr = retrain(T, S, wt, 0.995, masks=masks); Tr = dict(A=Ar, c=cr)
        res, ab = tune(lambda a: code_mse(wt, a, Tr, [S], masks=masks)[S], [0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03], refine=True); a1 = ab[None]; v = []
        for sd in EVAL_SEEDS:
            w = torch.as_tensor(gauss(sd)); v.append(snr(code_mse(w, a1, Tr, [S], check_decode=(sd == 0), masks=masks)[S]))
        out[S] = dict(masks=masks, alpha=a1, snr=v, c=cr.tolist()); m = np.mean(v)
        P(f"   S={S:2d} generator (M1,M0)={masks} alpha*={a1:.2f} retrained SNR {m:.4f} +- {np.std(v, ddof=1):.4f}  gap_total {bound_snr(RTOT) - m:.3f}  gap_idx {bound_snr(RIDX) - m:.3f}  c={cr.tolist()}  [decoder check passed]")
    for a, b in ((8, 4), (16, 8), (32, 16)):
        d = np.array(out[a]["snr"]) - np.array(out[b]["snr"]); P(f"   S{a}-S{b}: {d.mean():+.4f} +- {d.std(ddof=1):.4f} dB (best generators, retrained tables)")
    json.dump(out, open(WD + "bestret_results.json", "w")); P(f"== bestret done in {time.time() - t0:.0f} s")

if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "gauss"; {"gauss": stage_gauss, "check": stage_check, "iq": stage_iq, "real": stage_real, "gensearch": stage_gensearch, "realiq": stage_realiq, "bestret": stage_bestret}[st]()
