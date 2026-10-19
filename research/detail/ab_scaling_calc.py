# AB: scalability / impact CALC for the appendix (research/paper/appendix_scaling.tex). CPU only, no GPU.
# Reads logs/r48k3.tensors (Qwen3-4B Model B build). Everything else is arithmetic on stated assumptions.
# usage: python3 research/detail/ab_scaling_calc.py > logs/ab_scaling_calc.log ; also writes the LaTeX table bodies to $AB_OUT (default research/paper/figs/ab_tables.json)
import re, math, ast, json, os, collections
ROOT = os.path.expanduser("~/repo")
OUT = os.environ.get("AB_OUT", os.devnull)
T = {}                                                   # LaTeX fragments
def fmt(x, d=2): return f"{x:,.{d}f}"
def p(*a): print(*a)

# ------------------------------------------------------------------ P1: exact byte accounting of the native file (Qwen3-4B, MEASURED inputs)
SHAPE = {'attn_q': (4096, 2560), 'attn_k': (1024, 2560), 'attn_v': (1024, 2560), 'attn_output': (2560, 4096), 'ffn_gate': (9728, 2560), 'ffn_up': (9728, 2560), 'ffn_down': (2560, 9728)}
BPB = [0, 32, 48, 64, 80, 96, 128, 192]                  # payload bytes per 256-weight row-block, classes 0..7 (r48k.py: 64*NW/8)
CLS_BPW = [0, 1, 1.5, 2, 2.5, 3, 4, 6]
tot = collections.Counter(); H = [0] * 8; trows = []
for l in open(ROOT + "/logs/r48k3.tensors"):
    m = re.match(r'^vrc\s+blk\.(\d+)\.(\S+)\.weight\s+(\S+)\s+blob .*hist (\[[^\]]*\]).*t_q ([\d.]+)s', l)
    if not m: continue
    li, nm, hist, tq = int(m.group(1)), m.group(2), ast.literal_eval(m.group(4)), float(m.group(5)); rows, n = SHAPE[nm]
    assert sum(hist) == rows * (n // 256)
    tot['N'] += rows * n; tot['payload'] += sum(h * b for h, b in zip(hist, BPB))
    tot['prescale'] += 2 * n; tot['perm16'] += 2 * rows; tot['perm_packed'] += (rows * max(1, math.ceil(math.log2(rows))) + 7) // 8
    tot['scale'] += 2 * rows * (n // 256); tot['tilewords'] += 4 * (rows // 32) * (n // 256); tot['hdr'] += 16; tot['tables'] += 2048
    tot['tq'] += tq; trows.append((li, nm, rows * n, tq))
    for k in range(8): H[k] += hist[k]
N = tot['N']; assert N == 3633315840
container = tot['payload'] + tot['prescale'] + tot['perm_packed'] + tot['scale'] + tot['tilewords'] + tot['hdr']
native = tot['payload'] + tot['prescale'] + tot['perm16'] + tot['scale'] + tot['tilewords'] + tot['hdr'] + tot['tables'] + 8 * 27437
p("== P1 byte accounting, Qwen3-4B, 252 linears (inputs: logs/r48k3.tensors class histograms; r48k.py side_bytes; r51_native.py layout)")
p(f"N_lin {N}  payload {tot['payload']} B  = {tot['payload']*8/N:.5f} bpw")
for k in ('prescale', 'perm_packed', 'perm16', 'scale', 'tilewords', 'tables', 'hdr'): p(f"  {k:12s} {tot[k]:>12d} B = {tot[k]*8/N:.6f} bpw")
p(f"  clamp delta  {8*27437:>12d} B = {8*27437*8/N:.6f} bpw (27,437 outliers x 8 B: 4 B index pair + 4 B f32)")
p(f"container blobs (payload+prescale+packed perm+scale+tilewords+hdr) {container} B; logged SUMMARY blobs 966271936 B; diff {container-966271936}")
p(f"native linear bytes (payload+prescale+int16 perm+scale+tilewords+2048 B tables+16 B hdr per tensor + 8 B per outlier) {native} B; GGUF tensor tables 967403816 B; diff {native-967403816}")
p(f"b_lin native = {native*8/N:.5f} bpw (paper 2.1301); b_pay = {tot['payload']*8/N:.5f}; fixed 17/256 = {17/256:.8f}; rest = {(native-tot['payload']-17/256*N/8)*8/N:.6f} bpw")
P_CLS = [h / sum(H) for h in H]
p("class mix (row-block share):", [round(x, 5) for x in P_CLS], " mean class rate bpw", sum(a * b for a, b in zip(P_CLS, CLS_BPW)))
reads = [0, 1, 1, 1, 2, 2, 2, 3]
p(f"mean table reads per 4-weight symbol {sum(a*b for a,b in zip(P_CLS,reads)):.4f} (paper: 1.06-1.20)")
T['p1_rows'] = [("payload (rate-class mix)", tot['payload'], None), ("per-row fp16 scale, 1 per 256 weights", tot['scale'], None), ("tile word, 1 per 32x256 tile", tot['tilewords'], None),
                ("column prescale c (fp16 per input column)", tot['prescale'], None), ("row permutation (int16 per row)", tot['perm16'], None), ("per-tensor tables (2048 B x 252)", tot['tables'], None),
                ("per-tensor header (16 B x 252)", tot['hdr'], None), ("clamp delta (8 B x 27,437)", 8 * 27437, None)]

# ------------------------------------------------------------------ P2: encode cost
p("\n== P2 encode cost, Qwen3-4B build (logs/r48k3.tensors)")
tq_sum = tot['tq']; wall = 4623.0
p(f"sum t_q over 252 tensors {tq_sum:.1f} s = {tq_sum/N*1e6:.4f} us/weight  ({tq_sum/wall*100:.1f}% of the 4623 s script wall time)")
p(f"whole script wall 4623 s / N = {wall/N*1e6:.4f} us/weight ({N/wall/1e6:.4f} Mweights/s)")
LT = {}
for l in open(ROOT + "/logs/r48k3.tensors"):
    m = re.match(r'^LAYER (\d+) vrc \[(\d+)s\]', l)
    if m: LT[int(m.group(1))] = int(m.group(2))
per = {k: LT[k] - (LT[k - 1] if k else 0) for k in LT}
p("per-layer wall s: min", min(per.values()), "max", max(per.values()), " layers 0-3:", [per[i] for i in range(4)], " layers 23-35 mean", sum(per[i] for i in range(23, 36)) / 13)
lay_w = N / 36
st_wall = sum(per[i] for i in range(23, 36)) / 13
st_tq = sum(t for (li, nm, w, t) in trows if li >= 23) / 13
p(f"steady-state layers 23-35: wall {st_wall:.1f} s/layer = {st_wall/lay_w*1e6:.4f} us/weight ; t_q {st_tq:.1f} s/layer = {st_tq/lay_w*1e6:.4f} us/weight")
byn = collections.defaultdict(lambda: [0, 0.0])
for (li, nm, w, t) in trows: byn[nm][0] += w; byn[nm][1] += t
for nm, (w, t) in byn.items(): p(f"  {nm:12s} n_in {SHAPE[nm][1]:5d} t_q {t:7.1f} s  {t/w*1e6:.3f} us/weight")
nbar4 = sum(SHAPE[nm][0] * SHAPE[nm][1] * SHAPE[nm][1] for nm in SHAPE) / sum(SHAPE[nm][0] * SHAPE[nm][1] for nm in SHAPE)
p(f"weight-averaged input width nbar (Qwen3-4B) {nbar4:.1f}")
# evaluation counts per weight, S=8 states, from best_cw() in r48k.py
W1 = [0, 4, 6, 8, 8, 8, 8, 8]; W2 = [0, 0, 0, 0, 2, 4, 8, 8]; W3 = [0, 0, 0, 0, 0, 0, 0, 8]; S = 8; TOP = 4
def ev_step(k):          # 4-dim distance evaluations per state per trellis step
    if k == 0: return 0
    K1 = 2 ** W1[k]; e = K1
    if W2[k]: e += 2 * min(TOP, K1 // 2) * 2 ** W2[k]
    if W3[k]: e += 2 * TOP * TOP * 2 ** W3[k]
    return e
ev_w = [S * ev_step(k) / 4 for k in range(8)]
p("4-dim distance evaluations per weight by class (S=8):", ev_w)
p(f"coordinator figure: 8 states x 2 branches x 128 candidates / 4 weights = {8*2*128/4:.0f} per weight (stage-1 only, class 3)")
mix_ev = sum(a * b for a, b in zip(P_CLS, ev_w))
p(f"class-mix weighted evaluations per weight = {mix_ev:.1f}")
intra = S * sum(4 * (124 - 4 * t) for t in range(31)) / 128
inter = lambda n: n / 2
p(f"intra-block feedback MACs per weight (8 states) {intra:.1f}; inter-block feedback MACs per weight n/2 = {inter(nbar4):.0f} at nbar {nbar4:.0f}")

MODELS = collections.OrderedDict()           # name: (L, d, q_dim, kv_dim, ffn, V, tied, src)
MODELS['Qwen3-4B'] = (36, 2560, 4096, 1024, 9728, 151936, True)
MODELS['Llama-3.1-8B'] = (32, 4096, 4096, 1024, 14336, 128256, False)
MODELS['Qwen3-14B'] = (40, 5120, 5120, 1024, 17408, 151936, False)
MODELS['Qwen3-32B'] = (64, 5120, 8192, 1024, 25600, 151936, False)
MODELS['Llama-3.1-70B'] = (80, 8192, 8192, 1024, 28672, 128256, False)
MODELS['Llama-3.1-405B'] = (126, 16384, 16384, 1024, 53248, 128256, False)
INFO = {}
for nm, (L, d, qd, kv, ff, V, tied) in MODELS.items():
    per_l = d * qd + 2 * d * kv + qd * d + 3 * d * ff
    Nl = L * per_l; Nh = V * d; Nin = 0 if tied else V * d
    # weight-averaged input width: q,k,v,gate,up have n=d ; o has n=qd ; down has n=ff
    nbar = (d * (d * qd + 2 * d * kv + 2 * d * ff) + qd * (qd * d) + ff * (ff * d)) / per_l
    maxmat = max(d * qd, d * ff)
    kvb = 2 * L * kv * 2                                  # fp16 KV bytes per token
    INFO[nm] = dict(L=L, d=d, ff=ff, N_lin=Nl, N_head=Nh, N_in=Nin, nbar=nbar, maxmat=maxmat, kv_per_tok=kvb, nmax=max(d, qd, ff))
    p(f"{nm:15s} N_lin {Nl:>15,d} head {Nh:>13,d} in-emb {Nin:>13,d} total {Nl+Nh+Nin:>15,d} nbar {nbar:8.1f} max matrix {maxmat:>12,d} kvB/tok {kvb}")
assert INFO['Qwen3-4B']['N_lin'] == N and INFO['Qwen3-4B']['N_head'] == 388956160

r_q = tq_sum / N * 1e6; r_e = wall / N * 1e6
NALL, LCAL = 544, 2048
def gpu_mem_gb(nm):
    I = INFO[nm]; X = NALL * LCAL * I['d'] * 2; Hs = 3 * I['nmax'] ** 2 * 4; W = I['maxmat'] * 4
    return X / 1e9, Hs / 1e9, W / 1e9, (X + Hs + W) / 1e9
p("\nGPU memory formula check (X buffer + 3 n_max^2 fp32 + largest fp32 matrix): Qwen3-4B ->", [round(v, 2) for v in gpu_mem_gb('Qwen3-4B')], "GB; measured peak 9.36 GB (formula / measured = %.2f)" % (gpu_mem_gb('Qwen3-4B')[3] / 9.36))
# --- indicative two-term fits of the per-tensor quantizer time t_q (252 tensors, non-negative least squares); regressors are physically motivated but the mechanism is NOT isolated
import numpy as np
try:
    from scipy.optimize import nnls
except Exception:
    nnls = None
tens = []
for l in open(ROOT + "/logs/r48k3.tensors"):
    m = re.match(r'^vrc\s+blk\.(\d+)\.(\S+)\.weight\s+(\S+)\s+blob .*hist (\[[^\]]*\]).*t_q ([\d.]+)s', l)
    if m:
        nm_, hist = m.group(2), ast.literal_eval(m.group(4)); r_, n_ = SHAPE[nm_]
        tens.append((int(m.group(1)), nm_, r_, n_, sum(h * 256 * e for h, e in zip(hist, ev_w)), float(m.group(5))))
def nnls_fit(cols, label, lo=0):
    sel = [t for t in tens if t[0] >= lo]; A = np.array([[c(t) for c in cols] for t in sel]); y = np.array([t[5] for t in sel]); sc = A.max(0)
    coef, _ = nnls(A / sc, y); coef = coef / sc; pred = A @ coef; r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(); sh = (A * coef).sum(0)
    p(f"FIT {label} (layers>={lo}, n={len(sel)}): coef {coef} R2 {r2:.3f} shares {np.round(sh / sh.sum(), 3)} sum_pred {pred.sum():.1f} s vs sum_t_q {y.sum():.1f} s")
    return coef, r2, sh / sh.sum()
FIT = {}
if nnls:
    FIT['A'] = nnls_fit([lambda t: t[4] / 1e9, lambda t: t[2] * t[3] ** 2 / 1e12], "A [Gevals, m*n^2/1e12]")
    FIT['B'] = nnls_fit([lambda t: t[4] / 1e9, lambda t: t[3] ** 3 / 1e12], "B [Gevals, n^3/1e12]")
    FIT['A8'] = nnls_fit([lambda t: t[4] / 1e9, lambda t: t[2] * t[3] ** 2 / 1e12], "A [Gevals, m*n^2/1e12]", lo=8)
    FIT['B8'] = nnls_fit([lambda t: t[4] / 1e9, lambda t: t[3] ** 3 / 1e12], "B [Gevals, n^3/1e12]", lo=8)
    FIT['E'] = nnls_fit([lambda t: t[4] / 1e9], "E [Gevals only]", lo=8)
    FIT['C3'] = nnls_fit([lambda t: t[4] / 1e9, lambda t: t[3] ** 3 / 1e12, lambda t: t[2] * t[3] ** 2 / 1e12], "C [Gevals, n^3, m*n^2]")
    p("same-mix control (similar class mixes, same N=24.9M weights): median t_q ffn_gate", np.median([t[5] for t in tens if t[1] == 'ffn_gate']), "ffn_up", np.median([t[5] for t in tens if t[1] == 'ffn_up']), "ffn_down", np.median([t[5] for t in tens if t[1] == 'ffn_down']),
      "; mean evals per weight: gate %.0f up %.0f down %.0f" % tuple(np.mean([t[4] / (t[2] * t[3]) for t in tens if t[1] == k]) for k in ('ffn_gate', 'ffn_up', 'ffn_down')))
def tensors_of(nm):
    I = MODELS[nm]; L, d, qd, kv, ff = I[0], I[1], I[2], I[3], I[4]
    return L, [(qd, d), (kv, d), (kv, d), (d, qd), (ff, d), (ff, d), (d, ff)]            # (m, n) per layer
def fit_time(nm, key):
    (a, b), r2, sh = FIT[key]; L, ts = tensors_of(nm); ev = mix_ev * 1.0
    if key.startswith('A'): return L * sum(a * ev * m_ * n_ / 1e9 + b * m_ * n_ ** 2 / 1e12 for m_, n_ in ts) / 3600
    return L * sum(a * ev * m_ * n_ / 1e9 + b * n_ ** 3 / 1e12 for m_, n_ in ts) / 3600
rows_p2 = []
for nm in ['Qwen3-4B', 'Llama-3.1-8B', 'Qwen3-14B', 'Qwen3-32B', 'Llama-3.1-70B', 'Llama-3.1-405B']:
    I = INFO[nm]; Nl = I['N_lin']
    lo = Nl * r_q / 3600e6; ce = Nl * r_e / 3600e6
    fa = fit_time(nm, 'A') if nnls else float('nan'); fb = fit_time(nm, 'B') if nnls else float('nan')
    X, Hs, W, tot_ = gpu_mem_gb(nm)
    rows_p2.append((nm, Nl, lo, ce, fa, fb, tot_, tot_ / 0.74))
    p(f"P2 {nm:15s} N_lin {Nl/1e9:8.2f}e9  GPU-h: linear(quantizer-only 0.762 us/w) {lo:8.2f}  linear(whole script 1.272 us/w) {ce:8.2f}  fitA(evals+m n^2) {fa:8.2f}  fitB(evals+n^3) {fb:8.2f}   [X {X:.1f} GB, 3H {Hs:.1f} GB, W {W:.1f} GB, formula sum {tot_:.1f} GB, scaled by 1/0.74 -> {tot_/0.74:.1f} GB]")
T['p2'] = rows_p2
host_h = lambda nm: 2 * (3 * INFO[nm]['d'] ** 2 + INFO[nm]['ff'] ** 2) * 4 / 1e9
p("host RAM for the per-layer Hessians kept on CPU (fit+eval, fp32, groups in/o/mlp/down):", {nm: round(host_h(nm), 2) for nm in INFO}, "GB")
cal_flops = lambda nm: 2 * INFO[nm]['N_lin'] * NALL * LCAL
chol = lambda nm: INFO[nm]['L'] * (3 * INFO[nm]['d'] ** 3 / 3 + INFO[nm]['ff'] ** 3 / 3)
for nm in ['Qwen3-4B', 'Llama-3.1-405B']: p(f"  {nm}: calibration forward {cal_flops(nm):.3g} FLOP; Cholesky (fit set) {chol(nm):.3g} FLOP; ratio {chol(nm)/cal_flops(nm):.4f}")

# ------------------------------------------------------------------ P3/P4/P5
p("\n== P3 bytes per token and bandwidth upper bound")
B_LIN = {'fp16': 16.0, 'Q8_0': 8.5, 'Q4_K_M': 4.85, 'IQ2_XXS': 2.141, 'Model B': 2.1301}
B_HEAD = {'fp16': 16.0, 'Q8_0': 8.5, 'Q4_K_M': 4.85, 'IQ2_XXS': 5.5, 'Model B': 5.5}
B_IN = {'fp16': 16.0, 'Q8_0': 8.5, 'Q4_K_M': 4.85, 'IQ2_XXS': 2.625, 'Model B': 2.625}          # untied input embedding: Q2_K in the IQ2_XXS mix (cross-check below)
META = 6e6
def wbytes(nm, f):                                           # weights read per generated token
    I = INFO[nm]; return (B_LIN[f] * I['N_lin'] + B_HEAD[f] * I['N_head']) / 8
def fbytes(nm, f):                                           # file size
    I = INFO[nm]; return wbytes(nm, f) + B_IN[f] * I['N_in'] / 8 + META
anch = 967403816 + 388956160 * 5.5 / 8
p(f"anchor Qwen3-4B Model B: linear 967,403,816 B + head {388956160*5.5/8:.0f} B = {anch:.0f} B per token; tg 201.7 -> {201.7*anch/1e9:.2f} GB/s effective")
for bw in (672, 1008, 3350): p(f"  bound at {bw} GB/s: {bw*1e9/anch:.1f} tok/s; measured 201.7 = {201.7/(bw*1e9/anch):.3f} of bound")
p(f"  IQ2_XXS 4B file 1,246,621,312 B: 203.1 t/s -> {203.1*1246621312/1e9:.1f} GB/s ({203.1*1246621312/672e9:.3f} of 672)")
q8 = 4280405408; q8t = sorted([118.9, 119.8, 121.8])[1]
p(f"  Q8_0 carrier 4,280,405,408 B, tg128 median of 3 (logs/r51_d1_q8ceiling.log) {q8t} -> {q8t*q8/1e9:.1f} GB/s = {q8t*q8/672e9:.3f} of 672")
p(f"  model B effective / Q8_0 effective = {201.7*anch/(q8t*q8):.3f}")
p("cross-check file sizes: Llama-3.1-8B IQ2_XXS CALC", f"{fbytes('Llama-3.1-8B','IQ2_XXS'):,.0f}", "vs MEASURED local file 2,399,212,576 B (logs/l8_build.log); Q4_K_M CALC", f"{fbytes('Llama-3.1-8B','Q4_K_M'):,.0f}", "vs llama.cpp README 4.58 GiB =", f"{4.58*2**30:,.0f}", "; Q8_0 CALC", f"{fbytes('Llama-3.1-8B','Q8_0'):,.0f}", "vs MEASURED local Q8_0 8,540,771,136 B")
p("cross-check Qwen3-4B Model B file CALC", f"{fbytes('Qwen3-4B','Model B'):,.0f}", "vs 1,241,608,288 (tied embedding is the head)")
GPU = [("12 GiB laptop (RTX 5070 Ti Laptop class)", 672e9, 12 * 2**30), ("RTX 4090 (24 GiB)", 1008e9, 24 * 2**30), ("H100 SXM (80 GiB)", 3350e9, 80 * 2**30)]
FMTS = ['fp16', 'Q4_K_M', 'IQ2_XXS', 'Model B']; BIG = ['Llama-3.1-8B', 'Llama-3.1-70B', 'Llama-3.1-405B']
rows_p3 = []
for nm in BIG:
    for f in FMTS:
        wb = wbytes(nm, f); fs = fbytes(nm, f); cells = []
        for g, bw, cap in GPU:
            cells.append(None if fs > 0.9 * cap else bw / wb)
        rows_p3.append((nm, f, wb, fs, cells))
        p(f"P3 {nm:15s} {f:8s} weights read/token {wb/1e9:9.2f} GB  file {fs/1e9:9.2f} GB  bound t/s: " + "  ".join("n/f" if c is None else f"{c:8.1f}" for c in cells))
T['p3'] = rows_p3
T['p3_anchor'] = dict(anch=anch, eff=201.7 * anch / 1e9, frac=201.7 * anch / 672e9, q8_eff=q8t * q8 / 1e9, q8_frac=q8t * q8 / 672e9, iq2_eff=203.1 * 1246621312 / 1e9)

p("\n== P4 slab overhead")
for nm in ['Qwen3-4B', 'Llama-3.1-8B', 'Llama-3.1-70B', 'Llama-3.1-405B']:
    I = INFO[nm]; slab = I['maxmat'] * 34 / 32
    p(f"P4 {nm:15s} Q8_0 slab (largest matrix {I['maxmat']:,d} weights x 34/32) {slab/1e6:9.1f} MB ; Model B weights {wbytes(nm,'Model B')/1e9:8.2f} GB ; ratio {slab/wbytes(nm,'Model B')*100:.2f}%")
p(f"Qwen3-4B: measured +56 MiB = {56*2**20/1e6:.1f} MB; slab {INFO['Qwen3-4B']['maxmat']*34/32/1e6:.2f} MB = {INFO['Qwen3-4B']['maxmat']*34/32/2**20:.1f} MiB; undecomposed rest {56-INFO['Qwen3-4B']['maxmat']*34/32/2**20:.1f} MiB; share of peak 56/3218 = {56/3218*100:.2f}% ; 56 MiB / weights {56*2**20/wbytes('Qwen3-4B','Model B')*100:.2f}%")
p("approx 1/L law: slab/weights ~", {nm: round(INFO[nm]['maxmat'] * 34 / 32 / wbytes(nm, 'Model B') * INFO[nm]['L'], 3) for nm in INFO})

p("\n== P5 table size")
p("tables_blob(): 4 (magic) + 1 + 1024 (A) + 1024 (B) + 4 (c) + 8 + 8 (shifts) =", 4 + 1 + 1024 + 1024 + 4 + 16, "B; native layout reserves 2048 B per tensor ->", 2048 * 252, "B =", f"{2048*252/1241608288*100:.4f}% of the file")
p("per-thread-block activation table: 256 entries x 32 chunks x 4 B =", 256 * 32 * 4, "B (paper: 32 KB)")

# ------------------------------------------------------------------ Impact
p("\n== Impact: file sizes (GB = 1e9 B), CALC")
rows_imp = []
for nm in BIG:
    sz = {f: fbytes(nm, f) for f in ['fp16', 'Q8_0'] + FMTS[1:]}
    rows_imp.append((nm, sz))
    p(f"{nm:15s} " + "  ".join(f"{f} {v/1e9:8.2f}" for f, v in sz.items()) + f"   fp16/ours {sz['fp16']/sz['Model B']:.2f}x  Q4KM/ours {sz['Q4_K_M']/sz['Model B']:.2f}x  IQ2/ours {sz['IQ2_XXS']/sz['Model B']:.4f}x")
T['imp'] = rows_imp
# sensitivity of b_lin(ours): 2.05 .. 2.30 ; Q4_K_M 4.85 .. 4.90
for nm in BIG:
    for bl in (2.05, 2.13, 2.30):
        I = INFO[nm]; s = (bl * I['N_lin'] + 5.5 * I['N_head']) / 8 + 2.625 * I['N_in'] / 8 + META
        p(f"sens {nm} ours at b_lin={bl}: {s/1e9:.2f} GB; fp16/ours {fbytes(nm,'fp16')/s:.2f}x")
GIB = 2**30
CTX = 8192
def kv(nm, seqs): return INFO[nm]['kv_per_tok'] * CTX * seqs
def ngpu(nm, f, cap, seqs, reserve=1 * GIB, usable=0.90): return math.ceil((fbytes(nm, f) + kv(nm, seqs) + reserve) / (usable * cap))
rows_g = []
p("KV bytes per token (fp16):", {nm: INFO[nm]['kv_per_tok'] for nm in BIG}, " KV for 8192 tokens x1:", {nm: round(kv(nm, 1) / 1e9, 2) for nm in BIG}, "GB; x32:", {nm: round(kv(nm, 32) / 1e9, 1) for nm in BIG}, "GB")
for nm in BIG:
    for f in ['fp16', 'Q4_K_M', 'IQ2_XXS', 'Model B']:
        r = [ngpu(nm, f, 24 * GIB, s) for s in (1, 32)] + [ngpu(nm, f, 80 * GIB, s) for s in (1, 32)]
        rows_g.append((nm, f, r)); p(f"GPUs {nm:15s} {f:8s} 24GiB: ctx8k x1 {r[0]:3d}  x32 {r[1]:3d} | 80GiB: x1 {r[2]:3d}  x32 {r[3]:3d}")
T['gpus'] = rows_g
PR = {24: (0.89, 1.29), 80: (2.89, 4.29)}           # USD per GPU-hour: RunPod RTX 4090 secure / Lambda A10 24GB ; RunPod H100 PCIe / Lambda 1x H100 SXM
rows_c = []
for nm in BIG:
    for f in ['fp16', 'Q4_K_M', 'Model B']:
        lo = min(ngpu(nm, f, 24 * GIB, 1) * PR[24][0], ngpu(nm, f, 80 * GIB, 1) * PR[80][0]); hi = min(ngpu(nm, f, 24 * GIB, 1) * PR[24][1], ngpu(nm, f, 80 * GIB, 1) * PR[80][1])
        # pessimistic: only the 80 GiB class considered (24 GiB cards lack the bandwidth for large models) -> report both
        lo80 = ngpu(nm, f, 80 * GIB, 1) * PR[80][0]; hi80 = ngpu(nm, f, 80 * GIB, 1) * PR[80][1]
        rows_c.append((nm, f, lo, hi, lo80, hi80)); p(f"COST {nm:15s} {f:8s} replica $/h any class: {lo:6.2f}-{hi:6.2f} ; 80GiB only: {lo80:6.2f}-{hi80:6.2f}")
T['cost'] = rows_c
rows_d = []
for nm in BIG:
    s = {f: fbytes(nm, f) for f in ['fp16', 'Q4_K_M', 'Model B']}
    for ref in ('fp16', 'Q4_K_M'):
        sv = s[ref] - s['Model B']
        rows_d.append((nm, ref, sv, sv * 1e3 / 1e12, sv * 1e6 / 1e15, sv * 1e6 / 1e9 * 0.09))
        p(f"DL {nm:15s} saved vs {ref:7s} {sv/1e9:8.2f} GB/download ; x1e3 = {sv*1e3/1e12:8.2f} TB ; x1e6 = {sv*1e6/1e15:8.3f} PB ; at $0.09/GB list: ${sv*1e6/1e9*0.09:,.0f} per 1M downloads")
T['dl'] = rows_d
p("energy proxy (weights bytes read per token) ratios ours/fp16 and ours/Q4_K_M:", {nm: (round(wbytes(nm, 'Model B') / wbytes(nm, 'fp16'), 4), round(wbytes(nm, 'Model B') / wbytes(nm, 'Q4_K_M'), 4)) for nm in BIG})

# ------------------------------------------------------------------ Quality vs scale (published tables, copied by hand; ratio = ppl / fp16 ppl of the same table)
p("\n== Published 2-bit perplexity across scale (hand-copied from arXiv v4 tables; ratio CALC)")
PUB = [
 ("QTIP T5 (Wikitext2, ctx 4096)", "Llama-2", ["7B", "13B", "70B"], {"FP16": [5.12, 4.57, 3.12], "QTIP 2.00": [5.86, 5.11, 3.70], "QuIP# 2.00": [6.19, 5.35, 3.91], "AQLM 2+": [6.14, 5.33, 3.83]}),
 ("AQLM T1 (Wikitext2, no e2e FT)", "Llama-2", ["7B", "13B", "70B"], {"FP16": [5.12, 4.57, 3.12], "AQLM 2.02/1.97/2.07": [6.59, 5.60, 3.94], "QuIP# 2.02/2.01/2.01": [8.22, 6.06, 4.16]}),
 ("QTIP T5 (Wikitext2, ctx 2048)", "Llama-1", ["7B", "13B", "30B", "65B"], {"FP16": [5.68, 5.09, 4.10, 3.53], "QTIP 2.00": [6.52, 5.80, 4.83, 4.21], "QuIP# 2.00": [6.86, 5.97, 5.02, 4.36]}),
 ("QTIP T7 (Wikitext2, ctx 8192)", "Llama-3", ["8B", "70B"], {"BF16": [5.54, 2.59], "QTIP 2.00": [7.33, 4.97], "QuIP# 2.00": [7.84, 5.77]}),
 ("QTIP T8 (Wikitext2, ctx 8192, instruct)", "Llama-3.1", ["8B", "70B", "405B"], {"BF16 (405B: Meta FP8)": [6.50, 3.52, 1.70], "QTIP 2-bit": [7.82, 5.08, 3.29]}),
]
rows_pub = []
for src, fam, sizes, d in PUB:
    base = list(d.values())[0]
    for meth, v in list(d.items())[1:]:
        ratios = [a / b for a, b in zip(v, base)]; nll = [math.log(r) for r in ratios]; gap = [a - b for a, b in zip(v, base)]
        rows_pub.append((src, fam, meth, sizes, v, base, ratios, nll, gap))
        p(f"{src:40s} {fam:9s} {meth:24s} sizes {sizes} ppl {v} fp16 {base} ratio {[round(r,3) for r in ratios]} dNLL {[round(x,3) for x in nll]} abs gap {[round(g,2) for g in gap]}")
T['pub'] = rows_pub
zs = {"QTIP 2-bit ArcC (T6)": ([40.0, 45.6, 51.1], [35.7, 41.4, 48.0]), "QTIP 2-bit ArcE (T6)": ([69.3, 73.3, 77.7], [65.6, 70.8, 76.3]), "QTIP 2-bit Wino (T6)": ([67.3, 69.6, 77.0], [64.7, 67.6, 75.1]),
      "AQLM 2-bit avg5 (T1)": ([62.35, 65.38, 70.17], [57.28, 61.32, 68.75])}
for k, (f16, q) in zs.items(): p(f"{k:24s} fp16 {f16} quant {q} drop {[round(a-b,2) for a,b in zip(f16,q)]}  (Llama-2 7B/13B/70B)")
T['zs'] = zs
# our points
p("\nOur points: Qwen3-4B 19.0932/12.4567 =", round(19.0932 / 12.4567, 4), "IQ2_XXS", round(28.6083 / 12.4567, 4), "; excess-loss cut", round(1 - (19.0932 / 12.4567 - 1) / (28.6083 / 12.4567 - 1), 4))
p("Gemma-3-4B carrier 17.1804/8.4447 =", round(17.1804 / 8.4447, 4), " native 17.1806/8.4447 =", round(17.1806 / 8.4447, 4), " IQ2_XXS", round(19.4229 / 8.4447, 4), "; excess-loss cut", round(1 - (17.1804 / 8.4447 - 1) / (19.4229 / 8.4447 - 1), 4))
p("dNLL (ln ratio) Qwen3-4B ours", round(math.log(19.0932 / 12.4567), 4), "IQ2_XXS", round(math.log(28.6083 / 12.4567), 4), "; Gemma ours", round(math.log(17.1804 / 8.4447), 4), "IQ2_XXS", round(math.log(19.4229 / 8.4447), 4))

# ------------------------------------------------------------------ depth sensitivity and fit margins
p("\n== KV depth sensitivity of the byte-ratio (bytes read per token = weights + KV_per_token x depth), CALC")
rows_dep = []
for nm in ['Llama-3.1-8B', 'Llama-3.1-70B']:
    for D in (0, 4096, 32768):
        kvb = INFO[nm]['kv_per_tok'] * D; b = {f: wbytes(nm, f) + kvb for f in ['fp16', 'Q4_K_M', 'Model B']}
        rows_dep.append((nm, D, kvb, b['fp16'] / b['Model B'], b['Q4_K_M'] / b['Model B']))
        p(f"DEPTH {nm:15s} depth {D:6d} KV read {kvb/1e9:7.3f} GB  fp16/ours {b['fp16']/b['Model B']:.2f}x  Q4KM/ours {b['Q4_K_M']/b['Model B']:.2f}x")
T['dep'] = rows_dep
w4 = anch; kv4 = 147456 * 4096
p(f"Qwen3-4B depth check: bytes ratio d0/d4096 = {w4/(w4+kv4):.4f} (weights {w4/1e9:.3f} GB, KV {kv4/1e9:.3f} GB) vs measured tg ratio 159.7/201.7 = {159.7/201.7:.4f} (logs/r51_ppc2.log REALSUM)")
p("\nfit margins for the single-GPU cells: (0.9*cap - need)/(0.9*cap)")
for nm in BIG:
    for f in ['Q4_K_M', 'Model B']:
        for capn, cap in (("24GiB", 24 * GIB), ("80GiB", 80 * GIB)):
            need = fbytes(nm, f) + kv(nm, 1) + GIB
            if need <= 0.9 * cap: p(f"  {nm:15s} {f:8s} {capn}: need {need/GIB:6.2f} GiB of {0.9*cap/GIB:6.2f} usable, margin {(0.9*cap-need)/(0.9*cap)*100:5.1f}%")
json.dump(T, open(OUT, "w"), default=str)
