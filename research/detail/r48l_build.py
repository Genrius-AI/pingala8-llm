# R48l = r52a_build.py (disk-resident hidden states, ARMS, same Model B front end: kernel-exact VRii-C Pingala-8, r48k3 tables, NSEQ 512 x 2048 fit +
# NEV 32 eval, DAMP 0.1, ALPHA 1.05, NSAL 4, R48K shifts) ported to Llama-3.1-8B (arch llama, 32 layers, GQA 32/8, rope theta 5e5 + llama3 scaling, untied
# output, no QK-norm). Patches vs r52a_build.py are marked LLAMA. No retune: tables/settings are the Qwen3-4B ones (as for Gemma, r48g_build.py).
#   * source W = bf16 safetensors of unsloth/Meta-Llama-3.1-8B; attn_q / attn_k rows are re-permuted exactly like convert_hf_to_gguf.py (LlamaModel.permute),
#     so blobs, carrier and native file are in GGUF row order; each W is checked against the Q8_0 reference (rel err < 0.05) before it is encoded.
#   * imatrix (c, sw) from IMATRIX (llama-imatrix on the Q8_0, calib_c4 -c 512 --chunks 200), mix = llama-quantize IQ2_XXS of the Q8_0 with that imatrix.
#   * budgets: IQ slots 0.9930347687917909 x IQ2_XXS tensor bytes (Gemma convention), non-IQ slots the mix bytes; DevA reference = the IQ2_XXS mix tensor.
#   * ARMS name:NS:TABDIR:mode[:carrier], mode std | flat.  flat = R57/R56 "rateoff" class rule (ONE rate class per tensor = the class whose b/w is nearest the
#     tensor index budget; IQ2_XXS slot -> 3 (2.0), Q4_K slot (attn_v) -> 6 (4.0), Q2_K slot (ffn_down L0-3) -> 4 (2.5)), byte carry off, NSAL salient floor kept
#     (tile-blocks of the top-4 |w| clamp to class >= 6, as flattrim did on the gate-failing tensors; here on every tensor).
#   * BOS + 2047 calibration tokens per sequence (llama-perplexity puts BOS at every chunk start).
#   * lean memory (VRAM cap GPUFRAC, RAM <= ~5.5 GB): Hessians accumulated in place, ffn_down Hessians spilled to disk, n > GMAX Cholesky chain done in
#     place with LAPACK (scipy dpotrf/dpotri/dpotrf) on the CPU instead of 3 double copies.
# usage: r48l_build.py MIX OUTDIR   env: IMATRIX Q8REF ARMS ENC_LAYERS KINDS NSEQ NEV DAMP ALPHA NSAL LB SPS XDIR GPUFRAC GMAX HFREPO
import sys, os, math, json, time, re, hashlib, numpy as np, torch
sys.path.insert(0, "llama.cpp/gguf-py"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader, GGMLQuantizationType as QT
from gguf.quants import dequantize, quantize
from transformers import AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding     # LLAMA
from safetensors import safe_open
from huggingface_hub import snapshot_download
import scipy.linalg.lapack as lp
import r48_a1gpu as A, r52a_k as RK
K = RK.K
torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = True; dev = "cuda"
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPUFRAC", "0.40")))
mix, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
NSEQ, NEV, L = int(os.environ.get("NSEQ", "512")), int(os.environ.get("NEV", "32")), 2048
DAMP = float(os.environ.get("DAMP", "0.1")); ALPHA = float(os.environ.get("ALPHA", "1.05")); MARGIN = int(os.environ.get("MARGIN", "96")); CFLOOR = 1e-2
NSAL = int(os.environ.get("NSAL", "4")); LB = int(os.environ.get("LB", "2")); SPS = int(os.environ.get("SPS", "32")); GMAX = int(os.environ.get("GMAX", "10000"))
ENC = set(int(x) for x in os.environ["ENC_LAYERS"].split(",")) if os.environ.get("ENC_LAYERS") else None
KINDS = set(os.environ["KINDS"].split(",")) if os.environ.get("KINDS") else None
IQ = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS"}
LIN = re.compile(r"blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$")
HFN = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
       "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}                                           # LLAMA
IM = {t.name[:-8]: np.asarray(t.data, np.float64).ravel() for t in GGUFReader(os.environ["IMATRIX"]).tensors if t.name.endswith(".in_sum2")}
mixt = {t.name: t for t in GGUFReader(mix).tensors}
q8r = {t.name: t for t in GGUFReader(os.environ["Q8REF"]).tensors}                                                                   # LLAMA: sanity reference for W / row order
E1R_FRAC = float(os.environ.get("E1R_FRAC", "0.9930347687917909")); E1R = {t: int(mixt[t].n_bytes * E1R_FRAC) for t in mixt}
ARMS = []
for a in os.environ["ARMS"].split(","):
    p = a.split(":"); ARMS.append(dict(name=p[0], ns=int(p[1]), tab=p[2], mode=p[3], car=p[4] if len(p) > 4 and p[4] else None))
assert all(a["mode"] in ("std", "flat") for a in ARMS), "modes: std | flat"
TRIM = set(json.load(open(os.environ["FLATTRIM_FILE"]))["moved"]) if os.environ.get("FLATTRIM_FILE") else set()      # LLAMA: class 3 -> 2 on these tensors (l8_trim.py)
SALMODE = os.environ.get("SALMODE", "gate")      # gate: NSAL floor only for tensors whose unfloored encode fails max|Wq|/max|w| >= 0.9 (flattrim semantics); all: NSAL on every tensor
TK, GK, DK, DST = {}, {}, {}, {}
for a in ARMS:
    v = a["name"]; RK.set_ns(a["ns"]); os.makedirs(os.path.join(OUT, "blobs_" + v), exist_ok=True)
    TK[v] = torch.load(f"{a['tab']}/tables_vrc.pt"); J_ = json.load(open(f"{a['tab']}/tables_vrc.json"))
    GK[v] = {int(x): y for x, y in J_["G"].items()}; DK[v] = torch.tensor([J_["D"][str(i)] for i in range(8)], dtype=torch.float64)
    open(os.path.join(OUT, "blobs_" + v, "__vrk_tables__.blob"), "wb").write(RK.tables_blob(TK[v], True))
    if a["car"]: DST[v] = {t.name: t for t in GGUFReader(a["car"], "r+").tensors}
def put(v, name, r0, Ar):
    dt = DST[v][name]; Ar = np.asarray(Ar, dtype=np.float32)
    if dt.tensor_type == QT.Q8_0: dt.data[r0:r0 + Ar.shape[0]] = quantize(Ar, QT.Q8_0)
    else: dt.data[r0:r0 + Ar.shape[0]] = Ar.astype(dt.data.dtype)
BASE = {t: (E1R[t] if mixt[t].tensor_type.name in IQ else int(mixt[t].n_bytes)) for t in mixt if LIN.match(t)}
SHP = {t: (int(mixt[t].shape[1]), int(mixt[t].shape[0])) for t in BASE}          # (rows, n); GGUF shape is [n, rows]
# ---- LLAMA: pre-flight byte plan (exact class bytes of the flat rule) before any GPU time is spent
def flat_class(name):
    if name in TRIM: return 2
    rows, n = SHP[name]; bud = 8 * (BASE[name] - MARGIN - K.side_bytes(rows, n)); r = bud / (rows * n)
    return int(torch.argmin((K.CR[1:] - r).abs())) + 1
def cls_bytes(name, k):
    rows, n = SHP[name]; return K.side_bytes(rows, n) + rows * (n // 256) * K.BPB[k]
MIXLIN = sum(int(mixt[t].n_bytes) for t in BASE); PLAN = {t: cls_bytes(t, flat_class(t)) for t in BASE}; PH = {}
for t in BASE: PH[(mixt[t].tensor_type.name, flat_class(t))] = PH.get((mixt[t].tensor_type.name, flat_class(t)), 0) + 1
SALMAX = len(BASE) * NSAL * 32 * (K.BPB[6] - K.BPB[3]) if os.environ.get("SALMODE", "gate") == "all" else 0
print(f"R48l PLAN flat: slot/class histogram {PH}; linear bytes: mix {MIXLIN}, 0.993-budget {sum(BASE.values())}, flat plan {sum(PLAN.values())} "
      f"(+ <= {SALMAX} sal; gate mode adds NSAL tile-blocks only on gate-failing tensors) trim {sorted(TRIM)} -> vs mix {sum(PLAN.values()) - MIXLIN:+d}, with max sal {sum(PLAN.values()) + SALMAX - MIXLIN:+d}; vs 0.993 budget {sum(PLAN.values()) - sum(BASE.values()):+d}", flush=True)
if any(a["mode"] == "flat" for a in ARMS) and sum(PLAN.values()) + SALMAX > MIXLIN and not os.environ.get("ALLOW_OVER"):
    sys.exit("ABORT flat plan exceeds the IQ2_XXS linear bytes even before the native-file extras; set ALLOW_OVER=1 to override")
# flat allocator: r52a_k.alloc rateoff path (nearest class) + salient floor (class >= 6) like flattrim
FLAT = [False]; CURN = [None]; _alloc_rk = K.alloc
def _alloc_flat(E, budget, CD, floor=None):
    if not FLAT[0]: return _alloc_rk(E, budget, CD, floor)
    RK.RATEOFF[0] = True
    try: c = _alloc_rk(E, budget, CD, None)
    finally: RK.RATEOFF[0] = False
    if CURN[0] in TRIM: c = torch.full_like(c, 2)
    return c if floor is None else torch.where(floor, c.clamp(min=6), c)
K.alloc = _alloc_flat
print(f"R48l build: ARMS {[(a['name'], a['ns'], a['tab'], a['mode'], a['car']) for a in ARMS]} ENC {sorted(ENC) if ENC else 'all'} KINDS {sorted(KINDS) if KINDS else 'all'} "
      f"NSEQ {NSEQ} NEV {NEV} LB {LB} SPS {SPS} GMAX {GMAX} S1 {K.S1} S2 {K.S2}", flush=True)
RESUME_MODE = bool(os.environ.get("RESUME")) and os.path.exists(os.path.join(OUT, "stats.json"))      # LLAMA: restart after a kill; carrier/blobs of finished layers are kept
if DST and not RESUME_MODE:                               # non-linear tensors (token_embd, output, ...) from the IQ2_XXS mix into every carrier
    for name, mt in mixt.items():
        ty = mt.tensor_type.name
        if ty in ("F16", "F32", "BF16") or LIN.match(name): continue
        sh_ = [int(x) for x in next(iter(DST.values()))[name].shape]; cols = sh_[0]; rows = int(np.prod(sh_[1:])) if len(sh_) > 1 else 1   # shape from carrier
        for r0 in range(0, rows, 16384):
            Ar = dequantize(mt.data[r0:r0 + 16384], mt.tensor_type).reshape(-1, cols)
            for v in DST: put(v, name, r0, Ar)
        print(f"NONLIN {name} {ty} {rows}x{cols}", flush=True)
P = os.environ.get("HFREPO", "unsloth/Meta-Llama-3.1-8B"); SNAP = snapshot_download(P, local_files_only=True)
WM = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"] if os.path.exists(os.path.join(SNAP, "model.safetensors.index.json")) else {k: "model.safetensors" for k in safe_open(os.path.join(SNAP, "model.safetensors"), framework="pt").keys()}
def tensor(key):
    with safe_open(os.path.join(SNAP, WM[key]), framework="pt", device="cpu") as f: return f.get_tensor(key)
cfg = AutoConfig.from_pretrained(P); cfg._attn_implementation = "sdpa"; NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads
def permute(w, n_head, n_head_kv):                         # LLAMA: = convert_hf_to_gguf LlamaModel.permute (RoPE row order of llama.cpp)
    if n_head_kv is not None and n_head != n_head_kv: n_head = n_head_kv
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)
def getW(li, k):
    w = tensor(f"model.layers.{li}.{HFN[k]}.weight").to(torch.float32)
    return permute(w, NH, NH) if k == "attn_q" else permute(w, NH, NKV) if k == "attn_k" else w
tok = AutoTokenizer.from_pretrained(P)
ids = tok(open("models/r35_hcal.txt").read(), return_tensors="pt", add_special_tokens=False).input_ids[0]
assert ids.numel() >= (NSEQ + NEV) * (L - 1), f"only {ids.numel()} tokens for fit+eval"
ids = torch.cat([torch.full((NSEQ + NEV, 1), tok.bos_token_id), ids[:(NSEQ + NEV) * (L - 1)].view(NSEQ + NEV, L - 1)], 1); NALL = NSEQ + NEV      # LLAMA: BOS + 2047
def grp(b0): return "fit" if b0 < NSEQ else "eval"
XDIR = os.environ.get("XDIR", os.path.join(OUT, "xs")); os.makedirs(XDIR, exist_ok=True); D_ = cfg.hidden_size
SH = [(s0, min(NALL, s0 + SPS)) for s0 in range(0, NALL, SPS)]
def xpath(i): return os.path.join(XDIR, f"x{i}.f16")
def xread(i): return torch.from_numpy(np.fromfile(xpath(i), np.float16).reshape(SH[i][1] - SH[i][0], L, D_))
def dropcache(f): os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
def xwrite(i, T_):
    with open(xpath(i), "wb") as f:
        f.write(T_.cpu().numpy().tobytes()); f.flush(); os.fdatasync(f.fileno()); dropcache(f)
E = tensor("model.embed_tokens.weight").to(dev, torch.float16)
for i, (s0, s1) in enumerate(SH): xwrite(i, E[ids[s0:s1].to(dev)])
del E; torch.cuda.empty_cache()
pos = torch.arange(L, device=dev)[None]; PE = LlamaRotaryEmbedding(config=cfg).to(dev)(torch.zeros(1, L, D_, dtype=torch.float16, device=dev), pos)
HK = {"attn_q": "in", "attn_k": "in", "attn_v": "in", "attn_output": "o", "ffn_gate": "mlp", "ffn_up": "mlp", "ffn_down": "down"}
HMEM, HSPILL = {}, {}
def hput(kk, T_):                                          # LLAMA: ffn_down Hessians (14336^2 fp32 = 822 MB each) are spilled to disk
    if kk[0] == "down":
        p = os.path.join(XDIR, f"H_{kk[0]}_{kk[1]}.npy")
        with open(p, "wb") as f: np.save(f, T_.numpy()); f.flush(); os.fdatasync(f.fileno()); dropcache(f)
        HSPILL[kk] = p
    else: HMEM[kk] = T_
def hget(kk):
    if kk in HSPILL:
        with open(HSPILL[kk], "rb") as f: r = torch.from_numpy(np.load(f)); dropcache(f); return r
    return HMEM[kk]
def dmet(W, Wq, Hc): Hg = Hc.to(dev); r = A.dmetric(W, Wq, Hg); del Hg; return r
fw, ifw = A.make_fw(dev); _, ifwc = A.make_fw("cpu"); CARRY = {}; TOT = {}; STATS = []; T0 = time.time(); ROWS = {}; RESUME = 0
if RESUME_MODE:
    STATS = json.load(open(os.path.join(OUT, "stats.json"))); per = len(ARMS) * len([k for k in HK if KINDS is None or k in KINDS]); RESUME = len(STATS) // per; STATS = STATS[:RESUME * per]
    for st_ in STATS: ROWS.setdefault(st_["v"], []).append(st_); TOT[st_["v"]] = TOT.get(st_["v"], 0) + st_["bytes"]
    print(f"RESUME from layer {RESUME}: {len(STATS)} tensor stats kept, {sum(TOT.values())} blob bytes", flush=True)

def k_tensor_l(W, c, Hf, sw, budget_fn, T, CD, G, tr, alpha=1.05, dev="cuda", damp=0.1, nsal=4):
    """r48k.k_tensor with a lean memory profile (Hf is a CPU fp32 tensor; n > GMAX Cholesky chain in place on the CPU). Same maths."""
    fw, ifw = A.make_fw(dev); rows, n = W.shape; nbk = n // 256
    Y = fw(W * c); rmsb = Y.view(rows, -1, 256).pow(2).mean(2).sqrt().clamp_min(1e-12).cpu()
    Hg = Hf.to(dev); Hg /= c[:, None]; Hg /= c[None, :]
    t_ = fw(Hg); del Hg; t_ = t_.T.contiguous(); Hp = fw(t_); del t_
    if n <= GMAX:
        Hq = Hp.double(); del Hp; Hq.diagonal().add_(damp * float(Hq.diag().mean())); Lc = torch.linalg.cholesky(Hq); del Hq
        Hi = torch.cholesky_inverse(Lc); del Lc; Uc = torch.linalg.cholesky(Hi, upper=True).cpu(); del Hi
        dg = Uc.diagonal().clone(); Ucf = Uc.float(); del Uc
    else:                                                  # in-place LAPACK: one 8 B x n^2 array (1.64 GB at n = 14336), then a fp32 copy
        Hn = Hp.cpu().numpy(); del Hp; torch.cuda.empty_cache(); H64 = Hn.astype(np.float64); del Hn
        H64.flat[::n + 1] += damp * float(H64.diagonal().mean()); F = H64.T                    # symmetric: F-ordered view of the same matrix
        F, info = lp.dpotrf(F, lower=0, overwrite_a=1); assert info == 0, f"dpotrf {info}"
        F, info = lp.dpotri(F, lower=0, overwrite_c=1); assert info == 0, f"dpotri {info}"
        F, info = lp.dpotrf(F, lower=0, overwrite_a=1); assert info == 0, f"dpotrf2 {info}"     # upper factor U of inv(H), A = U^T U (torch upper=True)
        dg = torch.from_numpy(np.ascontiguousarray(F.diagonal())); Ucf = torch.from_numpy(F).float().triu_(); del F, H64
    wb = dg.pow(-2).view(nbk, 256).mean(1); wb = (wb / wb.mean()).double(); Ucg = Ucf.to(dev).contiguous(); del Ucf
    budget = budget_fn(rows, n)
    Ebk = (Y.view(rows, nbk, 256) ** 2).sum(2).double().cpu()
    prm = torch.argsort(Ebk.clamp_min(1e-30).log().mean(1)); Es = Ebk[prm] * wb[None]
    floor = None
    if nsal:
        top = torch.topk(W.abs().flatten(), nsal).indices.cpu(); inv = torch.argsort(prm)
        floor = torch.zeros(rows // 32, nbk, dtype=torch.bool); floor[inv[top // n] // 32, (top % n) // 256] = True
    cs = K.alloc(Es.view(rows // 32, 32, nbk).sum(1), budget / 32, CD, floor)
    clsS = cs.repeat_interleave(32, 0); cls = torch.empty_like(clsS); cls[prm] = clsS
    Gk = torch.tensor([1.0] + [G[k] for k in range(1, K.NC)], dtype=torch.float32)
    S16 = (alpha * rmsb / Gk[cls]).to(torch.float16); S16[cls == 0] = 0; Sb = S16.float()
    old = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = False
    TG = {k_: v.to(dev) for k_, v in T.items()}
    try: Yq, IDX = K.encode(Y, Ucg, cls.to(dev), Sb.to(dev), TG, tr)
    finally: torch.backends.cuda.matmul.allow_tf32 = old
    wmax = float(np.float16(float(W.abs().max()) * 1.001)); Wq = (ifw(Yq) / c).clamp(-wmax, wmax)
    del Y, Ucg
    return dict(Yq=Yq, Wq=Wq, IDX=IDX, cls=cls, cs=cs, prm=prm, S16=S16, Sb=Sb, wmax=wmax, idx_bits=float((256 * K.CR[cls]).sum()), budget=budget,
                hist=torch.bincount(cls.flatten(), minlength=K.NC).tolist(), nfloor=int(floor.sum()) if floor is not None else 0)

for li in range(cfg.num_hidden_layers):
    if ENC is not None and li > max(ENC): break
    enc = (ENC is None or li in ENC) and li >= RESUME
    layer = LlamaDecoderLayer(cfg, li); pre = f"model.layers.{li}."
    layer.load_state_dict({k[len(pre):]: tensor(k) for k in WM if k.startswith(pre)}); layer = layer.to(dev, torch.float16).eval()
    H = {}; CUR = ["fit"]
    def mk(k):
        def hook(m_, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float(); kk = (k, CUR[0])
            if kk not in H: H[kk] = torch.zeros(x.shape[1], x.shape[1], device=dev)
            H[kk].addmm_(x.T, x)                           # LLAMA: in place (no 822 MB temp for ffn_down)
        return hook
    hs = [layer.self_attn.q_proj.register_forward_pre_hook(mk("in")), layer.self_attn.o_proj.register_forward_pre_hook(mk("o")),
          layer.mlp.gate_proj.register_forward_pre_hook(mk("mlp")), layer.mlp.down_proj.register_forward_pre_hook(mk("down"))] if enc else []
    with torch.no_grad():
        for i, (s0, s1) in enumerate(SH):
            Xg = xread(i).to(dev)
            for b0 in range(0, s1 - s0, LB):
                CUR[0] = grp(s0 + b0); o = layer(Xg[b0:b0 + LB], position_embeddings=PE, position_ids=pos); Xg[b0:b0 + LB] = o[0] if isinstance(o, tuple) else o
            if not bool(torch.isfinite(Xg).all()): sys.exit(f"ABORT non-finite hidden states after layer {li}")
            xwrite(i, Xg); mx_ = float(Xg.abs().max()); del Xg
    for h_ in hs: h_.remove()
    for kk in list(H): hput(kk, (H[kk] / ({"fit": NSEQ, "eval": NEV}[kk[1]] * L)).cpu()); del H[kk]
    del layer; torch.cuda.empty_cache()
    print(f"PROP {li} [{time.time() - T0:.0f}s] enc {enc} max|X| {mx_:.1f} gpumax {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
    if not enc: continue
    for k, hk in HK.items():
        if KINDS is not None and k not in KINDS: continue
        name = f"blk.{li}.{k}.weight"; ty = mixt[name].tensor_type.name; base = BASE[name]
        W = getW(li, k).to(dev); rows, n = W.shape; s = IM[name]; N = rows * n
        assert (rows, n) == SHP[name] and len(s) == n, (name, W.shape, SHP[name], len(s))
        Wr = torch.as_tensor(np.asarray(dequantize(q8r[name].data, q8r[name].tensor_type), np.float32).reshape(rows, n), device=dev)
        rq = float((W - Wr).norm() / W.norm()); del Wr
        assert rq < 0.05, f"{name}: W (GGUF row order) vs Q8_0 reference rel err {rq:.4f} (row order / source mismatch)"
        c = torch.as_tensor(np.sqrt(np.maximum(s / s.mean(), CFLOOR)).astype(np.float16).astype(np.float32), device=dev)
        sw = torch.as_tensor(s / s.mean(), dtype=torch.float32, device=dev)
        WA = torch.as_tensor(np.asarray(dequantize(mixt[name].data, mixt[name].tensor_type), np.float32).reshape(rows, n), device=dev); DevA = dmet(W, WA, hget((hk, "eval"))); del WA
        for a in ARMS:
            v = a["name"]; RK.set_ns(a["ns"]); FLAT[0] = a["mode"] == "flat"; t_q = time.time()
            tgt_bytes = base - MARGIN
            cy = 0.0 if a["mode"] == "flat" else CARRY.get(v, 0.0)
            def bud(r_, n_): return 8 * (tgt_bytes - K.side_bytes(r_, n_)) - cy
            CURN[0] = name; resal = 0
            R = k_tensor_l(W, c, hget((hk, "fit")), sw, bud, TK[v], DK[v], GK[v], True, alpha=ALPHA, dev=dev, damp=DAMP, nsal=(NSAL if (a["mode"] == "std" or SALMODE == "all") else 0))
            if a["mode"] == "flat" and SALMODE == "gate" and float(R["Wq"].abs().max() / W.abs().max()) < 0.9:      # LESSONS 9 gate failed: redo with the salient floor
                resal = 1; del R; torch.cuda.empty_cache()
                R = k_tensor_l(W, c, hget((hk, "fit")), sw, bud, TK[v], DK[v], GK[v], True, alpha=ALPHA, dev=dev, damp=DAMP, nsal=NSAL)
            torch.cuda.synchronize(); t_q = time.time() - t_q
            blob = K.pack_blob(c.half().cpu().numpy(), R["prm"], R["S16"].numpy(), R["cs"], R["IDX"].cpu().numpy(), R["wmax"], True)
            hdr, Yd = K.unpack_blob(blob, TK[v])
            if not torch.equal(Yd, R["Yq"].cpu()): sys.exit(f"ABORT decode mismatch on {name} {v}")
            Wn = (ifwc(Yd) / hdr["c"]).clamp(-hdr["wmax"], hdr["wmax"]); del Yd
            open(os.path.join(OUT, "blobs_" + v, name + ".blob"), "wb").write(K.relayout(blob))
            if v in DST:
                Wnc = Wn.numpy()
                for r0 in range(0, rows, 16384): put(v, name, r0, Wnc[r0:r0 + 16384])
            Wq = Wn.to(dev); del Wn
            Dw = float(((Wq - W) ** 2 * sw).sum() / (W ** 2 * sw).sum()); Dfit = dmet(W, Wq, hget((hk, "fit"))); Dev = dmet(W, Wq, hget((hk, "eval")))
            MR = float(Wq.abs().max() / W.abs().max())
            CARRY[v] = cy + 8 * len(blob) - 8 * tgt_bytes; TOT[v] = TOT.get(v, 0) + len(blob)
            st = dict(name=name, v=v, slot=ty, rows=rows, n=n, bytes=len(blob), base=base, bpw=8 * len(blob) / N, Dw=Dw, Dfit=Dfit, Dev=Dev, DevA=DevA, rDevA=Dev / DevA,
                      maxratio=MR, q8rel=rq, resal=resal, hist=R["hist"], nfloor=R["nfloor"], t_q=t_q, sha=hashlib.sha256(blob).hexdigest()[:16]); STATS.append(st); ROWS.setdefault(v, []).append(st)
            print(f"{v:8s} {name:28s} {ty:7s} blob {8 * len(blob) / N:.4f} bpw (base {8 * base / N:.4f}) Dw {Dw:.5f} Dfit {Dfit:.6f} Dev {Dev:.6f} IQ2_Dev {DevA:.6f} "
                  f"Dev/IQ2 {Dev / DevA:.4f} maxWq/maxw {MR:.4f} q8rel {rq:.4f} hist {R['hist']} floor {R['nfloor']} resal {resal} t_q {t_q:.1f}s gpumax {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
            del R, Wq, blob; torch.cuda.empty_cache()
        del W; torch.cuda.empty_cache()
    HMEM.clear()
    for p_ in list(HSPILL.values()): os.remove(p_)
    HSPILL.clear()
    for v in ROWS:
        print(f"LAYER {li} {v} [{time.time() - T0:.0f}s] geomean Dev/IQ2_XXS-Dev all-so-far {math.exp(np.mean([math.log(r['rDevA']) for r in ROWS[v]])):.4f} bytes {TOT[v]} maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    json.dump(STATS, open(os.path.join(OUT, "stats.json"), "w")); json.dump(STATS, open(os.path.join(OUT, "stats_vrk.json"), "w"))
for v in ROWS:
    rr_ = [r for r in ROWS[v] if r["slot"] in IQ]
    print(f"SUMMARY {v} n {len(ROWS[v])} blobs {TOT[v]} B geomean Dev/IQ2_XXS-Dev IQ {math.exp(np.mean([math.log(r['rDevA']) for r in rr_])):.4f} all {math.exp(np.mean([math.log(r['rDevA']) for r in ROWS[v]])):.4f} "
          f"geomean Dw {math.exp(np.mean([math.log(r['Dw']) for r in ROWS[v]])):.6f} maxratio>=0.9 {sum(r['maxratio'] >= 0.9 for r in ROWS[v])}/{len(ROWS[v])} min {min(r['maxratio'] for r in ROWS[v]):.4f}", flush=True)
for i in range(len(SH)): os.remove(xpath(i))
print(f"GATES done [{time.time() - T0:.0f}s]", flush=True)
