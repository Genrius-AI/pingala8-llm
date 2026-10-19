# R48k: KERNEL-EXACT build (r48k.py: VRi-kernel + VRii-C ung16, r49_vr.cu layout), copy of r48_build.py front end.
# Each tensor is encoded in BOTH variants (and layers 0-3 IQ slots once more for vrc with BOOST extra bytes from the IQ2_XXS slack).
# usage: r48k_build.py MIX CARRIER_VRI CARRIER_VRC BLOBROOT
# (front end notes from r48_build.py follow)
# R48 Model B build: A1 (VRtile-B2-c8-Hw, research/detail/r48_a1gpu.py = GPU port of r47_a1.py) on ALL 252 linears of
# Qwen3-4B. Front end (calibration text, NSEQ 512 x 2048, fp16 layer-by-layer propagation, H hooks) = r45b_build.py
# (= E1r's r35_ldlq.py). Extra NEV held-out sequences give a per-tensor eval H for Deval.
# Per-tensor real-blob byte budget: the 212 IQ2_XXS-slot tensors get E1r's real blob bytes (the same budget Model A used,
# r36_e1.vedq.stats.json); the 40 other linears (36 attn_v Q4_K, ffn_down 0-3 Q2_K) get their bytes in the IQ2_XXS mix.
# Budget minus MARGIN bytes per tensor (container directory), running surplus carried. Each blob is decoded at once
# (Yq bit-identical required) and the Q8_0 carrier is written from the DECODED blob.
# usage: r48_build.py MIX_GGUF OUT_Q8_GGUF BLOBDIR   env NSEQ NEV DAMP ALPHA MAXL MARGIN
import sys, os, math, json, time, re, hashlib, numpy as np, torch
sys.path.insert(0, "~/repo/llama.cpp/gguf-py"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader, GGMLQuantizationType as QT
from gguf.quants import dequantize, quantize
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
from safetensors import safe_open
from huggingface_hub import snapshot_download
import r48_a1gpu as A
torch.set_num_threads(16); torch.backends.cuda.matmul.allow_tf32 = True; dev = "cuda"
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPUFRAC", "0.80")))
mix, outI, outC, BROOT = sys.argv[1:5]; BL = {v: os.path.join(BROOT, "blobs_" + v) for v in ("vri", "vrc", "vrcB")}
for d_ in BL.values(): os.makedirs(d_, exist_ok=True)
import r48k as K
NSEQ, NEV, L = int(os.environ.get("NSEQ", "512")), int(os.environ.get("NEV", "32")), 2048
DAMP = float(os.environ.get("DAMP", "0.1")); ALPHA = float(os.environ.get("ALPHA", "1.05")); MARGIN = int(os.environ.get("MARGIN", "96")); CFLOOR = 1e-2
IQ = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS"}
LIN = re.compile(r"blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$")
F16 = "~/repo/models/qwen3-4B-f16.gguf"; MODELA = "~/repo/work/r45b/4B-r45b-ung8.gguf"
IM = {t.name[:-8]: np.asarray(t.data, np.float64).ravel() for t in GGUFReader("~/repo/models/qwen3-4B.imatrix").tensors if t.name.endswith(".in_sum2")}
src = {t.name: t for t in GGUFReader(F16).tensors}
mixt = {t.name: t for t in GGUFReader(mix).tensors}
DST = {"vri": {t.name: t for t in GGUFReader(outI, "r+").tensors}, "vrc": {t.name: t for t in GGUFReader(outC, "r+").tensors}}
mA = {t.name: t for t in GGUFReader(MODELA).tensors}                    # Model A carrier, READ ONLY (reference Deval)
E1R = {s_["name"]: s_["words_bytes"] + s_["side_bytes"] for s_ in json.load(open("~/repo/research/work/r36_e1.vedq.stats.json"))}
E1R_DOUT = {}
for l_ in open("~/repo/logs/r36_e1.tensors", errors="ignore"):
    m_ = re.match(r"(\S+)\s+gptq bpw .* postclamp ([0-9.]+)", l_)
    if m_: E1R_DOUT[m_.group(1)] = float(m_.group(2))
def put(name, r0, Ar, v=None):
    if v is None:
        for v_ in DST: put(name, r0, Ar, v_)
        return
    dt = DST[v][name]; Ar = np.asarray(Ar, dtype=np.float32)
    if dt.tensor_type == QT.Q8_0: dt.data[r0:r0 + Ar.shape[0]] = quantize(Ar, QT.Q8_0)
    else: dt.data[r0:r0 + Ar.shape[0]] = Ar.astype(dt.data.dtype)
TK = {}; GK = {}; DK = {}; TABDIR = os.environ.get("TABDIR", "~/repo/research/work/r48k")
for v in ("vri", "vrc"):
    TK[v] = torch.load(f"{TABDIR}/tables_{v}.pt"); J_ = json.load(open(f"{TABDIR}/tables_{v}.json"))
    GK[v] = {int(a): b for a, b in J_["G"].items()}; DK[v] = torch.tensor([J_["D"][str(i)] if str(i) in J_["D"] else J_["D"][i] for i in range(8)], dtype=torch.float64)
    open(os.path.join(BL[v], "__vrk_tables__.blob"), "wb").write(K.tables_blob(TK[v], v == "vrc"))
open(os.path.join(BL["vrcB"], "__vrk_tables__.blob"), "wb").write(K.tables_blob(TK["vrc"], True))
BOOST = float(os.environ.get("BOOST", "5.9e6")); ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]; NSAL = int(os.environ.get("NSAL", "4"))
print(f"R48k build: ONLY {ONLY} NS {K.NSC[0]} TABDIR {TABDIR} S1 {K.S1} S2 {K.S2} NSEQ {NSEQ} NEV {NEV} DAMP {DAMP} ALPHA {ALPHA} MARGIN {MARGIN} BOOST {BOOST} NSAL {NSAL} G {GK} D {[[round(float(x), 5) for x in DK[v]] for v in DK]}", flush=True)
kept = 0
for name, mt in mixt.items():
    ty = mt.tensor_type.name
    if ty in ("F16", "F32", "BF16") or LIN.match(name): continue
    rows, cols = src[name].data.shape
    for r0 in range(0, rows, 16384): put(name, r0, dequantize(mt.data[r0:r0 + 16384], mt.tensor_type).reshape(-1, cols))
    kept += int(mt.n_bytes); print(f"{name:32s} KEEP {ty}", flush=True)
P = "Qwen/Qwen3-4B"; SNAP = snapshot_download(P, local_files_only=True)
WM = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
def tensor(key):
    with safe_open(os.path.join(SNAP, WM[key]), framework="pt", device="cpu") as f: return f.get_tensor(key)
cfg = AutoConfig.from_pretrained(P); cfg._attn_implementation = "sdpa"
tok = AutoTokenizer.from_pretrained(P)
ids = tok(open("models/r35_hcal.txt").read(), return_tensors="pt").input_ids[0]
assert ids.numel() >= (NSEQ + NEV) * L, f"only {ids.numel()} tokens for fit+eval"
ids = ids[:(NSEQ + NEV) * L].view(NSEQ + NEV, L); NALL = NSEQ + NEV
def grp(b0): return "fit" if b0 < NSEQ else "eval"
E = tensor("model.embed_tokens.weight").to(dev, torch.float16); X = torch.empty(NALL, L, cfg.hidden_size, dtype=torch.float16, device=dev)
for b0 in range(0, NALL, 16): X[b0:b0 + 16] = E[ids[b0:b0 + 16].to(dev)]
del E; torch.cuda.empty_cache()
pos = torch.arange(L, device=dev)[None]; PE = Qwen3RotaryEmbedding(config=cfg).to(dev)(X[:1], pos)
HK = {"attn_q": "in", "attn_k": "in", "attn_v": "in", "attn_output": "o", "ffn_gate": "mlp", "ffn_up": "mlp", "ffn_down": "down"}
fw, ifw = A.make_fw(dev); _, ifwc = A.make_fw("cpu"); CARRY = {}; TOT = {}; STATS = []; T0 = time.time(); ROWS = {}
EARLY_BASE = sum((E1R[t] if mixt[t].tensor_type.name in IQ else 0) for t in mixt if LIN.match(t) and int(t.split('.')[1]) <= 3)
for li in range(cfg.num_hidden_layers):
    layer = Qwen3DecoderLayer(cfg, li); pre = f"model.layers.{li}."
    layer.load_state_dict({k[len(pre):]: tensor(k) for k in WM if k.startswith(pre)}); layer = layer.to(dev, torch.float16).eval()
    H = {}; CUR = ["fit"]
    def mk(k):
        def hook(m_, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float(); kk = (k, CUR[0]); H[kk] = H.get(kk, 0) + x.T @ x
        return hook
    hs = [layer.self_attn.q_proj.register_forward_pre_hook(mk("in")), layer.self_attn.o_proj.register_forward_pre_hook(mk("o")),
          layer.mlp.gate_proj.register_forward_pre_hook(mk("mlp")), layer.mlp.down_proj.register_forward_pre_hook(mk("down"))]
    with torch.no_grad():
        for b0 in range(0, NALL, 4):
            CUR[0] = grp(b0); o = layer(X[b0:b0 + 4], position_embeddings=PE, position_ids=pos); X[b0:b0 + 4] = o[0] if isinstance(o, tuple) else o
    for h_ in hs: h_.remove()
    for kk in list(H): H[kk] = (H[kk] / ({"fit": NSEQ, "eval": NEV}[kk[1]] * L)).cpu()
    del layer; torch.cuda.empty_cache()
    if not all(bool(torch.isfinite(X[b0:b0 + 16]).all()) for b0 in range(0, NALL, 16)): sys.exit(f"ABORT non-finite hidden states after layer {li}")
    for k, hk in HK.items():
        name = f"blk.{li}.{k}.weight"
        ty = mixt[name].tensor_type.name; base = E1R[name] if ty in IQ else int(mixt[name].n_bytes)
        W = torch.as_tensor(np.asarray(src[name].data, np.float32), device=dev); rows, n = W.shape; s = IM[name]; N = rows * n
        c = torch.as_tensor(np.sqrt(np.maximum(s / s.mean(), CFLOOR)).astype(np.float16).astype(np.float32), device=dev)
        sw = torch.as_tensor(s / s.mean(), dtype=torch.float32, device=dev); Hf = H[(hk, "fit")].to(dev); He = H[(hk, "eval")].to(dev)
        WA = torch.as_tensor(np.asarray(dequantize(mA[name].data, mA[name].tensor_type), np.float32).reshape(rows, n), device=dev); DevA = A.dmetric(W, WA, He); del WA
        runs = [("vri", 0.0), ("vrc", 0.0)] + ([("vrcB", BOOST * base / EARLY_BASE)] if (li <= 3 and ty in IQ) else [])
        if ONLY: runs = [r_ for r_ in runs if r_[0] in ONLY]
        for v, extra in runs:
            t_q = time.time(); vv = "vrc" if v == "vrcB" else v; tgt_bytes = base - MARGIN + int(extra)
            cy = CARRY.get(v, 0.0)
            def bud(r_, n_): return 8 * (tgt_bytes - K.side_bytes(r_, n_)) - cy
            R = K.k_tensor(W, c, Hf, sw, bud, TK[vv], DK[vv], GK[vv], vv == "vrc", alpha=ALPHA, dev=dev, damp=DAMP, nsal=NSAL); torch.cuda.synchronize(); t_q = time.time() - t_q
            blob = K.pack_blob(c.half().cpu().numpy(), R["prm"], R["S16"].numpy(), R["cs"], R["IDX"].cpu().numpy(), R["wmax"], vv == "vrc")
            hdr, Yd = K.unpack_blob(blob, TK[vv])
            if not torch.equal(Yd, R["Yq"].cpu()): sys.exit(f"ABORT decode mismatch on {name} {v}")
            Wn = (ifwc(Yd) / hdr["c"]).clamp(-hdr["wmax"], hdr["wmax"]); gdiff = float((Wn.to(dev) - R["Wq"]).abs().max())
            open(os.path.join(BL[v], name + ".blob"), "wb").write(blob)
            if v != "vrcB":
                Wnc = Wn.numpy()
                for r0 in range(0, rows, 16384): put(name, r0, Wnc[r0:r0 + 16384], v)
            Wq = Wn.to(dev); Dw = float(((Wq - W) ** 2 * sw).sum() / (W ** 2 * sw).sum()); Dfit = A.dmetric(W, Wq, Hf); Dev = A.dmetric(W, Wq, He)
            MR = float(Wq.abs().max() / W.abs().max()); rE = Dfit / E1R_DOUT[name] if name in E1R_DOUT else float("nan")
            CARRY[v] = cy + 8 * len(blob) - 8 * tgt_bytes; TOT[v] = TOT.get(v, 0) + len(blob)
            st = dict(name=name, v=v, slot=ty, rows=rows, n=n, bytes=len(blob), base=base, bpw=8 * len(blob) / N, Dw=Dw, Dfit=Dfit, Dev=Dev, DevA=DevA, rDevA=Dev / DevA,
                      rE1r_fit=rE, maxratio=MR, hist=R["hist"], nfloor=R["nfloor"], sha=hashlib.sha256(blob).hexdigest()[:16]); STATS.append(st)
            if v != "vrcB": ROWS.setdefault(v, []).append(st)
            print(f"{v:4s} {name:28s} {ty:7s} blob {8 * len(blob) / N:.4f} bpw (base {8 * base / N:.4f}) Dw {Dw:.5f} Dfit {Dfit:.6f} Dev {Dev:.6f} ModelA_Dev {DevA:.6f} "
                  f"Dev/A {Dev / DevA:.3f} Dfit/E1r {rE:.3f} maxWq/maxw {MR:.4f} hist {R['hist']} floor {R['nfloor']} gpu_vs_dec {gdiff:.1e} t_q {t_q:.1f}s", flush=True)
            del R, Yd, Wn, Wq, blob; torch.cuda.empty_cache()
        del W, Hf, He; torch.cuda.empty_cache()
    del H
    for v in ROWS:
        rr_ = [r for r in ROWS[v] if r["slot"] in IQ]
        print(f"LAYER {li} {v} [{time.time() - T0:.0f}s] geomean Dev/ModelA IQslots {math.exp(np.mean([math.log(r['rDevA']) for r in rr_])):.4f} all {math.exp(np.mean([math.log(r['rDevA']) for r in ROWS[v]])):.4f} median Dfit/E1r {np.nanmedian([r['rE1r_fit'] for r in rr_]):.4f} maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    json.dump(STATS, open(os.path.join(BROOT, "stats_vrk.json"), "w"))
    if li + 1 >= int(os.environ.get("MAXL", "99")): print("MAXL stop", flush=True); break
for v in ROWS:
    rr_ = [r for r in ROWS[v] if r["slot"] in IQ]
    print(f"SUMMARY {v} n {len(ROWS[v])} blobs {TOT[v]} B geomean Dev/ModelA IQ212 {math.exp(np.mean([math.log(r['rDevA']) for r in rr_])):.4f} all {math.exp(np.mean([math.log(r['rDevA']) for r in ROWS[v]])):.4f} "
          f"median Dfit/E1r {np.nanmedian([r['rE1r_fit'] for r in rr_]):.4f} maxratio>=0.9 {sum(r['maxratio'] >= 0.9 for r in ROWS[v])}/{len(ROWS[v])} min {min(r['maxratio'] for r in ROWS[v]):.4f}", flush=True)
B_ = [r for r in STATS if r["v"] == "vrcB"]
if B_:
    base_ = {r["name"]: r for r in ROWS["vrc"]}
    print(f"BOOST vrcB extra bytes {sum(r['bytes'] - base_[r['name']]['bytes'] for r in B_)} on {len(B_)} tensors; geomean Dev boost/base {math.exp(np.mean([math.log(r['Dev'] / base_[r['name']]['Dev']) for r in B_])):.4f}", flush=True)
print(f"GATES done [{time.time() - T0:.0f}s]", flush=True)
