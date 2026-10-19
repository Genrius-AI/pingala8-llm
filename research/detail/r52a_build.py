# R52a ablation build = r48k_build.py front end (same calibration text, NSEQ 512 x 2048 fit + NEV 32 eval, fp16 layer propagation,
# same per-tensor byte budgets, alpha, damp, shift fix) with:
#   * hidden states X kept in fp16 shard files on disk (VRAM <= 5 GB rule; r48k builds held X on GPU, 9.36 GB peak)
#   * ENC_LAYERS: encode only these layers (proxy), the rest are only propagated
#   * ARMS = name:NS:TABDIR:mode[:carrier] , mode in std | rateoff | avb | mirror | flattrim   (avb = attn_v at AVB bpw, saved bytes spread over the
#     212 IQ2_XXS-slot tensors in proportion to their base bytes; mirror = R56 control: std allocator, byte carry ON, nsal=0, per-tensor
#     byte target = the rate-off arm's blob bytes from MIRROR_STATS (default work/r52a/f1/stats.json, v == roff);
#   flattrim = R57 size-matched flat arm: fixed class per tensor from FLATTRIM_FILE (research/work/r57/flattrim_cls.json: cls map + sal list; sal tensors get
#   the NSAL salient floor), byte carry off like rateoff). Carrier (Q8_0 GGUF, pre-copied template) only for full builds.
# usage: r52a_build.py MIX OUTDIR    env ARMS ENC_LAYERS NSEQ NEV DAMP ALPHA NSAL AVB XDIR GPUFRAC
import sys, os, math, json, time, re, hashlib, numpy as np, torch
sys.path.insert(0, "~/repo/llama.cpp/gguf-py"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader, GGMLQuantizationType as QT
from gguf.quants import dequantize, quantize
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
from safetensors import safe_open
from huggingface_hub import snapshot_download
import r48_a1gpu as A, r52a_k as RK
K = RK.K
torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = True; dev = "cuda"
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPUFRAC", "0.40")))
mix, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
NSEQ, NEV, L = int(os.environ.get("NSEQ", "512")), int(os.environ.get("NEV", "32")), 2048
DAMP = float(os.environ.get("DAMP", "0.1")); ALPHA = float(os.environ.get("ALPHA", "1.05")); MARGIN = int(os.environ.get("MARGIN", "96")); CFLOOR = 1e-2
NSAL = int(os.environ.get("NSAL", "4")); AVB = float(os.environ.get("AVB", "2.0625"))
ENC = set(int(x) for x in os.environ["ENC_LAYERS"].split(",")) if os.environ.get("ENC_LAYERS") else None
IQ = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS"}
LIN = re.compile(r"blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$")
F16 = "~/repo/models/qwen3-4B-f16.gguf"; MODELA = "~/repo/work/r45b/4B-r45b-ung8.gguf"
IM = {t.name[:-8]: np.asarray(t.data, np.float64).ravel() for t in GGUFReader("~/repo/models/qwen3-4B.imatrix").tensors if t.name.endswith(".in_sum2")}
src = {t.name: t for t in GGUFReader(F16).tensors}
mixt = {t.name: t for t in GGUFReader(mix).tensors}
mA = {t.name: t for t in GGUFReader(MODELA).tensors}
E1R = {s_["name"]: s_["words_bytes"] + s_["side_bytes"] for s_ in json.load(open("~/repo/research/work/r36_e1.vedq.stats.json"))}
ARMS = []
for a in os.environ["ARMS"].split(","):
    p = a.split(":"); ARMS.append(dict(name=p[0], ns=int(p[1]), tab=p[2], mode=p[3], car=p[4] if len(p) > 4 and p[4] else None))
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
# avb: attn_v target bytes and the per-IQ-tensor extra
BASE = {t: (E1R[t] if mixt[t].tensor_type.name in IQ else int(mixt[t].n_bytes)) for t in mixt if LIN.match(t)}
AVT = {t: int(round(AVB * np.prod(src[t].data.shape) / 8)) for t in BASE if ".attn_v." in t}
SAVED = sum(BASE[t] - AVT[t] for t in AVT); IQB = sum(BASE[t] for t in BASE if mixt[t].tensor_type.name in IQ)
TGT = {}
TRIM = json.load(open(os.environ.get("FLATTRIM_FILE", "~/repo/research/work/r57/flattrim_cls.json"))) if any(a["mode"] == "flattrim" for a in ARMS) else None
if TRIM: assert all(t in TRIM["cls"] for t in BASE), "flattrim: class map incomplete"
FLAT = [None]; _alloc_rk = K.alloc
def _alloc_flat(E, budget, CD, floor=None):          # flattrim only: fixed class per tensor; salient floor (class >= 6) where nsal gave one
    if FLAT[0] is None: return _alloc_rk(E, budget, CD, floor)
    c = torch.full(E.shape, FLAT[0], dtype=torch.long)
    return c if floor is None else torch.where(floor, c.clamp(min=6), c)
K.alloc = _alloc_flat
if any(a["mode"] == "mirror" for a in ARMS):
    TGT = {s_["name"]: int(s_["bytes"]) for s_ in json.load(open(os.environ.get("MIRROR_STATS", "~/repo/work/r52a/f1/stats.json"))) if s_["v"] == "roff"}
    assert all(t in TGT for t in BASE), f"mirror: no roff byte target for {[t for t in BASE if t not in TGT][:3]}"
print(f"R52a build: ARMS {[(a['name'], a['ns'], a['tab'], a['mode'], a['car']) for a in ARMS]} ENC {sorted(ENC) if ENC else 'all'} NSEQ {NSEQ} NEV {NEV} "
      f"S1 {K.S1} S2 {K.S2} avb AVB {AVB} saved {SAVED} B over IQ base {IQB} B (x{1 + SAVED / IQB:.5f})", flush=True)
for name, mt in mixt.items():
    ty = mt.tensor_type.name
    if ty in ("F16", "F32", "BF16") or LIN.match(name) or not DST: continue
    rows, cols = src[name].data.shape
    for r0 in range(0, rows, 16384):
        Ar = dequantize(mt.data[r0:r0 + 16384], mt.tensor_type).reshape(-1, cols)
        for v in DST: put(v, name, r0, Ar)
P = "Qwen/Qwen3-4B"; SNAP = snapshot_download(P, local_files_only=True)
WM = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
def tensor(key):
    with safe_open(os.path.join(SNAP, WM[key]), framework="pt", device="cpu") as f: return f.get_tensor(key)
cfg = AutoConfig.from_pretrained(P); cfg._attn_implementation = "sdpa"
tok = AutoTokenizer.from_pretrained(P)
ids = tok(open("models/r35_hcal.txt").read(), return_tensors="pt").input_ids[0]
assert ids.numel() >= (NSEQ + NEV) * L
ids = ids[:(NSEQ + NEV) * L].view(NSEQ + NEV, L); NALL = NSEQ + NEV
def grp(b0): return "fit" if b0 < NSEQ else "eval"
# ---- X on disk: shards of SPS sequences (fp16, < 1 GB each)
XDIR = os.environ.get("XDIR", os.path.join(OUT, "xs")); os.makedirs(XDIR, exist_ok=True); SPS = 64; D_ = cfg.hidden_size
SH = [(s0, min(NALL, s0 + SPS)) for s0 in range(0, NALL, SPS)]
def xpath(i): return os.path.join(XDIR, f"x{i}.f16")
def xread(i): return torch.from_numpy(np.fromfile(xpath(i), np.float16).reshape(SH[i][1] - SH[i][0], L, D_))
def xwrite(i, T_):
    with open(xpath(i), "wb") as f:
        f.write(T_.cpu().numpy().tobytes()); f.flush(); os.fdatasync(f.fileno()); os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
E = tensor("model.embed_tokens.weight").to(dev, torch.float16)
for i, (s0, s1) in enumerate(SH): xwrite(i, E[ids[s0:s1].to(dev)])
del E; torch.cuda.empty_cache()
pos = torch.arange(L, device=dev)[None]; PE = Qwen3RotaryEmbedding(config=cfg).to(dev)(torch.zeros(1, L, D_, dtype=torch.float16, device=dev), pos)
HK = {"attn_q": "in", "attn_k": "in", "attn_v": "in", "attn_output": "o", "ffn_gate": "mlp", "ffn_up": "mlp", "ffn_down": "down"}
fw, ifw = A.make_fw(dev); _, ifwc = A.make_fw("cpu"); CARRY = {}; TOT = {}; STATS = []; T0 = time.time(); ROWS = {}
# RESUME_LI=N|auto (2026-10-17, after a CUDA "unknown error" crash): layers < N are already encoded (blobs, carrier rows, stats.json kept);
# X is re-propagated from the embeddings through them in groups of 6 layers per shard pass (same batches of 4, no Hessian hooks), then the loop goes on at N.
R0 = os.environ.get("RESUME_LI", "0")
if R0 != "0":
    assert all(a["mode"] in ("rateoff", "flattrim") for a in ARMS), "resume: needs byte carry off (rateoff/flattrim)"
    OLD = json.load(open(os.path.join(OUT, "stats.json")))
    R0 = min(sum(1 for s_ in OLD if s_["v"] == a["name"]) // 7 for a in ARMS) if R0 == "auto" else int(R0)
    STATS = [s_ for s_ in OLD if int(s_["name"].split(".")[1]) < R0]
    for s_ in STATS: ROWS.setdefault(s_["v"], []).append(s_); TOT[s_["v"]] = TOT.get(s_["v"], 0) + s_["bytes"]
    assert all(len(ROWS.get(a["name"], [])) == 7 * R0 for a in ARMS), "resume: stats.json does not cover layers < RESUME_LI"
    def _ly(li):
        ly = Qwen3DecoderLayer(cfg, li); pre = f"model.layers.{li}."
        ly.load_state_dict({k[len(pre):]: tensor(k) for k in WM if k.startswith(pre)}); return ly.to(dev, torch.float16).eval()
    for g0 in range(0, R0, 6):
        LS = [_ly(li) for li in range(g0, min(R0, g0 + 6))]
        with torch.no_grad():
            for i, (s0, s1) in enumerate(SH):
                Xg = xread(i).to(dev)
                for b0 in range(0, s1 - s0, 4):
                    x_ = Xg[b0:b0 + 4]
                    for ly in LS: o = ly(x_, position_embeddings=PE, position_ids=pos); x_ = o[0] if isinstance(o, tuple) else o
                    Xg[b0:b0 + 4] = x_
                if not bool(torch.isfinite(Xg).all()): sys.exit(f"ABORT non-finite hidden states in resume group {g0}")
                xwrite(i, Xg); del Xg
        del LS; torch.cuda.empty_cache(); print(f"RESUME PROP layers {g0}-{min(R0, g0 + 6) - 1} [{time.time() - T0:.0f}s]", flush=True)
    print(f"RESUME from layer {R0}: {len(STATS)} stats rows kept, bytes {TOT}", flush=True)
R0 = int(R0)
for li in range(cfg.num_hidden_layers):
    if li < R0: continue
    if ENC is not None and li > max(ENC): break
    enc = ENC is None or li in ENC
    layer = Qwen3DecoderLayer(cfg, li); pre = f"model.layers.{li}."
    layer.load_state_dict({k[len(pre):]: tensor(k) for k in WM if k.startswith(pre)}); layer = layer.to(dev, torch.float16).eval()
    H = {}; CUR = ["fit"]
    def mk(k):
        def hook(m_, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float(); kk = (k, CUR[0]); H[kk] = H.get(kk, 0) + x.T @ x
        return hook
    hs = [layer.self_attn.q_proj.register_forward_pre_hook(mk("in")), layer.self_attn.o_proj.register_forward_pre_hook(mk("o")),
          layer.mlp.gate_proj.register_forward_pre_hook(mk("mlp")), layer.mlp.down_proj.register_forward_pre_hook(mk("down"))] if enc else []
    with torch.no_grad():
        for i, (s0, s1) in enumerate(SH):
            Xg = xread(i).to(dev)
            for b0 in range(0, s1 - s0, 4):
                CUR[0] = grp(s0 + b0); o = layer(Xg[b0:b0 + 4], position_embeddings=PE, position_ids=pos); Xg[b0:b0 + 4] = o[0] if isinstance(o, tuple) else o
            if not bool(torch.isfinite(Xg).all()): sys.exit(f"ABORT non-finite hidden states after layer {li}")
            xwrite(i, Xg); del Xg
    for h_ in hs: h_.remove()
    for kk in list(H): H[kk] = (H[kk] / ({"fit": NSEQ, "eval": NEV}[kk[1]] * L)).cpu()
    del layer; torch.cuda.empty_cache()
    print(f"PROP {li} [{time.time() - T0:.0f}s] enc {enc}", flush=True)
    if not enc: continue
    for k, hk in HK.items():
        name = f"blk.{li}.{k}.weight"; ty = mixt[name].tensor_type.name; base = BASE[name]
        W = torch.as_tensor(np.asarray(src[name].data, np.float32), device=dev); rows, n = W.shape; s = IM[name]; N = rows * n
        c = torch.as_tensor(np.sqrt(np.maximum(s / s.mean(), CFLOOR)).astype(np.float16).astype(np.float32), device=dev)
        sw = torch.as_tensor(s / s.mean(), dtype=torch.float32, device=dev); Hf = H[(hk, "fit")].to(dev); He = H[(hk, "eval")].to(dev)
        WA = torch.as_tensor(np.asarray(dequantize(mA[name].data, mA[name].tensor_type), np.float32).reshape(rows, n), device=dev); DevA = A.dmetric(W, WA, He); del WA
        for a in ARMS:
            v = a["name"]; RK.set_ns(a["ns"]); RK.RATEOFF[0] = a["mode"] == "rateoff"; t_q = time.time(); FLAT[0] = TRIM["cls"][name] if a["mode"] == "flattrim" else None
            if a["mode"] == "avb": tgt_bytes = (AVT[name] if name in AVT else base + (int(SAVED * base / IQB) if ty in IQ else 0)) - MARGIN
            elif a["mode"] == "mirror": tgt_bytes = TGT[name] - MARGIN
            else: tgt_bytes = base - MARGIN
            cy = 0.0 if a["mode"] in ("rateoff", "flattrim") else CARRY.get(v, 0.0)
            def bud(r_, n_): return 8 * (tgt_bytes - K.side_bytes(r_, n_)) - cy
            R = K.k_tensor(W, c, Hf, sw, bud, TK[v], DK[v], GK[v], True, alpha=ALPHA, dev=dev, damp=DAMP, nsal=(NSAL if name in TRIM["sal"] else 0) if a["mode"] == "flattrim" else (0 if a["mode"] in ("rateoff", "mirror") else NSAL))
            torch.cuda.synchronize(); t_q = time.time() - t_q
            blob = K.pack_blob(c.half().cpu().numpy(), R["prm"], R["S16"].numpy(), R["cs"], R["IDX"].cpu().numpy(), R["wmax"], True)
            hdr, Yd = K.unpack_blob(blob, TK[v])
            if not torch.equal(Yd, R["Yq"].cpu()): sys.exit(f"ABORT decode mismatch on {name} {v}")
            Wn = (ifwc(Yd) / hdr["c"]).clamp(-hdr["wmax"], hdr["wmax"])
            open(os.path.join(OUT, "blobs_" + v, name + ".blob"), "wb").write(K.relayout(blob))
            if v in DST:
                Wnc = Wn.numpy()
                for r0 in range(0, rows, 16384): put(v, name, r0, Wnc[r0:r0 + 16384])
            Wq = Wn.to(dev); Dw = float(((Wq - W) ** 2 * sw).sum() / (W ** 2 * sw).sum()); Dfit = A.dmetric(W, Wq, Hf); Dev = A.dmetric(W, Wq, He)
            MR = float(Wq.abs().max() / W.abs().max())
            CARRY[v] = cy + 8 * len(blob) - 8 * tgt_bytes; TOT[v] = TOT.get(v, 0) + len(blob)
            st = dict(name=name, v=v, slot=ty, rows=rows, n=n, bytes=len(blob), base=base, bpw=8 * len(blob) / N, Dw=Dw, Dfit=Dfit, Dev=Dev, DevA=DevA, rDevA=Dev / DevA,
                      maxratio=MR, hist=R["hist"], nfloor=R["nfloor"], t_q=t_q, sha=hashlib.sha256(blob).hexdigest()[:16]); STATS.append(st); ROWS.setdefault(v, []).append(st)
            print(f"{v:8s} {name:28s} {ty:7s} blob {8 * len(blob) / N:.4f} bpw (base {8 * base / N:.4f}) Dw {Dw:.5f} Dfit {Dfit:.6f} Dev {Dev:.6f} ModelA_Dev {DevA:.6f} "
                  f"Dev/A {Dev / DevA:.4f} maxWq/maxw {MR:.4f} hist {R['hist']} floor {R['nfloor']} t_q {t_q:.1f}s gpumax {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
            del R, Yd, Wn, Wq, blob; torch.cuda.empty_cache()
        del W, Hf, He; torch.cuda.empty_cache()
    del H
    for v in ROWS:
        print(f"LAYER {li} {v} [{time.time() - T0:.0f}s] geomean Dev/ModelA all-so-far {math.exp(np.mean([math.log(r['rDevA']) for r in ROWS[v]])):.4f} bytes {TOT[v]} maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    json.dump(STATS, open(os.path.join(OUT, "stats.json"), "w"))
for v in ROWS:
    rr_ = [r for r in ROWS[v] if r["slot"] in IQ]
    print(f"SUMMARY {v} n {len(ROWS[v])} blobs {TOT[v]} B geomean Dev/ModelA IQ {math.exp(np.mean([math.log(r['rDevA']) for r in rr_])):.4f} all {math.exp(np.mean([math.log(r['rDevA']) for r in ROWS[v]])):.4f} "
          f"geomean Dw {math.exp(np.mean([math.log(r['Dw']) for r in ROWS[v]])):.6f} maxratio>=0.9 {sum(r['maxratio'] >= 0.9 for r in ROWS[v])}/{len(ROWS[v])} min {min(r['maxratio'] for r in ROWS[v]):.4f}", flush=True)
for i in range(len(SH)): os.remove(xpath(i))
print(f"GATES done [{time.time() - T0:.0f}s]", flush=True)
