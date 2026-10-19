# R48l propagation gate: the builder's layer propagation (fp16, no mask arg -> sdpa causal, llama3 rope scaling, BOS + 2047 tokens) must give a sane
# language-model loss when the stack is closed with the final norm + lm_head. usage: r48l_gate.py [NSEQ=4]   (GPU, ~2 GB VRAM, streams one layer at a time)
import sys, os, json, math, numpy as np, torch
from transformers import AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from safetensors import safe_open
from huggingface_hub import snapshot_download
dev = "cuda"; P = os.environ.get("HFREPO", "unsloth/Meta-Llama-3.1-8B"); SNAP = snapshot_download(P, local_files_only=True); L = 2048; NS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
WM = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"] if os.path.exists(os.path.join(SNAP, "model.safetensors.index.json")) else {k: "model.safetensors" for k in safe_open(os.path.join(SNAP, "model.safetensors"), framework="pt").keys()}
def tensor(key):
    with safe_open(os.path.join(SNAP, WM[key]), framework="pt", device="cpu") as f: return f.get_tensor(key)
cfg = AutoConfig.from_pretrained(P); cfg._attn_implementation = "sdpa"; tok = AutoTokenizer.from_pretrained(P)
ids = tok(open("models/r35_hcal.txt").read(), return_tensors="pt", add_special_tokens=False).input_ids[0][:NS * (L - 1)].view(NS, L - 1)
ids = torch.cat([torch.full((NS, 1), tok.bos_token_id), ids], 1).to(dev)
E = tensor("model.embed_tokens.weight").to(dev, torch.float16); X = E[ids]; del E
pos = torch.arange(L, device=dev)[None]; PE = LlamaRotaryEmbedding(config=cfg).to(dev)(torch.zeros(1, L, cfg.hidden_size, dtype=torch.float16, device=dev), pos)
for li in range(cfg.num_hidden_layers):
    layer = LlamaDecoderLayer(cfg, li); pre = f"model.layers.{li}."
    layer.load_state_dict({k[len(pre):]: tensor(k) for k in WM if k.startswith(pre)}); layer = layer.to(dev, torch.float16).eval()
    with torch.no_grad():
        for b in range(NS):
            o = layer(X[b:b + 1], position_embeddings=PE, position_ids=pos); X[b:b + 1] = o[0] if isinstance(o, tuple) else o
    if not bool(torch.isfinite(X).all()): sys.exit(f"GATE llama FAIL non-finite after layer {li}")
    print(f"layer {li} max|X| {float(X.abs().max()):.1f}", flush=True); del layer; torch.cuda.empty_cache()
nw = tensor("model.norm.weight").to(dev, torch.float32); Xf = X.float(); Xn = (Xf * torch.rsqrt(Xf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * nw).half()
Wh = tensor("lm_head.weight" if "lm_head.weight" in WM else "model.embed_tokens.weight").to(dev, torch.float16); nll = 0.0; cnt = 0   # tied embeddings fallback for the Llama-3.2-1B test
with torch.no_grad():
    for b in range(NS):
        t = ids[b, 1:]
        for c0 in range(0, L - 1, 512):                      # chunked logits (VRAM cap)
            lg = (Xn[b, c0:min(c0 + 512, L - 1)] @ Wh.T).float(); lp_ = torch.log_softmax(lg, -1)
            nll += float(-lp_.gather(1, t[c0:c0 + lg.shape[0], None]).sum()); cnt += lg.shape[0]
ppl = math.exp(nll / cnt); print(f"GATE llama ppl {ppl:.3f} over {cnt} tokens ({NS} seqs x {L}) {'PASS' if ppl < 40 else 'FAIL'}")
