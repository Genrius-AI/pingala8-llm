# l8_trim: a-priori trim list for the flat Llama-3.1-8B Model B (no Dev from a previous run exists). Pre-registered rule (written before the build):
#   flat classes put the 188 IQ2_XXS-slot tensors at class 3 (2.0 index b/w), which is +1.0 MB... over/under the IQ2_XXS tensor bytes only marginally; the
#   0.993 x IQ2_XXS budget (Gemma/Qwen convention) needs -10.9 MB = exactly 3 class steps 3->2 on 58.7M-weight tensors (N/16 = 3,670,016 B each).
#   Candidates: ffn_gate/ffn_up/ffn_down of layers 4..30 (IQ2_XXS slots, N = 58,720,256). A class step multiplies a tensor's weighted error by ~2 (first-order
#   D ~ 4^-R), so the damage is proportional to the tensor's calibration-weighted energy  S_t = sum_ij w_ij^2 * imatrix_j  (imatrix = sum of squared inputs over the same
#   calibration text, so S_t ~ expected squared output energy of W x); the saving is identical for all candidates. Pick the 3 smallest S_t, at most one per layer and
#   no two within one layer of each other. usage: l8_trim.py MIX IMATRIX OUT.json   (CPU, ~2 min)
import sys, os, json, re, numpy as np, torch
sys.path.insert(0, "llama.cpp/gguf-py")
from gguf import GGUFReader
from safetensors import safe_open
from huggingface_hub import snapshot_download
mix, imx, out = sys.argv[1:4]
IM = {t.name[:-8]: np.asarray(t.data, np.float64).ravel() for t in GGUFReader(imx).tensors if t.name.endswith(".in_sum2")}
slot = {t.name: t.tensor_type.name for t in GGUFReader(mix).tensors}
SNAP = snapshot_download("unsloth/Meta-Llama-3.1-8B", local_files_only=True); WM = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
HFN = {"ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
S = {}
for li in range(4, 31):
    for k, hf in HFN.items():
        name = f"blk.{li}.{k}.weight"; assert slot[name] == "IQ2_XXS", (name, slot[name])
        key = f"model.layers.{li}.{hf}.weight"
        with safe_open(os.path.join(SNAP, WM[key]), framework="pt", device="cpu") as f: w = f.get_tensor(key).float()
        S[name] = float((w.pow(2).sum(0).double().numpy() * IM[name]).sum()); assert w.numel() == 58720256
rank = sorted(S, key=S.get); moved = []; lay = lambda n: int(n.split(".")[1])
for n in rank:
    if all(abs(lay(n) - lay(m)) > 1 for m in moved): moved.append(n)
    if len(moved) == 3: break
json.dump(dict(moved=moved, saving_each=58720256 // 16, rank=[(n, S[n]) for n in rank[:12]], median=float(np.median(list(S.values()))), rule="3 smallest sum_ij w^2*imatrix_j among ffn_* L4-30, <=1 per layer, not adjacent"), open(out, "w"), indent=1)
print("TRIM moved", moved, "S", [round(S[m], 3) for m in moved], "median S", round(float(np.median(list(S.values()))), 3), "min/max", round(min(S.values()), 3), round(max(S.values()), 3))
