# R51: exact clamp restore as a sparse delta (coordinator decision 2026-10-05). Per tensor: W = ifwht256(Yq)/c (r48k.unpack_blob, seed-0 signs),
# entries with |W| > wmax (ORIGINAL row/col indices, sorted by row then col), delta = clamp(W) - W. Kernel: y = W_vr x + Delta x.
# Writes research/work/r51t/vr_delta_pingala8.npz (rows u16, cols u16, d32 f32, d16 f16 per tensor) and research/work/r51t/delta_counts.json.
import sys, os, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import r48k as K, r48_a1gpu as A
torch.set_num_threads(int(os.environ.get("NT", "8")))
bd = sys.argv[1]; od = sys.argv[2]; os.makedirs(od, exist_ok=True)
T, tr = K.tables_from_blob(open(os.path.join(bd, "__vrk_tables__.blob"), "rb").read()); assert tr == 2
SJ = os.environ.get("R51D_STATS", os.path.join(os.path.dirname(bd.rstrip("/")), "stats_vrk.json")); VAR = os.environ.get("R51D_VAR", "vrc")   # R57: env overrides, defaults = original
st = [s["name"] for s in json.load(open(SJ)) if s["v"] == VAR]
_, ifw = A.make_fw("cpu"); out = {}; cnt = {}; e16max = (0.0, ""); tot = 0
for nm in sorted(st, key=lambda s: (int(s.split(".")[1]), s)):
    h, Yq = K.unpack_blob(open(os.path.join(bd, nm + ".blob"), "rb").read(), T); W = ifw(Yq) / h["c"]; wm = h["wmax"]
    r, c = torch.nonzero(W.abs() > wm, as_tuple=True); w = W[r, c]; d = (w.clamp(-wm, wm) - w).numpy().astype(np.float32)
    assert h["rows"] < 65536 and h["n"] < 65536; cnt[nm] = int(len(d)); tot += len(d)
    if len(d):
        out[nm + "/rows"] = r.numpy().astype(np.uint16); out[nm + "/cols"] = c.numpy().astype(np.uint16); out[nm + "/d32"] = d; out[nm + "/d16"] = d.astype(np.float16)
        e = float(np.abs((w.numpy() + d.astype(np.float16).astype(np.float32)) - w.clamp(-wm, wm).numpy()).max())
        if e > e16max[0]: e16max = (e, nm)
np.savez(os.path.join(od, os.environ.get("R51D_OUT", "vr_delta_pingala8.npz")), **out); json.dump(dict(total=tot, per_tensor=cnt), open(os.path.join(od, "delta_counts.json"), "w"), indent=0)
print(f"DELTA tensors_with_delta {sum(v > 0 for v in cnt.values())}/{len(cnt)} entries {tot} bytes_fp16 {6 * tot} bytes_f32 {8 * tot}; "
      f"fp16-delta max |W+d16 - clamp(W)| = {e16max[0]:.4e} ({e16max[1]})")
