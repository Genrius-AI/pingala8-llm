# R48k: decode real blobs (r48k.unpack_blob) and compare with the evaluated Q8_0 carrier and Model A, Deval on r37 eval H.
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "~/repo/llama.cpp/gguf-py")
import r48k as K, r48_a1gpu as A
from gguf import GGUFReader, GGMLQuantizationType as QT
from gguf.quants import dequantize, quantize
bd, car = sys.argv[1], sys.argv[2]; names = sys.argv[3].split(",")
T, tr = K.tables_from_blob(open(os.path.join(bd, "__vrk_tables__.blob"), "rb").read()); _, ifw = A.make_fw("cpu")
C = {t.name: t for t in GGUFReader(car).tensors}; F = {t.name: t for t in GGUFReader("~/repo/models/qwen3-4B-f16.gguf").tensors}
MA = {t.name: t for t in GGUFReader("~/repo/work/r45b/4B-r45b-ung8.gguf").tensors}
print("tables tr", tr, "NS", K.NSC[0], "S1", K.S1, "S2", K.S2)
for nm in names:
    name = nm + ".weight"; h, Yq = K.unpack_blob(open(os.path.join(bd, name + ".blob"), "rb").read(), T)
    Wd = (ifw(Yq) / h["c"]).clamp(-h["wmax"], h["wmax"]); W = torch.as_tensor(np.asarray(F[name].data, np.float32)); rows, n = W.shape
    Wc = torch.as_tensor(np.asarray(dequantize(C[name].data, C[name].tensor_type), np.float32).reshape(rows, n))
    Wa = torch.as_tensor(np.asarray(dequantize(MA[name].data, MA[name].tensor_type), np.float32).reshape(rows, n))
    q8eq = np.array_equal(np.asarray(quantize(Wd.numpy(), QT.Q8_0)).view(np.uint8).reshape(-1), np.asarray(C[name].data).view(np.uint8).reshape(-1))
    He = torch.as_tensor(np.load(f"~/repo/research/work/r37_H_{nm}_eval.npy")).float()
    D = lambda Wq: float((((Wq - W) @ He) * (Wq - W)).sum() / ((W @ He) * W).sum())
    print(f"{nm}: carrier==Q8_0(decoded) bytes {q8eq}; max|car-dec| {float((Wc - Wd).abs().max()):.3e} rel {float((Wc - Wd).norm() / Wd.norm()):.2e}; "
          f"Deval(r37 H) decoded {D(Wd):.6f} carrier {D(Wc):.6f} ModelA {D(Wa):.6f} ratio dec/A {D(Wd) / D(Wa):.3f}", flush=True)
