# R48k: write the layers 0-3 boosted VRii-C blobs (blobs_vrcB) into the VRii-C carrier (decode -> Q8_0), for the boost ppl run.
# usage: r48k_swapB.py BLOBDIR_vrcB CARRIER
import sys, os, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "~/repo/llama.cpp/gguf-py")
import r48k as K, r48_a1gpu as A
from gguf import GGUFReader, GGMLQuantizationType as QT
from gguf.quants import quantize
bd, car = sys.argv[1:3]; T, _ = K.tables_from_blob(open(os.path.join(bd, "__vrk_tables__.blob"), "rb").read()); _, ifw = A.make_fw("cpu")
dst = {t.name: t for t in GGUFReader(car, "r+").tensors}; k = 0
for fn in sorted(glob.glob(os.path.join(bd, "blk.*.blob"))):
    name = os.path.basename(fn)[:-5]; h, Yq = K.unpack_blob(open(fn, "rb").read(), T); Wn = (ifw(Yq) / h["c"]).clamp(-h["wmax"], h["wmax"]).numpy()
    for r0 in range(0, Wn.shape[0], 16384): dst[name].data[r0:r0 + 16384] = quantize(Wn[r0:r0 + 16384], QT.Q8_0)
    k += 1; print("SWAPB", name, flush=True)
print(f"SWAPPED_B {k} tensors")
