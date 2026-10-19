# R48k: convert every VRK1 blob in BLOBDIR to the kernel tile-block-major VRK2 layout (lossless; decode checked) and update stats sha.
import sys, os, glob, json, hashlib, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r48k as K
bd = sys.argv[1]; T, _ = K.tables_from_blob(open(os.path.join(bd, "__vrk_tables__.blob"), "rb").read()); sf = os.path.join(os.path.dirname(bd.rstrip("/")), "stats_vrk.json")
st = json.load(open(sf)); v = os.path.basename(bd.rstrip("/"))[6:]; idx = {s["name"]: s for s in st if s["v"] == v}; n = 0
for fn in sorted(glob.glob(os.path.join(bd, "blk.*.blob"))):
    b = open(fn, "rb").read()
    if b[:4] != b"VRK1": idx[os.path.basename(fn)[:-5]]["sha"] = hashlib.sha256(b).hexdigest()[:16]; continue
    b2 = K.relayout(b); assert len(b2) == len(b)
    assert K.to_vrk1(b2) == b, fn                                    # lossless byte permutation (decode equality is checked by container verify)
    open(fn, "wb").write(b2); idx[os.path.basename(fn)[:-5]]["sha"] = hashlib.sha256(b2).hexdigest()[:16]; n += 1
json.dump(st, open(sf, "w")); print(f"RELAYOUT {bd}: {n} blobs VRK1 -> VRK2, round trip VRK2->VRK1 byte-identical, sizes unchanged")
