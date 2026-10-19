# R48k REAL FILE (kernel formats; copy of r48_container.py with the r48k codec): container for an r48_build.py run. Items: GGUF meta of the IQ2_XXS mix, the A1 global tables, one A1 blob per
# coded linear, raw mix bytes for every other tensor. Same part format as r36_container.py (imported read-only).
# Variant "all": all 252 linears A1. Variant "iq212": A1 only on the 212 IQ2_XXS-slot tensors, the 40 others raw mix bytes
# (= Model A layout). Verify decodes the container (CPU) and requires decode -> Q8_0 == the evaluated carrier, tensor by tensor.
# usage: r48_container.py encode BLOBDIR MIX PREFIX VARIANT | verify PREFIX BLOBDIR CARRIER MIX VARIANT
import sys, os, json, hashlib, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "r36_rans"))
sys.path.insert(0, "~/repo/llama.cpp/gguf-py")
import r36_container as R36, r48_a1gpu as A, r48k as K
MODELA_BYTES, IQ2_BYTES, E1R_BYTES = 1240465501, 1246621312, 1240467793
def use_blob(ty, variant): return variant == "all" or ty == "IQ2_XXS"
def cmd_encode(blobdir, mix, prefix, variant, part_mb=640):
    from gguf import GGUFReader
    r = GGUFReader(mix); raw = np.memmap(mix, np.uint8, mode="r"); cnt = dict(T=0, B=0, tb=0)
    def items():
        yield "__gguf_meta__", "M", "META", bytes(raw[: r.data_offset])
        yield "__vrk_tables__", "A", "TAB", open(os.path.join(blobdir, "__vrk_tables__.blob"), "rb").read()
        for t in r.tensors:
            ty = t.tensor_type.name; fn = os.path.join(blobdir, t.name + ".blob")
            if os.path.exists(fn) and use_blob(ty, variant):
                b = open(fn, "rb").read(); cnt["T"] += 1; cnt["tb"] += len(b); yield t.name, "T", ty, b
            else:
                assert ty in ("F32", "Q5_K", "Q4_K", "Q2_K", "Q6_K", "Q8_0", "F16"), f"{t.name} {ty} has no blob"
                cnt["B"] += 1; o = int(t.data_offset); yield t.name, "B", ty, bytes(raw[o:o + int(t.n_bytes)])
    extra = dict(fmt="VEDQR48K kernel-exact VR47/R49 (r49_vr.cu cpu_vr semantics), Had256, variant " + variant, src_mix=os.path.basename(mix),
                 recon="W=clamp(ifwht256(T[cls][label]*S)/c,-wmax,wmax); ifwht256(Z)=(H256 Z/16)*S, signs = r36 seed-0 signs")
    paths = R36.write_parts(prefix, items(), part_mb * 1_000_000, extra)
    tot = sum(os.path.getsize(p) for p in paths)
    print(f"PARTS {' '.join(f'{os.path.basename(p)}={os.path.getsize(p)}' for p in paths)}")
    print(f"REAL_BYTES {tot} ({tot / 1e6:.3f} MB) variant {variant}: vs IQ2_XXS {IQ2_BYTES}: {tot - IQ2_BYTES:+d} B; vs Model A {MODELA_BYTES}: {tot - MODELA_BYTES:+d} B; vs E1r {E1R_BYTES}: {tot - E1R_BYTES:+d} B")
    print(f"A1 tensors {cnt['T']}, blob bytes {cnt['tb']}; raw tensors {cnt['B']}; meta {r.data_offset} B")
def cmd_verify(prefix, blobdir, carrier, mix, variant):
    from gguf import GGUFReader, GGMLQuantizationType as QT
    from gguf.quants import dequantize, quantize
    torch.set_num_threads(int(os.environ.get("NT", "4")))
    items, extra = R36.read_parts(prefix); car = {t.name: t for t in GGUFReader(carrier).tensors}
    vb = os.path.basename(blobdir.rstrip("/"))[6:]; order = ["vrc", "vrcB"] if vb == "vrcBm" else [vb]; st = {}
    for v_ in order: st.update({s["name"]: s for s in json.load(open(os.path.join(os.path.dirname(blobdir.rstrip("/")), "stats_vrk.json"))) if s["v"] == v_})
    _, ifw = A.make_fw("cpu"); codes = None; nT = okS = okQ8 = nB = 0; mis = []
    for name, kind, ty, blob in items:
        if kind == "M": assert bytes(blob) == open(mix, "rb").read(len(blob)), "meta differs"; print("VERIFY gguf meta == mix header", flush=True); continue
        if kind == "A": codes, _ = K.tables_from_blob(bytes(blob)); continue
        ct = car[name]; cb = np.asarray(ct.data).view(np.uint8).reshape(-1)
        if kind == "T":
            nT += 1; b = bytes(blob); okS += hashlib.sha256(b).hexdigest()[:16] == st[name]["sha"]
            h, Yq = K.unpack_blob(b, codes); Wn = (ifw(Yq) / h["c"]).clamp(-h["wmax"], h["wmax"]).numpy()
            q8 = np.asarray(quantize(Wn, QT.Q8_0)).view(np.uint8).reshape(-1)
        else:
            nB += 1
            if ty in ("F32", "F16"): q8 = np.frombuffer(blob, np.uint8)
            else:
                rws = int(ct.n_elements // int(ct.shape[0])); dq = np.asarray(dequantize(np.frombuffer(blob, np.uint8).reshape(rws, -1), QT[ty]), np.float32).reshape(rws, -1)
                q8 = (np.asarray(quantize(dq, QT.Q8_0)) if ct.tensor_type == QT.Q8_0 else dq.astype(np.asarray(ct.data).dtype)).view(np.uint8).reshape(-1)
        eq = q8.size == cb.size and np.array_equal(q8, cb); okQ8 += eq
        if not eq: mis.append((name, kind, float(np.mean(q8 != cb)) if q8.size == cb.size else -1.0))
    print(f"VERIFY A1 tensors {nT}: blob sha == build sha {okS}/{nT}")
    print(f"VERIFY carrier tensor bytes reproduced from the container (decode -> Q8_0): {okQ8}/{nT + nB} (A1 {nT}, raw {nB}); mismatches {mis[:10]}")
if __name__ == "__main__":
    c = sys.argv[1]
    if c == "encode": cmd_encode(*sys.argv[2:6])
    elif c == "verify": cmd_verify(*sys.argv[2:7])
