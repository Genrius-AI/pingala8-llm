# R51 item 4: native Pingala-8 Model B GGUF for the fast VR kernel, written as gguf-split shards (each <= 1,000,000,000 B).
# Source: VRK2 blobs work/r48k3/blobs_vrc (+ __vrk_tables__ tr 2 = 8 gana states), KV meta + non-linear tensors from the IQ2_XXS mix,
# clamp deltas research/work/r51t/vr_delta_pingala8.npz (r51_delta.py). Kernel bytes from r51_transcode.transcode (bit-checked, transcoder_r51.md).
# Tensors per linear L = blk.N.<attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down>:
#   L.weight      I8  [kbytes]  kernel tensor bytes (vr49.cuh layout: A|B|c,NS,TRL,mode | M | S [b][r] | pad16 | D), sorted rows
#   L.vr_prm      I16 [rows]    sorted row -> original row (y_orig[prm[r]] = y_sorted[r]); rows = ne0, logical shape (K, rows)
#   L.vr_c        F16 [K]       per input column prescale: x' = H256 S (x / c) / 16 per 256-block (r36 seed-0 signs S)
#   L.vr_didx     I16 [cnt, 2]  clamp-restore delta: row 0 = original rows, row 1 = original cols (sorted by row) (only if cnt > 0)
#   L.vr_dval     F32 [cnt]     delta = clamp(W) - W at those entries; y += Delta x (exact, transcoder_r51.md)
# Global: vr.had_signs I8 [256] (r48_a1gpu.make_fw seed-0 signs). KV: vr.format, vr.ns 108, vr.trl 2, vr.n_linear 252.
# usage: r51_native.py plan OUT | dryrun OUT | build OUT      (OUT = path stem, shards OUT-0000i-of-0000n.gguf)
import sys, os, json, struct, time, hashlib, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "~/repo/llama.cpp/gguf-py")
import gguf, r51_transcode as TC, r48_a1gpu as A1
from gguf import GGUFReader, GGUFWriter, GGUFValueType
ROOT = "~/repo"; E = os.environ.get   # R57: every path/constant below can be overridden by R51N_* env vars; the defaults are the original Model B values
BD = E("R51N_BD", f"{ROOT}/work/r48k3/blobs_vrc"); MIX = E("R51N_MIX", f"{ROOT}/work/r45b/4B-IQ2_XXS-r45b.gguf")
STATS = E("R51N_STATS", f"{ROOT}/work/r48k3/stats_vrk.json"); VAR = E("R51N_VAR", "vrc")        # stats json + the variant tag ("v") whose 252 entries are the linears
NPZ = E("R51N_NPZ", f"{ROOT}/research/work/r51t/vr_delta_pingala8.npz"); BUDGET = int(E("R51N_BUDGET", "1246621312")); CONTAINER = int(E("R51N_CONTAINER", "1240445591")); SHARD_MAX = 1_000_000_000
SPLIT_DATA = SHARD_MAX - 8_000_000                                     # data bytes per shard; leaves room for the 5.96 MB KV header + padding
DRY3 = ["blk.0.attn_v.weight", "blk.6.ffn_down.weight", "blk.16.attn_q.weight"]
al = lambda n: (n + 31) // 32 * 32

def kbytes_of(blob):
    p = TC.parse_vrk2(blob); return TC.kbytes_size(p["rows"], p["n"], len(p["stream"]) // 4), p["rows"], p["n"]

def plan_tensors(names_only=None):
    """Ordered list of (name, kind, shape (numpy order), np dtype, nbytes, raw_type, src) for the native file."""
    r = GGUFReader(MIX); z = np.load(NPZ); lin = {s["name"] for s in json.load(open(STATS)) if s["v"] == VAR}
    assert len(lin) == 252; out = []
    for t in r.tensors:
        if names_only is not None and t.name not in names_only: continue
        if t.name in lin:
            blob = open(os.path.join(BD, t.name + ".blob"), "rb").read(); kb, rows, n = kbytes_of(blob); base = t.name[:-len(".weight")]
            out.append((t.name, "vr", (kb,), np.int8, kb, None, t.name))
            out.append((base + ".vr_prm", "prm", (rows,), np.int16, 2 * rows, None, t.name))
            out.append((base + ".vr_c", "c", (n,), np.float16, 2 * n, None, t.name))
            if t.name + "/rows" in z.files:
                cnt = len(z[t.name + "/rows"])
                out.append((base + ".vr_didx", "didx", (2, cnt), np.int16, 4 * cnt, None, t.name))
                out.append((base + ".vr_dval", "dval", (cnt,), np.float32, 4 * cnt, None, t.name))
        else:
            out.append((t.name, "raw", tuple(t.data.shape), t.data.dtype, int(t.data.nbytes), t.tensor_type, t.name))
    out.append(("vr.had_signs", "signs", (256,), np.int8, 256, None, None))
    return r, out

def make_writer(r, out, stem, split_data):
    w = GGUFWriter(stem + ".gguf", r.fields["general.architecture"].contents(), split_max_size=split_data)
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."): continue
        vt = f.types[0]; w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == GGUFValueType.ARRAY else None)
    w.add_key_value("vr.format", "VRii-C Pingala-8 (8 gana states, Ungerboeck ff labels, NS 108, TRL 2), r51 native", GGUFValueType.STRING)
    w.add_key_value("vr.ns", 108, GGUFValueType.UINT32); w.add_key_value("vr.trl", 2, GGUFValueType.UINT32)
    w.add_key_value("vr.n_linear", sum(1 for o in out if o[1] == "vr"), GGUFValueType.UINT32)
    w.add_key_value("vr.recon", "y_orig[prm[r]] = sum_k Wvr_sorted[r,k] * xh[k] + (Delta x)[orig]; xh = per-256 (H256 diag(signs) (x/c))/16; "
                    "Wvr = vr49.cuh vr_wint*vr_wscale; Delta from vr_didx/vr_dval restores clamp(W,-wmax,wmax) exactly", GGUFValueType.STRING)
    for name, kind, shp, dt, nb, raw, _ in out:
        w.add_tensor_info(name, shp, np.dtype(dt), nb, raw_dtype=raw)
    return w

def tensor_data(kind, src, r_by_name, z, tab, cache):
    if kind == "raw": return np.asarray(r_by_name[src].data)
    if kind == "signs":
        g = torch.Generator().manual_seed(0); return (torch.randint(0, 2, (256,), generator=g) * 2 - 1).numpy().astype(np.int8)   # = make_fw SG
    if cache.get("name") != src:
        kb, side = TC.transcode(open(os.path.join(BD, src + ".blob"), "rb").read(), tab); cache.clear(); cache.update(name=src, kb=kb, side=side)
    kb, side = cache["kb"], cache["side"]
    if kind == "vr": return np.frombuffer(kb, np.int8)
    if kind == "prm": assert side["prm"].max() < 32768; return side["prm"].astype(np.int16)
    if kind == "c": return np.asarray(side["c16"], np.float16)
    if kind == "didx": return np.stack([z[src + "/rows"], z[src + "/cols"]]).astype(np.uint16).view(np.int16)
    if kind == "dval": return z[src + "/d32"].astype(np.float32)

def write(stem, names_only=None, split_data=SPLIT_DATA, log=print):
    r, out = plan_tensors(names_only); rb = {t.name: t for t in r.tensors}; z = np.load(NPZ); tab = open(os.path.join(BD, "__vrk_tables__.blob"), "rb").read()
    w = make_writer(r, out, stem, split_data); w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    paths = [f.name for f in w.fout]; cache = {}; t0 = time.time()
    for i, (name, kind, shp, dt, nb, raw, src) in enumerate(out):
        a = tensor_data(kind, src, rb, z, tab, cache); assert a.nbytes == nb, (name, a.nbytes, nb)
        w.write_tensor_data(a)
        if kind == "vr" and (i % 40 == 0): log(f"  wrote {i}/{len(out)} {name} t {time.time() - t0:.0f}s")
    w.close(); sizes = [os.path.getsize(p) for p in paths]
    return paths, sizes, out

def header_sizes(stem, names_only=None):
    """Exact shard sizes without writing tensor data: write header + KV + tensor infos only, then add aligned data sizes."""
    r, out = plan_tensors(names_only); w = make_writer(r, out, stem, SPLIT_DATA)
    shard_t = [list(t.values()) for t in w.tensors]; w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    hdr = [f.tell() for f in w.fout]; paths = [f.name for f in w.fout]; w.close()
    for p in paths: os.remove(p)
    return [al(h) + sum(al(ti.nbytes) for ti in ts) for h, ts in zip(hdr, shard_t)], hdr, [len(ts) for ts in shard_t], out, paths

def cmd_plan(stem):
    sizes, hdr, nts, out, paths = header_sizes(stem); tot = sum(sizes)
    by = {}; [by.__setitem__(k, by.get(k, 0) + al(nb)) for _, k, _, _, nb, _, _ in out]
    nlin = sum(1 for o in out if o[1] == "vr")
    print(f"PLAN shards {len(sizes)}: " + "; ".join(f"{os.path.basename(p)} {s} B ({n} tensors, header {h} B)" for p, s, n, h in zip(paths, sizes, nts, hdr)))
    print(f"PLAN tensors {len(out)} (linears {nlin}); data bytes by kind (32-aligned): {by}")
    print(f"PLAN TOTAL {tot} B; budget {BUDGET} -> margin {BUDGET - tot:+d} B ({'OK' if tot <= BUDGET else 'OVER'}); vs container {CONTAINER}: {tot - CONTAINER:+d} B; "
          f"max shard {max(sizes)} ({'OK' if max(sizes) <= SHARD_MAX else 'OVER'} 1e9)")
    r = GGUFReader(MIX); slot = sum(int(t.n_bytes) for t in r.tensors if any(o[0] == t.name and o[1] == "vr" for o in out))
    vr = by["vr"]; print(f"PLAN alt 'padded slot' layout (linears keep mix type/shape, VR bytes + zero pad): linear data {slot} vs tight {vr} -> total >= {tot - vr + slot} B "
                         f"({'OVER' if tot - vr + slot > BUDGET else 'ok'} budget)")
    return tot

def cmd_dryrun(stem):
    t0 = time.time(); paths, sizes, out = write(stem, set(DRY3), split_data=5_000_000)                 # small split size: force >= 2 shards
    print(f"DRYRUN wrote {len(paths)} shards for {DRY3}: " + "; ".join(f"{os.path.basename(p)} {s} B" for p, s in zip(paths, sizes)) + f" total {sum(sizes)} t {time.time() - t0:.0f}s")
    rd = {}; [rd.update({t.name: t for t in GGUFReader(p).tensors}) for p in paths]
    tab = open(os.path.join(BD, "__vrk_tables__.blob"), "rb").read(); z = np.load(NPZ); ok = True
    for nm in DRY3:
        kb, side = TC.transcode(open(os.path.join(BD, nm + ".blob"), "rb").read(), tab); b = nm[:-7]
        e1 = bytes(np.asarray(rd[nm].data).view(np.uint8)) == kb; e2 = np.array_equal(np.asarray(rd[b + ".vr_prm"].data).astype(np.int64), side["prm"])
        e3 = np.array_equal(np.asarray(rd[b + ".vr_c"].data).view(np.uint16), np.asarray(side["c16"]).view(np.uint16))
        e4 = (b + ".vr_didx" not in rd and nm + "/rows" not in z.files) or (np.array_equal(np.asarray(rd[b + ".vr_didx"].data).view(np.uint16)[0], z[nm + "/rows"])
              and np.array_equal(np.asarray(rd[b + ".vr_didx"].data).view(np.uint16)[1], z[nm + "/cols"]) and np.array_equal(np.asarray(rd[b + ".vr_dval"].data), z[nm + "/d32"]))
        ok &= e1 and e2 and e3 and e4
        print(f"DRYRUN readback {nm}: types {rd[nm].tensor_type.name}/{rd[b + '.vr_prm'].tensor_type.name}/{rd[b + '.vr_c'].tensor_type.name}"
              f"{'/' + rd[b + '.vr_didx'].tensor_type.name + '/' + rd[b + '.vr_dval'].tensor_type.name if b + '.vr_didx' in rd else ''} "
              f"kernel bytes == transcode {e1}, prm {e2}, c {e3}, delta {e4}")
    print(f"DRYRUN readback {'PASS' if ok else 'FAIL'}"); return paths

def cmd_build(stem):
    t0 = time.time(); paths, sizes, out = write(stem, None, log=lambda s: print(s, flush=True)); tot = sum(sizes)
    for p, s in zip(paths, sizes):
        h = hashlib.sha256(); f = open(p, "rb")
        for blk in iter(lambda: f.read(1 << 24), b""): h.update(blk)
        print(f"SHARD {p} {s} B sha256 {h.hexdigest()} {'OK' if s <= SHARD_MAX else 'OVER 1e9'}", flush=True)
    print(f"NATIVE TOTAL {tot} B budget {BUDGET} margin {BUDGET - tot:+d} vs container {CONTAINER} {tot - CONTAINER:+d}; tensors {len(out)} t {time.time() - t0:.0f}s")


def cmd_verify(stem):
    """Read every shard back; every tensor must equal freshly regenerated data (raw = mix bytes, VR = transcode, side = npz/transcode)."""
    import glob
    paths = sorted(glob.glob(stem + "-0000*-of-*.gguf")); rd = {}
    for p in paths: rd.update({t.name: t for t in GGUFReader(p).tensors})
    r, out = plan_tensors(); rb = {t.name: t for t in r.tensors}; z = np.load(NPZ); tab = open(os.path.join(BD, "__vrk_tables__.blob"), "rb").read(); cache = {}
    bad = [nm for nm, kind, shp, dt, nb, raw, src in out
           if not (nm in rd and bytes(np.asarray(rd[nm].data).view(np.uint8)) == bytes(np.ascontiguousarray(tensor_data(kind, src, rb, z, tab, cache)).view(np.uint8)))]
    kv = GGUFReader(paths[0]).fields
    print(f"VERIFY shards {len(paths)} tensors in file {len(rd)} expected {len(out)}; mismatches {len(bad)} {bad[:5]}; "
          f"split.count {kv['split.count'].contents()} vr.ns {kv['vr.ns'].contents()} vr.trl {kv['vr.trl'].contents()} {'PASS' if not bad and len(rd) == len(out) else 'FAIL'}")
if __name__ == "__main__":
    {"plan": cmd_plan, "dryrun": cmd_dryrun, "build": cmd_build, "verify": cmd_verify}[sys.argv[1]](sys.argv[2])
