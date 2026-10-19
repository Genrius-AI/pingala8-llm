# b3p_native.py (R59 solver, B3+; prereg research/agents/solver_b3plus_r59.md). B3+ = all 252 linears from the f1 roff blobs, stored in the
# "VR3" native layout: natural row order (no L.vr_prm I16 [rows]) and no u32 tile meta M. Every rate-off tensor has one class c, so tile (g, b)
# sits at payload word (g*nb + b) * 64 * NW(c); c is header word 515 (VRN_VR3_TAG | c). KV vr.layout = 3.
#   VR3 file tensor:  A | B | cs, NS, TRL, VR3_TAG|c   (2064 B) | S fp16 [b][r] (64 nt B) | D (nt * 256 NW(c) B)
# CUDA kernels are unchanged: llama-model-loader expands VR3 on the host (llama-vr3.h vrn3_expand) into the v2 in-memory bytes
# (M[i] = i*64*NW(c) | c<<29, word 515 = 0, S at 2064 + 4 nt, D at align16(2064 + 68 nt)) with prm = pinv = identity. The expansion is
# byte-checked here against r51_transcode.transcode of the natural-order VRK2 blob.
# Every stage is CPU, resumable (per-stage json in $B3P_OUT), chunked (peak RSS < 2.5 GB), and waits while ~/repo/logs/commit_low exists.
# usage (CPU=1 bin/job python3 b3p_native.py <cmd>):
#   scan                     header/meta of every B3 and B3+ blob: constant class, meta == arithmetic offsets, stream length, byte sums
#   delta OUTDIR             clamp-delta npz for B3+ (B3 npz entries for byte-identical blobs; chunked r51_delta recompute otherwise,
#                            validated bit-exactly on B3's own blk.4/blk.5 ffn_gate entries first)
#   plan                     exact native totals from the GGUF writer headers: B3 v2 (must equal 1245915872), B3 v3, B3+ v3, B3+ v2
#   check all|NAME...        VR3 bytes: expansion == transcode(natural blob) byte-exact; cpu_vr_port(expanded) == r48k(original VRK2) bit-exact
#   dump OUTDIR NAME...      VR3 tensor bytes + expected v2 bytes (input of the C test b3p_vr3_test.c)
#   build STEM | verify STEM VR3 native shards; SET=b3p (default) or SET=b3 (B3's own blobs in VR3, the GPU layout check); B3P_NPZ = delta npz
import sys, os, json, math, struct, time, hashlib, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "~/repo/llama.cpp/gguf-py")
import r48k as K, r48_a1gpu as A1, r51_transcode as TC, r51_native as RN
from gguf import GGUFReader, GGUFValueType
torch.set_num_threads(int(os.environ.get("NT", "6")))
ROOT = "~/repo"; E = os.environ.get
B3BD = f"{ROOT}/work/r59b3/blobs_b3"; F1BD = f"{ROOT}/work/r52a/f1/blobs_roff"; B3NPZ = f"{ROOT}/work/r59b3/delta/vr_delta_b3.npz"
STATS = f"{ROOT}/work/r59b3/stats_b3.json"; B3_BYTES = 1245915872
OUT = E("B3P_OUT", f"{ROOT}/work/b3p/state"); os.makedirs(OUT, exist_ok=True)
NW = TC.NW; HDR = TC.HDR; VR3_TAG = 0x56523300; al = RN.al
NAMES = [s["name"] for s in json.load(open(STATS)) if s["v"] == "b3"]; assert len(NAMES) == 252
B3P = {nm: os.path.join(F1BD, nm + ".blob") for nm in NAMES}                       # option (a): every linear from f1 roff
B3 = {nm: os.path.join(B3BD, nm + ".blob") for nm in NAMES}
TAB = open(os.path.join(F1BD, "__vrk_tables__.blob"), "rb").read(); assert TAB == open(os.path.join(B3BD, "__vrk_tables__.blob"), "rb").read()
T, _tr = K.tables_from_blob(TAB); assert _tr == 2                                     # Pingala-8 (sets r48k NS 8)

def wait_commit():
    while os.path.exists(f"{ROOT}/logs/commit_low"): print("commit_low present, waiting 60 s", flush=True); time.sleep(60)

def cache(name, fn):
    p = os.path.join(OUT, name + ".json")
    if os.path.exists(p): return json.load(open(p))
    wait_commit(); r = fn(); json.dump(r, open(p + ".tmp", "w"), indent=0); os.replace(p + ".tmp", p); return r

def head(path):
    with open(path, "rb") as f:
        mg, rows, n, tr, pb, wm = struct.unpack(HDR, f.read(16)); assert mg == b"VRK2"; nb = n // 256; nt = rows // 32 * nb
        o = 16 + 2 * n + (rows * pb + 7) // 8 + 2 * rows * nb; f.seek(o); meta = np.frombuffer(f.read(4 * nt), "<u4")
    sl = os.path.getsize(path) - o - 4 * nt; c = int(meta[0] >> 29)
    const = bool(((meta >> 29) == c).all()); arith = bool(((meta & 0x1FFFFFFF).astype(np.int64) == np.arange(nt) * 64 * NW[c]).all())
    return dict(rows=rows, n=n, nb=nb, nt=nt, c=c, const=const, arith=arith, slen=sl, slen_ok=sl == nt * 256 * NW[c], size=os.path.getsize(path),
                kb2=TC.kbytes_size(rows, n, sl // 4), kb3=2064 + 64 * nt + sl)

def cmd_scan():
    def run():
        res = {}
        for tag, mp in (("b3", B3), ("b3p", B3P)):
            hs = {nm: head(p) for nm, p in mp.items()}
            res[tag] = dict(n=len(hs), const=sum(h["const"] for h in hs.values()), arith=sum(h["arith"] for h in hs.values()), slen_ok=sum(h["slen_ok"] for h in hs.values()),
                            classes={str(c): sum(h["c"] == c for h in hs.values()) for c in range(8)}, blob=sum(h["size"] for h in hs.values()),
                            M=sum(4 * h["nt"] for h in hs.values()), prm_i16=sum(2 * h["rows"] for h in hs.values()),
                            kb2=sum(al(h["kb2"]) for h in hs.values()), kb3=sum(al(h["kb3"]) for h in hs.values()))
        res["same_blob"] = [nm for nm in NAMES if os.path.realpath(B3[nm]) == os.path.realpath(B3P[nm])]
        return res
    r = cache("scan", run)
    for tag in ("b3", "b3p"):
        x = r[tag]; print(f"SCAN {tag}: {x['n']} tensors, constant-class {x['const']}, meta==arithmetic {x['arith']}, stream len ok {x['slen_ok']}, classes "
                          f"{ {k: v for k, v in x['classes'].items() if v} }, blob bytes {x['blob']}, M {x['M']}, vr_prm I16 {x['prm_i16']}, "
                          f"kernel tensor bytes (32-al) v2 {x['kb2']} -> VR3 {x['kb3']} (-{x['kb2'] - x['kb3']})")
    print(f"SCAN b3p blobs that are the same file as b3: {len(r['same_blob'])}/252; differ: {[n for n in NAMES if n not in r['same_blob']]}")
    return r

# ---------------------------------------------------------------- chunked VRK2 helpers (identity-perm sub-blobs of whole tiles)
def sub_vrk2(R, n, tr, wm, c16, S, c, st):
    pb = max(1, math.ceil(math.log2(R))); ntl = R // 32 * (n // 256)
    meta = (np.arange(ntl, dtype=np.uint64) * np.uint64(64 * NW[c])) | (np.uint64(c) << np.uint64(29))
    return (struct.pack(HDR, b"VRK2", R, n, tr, pb, np.float16(wm)) + np.asarray(c16, np.float16).tobytes() + A1._pack(np.arange(R), pb).tobytes()
            + np.ascontiguousarray(S, np.float16).tobytes() + meta.astype("<u4").tobytes() + bytes(st))

def gstep(n): return max(1, (1 << 22) // (32 * n))                                    # tiles per chunk (~4M weights)

def sorted_chunks(p):
    """yield (g0, g1, Yq rows 32g0..32g1 in SORTED order) from the shipped r48k decoder on byte slices of the original VRK2 stream."""
    rows, n = p["rows"], p["n"]; nb = n // 256; tiles = rows // 32; c = int(p["meta"].flat[0] >> 29); tb = 256 * NW[c]
    assert ((p["meta"] >> 29) == c).all() and ((p["meta"].ravel() & 0x1FFFFFFF).astype(np.int64) == np.arange(tiles * nb) * 64 * NW[c]).all()
    for g0 in range(0, tiles, gstep(n)):
        g1 = min(tiles, g0 + gstep(n)); R = 32 * (g1 - g0)
        sub = sub_vrk2(R, n, p["tr"], p["wmax"], p["c16"], p["S"][32 * g0:32 * g1], c, p["stream"][g0 * nb * tb:g1 * nb * tb].tobytes())
        _, Yq = K.unpack_blob(sub, T); yield g0, g1, Yq

def delta_of(path):
    """= r51_delta.py on one blob (W = ifw(Yq)/c, |W| > fp16 wmax, original rows, row-major order), decoded in chunks of sorted rows."""
    p = TC.parse_vrk2(open(path, "rb").read()); _, ifw = A1.make_fw("cpu"); wm = float(np.float16(p["wmax"])); c = torch.as_tensor(p["c16"].astype(np.float32))
    R_, C_, D_ = [], [], []
    for g0, g1, Yq in sorted_chunks(p):
        W = ifw(Yq) / c; r, cc = torch.nonzero(W.abs() > wm, as_tuple=True); w = W[r, cc]
        R_.append(p["prm"][32 * g0 + r.numpy()]); C_.append(cc.numpy()); D_.append((w.clamp(-wm, wm) - w).numpy().astype(np.float32))
    R_, C_, D_ = (np.concatenate(a) for a in (R_, C_, D_)); o = np.lexsort((C_, R_))
    return R_[o].astype(np.uint16), C_[o].astype(np.uint16), D_[o]

def cmd_delta(od):
    os.makedirs(od, exist_ok=True); z = np.load(B3NPZ); t0 = time.time()
    for nm in ("blk.4.ffn_gate.weight", "blk.5.ffn_gate.weight"):                          # the chunked recompute must reproduce B3's npz bit for bit
        def run(nm=nm):
            r, c, d = delta_of(B3[nm])
            ok = np.array_equal(r, z[nm + "/rows"]) and np.array_equal(c, z[nm + "/cols"]) and np.array_equal(d.view(np.uint32), z[nm + "/d32"].view(np.uint32))
            return dict(name=nm, cnt=int(len(d)), cnt_b3npz=int(len(z[nm + "/rows"])), bit_equal=bool(ok))
        v = cache("dval_" + nm, run); print(f"DELTA-VALIDATE B3 {nm}: chunked recompute {v['cnt']} vs B3 npz {v['cnt_b3npz']} entries, bit-equal {v['bit_equal']} t {time.time() - t0:.0f}s", flush=True)
        assert v["bit_equal"]
    same = {nm for nm in NAMES if os.path.realpath(B3[nm]) == os.path.realpath(B3P[nm])}; out = {}; cnt = {}
    for nm in NAMES:
        if nm in same:
            if nm + "/rows" in z.files:
                for k in ("rows", "cols", "d32", "d16"): out[nm + "/" + k] = z[nm + "/" + k]
            cnt[nm] = int(len(z[nm + "/rows"])) if nm + "/rows" in z.files else 0
        else:
            wait_commit(); r, c, d = delta_of(B3P[nm]); cnt[nm] = int(len(d))
            if len(d): out[nm + "/rows"], out[nm + "/cols"], out[nm + "/d32"], out[nm + "/d16"] = r, c, d, d.astype(np.float16)
            print(f"DELTA recomputed {nm} (f1): {len(d)} entries (B3 f4 version {len(z[nm + '/rows']) if nm + '/rows' in z.files else 0}) t {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(od, "vr_delta_b3p.npz"), **out); tot = sum(cnt.values()); b3t = sum(len(z[f]) for f in z.files if f.endswith("/rows"))
    for p in (os.path.join(od, "delta_counts.json"), os.path.join(OUT, "dcount_b3p.json")): json.dump(dict(total=tot, per_tensor=cnt), open(p, "w"), indent=0)
    print(f"DELTA b3p tensors_with_delta {sum(v > 0 for v in cnt.values())}/252 entries {tot} (B3 {b3t}, {tot - b3t:+d}); bytes_f32 {8 * tot}")

# ---------------------------------------------------------------- native plan / writer (r51_native structure; layout 3 drops vr_prm, smaller weight)
def plan_out(mp, layout, counts):
    r = GGUFReader(RN.MIX); hs = {nm: head(p) for nm, p in mp.items()}; out = []
    for t in r.tensors:
        if t.name in mp:
            h = hs[t.name]; base = t.name[:-len(".weight")]; kb = h["kb2"] if layout == 2 else h["kb3"]
            out.append((t.name, "vr", (kb,), np.int8, kb, None, t.name))
            if layout == 2: out.append((base + ".vr_prm", "prm", (h["rows"],), np.int16, 2 * h["rows"], None, t.name))
            out.append((base + ".vr_c", "c", (h["n"],), np.float16, 2 * h["n"], None, t.name))
            cnt = counts.get(t.name, 0)
            if cnt:
                out.append((base + ".vr_didx", "didx", (2, cnt), np.int16, 4 * cnt, None, t.name)); out.append((base + ".vr_dval", "dval", (cnt,), np.float32, 4 * cnt, None, t.name))
        else: out.append((t.name, "raw", tuple(t.data.shape), t.data.dtype, int(t.data.nbytes), t.tensor_type, t.name))
    out.append(("vr.had_signs", "signs", (256,), np.int8, 256, None, None)); return r, out

def writer(r, out, stem, layout):
    w = RN.make_writer(r, out, stem, RN.SPLIT_DATA)
    if layout == 3: w.add_key_value("vr.layout", 3, GGUFValueType.UINT32)            # VR3: natural rows, no vr_prm, no M (llama-vr3.h vrn3_expand)
    return w

def plan_total(mp, layout, counts, stem):
    r, out = plan_out(mp, layout, counts); w = writer(r, out, stem, layout); shard_t = [list(t.values()) for t in w.tensors]
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file(); hdr = [f.tell() for f in w.fout]; paths = [f.name for f in w.fout]; w.close()
    for p in paths: os.remove(p)
    sizes = [al(h) + sum(al(ti.nbytes) for ti in ts) for h, ts in zip(hdr, shard_t)]
    by = {}; [by.__setitem__(k, by.get(k, 0) + al(nb)) for _, k, _, _, nb, _, _ in out]
    return dict(total=sum(sizes), shards=sizes, hdr=hdr, by=by, ntens=len(out))

def counts_b3():
    z = np.load(B3NPZ); return {f[:-5]: int(len(z[f])) for f in z.files if f.endswith("/rows")}

def counts_b3p():
    pc = os.path.join(OUT, "dcount_b3p.json"); assert os.path.exists(pc), "run delta first"; return json.load(open(pc))["per_tensor"]

def cmd_plan():
    st = os.path.join(OUT, "plan_tmp"); res = {}
    for tag, mp, lay, cn in (("B3_v2", B3, 2, counts_b3()), ("B3_v3", B3, 3, counts_b3()), ("B3p_v3", B3P, 3, counts_b3p()), ("B3p_v2", B3P, 2, counts_b3p())):
        res[tag] = cache("plan_" + tag, lambda mp=mp, lay=lay, cn=cn: plan_total(mp, lay, cn, st))
        x = res[tag]; print(f"PLAN {tag}: TOTAL {x['total']} B (vs B3 {x['total'] - B3_BYTES:+d}; vs IQ2_XXS {x['total'] - RN.BUDGET:+d}); shards {x['shards']} "
                            f"headers {x['hdr']}; tensors {x['ntens']}; by kind {x['by']}", flush=True)
    assert res["B3_v2"]["total"] == B3_BYTES, "planner does not reproduce B3"
    print(f"PLAN layout saving on B3 (v2 -> VR3): {B3_BYTES - res['B3_v3']['total']} B; B3+ VR3 vs B3 {res['B3p_v3']['total'] - B3_BYTES:+d} B, vs IQ2_XXS "
          f"{res['B3p_v3']['total'] - RN.BUDGET:+d} B ({'size-matched' if res['B3p_v3']['total'] <= RN.BUDGET else 'OVER'}); B3+ in the v2 layout would be "
          f"{res['B3p_v2']['total'] - RN.BUDGET:+d} B vs IQ2_XXS ({'fits' if res['B3p_v2']['total'] <= RN.BUDGET else 'OVER'})")

# ---------------------------------------------------------------- VR3 bytes and the loader expansion (Python mirror of llama-vr3.h)
def natural_parts(blob):
    """the tensor with rows in natural order (identity perm): stream (tiles, nb, 32, bpb) and S (rows, nb). Rows are coded independently."""
    p = TC.parse_vrk2(blob); rows, n = p["rows"], p["n"]; nb = n // 256; tiles = rows // 32; c = int(p["meta"].flat[0] >> 29); bpb = 8 * NW[c]
    assert ((p["meta"] >> 29) == c).all() and ((p["meta"].ravel() & 0x1FFFFFFF).astype(np.int64) == np.arange(tiles * nb) * 64 * NW[c]).all(), "not constant-class"
    st = p["stream"].reshape(tiles, nb, 32, bpb).transpose(0, 2, 1, 3).reshape(rows, nb, bpb)          # VRK2 tile-block-major -> (sorted row, block, bytes)
    nat = np.empty_like(st); nat[p["prm"]] = st; st2 = np.ascontiguousarray(nat.reshape(tiles, 32, nb, bpb).transpose(0, 2, 1, 3))
    Sn = np.empty_like(p["S"]); Sn[p["prm"]] = p["S"]
    return p, st2, Sn, dict(rows=rows, n=n, c=c, c16=p["c16"])

def natural_vrk2(blob):
    p, st2, Sn, sd = natural_parts(blob); return sub_vrk2(sd["rows"], sd["n"], p["tr"], p["wmax"], p["c16"], Sn, sd["c"], st2.tobytes()), sd

def vr3_from_v2(kb2, rows, n, c):
    nt = rows // 32 * (n // 256); offD = (2064 + 68 * nt + 15) & ~15; h = bytearray(kb2[:2064])
    assert struct.unpack_from("<I", h, 2060)[0] == 0; struct.pack_into("<I", h, 2060, VR3_TAG | c)
    assert (np.frombuffer(kb2, "<u4", nt, 2064) == ((np.arange(nt, dtype=np.uint64) * np.uint64(64 * NW[c])) | np.uint64(c << 29)).astype(np.uint32)).all()
    return bytes(h) + kb2[2064 + 4 * nt:2064 + 68 * nt] + kb2[offD:]

def vrn3_rows(fbytes, K, w515):
    """= llama-vr3.h vrn3_rows: rows from the file size, K and the class tag, or -1."""
    if (w515 & 0xFFFFFFF8) != VR3_TAG or K % 256 or fbytes < 2064: return -1
    c = w515 & 7; per = 64 + 256 * NW[c]
    if c == 0 or (fbytes - 2064) % per: return -1
    nt = (fbytes - 2064) // per; nb = K // 256
    if nt % nb: return -1
    rows = 32 * (nt // nb); return rows if 0 < rows < 32768 else -1

def expand_v2(kb3, K):
    """= llama-vr3.h vrn3_expand: VR3 file bytes -> v2 in-memory kernel bytes (rows, bytes)."""
    w515 = struct.unpack_from("<I", kb3, 2060)[0]; rows = vrn3_rows(len(kb3), K, w515); assert rows > 0; c = w515 & 7
    nt = rows // 32 * (K // 256); offD = (2064 + 68 * nt + 15) & ~15; out = bytearray(TC.kbytes_size(rows, K, nt * 64 * NW[c]))
    out[:2060] = kb3[:2060]                                                              # word 515 stays 0 (the v2 file value; runtime loader mode)
    out[2064:2064 + 4 * nt] = ((np.arange(nt, dtype=np.uint64) * np.uint64(64 * NW[c])) | np.uint64(c << 29)).astype("<u4").tobytes()
    out[2064 + 4 * nt:2064 + 68 * nt] = kb3[2064:2064 + 64 * nt]; out[offD:] = kb3[2064 + 64 * nt:]
    return rows, bytes(out)

FULL_MAX = 12 << 20                                                                   # tensors up to 12M weights: also one-shot transcode

def vr3_bytes(blob):
    """VR3 bytes from the shipped transcoder run on whole-tile chunks of the natural-order tensor (bounded memory); kb2n = expand(VR3) =
    the v2 bytes the loader uploads. Tensors <= FULL_MAX weights: expand(VR3) must equal the one-shot transcode of the natural blob."""
    p, st2, Sn, sd = natural_parts(blob); rows, n, c = sd["rows"], sd["n"], sd["c"]; nb = n // 256; tiles = rows // 32; hdr = None; Sp, Dp = [], []
    for g0 in range(0, tiles, gstep(n)):
        g1 = min(tiles, g0 + gstep(n)); R = 32 * (g1 - g0); ntl = (g1 - g0) * nb
        kb, side = TC.transcode(sub_vrk2(R, n, p["tr"], p["wmax"], p["c16"], Sn[32 * g0:32 * g1], c, st2[g0:g1].tobytes()), TAB)
        assert hdr is None or kb[:2064] == hdr; hdr = kb[:2064]
        Sp.append(np.frombuffer(kb, np.float16, nb * R, 2064 + 4 * ntl).reshape(nb, R)); Dp.append(kb[(2064 + 68 * ntl + 15) & ~15:])
    h = bytearray(hdr); assert struct.unpack_from("<I", h, 2060)[0] == 0; struct.pack_into("<I", h, 2060, VR3_TAG | c)
    kb3 = bytes(h) + np.ascontiguousarray(np.concatenate(Sp, 1)).tobytes() + b"".join(Dp); del Sp, Dp
    xr, kb2n = expand_v2(kb3, n); ok = xr == rows and vr3_from_v2(kb2n, rows, n, c) == kb3
    if rows * n <= FULL_MAX:
        kf, sf = TC.transcode(natural_vrk2(blob)[0], TAB); ok = ok and kf == kb2n and (sf["prm"] == np.arange(rows)).all(); sd["full_cmp"] = True
    sd.update(kb2n=kb2n, exp_ok=bool(ok)); return kb3, sd

def port_chunks(kb, rows, n):
    """shipped r51_transcode.cpu_vr_port on whole-tile row chunks sliced out of v2 kernel bytes kb (constant class, arithmetic M rebased)."""
    nb = n // 256; ng = rows // 32; nt = ng * nb; M = np.frombuffer(kb, "<u4", nt, 2064); c = int(M[0] >> 29); tw = 64 * NW[c]
    S = np.frombuffer(kb, np.float16, nb * rows, 2064 + 4 * nt).reshape(nb, rows); D = np.frombuffer(kb, "<u4", offset=(2064 + 68 * nt + 15) & ~15)
    for g0 in range(0, ng, gstep(n)):
        g1 = min(ng, g0 + gstep(n)); R = 32 * (g1 - g0); ntl = (g1 - g0) * nb; offD = (2064 + 68 * ntl + 15) & ~15
        Ms = ((M[g0 * nb:g1 * nb] & 0x1FFFFFFF) - g0 * nb * tw) | (M[g0 * nb:g1 * nb] & 0xE0000000)
        sub = (bytes(kb[:2064]) + Ms.astype("<u4").tobytes() + np.ascontiguousarray(S[:, 32 * g0:32 * g1]).tobytes() + bytes(offD - 2064 - 68 * ntl)
               + D[g0 * nb * tw:g1 * nb * tw].tobytes())
        yield g0, g1, TC.cpu_vr_port(sub, R, n)

def cmd_check(names):
    names = NAMES if names == ["all"] else names; t00 = time.time(); nb_ = 0
    for nm in names:
        def run(nm=nm):
            t0 = time.time(); blob = open(B3P[nm], "rb").read(); p = TC.parse_vrk2(blob); rows, n = p["rows"], p["n"]
            kb3, sd = vr3_bytes(blob); kb2n = sd["kb2n"]; ref = np.empty((rows, n), np.float32)
            for g0, g1, Yq in sorted_chunks(p): ref[p["prm"][32 * g0:32 * g1]] = Yq.numpy()           # shipped r48k decode of the ORIGINAL VRK2 bytes
            bad = 0                                                                                    # shipped port on the bytes the loader uploads
            for g0, g1, got in port_chunks(kb2n, rows, n): bad += int((got.view(np.uint32) != ref[32 * g0:32 * g1].view(np.uint32)).sum())
            del ref
            return dict(name=nm, cls=sd["c"], rows=rows, n=n, kb2=TC.kbytes_size(rows, n, len(p["stream"]) // 4), kb3=len(kb3), prm_i16=2 * rows,
                        sha_kb3=hashlib.sha256(kb3).hexdigest()[:16], expand_eq_transcode=sd["exp_ok"], one_shot_cmp=bool(sd.get("full_cmp", False)), bad_weights=bad, n_weights=rows * n,
                        ok=bool(sd["exp_ok"] and bad == 0), t=round(time.time() - t0, 1))
        x = cache("check_" + nm, run); nb_ += not x["ok"]
        print(f"CHECK {nm} class {x['cls']} {x['rows']}x{x['n']}: v2 {x['kb2']} + vr_prm {x['prm_i16']} -> VR3 {x['kb3']} (-{x['kb2'] + x['prm_i16'] - x['kb3']} B); "
              f"expand(VR3) == transcode(natural) {x['expand_eq_transcode']}{' (one-shot)' if x.get('one_shot_cmp') else ' (chunks)'}; cpu_vr_port(expanded) vs r48k(original): {x['bad_weights']}/{x['n_weights']} differ; "
              f"{'PASS' if x['ok'] else 'FAIL'} t {x['t']}s", flush=True)
    print(f"CHECK SUMMARY {len(names) - nb_}/{len(names)} PASS t {time.time() - t00:.0f}s")

def cmd_negctl(nm):
    """negative controls for cmd_check: one corrupted D byte, S value, M word in the expanded bytes must change decoded weights."""
    blob = open(B3P[nm], "rb").read(); p = TC.parse_vrk2(blob); rows, n = p["rows"], p["n"]; nt = rows // 32 * (n // 256)
    kb3, sd = vr3_bytes(blob); kb2n = sd["kb2n"]; ref = np.empty((rows, n), np.float32)
    for g0, g1, Yq in sorted_chunks(p): ref[p["prm"][32 * g0:32 * g1]] = Yq.numpy()
    def ndiff(kb):
        return sum(int((got.view(np.uint32) != ref[32 * g0:32 * g1].view(np.uint32)).sum()) for g0, g1, got in port_chunks(kb, rows, n))
    offD = (2064 + 68 * nt + 15) & ~15; mid = offD + (len(kb2n) - offD) // 2
    flip = lambda kb, o, x: kb[:o] + bytes([kb[o] ^ x]) + kb[o + 1:]
    print(f"NEGCTL {nm}: ref nonzero {np.count_nonzero(ref)}/{ref.size}, std {ref.std():.4g}; clean {ndiff(kb2n)} differ; D byte flipped {ndiff(flip(kb2n, mid, 0x10))}; "
          f"S value (row 5, block 0) flipped {ndiff(flip(kb2n, 2064 + 4 * nt + 2 * 5 + 1, 0x04))}; M word 1 class bits {ndiff(flip(kb2n, 2064 + 4 + 3, 0x20))} differ")

def cmd_dump(od, names):
    os.makedirs(od, exist_ok=True)
    for nm in names:
        kb3, sd = vr3_bytes(open(B3P[nm], "rb").read()); open(os.path.join(od, nm + ".vr3"), "wb").write(kb3); open(os.path.join(od, nm + ".v2"), "wb").write(sd["kb2n"])
        print(f"DUMP {nm} K {sd['n']} rows {sd['rows']} class {sd['c']} vr3 {len(kb3)} v2 {len(sd['kb2n'])} python-expand-ok {sd['exp_ok']} one-shot-cmp {sd.get('full_cmp', False)}")

# ---------------------------------------------------------------- build / verify (SET=b3p default; SET=b3 = B3's own blobs in VR3 for the GPU layout check)
def build_set():
    s = E("SET", "b3p"); assert s in ("b3p", "b3")
    return (B3P, np.load(E("B3P_NPZ"))) if s == "b3p" else (B3, np.load(B3NPZ))

def tensor_data3(kind, src, rb, z, mp, cc):
    if kind in ("raw", "signs"): return RN.tensor_data(kind, src, rb, z, None, {})
    if kind == "didx": return np.stack([z[src + "/rows"], z[src + "/cols"]]).astype(np.uint16).view(np.int16)       # = r51_native.tensor_data
    if kind == "dval": return z[src + "/d32"].astype(np.float32)
    if cc.get("name") != src:
        kb3, sd = vr3_bytes(open(mp[src], "rb").read()); assert sd["exp_ok"], src; cc.clear(); cc.update(name=src, kb3=kb3, c16=sd["c16"])
    return np.frombuffer(cc["kb3"], np.int8) if kind == "vr" else np.asarray(cc["c16"], np.float16)

def cmd_build(stem):
    wait_commit(); mp, z = build_set(); cnt = {f[:-5]: int(len(z[f])) for f in z.files if f.endswith("/rows")}
    r, out = plan_out(mp, 3, cnt); rb = {t.name: t for t in r.tensors}
    w = writer(r, out, stem, 3); w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file(); paths = [f.name for f in w.fout]; cc = {}; t0 = time.time()
    for i, (name, kind, shp, dt, nb, raw, src) in enumerate(out):
        if kind == "vr": wait_commit()
        a = tensor_data3(kind, src, rb, z, mp, cc); assert a.nbytes == nb, (name, a.nbytes, nb); w.write_tensor_data(a)
        if kind == "vr" and i % 40 == 0: print(f"  wrote {i}/{len(out)} {name} t {time.time() - t0:.0f}s", flush=True)
    w.close(); sizes = [os.path.getsize(p) for p in paths]
    for p, s in zip(paths, sizes):
        h = hashlib.sha256(); f = open(p, "rb")
        for blk in iter(lambda: f.read(1 << 24), b""): h.update(blk)
        print(f"SHARD {p} {s} B sha256 {h.hexdigest()} {'OK' if s <= RN.SHARD_MAX else 'OVER 1e9'}", flush=True)
    print(f"NATIVE3 TOTAL {sum(sizes)} B vs B3 {sum(sizes) - B3_BYTES:+d} vs IQ2_XXS {sum(sizes) - RN.BUDGET:+d} ({'OK' if sum(sizes) <= RN.BUDGET else 'OVER'}); "
          f"tensors {len(out)} t {time.time() - t0:.0f}s")

def cmd_verify(stem):
    import glob
    wait_commit(); paths = sorted(glob.glob(stem + "-0000*-of-*.gguf")); rd = {}
    for p in paths: rd.update({t.name: t for t in GGUFReader(p).tensors})
    mp, z = build_set(); cnt = {f[:-5]: int(len(z[f])) for f in z.files if f.endswith("/rows")}
    r, out = plan_out(mp, 3, cnt); rb = {t.name: t for t in r.tensors}; cc = {}; bad = []
    for nm, kind, shp, dt, nb, raw, src in out:
        if nm not in rd or bytes(np.asarray(rd[nm].data).view(np.uint8)) != bytes(np.ascontiguousarray(tensor_data3(kind, src, rb, z, mp, cc)).view(np.uint8)): bad.append(nm)
    noprm = sum(1 for nm in rd if nm.endswith(".vr_prm")); lay = GGUFReader(paths[0]).fields["vr.layout"].contents()
    print(f"VERIFY3 shards {len(paths)} tensors {len(rd)} expected {len(out)}; mismatches {len(bad)} {bad[:5]}; vr_prm tensors {noprm}; vr.layout {lay} "
          f"{'PASS' if not bad and len(rd) == len(out) and noprm == 0 and lay == 3 else 'FAIL'}")

if __name__ == "__main__":
    a = sys.argv[1]
    {"scan": lambda: cmd_scan(), "delta": lambda: cmd_delta(sys.argv[2]), "plan": lambda: cmd_plan(), "check": lambda: cmd_check(sys.argv[2:]),
     "dump": lambda: cmd_dump(sys.argv[2], sys.argv[3:]), "negctl": lambda: [cmd_negctl(x) for x in sys.argv[2:]], "build": lambda: cmd_build(sys.argv[2]), "verify": lambda: cmd_verify(sys.argv[2])}[a]()
