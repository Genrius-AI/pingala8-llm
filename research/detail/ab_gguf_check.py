# AB: per-tensor check of the storage formula against the GGUF tensor tables of the native Qwen3-4B file (reads tensor headers only, CPU).
# usage: cd ~/repo && python3 research/detail/ab_gguf_check.py > logs/ab_gguf_check.log
import sys, re, ast, math, collections
sys.path.insert(0, "~/repo/llama.cpp/gguf-py")
from gguf import GGUFReader
SHAPE = {'attn_q': (4096, 2560), 'attn_k': (1024, 2560), 'attn_v': (1024, 2560), 'attn_output': (2560, 4096), 'ffn_gate': (9728, 2560), 'ffn_up': (9728, 2560), 'ffn_down': (2560, 9728)}
BPB = [0, 32, 48, 64, 80, 96, 128, 192]
hist = {}
for l in open('logs/r48k3.tensors'):
    m = re.match(r'^vrc\s+blk\.(\d+)\.(\S+)\.weight\s+(\S+)\s+blob .*hist (\[[^\]]*\])', l)
    if m: hist[(int(m.group(1)), m.group(2))] = ast.literal_eval(m.group(4))
sizes = collections.defaultdict(dict)
for p in ["work/r51t/native/4B-pingala8-vrnative-00001-of-00002.gguf", "work/r51t/native/4B-pingala8-vrnative-00002-of-00002.gguf"]:
    r = GGUFReader(p)
    for t in r.tensors:
        mm = re.match(r'blk\.(\d+)\.(\w+?)\.(weight|vr_prm|vr_c|vr_didx|vr_dval)$', t.name)
        if mm and mm.group(2) in SHAPE: sizes[(int(mm.group(1)), mm.group(2))][mm.group(3)] = int(t.n_bytes)
bad = 0; tot = collections.Counter(); nd = 0
for (li, nm), h in hist.items():
    rows, n = SHAPE[nm]; s = sizes[(li, nm)]
    P = sum(a * b for a, b in zip(h, BPB))
    exp_w = P + rows * (n // 256) * 2 + 4 * (rows // 32) * (n // 256)
    got_w = s['weight']
    d = got_w - exp_w
    tot['w_resid'] += d
    if s.get('vr_prm') != 2 * rows or s.get('vr_c') != 2 * n: bad += 1
    tot['prm'] += s.get('vr_prm', 0); tot['c'] += s.get('vr_c', 0); tot['didx'] += s.get('vr_didx', 0); tot['dval'] += s.get('vr_dval', 0); tot['w'] += got_w
    tot.setdefault('resid_set', set()).add(d) if False else None
    tot[('r', d)] += 1
print("tensors", len(hist), "mismatched prm/c sizes", bad)
print("weight-tensor residual (got - [payload + scales + tilewords]) distribution:", {k[1]: v for k, v in tot.items() if isinstance(k, tuple)})
print("sum weight tensors", tot['w'], "prm", tot['prm'], "c", tot['c'], "didx", tot['didx'], "dval", tot['dval'], "total", tot['w'] + tot['prm'] + tot['c'] + tot['didx'] + tot['dval'])
