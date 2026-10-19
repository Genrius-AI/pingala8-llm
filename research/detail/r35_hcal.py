# R35: Hessian calibration text, disjoint from models/calib_c4.txt (shard-0 docs 0..~?) and WikiText-2 test.
# Takes C4 train shard 00000 docs from index 200000 on until 6.0M chars; checks word 10-gram overlap vs both.
import gzip, json, glob
f = glob.glob('hf_cache/hub/datasets--allenai--c4/snapshots/*/en/c4-train.00000-of-01024.json.gz')[0]
out, n = [], 0
for i, line in enumerate(gzip.open(f, 'rt')):
    if i < 200000: continue
    t = json.loads(line)['text']; out.append(t); n += len(t)
    if n > 6_000_000: break
txt = '\n\n'.join(out); open('models/r35_hcal.txt', 'w').write(txt)
def g10(s):
    w = s.split(); return {' '.join(w[k:k + 10]) for k in range(len(w) - 9)}
G = g10(txt); C = g10(open('models/calib_c4.txt').read()); Wk = g10(open('datasets/wikitext2_test.raw').read())
print(f"R35_HCAL docs {len(out)} from idx 200000, chars {len(txt)}, 10-grams {len(G)}; overlap with calib_c4 {len(G & C)}, with wikitext2_test {len(G & Wk)}")
