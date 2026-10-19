# Build mmlu-test.bin (llama.cpp multiple-choice format) from cais/mmlu test split, all 57 subjects.
# Same layout as datasets/zs/mmlu-validation.bin: question text raw, 4 answer strings raw, one-hot labels, empty mc2.
import glob, struct, pandas as pd
S = sorted(glob.glob('hf_cache/hub/datasets--cais--mmlu/snapshots/*/*/test-00000-of-00001.parquet'))
S = [s for s in S if '/all/' not in s and '/auxiliary_train/' not in s]
rows = []
for f in S:
    df = pd.read_parquet(f)
    for q, ch, a in zip(df.question, df.choices, df.answer): rows.append((q, list(ch), int(a)))
def bs(s): b = s.encode(); return struct.pack('<I', len(b)) + b
blobs = []
for q, ch, a in rows:
    lab = [1 if i == a else 0 for i in range(len(ch))]
    blobs.append(bs(q) + struct.pack('<I', len(ch)) + b''.join(bs(c) for c in ch) + struct.pack('<%di' % len(ch), *lab) + struct.pack('<I', 0))
n = len(blobs); pos = []; o = 4 + 4 * n
for b in blobs: pos.append(o); o += len(b)
open('datasets/zs/mmlu-test-full.bin', 'wb').write(struct.pack('<I', n) + struct.pack('<%dI' % n, *pos) + b''.join(blobs))
print('subjects', len(S), 'tasks', n, 'bytes', o)
