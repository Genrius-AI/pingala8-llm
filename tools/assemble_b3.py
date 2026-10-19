#!/usr/bin/env python3
"""Assemble the B3 blob set from the two flat builds (research/work/r59/prereg_b3.txt and its amendment).

The original B3 directory was assembled by hand as a symlink farm; this script was written for the archive and reproduces it:
  blobs_b3/  = every blob of work/r52a/f1/blobs_roff (the one-class-per-tensor build), except
               blk.4.ffn_gate.weight and blk.5.ffn_gate.weight, taken from work/r52a/f4/blobs_flat (class 2)
  stats_b3.json = the f1 'roff' stats entries with those two entries replaced by the f4 'flat' ones, all relabelled v = 'b3'.
Checked against the original work/r59b3/stats_b3.json: identical as parsed JSON (252 entries).
usage: python3 tools/assemble_b3.py [F1_DIR=work/r52a/f1] [F4_DIR=work/r52a/f4] [OUT=work/r59b3]
"""
import json, os, sys
f1, f4, out = (sys.argv[1:] + ["work/r52a/f1", "work/r52a/f4", "work/r59b3"][len(sys.argv) - 1:])[:3]
FROM_F4 = ("blk.4.ffn_gate.weight", "blk.5.ffn_gate.weight")
bd = os.path.join(out, "blobs_b3"); os.makedirs(bd, exist_ok=True)
for fn in sorted(os.listdir(os.path.join(f1, "blobs_roff"))):
    src = os.path.join(f4, "blobs_flat", fn) if fn[:-len(".blob")] in FROM_F4 else os.path.join(f1, "blobs_roff", fn)
    dst = os.path.join(bd, fn)
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(os.path.abspath(src), dst)
a = [x for x in json.load(open(os.path.join(f1, "stats.json"))) if x["v"] == "roff"]
b = {x["name"]: x for x in json.load(open(os.path.join(f4, "stats.json"))) if x["v"] == "flat"}
st = [dict(b[x["name"]] if x["name"] in FROM_F4 else x, v="b3") for x in a]
json.dump(st, open(os.path.join(out, "stats_b3.json"), "w"))
print("B3 blobs", len(os.listdir(bd)), "stats", len(st), "from f4:", FROM_F4)
