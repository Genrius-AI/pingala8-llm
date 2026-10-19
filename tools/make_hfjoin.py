#!/usr/bin/env python3
r"""Build datasets/wikitext2_test_hfjoin.raw, the text used for every ctx-4096 perplexity number.

Recipe (verified against the file used in the paper): the WikiText-2 raw test split from the
Hugging Face hub (Salesforce/wikitext, config wikitext-2-raw-v1), rows joined with "\n\n".
Expected: 1,296,370 bytes, sha256 696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83,
299,078 Qwen3 tokens, 73 windows of 4096 at stride 4095.
"""
import hashlib, os, sys
from datasets import load_dataset

out = sys.argv[1] if len(sys.argv) > 1 else "datasets/wikitext2_test_hfjoin.raw"
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
data = "\n\n".join(ds["text"]).encode("utf-8")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "wb") as f:
    f.write(data)
print(out, len(data), hashlib.sha256(data).hexdigest())
