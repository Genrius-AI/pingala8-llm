#!/usr/bin/env python3
"""r59_kldc4 CPU input preparation (helper of r59_kldc4.sh; run it as `CPU=1 bin/job python -I ...`). Plain ASCII.
usage:
  r59_kldc4_prep.py split SRC OUTPREFIX MODEL_GGUF [--ctx 512] [--nparts 6] [--vocab 151936] [--cap 1000000000] [--tok BIN]
      Splits the KLD text SRC into NPARTS disjoint parts OUTPREFIX1.raw ... so that each part holds >= CPP*CTX tokens (+64 slack) and the base-logit
      file of CPP chunks stays under CAP bytes.  Stored bytes per token = 2*(2*((V+1)//2)+4) (llama-perplexity process_logits: nv uint16 per token);
      tokens stored per chunk = CTX-1-CTX//2.  Part 1 starts at byte 0 (its first CPP chunks are the old 12 x 512 run), the others start at evenly spaced
      line boundaries.  Prints "PARTS n= cpp= chunks= bytes_per_chunk= est_file_bytes=" and one "PART i ..." line each.
  r59_kldc4_prep.py c4 OUT MODEL_GGUF [--ntok 300500] [--lo 299500] [--hi 303000] [--calib models/calib_c4.txt ...]
      Downloads allenai/c4 en/c4-validation.00000-of-00008.json.gz (HF_HOME from env), joins the first documents with blank lines until the
      llama-tokenize count is in [lo, hi] (73 chunks of the ctx-4096 stride protocol need 298937..303031 tokens), writes OUT, checks overlap with the calibration files.
Token counts always come from llama-tokenize (--no-parse-special, i.e. what llama-perplexity does), never from an approximation."""
import os, sys, re, json, gzip, hashlib, subprocess, argparse, shutil

TOK_DEFAULT = os.path.expanduser("~/repo/llama.cpp/build/bin/llama-tokenize")


def tokcount(tok, model, data: bytes) -> int:
    r = subprocess.run([tok, "-m", model, "--stdin", "--no-parse-special", "--show-count", "--log-disable", "--ids"], input=data,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    m = re.search(rb"Total number of tokens: (\d+)", r.stdout)
    if r.returncode != 0 or not m:
        raise SystemExit(f"llama-tokenize failed rc={r.returncode}: {r.stderr[-300:]!r}")
    return int(m.group(1))


def bytes_per_chunk(ctx, vocab):
    nv = 2 * ((vocab + 1) // 2) + 4
    return (ctx - 1 - ctx // 2) * nv * 2


def cut_to_line(b: bytes, pos: int, forward=True) -> int:
    """nearest line boundary (index just after a newline) at or after pos (forward) / at or before pos."""
    if forward:
        i = b.find(b"\n", pos)
        return len(b) if i < 0 else i + 1
    i = b.rfind(b"\n", 0, pos)
    return 0 if i < 0 else i + 1


def cmd_split(a):
    src = open(a.src, "rb").read()
    per = bytes_per_chunk(a.ctx, a.vocab)
    hdr = 64 + 4 * a.ctx * 64                      # generous bound: magic/ints + tokens of up to 64 chunks (real header = 16 + 4*ctx*nchunk)
    cpp = min(a.cpp, (a.cap - hdr) // per)
    assert cpp >= 1
    need = cpp * a.ctx + 64
    total = tokcount(a.tok, a.model, src)
    ratio = len(src) / total
    print(f"SRC bytes {len(src)} tokens {total} bytes/token {ratio:.4f}; base-logit bytes/chunk {per}; chunks/part {cpp} -> est file {cpp * per + 16 + 4 * a.ctx * cpp} B (cap {a.cap})")
    span_est = int(need * ratio * 1.03)
    last_start = cut_to_line(src, max(0, len(src) - int(span_est * 1.5)), forward=False)
    starts = [0] + [cut_to_line(src, int(i * last_start / (a.nparts - 1))) for i in range(1, a.nparts)] if a.nparts > 1 else [0]
    out, prev_end = [], 0
    for i, s in enumerate(starts):
        if s < prev_end:
            raise SystemExit(f"part {i + 1} overlaps the previous part (start {s} < end {prev_end}); source too short for {a.nparts} parts")
        span = span_est
        for it in range(12):
            e = cut_to_line(src, s + span)
            n = tokcount(a.tok, a.model, src[s:e])
            if need <= n <= need + 700 or e >= len(src):
                break
            span = max(200, int(span * (need + 300) / max(n, 1)))
        if n < need:
            raise SystemExit(f"part {i + 1}: only {n} tokens < {need}")
        fn = f"{a.outprefix}{i + 1}.raw"
        open(fn, "wb").write(src[s:e])
        sha = hashlib.sha256(src[s:e]).hexdigest()[:16]
        out.append((i + 1, fn, s, e, n, sha)); prev_end = e
    print(f"PARTS n= {len(out)} cpp= {cpp} chunks= {len(out) * cpp} bytes_per_chunk= {per} est_file_bytes= {cpp * per + 16 + 4 * a.ctx * cpp}")
    for p in out:
        print("PART %d %s start %d end %d tokens %d sha256_16 %s" % p)


def fetch_c4(repo, fn):
    os.environ.pop("HF_HUB_OFFLINE", None); os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo, repo_type="dataset", filename=fn)
    except Exception as e:                                               # fall back to a plain HTTPS GET into the same cache dir
        print("hf_hub_download failed:", repr(e)[:200], "- trying plain https", flush=True)
        import urllib.request
        d = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/repo/hf_cache")), "r59_c4"); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, os.path.basename(fn))
        if not (os.path.exists(p) and os.path.getsize(p) > 30_000_000):
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/main/{fn}", p)
        return p


def cmd_c4(a):
    p = fetch_c4("allenai/c4", "en/c4-validation.00000-of-00008.json.gz")
    sz = os.path.getsize(p)
    print(f"C4 shard {p} bytes {sz} sha256_16 {hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]}")
    if not (20_000_000 < sz < 80_000_000):
        raise SystemExit(f"unexpected shard size {sz}")
    docs, nbytes = [], 0
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for ln in fh:
            t = json.loads(ln)["text"]
            docs.append(t); nbytes += len(t.encode()) + 2
            if nbytes > 1_900_000:
                break
    join = lambda k: "\n\n".join(docs[:k]).encode()
    cnt = lambda b: tokcount(a.tok, a.model, b)
    if cnt(join(len(docs))) < a.lo:
        raise SystemExit("not enough documents read for the token target")
    lo_k, hi_k = 1, len(docs)                                            # smallest k with count(first k docs) >= lo (count is monotone in k)
    while lo_k < hi_k:
        mid = (lo_k + hi_k) // 2
        if cnt(join(mid)) >= a.lo: hi_k = mid
        else: lo_k = mid + 1
    k = lo_k; data = join(k); n = cnt(data)
    if n > a.hi:                                                         # last document overshoots the window: cut it at a space/newline
        head = join(k - 1) + b"\n\n"; tail = docs[k - 1].encode()
        lo_i, hi_i = 0, len(tail)
        while hi_i - lo_i > 1:
            mid = (lo_i + hi_i) // 2
            if cnt(head + tail[:mid]) > a.hi: hi_i = mid
            else: lo_i = mid
        j = max(tail.rfind(b"\n", 0, lo_i), tail.rfind(b" ", 0, lo_i))
        data = (head + tail[:j]).rstrip(); n = cnt(data)
        if not (a.lo <= n <= a.hi):
            raise SystemExit(f"could not land in [{a.lo},{a.hi}] tokens (n={n})")
    open(a.out, "wb").write(data)
    chunks = (n - 2) // 4095
    print(f"C4 docs {k} bytes {len(data)} tokens {n} strided chunks (ctx 4096, stride 4095) {chunks} sha256_16 {hashlib.sha256(data).hexdigest()[:16]}")
    if chunks != 73:
        raise SystemExit(f"chunk count {chunks} != 73")
    # overlap check against calibration text (first 100 chars of every used document; calibration data is C4 train, validation is a disjoint split)
    cal = b""
    for c in a.calib:
        if os.path.exists(c):
            cal += open(c, "rb").read()
    if cal:
        hit = sum(1 for d in docs[:k] if len(d) > 100 and d[:100].encode() in cal)
        print(f"C4 overlap check: {hit} of {k} documents have their first 100 chars inside {','.join(a.calib)} ({len(cal)} bytes)")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("split"); s.add_argument("src"); s.add_argument("outprefix"); s.add_argument("model")
s.add_argument("--ctx", type=int, default=512); s.add_argument("--nparts", type=int, default=6); s.add_argument("--cpp", type=int, default=12)
s.add_argument("--vocab", type=int, default=151936); s.add_argument("--cap", type=int, default=1_000_000_000); s.add_argument("--tok", default=TOK_DEFAULT)
c = sub.add_parser("c4"); c.add_argument("out"); c.add_argument("model"); c.add_argument("--ntok", type=int, default=300500)
c.add_argument("--lo", type=int, default=299500); c.add_argument("--hi", type=int, default=303000); c.add_argument("--tok", default=TOK_DEFAULT)
c.add_argument("--calib", nargs="*", default=["models/calib_c4.txt", "datasets/c4_heldout.raw"])
a = ap.parse_args()
{"split": cmd_split, "c4": cmd_c4}[a.cmd](a)
