#!/usr/bin/env python3
"""r59_kldc4 report (helper of r59_kldc4.sh). usage: r59_kldc4_report.py [--complete | --done]
--complete : guard. exit 0 only if every number the report quotes comes from a FINISHED log: split.log + c4_prep.log, arms.txt, and for >= MINPARTS parts (default 5 = 60 chunks of 512)
             the Q8_0 base log ("Final estimate") and, for every arm, a KLD log with all 12 per-chunk lines whose last cumulative KLD equals the printed "Mean KLD"; plus,
             for every C4 arm, a perplexity log with all 73 windows and a "Final estimate" line.  Else prints INCOMPLETE [...] and exits 1; nothing is written.
--done     : exit 0 iff the report file exists (marker line) and the guard passes.
default    : refuse (exit 1) unless the guard passes; then write research/agents/eval_r59_kldc4.md (<= 400 words), copy it to notes/eval_r59_kldc4.txt,
             append one row to research/MODEL_B_PROGRESS.md (once).
Per-chunk KLD / top-1 are recovered from the CUMULATIVE per-chunk lines of llama-perplexity --kl-divergence (equal tokens per chunk: ctx-1-ctx//2 = 255):
   x_k = k*m_k - (k-1)*m_{k-1}; top-1 counts are rounded to integers.  Standard errors and paired 95% CIs (1.96*sd/sqrt(n)) use CHUNKS as the unit.
Per-window C4 NLL: the same differencing on the cumulative [k]ppl values (as research/detail/r59_paired.py), n = windows in the log.
Env overrides (selftest only): R59_LOGS R59_OUT R59_WIN R59_PROG R59_MINPARTS."""
import os, re, sys, math, glob, shutil, statistics, datetime

R = os.path.expanduser("~/repo"); os.chdir(R)
L = os.environ.get("R59_LOGS", "logs/r59")
OUT = os.environ.get("R59_OUT", "research/agents/eval_r59_kldc4.md")
WIN = os.environ.get("R59_WIN", "eval_r59_kldc4.txt")
PROG = os.environ.get("R59_PROG", "research/MODEL_B_PROGRESS.md")
MINPARTS = int(os.environ.get("R59_MINPARTS", "5"))
MARK = "r59 kld+c4"
CPP, TOKC, NWIN = 12, 255, 73            # chunks per part, scored tokens per 512-chunk, C4 windows
ORDER = ["mB", "b2", "xxs", "kt"]        # KLD arms (b2 only if listed in arms.txt)
C4ORDER = ["q8", "mB", "b2", "xxs", "kt"]
LAB = {"mB": "Model B", "b2": "Model B2", "xxs": "IQ2_XXS", "kt": "IQ2_KT-eq", "q8": "Q8_0 ref"}
WT2REF = {"mB": "logs/r36_hfj_r51_native_PPC2.log", "b2": "logs/r36_hfj_r57_flat_native.log", "xxs": "logs/r36_hfj_iq2xxs.log",
          "kt": "logs/r56_kteq_ppl.log", "q8": "logs/r36_hfj_fp16.log"}      # q8 slot: fp16 wikitext reference
OLD_BASE_PPL, OLD_B_KLD = 16.8826, 0.639950


def rd(p):
    try:
        return open(p, errors="replace", encoding="utf-8").read()
    except OSError:
        return ""


def arms():
    out = {}
    for ln in rd(f"{L}/arms.txt").splitlines():
        t = ln.split("\t")
        if len(t) >= 3 and t[0] in LAB and t[2].isdigit():
            out[t[0]] = int(t[2])
    return out


def nparts():
    m = re.search(r"PARTS n= (\d+) cpp= (\d+)", rd(f"{L}/split.log"))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def kld_log(path):
    """-> dict(kld=[per chunk], top=[per chunk %], final=Mean KLD, ftop=Same top p) or None if not a finished 12-chunk KLD log."""
    t = rd(path); rows = []
    for ln in t.splitlines():
        s = ln.split()
        if len(s) == 18 and s[0].isdigit() and s[13] == "%" and s[17] == "%":
            try:
                rows.append((int(s[0]), float(s[7]), float(s[14])))
            except ValueError:
                pass
    mk = re.search(r"Mean +KLD: +([0-9.]+)", t); mt = re.search(r"Same top p: +([0-9.]+)", t)
    if not (mk and mt) or [r[0] for r in rows[-CPP:]] != list(range(1, CPP + 1)):
        return None
    rows = rows[-CPP:]; kld, top, pk, pc = [], [], 0.0, 0
    for k, m, p in rows:
        kld.append(k * m - (k - 1) * pk); pk = m
        c = round(p / 100.0 * TOKC * k); top.append((c - pc) / TOKC * 100.0); pc = c
    if abs(rows[-1][1] - float(mk.group(1))) > 1e-4:
        return None
    return dict(kld=kld, top=top, final=float(mk.group(1)), ftop=float(mt.group(1)))


def ppl_log(path):
    """-> dict(nll=[per window], ppl=final cumulative, n) for a finished 73-window perplexity_v2 log, else None."""
    t = rd(path)
    m = re.findall(r"\[(\d+)\]([0-9]+\.[0-9]+)", t)
    if not m or "Final estimate" not in t:
        return None
    idx = [i for i, (k, _) in enumerate(m) if k == "1"]
    seq = m[idx[-1]:]
    if [int(k) for k, _ in seq] != list(range(1, NWIN + 1)) or not re.search(rf"computing over {NWIN} chunks", t):
        return None
    lp = [math.log(float(v)) for _, v in seq]
    return dict(nll=[(j + 1) * lp[j] - j * lp[j - 1] if j else lp[0] for j in range(NWIN)], ppl=float(seq[-1][1]), n=NWIN)


def wt2(key):
    m = re.findall(r"\[73\]([0-9]+\.[0-9]+)", rd(WT2REF[key]))
    return float(m[-1]) if m else None


def collect():
    """returns (missing, data) where data has kld[arm] -> (list per used part), used parts, c4[arm]."""
    miss = []; A = arms(); n, cpp = nparts()
    if not A: miss.append("arms.txt")
    for need in ("mB", "xxs", "kt"):
        if A and need not in A: miss.append(f"arm {need} not in arms.txt")
    if n == 0: miss.append("split.log (PARTS line)")
    elif cpp != CPP: miss.append(f"chunks per part {cpp} != {CPP}")
    if not re.search(r"strided chunks \(ctx 4096, stride 4095\) 73 ", rd(f"{L}/c4_prep.log")): miss.append("c4_prep.log (73-window line)")
    used, kl, base = [], {a: [] for a in A if a != "q8"}, {}
    for p in range(1, n + 1):
        bl = re.findall(r"Final estimate: PPL = ([0-9.]+)", rd(f"{L}/base_p{p}.log"))
        got = {a: kld_log(f"{L}/kld_{a}_p{p}.log") for a in kl}
        if bl and all(got.values()):
            used.append(p); base[p] = float(bl[-1])
            for a in kl: kl[a].append(got[a])
    if len(used) < MINPARTS:
        bad = [f"p{p}:" + "".join(a + " " for a in kl if not kld_log(f"{L}/kld_{a}_p{p}.log")) for p in range(1, n + 1) if p not in used]
        miss.append(f"only {len(used)} of {n} parts complete for all arms (need >= {MINPARTS}); incomplete " + ",".join(bad)[:200])
    c4 = {}
    for a in C4ORDER:
        if a == "b2" and "b2" not in A: continue
        c4[a] = ppl_log(f"{L}/c4_{a}.log")
        if c4[a] is None: miss.append(f"c4_{a}.log")
    return miss, dict(A=A, used=used, kld=kl, base=base, c4=c4, nparts=n)


def mean_se(x):
    n = len(x); return sum(x) / n, (statistics.stdev(x) / math.sqrt(n) if n > 1 else float("nan"))


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]; n = len(d); m = sum(d) / n; h = 1.96 * statistics.stdev(d) / math.sqrt(n)
    return m, m - h, m + h, sum(x > 0 for x in d), n


if "--complete" in sys.argv or "--done" in sys.argv:
    miss, _ = collect()
    if "--done" in sys.argv:
        sys.exit(0 if (os.path.exists(OUT) and MARK in rd(OUT) and not miss) else 1)
    print("INCOMPLETE", miss[:8], "..." if len(miss) > 8 else "") if miss else print("COMPLETE")
    sys.exit(1 if miss else 0)
miss, Dt = collect()
if miss:
    print("REFUSED: partial logs", miss[:8]); sys.exit(1)

A, used, KL, C4 = Dt["A"], Dt["used"], Dt["kld"], Dt["c4"]
flat = lambda a, key: [v for part in KL[a] for v in part[key]]
K = {a: flat(a, "kld") for a in KL}; T = {a: flat(a, "top") for a in KL}
nch = len(K["mB"])
# token-weighted means from the printed per-part statistics (equal tokens per part) must agree with the chunk means
for a in KL:
    tw = sum(p["final"] for p in KL[a]) / len(KL[a])
    assert abs(tw - sum(K[a]) / nch) < 5e-4, f"token-weighted mean KLD {tw} != chunk mean for {a}"
ntok = nch * TOKC
f4 = lambda x: f"{x:.4f}"; f2 = lambda x: f"{x:.2f}"; f1 = lambda x: f"{x:.1f}"; sg = lambda x: f"{x:+.4f}"
nm = lambda a: LAB[a]
today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
ktsha = (rd(f"{L}/kt_sha256.txt").split() or [""])[0]; r56sha = (rd("logs/r56_kteq_sha256.txt").split() or [""])[0]
sha_note = "KT-eq requant sha256 identical to the r56 quant" if ktsha and ktsha == r56sha else f"KT-eq requant sha256 {ktsha[:12]} differs from r56 {r56sha[:12]}"

m_ = re.search(r"C4 docs \d+ bytes \d+ tokens (\d+)", rd(f"{L}/c4_prep.log")); c4tok = f"{int(m_.group(1)):,}" if m_ else "?"
S = [f"# R59: large-sample KLD and C4 perplexity, Qwen3-4B at equal bytes",
     f"{today}. All numbers MEASURED by research/detail/r59_kldc4.sh (+_prep.py, _report.py); logs logs/r59/*. Reference checks: part-1 Q8_0 base ppl {Dt['base'][used[0]]:.4f} (r56 {OLD_BASE_PPL}), "
     f"Model B part-1 KLD {f4(sum(KL['mB'][0]['kld']) / CPP)} (pe52 {OLD_B_KLD}); {sha_note}.",
     f"KLD setup: WikiText-2 test, {len(used)} disjoint parts x {CPP} chunks x 512 = {nch} chunks, {ntok:,} scored tokens, base = Q8_0 logits. "
     f"SE = sd/sqrt(chunks). Paired difference = arm minus reference, 95% CI = 1.96 sd/sqrt({nch}).",
     "", "| arm | bytes | KLD +/- SE | top-1 % +/- SE | dKLD vs Model B [95% CI] | worse than B on |", "|---|---|---|---|---|---|"]
for a in ORDER:
    if a not in KL: continue
    m, se = mean_se(K[a]); tm, tse = mean_se(T[a])
    if a == "mB": pc, wc = "-", "-"
    else:
        d = paired(K[a], K["mB"]); pc, wc = f"{sg(d[0])} [{sg(d[1])}, {sg(d[2])}]", f"{d[3]}/{d[4]}"
    S.append(f"| {nm(a)} | {A[a]:,} | {f4(m)} +/- {f4(se)} | {f1(tm)} +/- {f1(tse)} | {pc} | {wc} |")
pairs = [(x, y) for x, y in (("b2", "mB"), ("b2", "kt"), ("kt", "mB")) if x in KL and y in KL]
extra = []
for x, y in pairs:
    if (x, y) in (("b2", "mB"), ("kt", "mB")): continue
    d = paired(K[x], K[y]); extra.append(f"{nm(x)} minus {nm(y)} dKLD {sg(d[0])} [{sg(d[1])}, {sg(d[2])}] ({d[3]}/{d[4]} chunks worse)")
if extra: S.append("Extra pairs: " + "; ".join(extra) + ".")
dt = paired(T["kt"], T["mB"])
S.append(f"Top-1 agreement, IQ2_KT-eq minus Model B: {dt[0]:+.2f} pts [{dt[1]:+.2f}, {dt[2]:+.2f}].")
S.append("")
S.append(f"C4 setup: allenai/c4 en validation shard 0, first {c4tok} tokens, {NWIN} windows, ctx 4096 stride 4095 (flags of r36_hfj.sh). Paired per-window NLL as r59_paired.py.")
S += ["| arm | C4 ppl | x Q8_0 | WikiText-2 ppl | dNLL vs Model B [95% CI] | worse than B on |", "|---|---|---|---|---|---|"]
q8 = C4["q8"]["ppl"]
for a in C4ORDER:
    if a not in C4: continue
    w = wt2(a)
    if a in ("mB", "q8"): pc, wc = "-", "-"
    else:
        d = paired(C4[a]["nll"], C4["mB"]["nll"]); pc, wc = f"{sg(d[0])} [{sg(d[1])}, {sg(d[2])}]", f"{d[3]}/{d[4]}"
    S.append(f"| {nm(a) if a != 'q8' else 'Q8_0 (WT2: fp16)'} | {f4(C4[a]['ppl'])} | {C4[a]['ppl'] / q8:.3f} | {f4(w) if w else 'n/a'} | {pc} | {wc} |")
# verdicts (computed, not asserted)
V = []
dk = paired(K["kt"], K["mB"])
V.append("VERDICT KLD: " + (f"Model B has LOWER KLD than IQ2_KT-eq by {dk[0]:.4f} [{dk[1]:.4f}, {dk[2]:.4f}], CI excludes 0: claim HOLDS on {nch} chunks." if dk[1] > 0 else
         (f"IQ2_KT-eq has LOWER KLD than Model B by {-dk[0]:.4f}, CI excludes 0: claim REVERSES." if dk[2] < 0 else
          f"difference {dk[0]:+.4f} [{dk[1]:+.4f}, {dk[2]:+.4f}] includes 0: claim NOT SUPPORTED at {nch} chunks.")))
if "b2" in KL:
    d2 = paired(K["kt"], K["b2"])
    V.append("Model B2 vs IQ2_KT-eq: " + (f"B2 lower by {d2[0]:.4f} [{d2[1]:.4f}, {d2[2]:.4f}], significant." if d2[1] > 0 else
             (f"KT lower by {-d2[0]:.4f}, significant." if d2[2] < 0 else f"difference {d2[0]:+.4f} [{d2[1]:+.4f}, {d2[2]:+.4f}] includes 0.")))
c_all = [a for a in C4 if a not in ("q8",)]
rk = lambda f: " < ".join(nm(a) for a in sorted(c_all, key=f))
V.append(f"C4 ppl order (low to high): {rk(lambda a: C4[a]['ppl'])}. WikiText-2 order: " +
         (rk(lambda a: wt2(a) or 1e9) if all(wt2(a) for a in c_all) else "n/a") + ".")
dc = paired(C4["kt"]["nll"], C4["mB"]["nll"])
V.append(f"C4 IQ2_KT-eq minus Model B dNLL {sg(dc[0])} [{sg(dc[1])}, {sg(dc[2])}] ({'KT better' if dc[2] < 0 else ('B better' if dc[1] > 0 else 'no significant difference')}).")
V.append("Caveats: one model, one seed, chunks treated as independent (adjacent text within a part), Q8_0 not fp16 as KLD base; C4 slice is 73 windows from one shard.")
S += ["", *V]
sec = "\n".join(S) + "\n"; nw = len(sec.split())
if nw > 400:                                   # drop optional lines, last first
    for pat in ("Caveats:", "Extra pairs:", "Top-1 agreement,"):
        S = [x for x in S if not x.startswith(pat)]; sec = "\n".join(S) + "\n"; nw = len(sec.split())
        if nw <= 400: break
if nw > 400:
    print("REFUSED: report is", nw, "words (> 400)"); sys.exit(2)
open(OUT, "w").write(sec)
try:
    shutil.copy(OUT, WIN)
except OSError as e:
    print("notes copy failed", e)
print("REPORT written", OUT, "words", nw)
if MARK not in rd(PROG):
    mb, mx, kt = mean_se(K["mB"]), mean_se(K["xxs"]), mean_se(K["kt"])
    row = (f"- {MARK} (research/agents/eval_r59_kldc4.md, {datetime.datetime.now():%Y-%m-%d}): Qwen3-4B KLD vs Q8_0 over {nch} chunks x 512 ({ntok:,} tokens, {len(used)} WikiText-2 parts): "
           f"Model B {f4(mb[0])}+/-{f4(mb[1])}, IQ2_KT-eq {f4(kt[0])}+/-{f4(kt[1])} (paired KT minus B {sg(dk[0])} [{sg(dk[1])}, {sg(dk[2])}]), IQ2_XXS {f4(mx[0])}+/-{f4(mx[1])}"
           + (f", Model B2 {f4(mean_se(K['b2'])[0])}" if "b2" in K else "") + f". C4 val shard 0 ppl (73 windows, ctx 4096): "
           + ", ".join(f"{nm(a)} {f4(C4[a]['ppl'])}" for a in C4ORDER if a in C4) + f"; C4 KT-eq minus B dNLL {sg(dc[0])} [{sg(dc[1])}, {sg(dc[2])}]. MEASURED logs/r59/*.")
    with open(PROG, "a") as fh:
        fh.write(row + "\n")
    print("progress row appended")
sys.exit(0)
