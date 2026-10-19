import sys, statistics
A = sys.argv[2].split(); R = {}
for ln in open(sys.argv[1]):
    if ln.startswith("PWR "):
        d = dict(kv.split("=") for kv in ln.split()[1:]); R[(d["arm"], d["test"], d["d"], int(d["round"]))] = d
for a in A:
    for t in ("pp512", "tg128"):
        for dd in ("0", "4096"):
            rs = [R[(a, t, dd, p)] for p in (1, 2, 3) if (a, t, dd, p) in R]
            if not rs: continue
            pr = [p for p in (1, 2, 3) if (a, t, dd, p) in R and ("iq2", t, dd, p) in R]
            rat = [float(R[(a, t, dd, p)]["tps"]) / float(R[("iq2", t, dd, p)]["tps"]) for p in pr]
            jr = [float(R[(a, t, dd, p)]["J_per_tok"]) / float(R[("iq2", t, dd, p)]["J_per_tok"]) for p in pr]
            med = lambda k: statistics.median(float(r[k]) for r in rs)
            print("REALSUM %-6s %s d=%-4s tps %.1f ratio_vs_iq2 %.4f [%s] J/tok %.5f J_ratio %.4f W %.1f mem_MHz %.0f sm_MHz %.0f peak_MiB %.0f" % (
                a, t, dd, med("tps"), statistics.median(rat), " ".join("%.3f" % x for x in rat), med("J_per_tok"), statistics.median(jr), med("mean_W"), med("mem_MHz_med"), med("sm_MHz_med"), max(float(r["peak_mem_MiB"]) for r in rs)))
