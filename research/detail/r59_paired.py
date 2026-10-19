# Paired per-window NLL differences from llama-perplexity cumulative logs (73 windows, ctx 4096). 2026-10-17.
# Per-window NLL_k = k*ln(P_k) - (k-1)*ln(P_{k-1}) from the running perplexity [k]; 95% CI = 1.96*sd/sqrt(73).
import re, math, statistics as st, sys
def per_chunk(f):
    m=re.findall(r'\[(\d+)\]([0-9.]+)', open(f,errors='ignore').read()); k=[int(a) for a,_ in m]; v=[float(b) for _,b in m]
    i=[j for j,x in enumerate(k) if x==1][-1]; v=v[i:i+73]; L=[math.log(x) for x in v]
    return [(j+1)*L[j]-j*L[j-1] if j else L[0] for j in range(73)]
a,b=per_chunk(sys.argv[1]),per_chunk(sys.argv[2]); d=[x-y for x,y in zip(a,b)]
print(f"dNLL {st.mean(d):+.4f} +/- {1.96*st.stdev(d)/math.sqrt(73):.4f} (95%), first better on {sum(x<0 for x in d)}/73")
