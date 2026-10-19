# Paired per-window NLL differences from llama-perplexity cumulative logs (N windows read from the logs, default 70; copy of r59_paired.py for the Llama-3.1-8B 70-window protocol). 2026-10-18.
# Per-window NLL_k = k*ln(P_k) - (k-1)*ln(P_{k-1}) from the running perplexity [k]; 95% CI = 1.96*sd/sqrt(len(d)).
import re, math, statistics as st, sys, os
NW=int(os.environ.get("NW","70"))
def per_chunk(f):
    m=re.findall(r'\[(\d+)\]([0-9.]+)', open(f,errors='ignore').read()); k=[int(a) for a,_ in m]; v=[float(b) for _,b in m]
    i=[j for j,x in enumerate(k) if x==1][-1]; v=v[i:i+NW]; L=[math.log(x) for x in v]
    return [(j+1)*L[j]-j*L[j-1] if j else L[0] for j in range(len(v))]
a,b=per_chunk(sys.argv[1]),per_chunk(sys.argv[2]); d=[x-y for x,y in zip(a,b)]
print(f"dNLL {st.mean(d):+.4f} +/- {1.96*st.stdev(d)/math.sqrt(len(d)):.4f} (95%), first better on {sum(x<0 for x in d)}/{len(d)}")
