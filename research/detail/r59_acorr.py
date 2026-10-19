# Paired per-window dNLL with lag-1 autocorrelation-corrected 95% CI (n_eff = n(1-r1)/(1+r1)) and moving-block bootstrap (B=5, 20000 draws, seed 0). 2026-10-17.
import sys, math, random, statistics as st
sys.path.insert(0,'research/detail')
import re
def per_chunk(f):
    m=re.findall(r'\[(\d+)\]([0-9.]+)', open(f,errors='ignore').read()); k=[int(a) for a,_ in m]; v=[float(b) for _,b in m]
    i=[j for j,x in enumerate(k) if x==1][-1]; v=v[i:i+73]; L=[math.log(x) for x in v]
    return [(j+1)*L[j]-j*L[j-1] if j else L[0] for j in range(73)]
a,b=per_chunk(sys.argv[1]),per_chunk(sys.argv[2]); d=[x-y for x,y in zip(a,b)]; n=len(d); m=st.mean(d); s=st.stdev(d)
r1=sum((d[i]-m)*(d[i+1]-m) for i in range(n-1))/sum((x-m)**2 for x in d)
neff=n*(1-r1)/(1+r1); ci=1.96*s/math.sqrt(neff)
random.seed(0); B=5;bs=[]
for _ in range(20000):
    idx=[]
    while len(idx)<n:
        st0=random.randrange(n-B+1); idx+=range(st0,st0+B)
    bs.append(st.mean(d[i] for i in idx[:n]))
bs.sort()
print(f"mean {m:+.4f} naive +/-{1.96*s/math.sqrt(n):.4f} lag1 {r1:+.2f} neff {neff:.1f} corrected +/-{ci:.4f} blockboot(B=5) [{bs[499]:+.4f},{bs[19499]:+.4f}]")
