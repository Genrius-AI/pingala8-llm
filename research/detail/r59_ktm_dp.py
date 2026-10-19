import json, gguf, numpy as np
d=json.load(open('research/detail/r56_kteq_design/imp2.json'))
im=gguf.GGUFReader('models/qwen3-4B.imatrix'); I={}; C={}
for t in im.tensors:
    if t.name.endswith('.in_sum2'): I[t.name[:-8]]=np.array(t.data,dtype=np.float64).ravel()
    elif t.name.endswith('.counts'): C[t.name[:-7]]=float(np.array(t.data).ravel()[0])
Z={l:I[f'blk.{l}.attn_output.weight'].sum()/C[f'blk.{l}.attn_output.weight'] for l in range(36)}
bpw={'iq2_kt':2.1375,'iq3_k':3.4375,'iq4_k':4.5}; D=lambda t:2.0**(-2*bpw[t])
rb={'attn_output':{'iq2_kt':2560*1092,'iq3_k':2560*1760},
    'attn_v':{'iq2_kt':1024*684,'iq3_k':1024*1100,'iq4_k':1024*1440},
    'ffn_down':{'iq2_kt':2560*2588,'iq3_k':2560*4180}}
S=lambda k,l:d[f'blk.{l}.{k}.weight']['S']
R=1.235; Rl={}  # residual energy at attn input (a) and ffn input (f)
for l in range(36):
    Rl[('a',l)]=R; R+=S('attn_output',l); Rl[('f',l)]=R; R+=S('ffn_down',l)
items=[]  # (name,kind,layer,cur,[(newtype,saved_bytes,abs_pen,rel_pen)])
for l in range(36):
    for k in ('attn_output','attn_v','ffn_down'):
        cur=d[f'blk.{l}.{k}.weight']['cur']
        if cur=='iq2_kt': continue
        if k=='attn_output': eff=S(k,l); r=Rl[('a',l)]
        elif k=='attn_v': eff=4*S(k,l)*S('attn_output',l)/Z[l]; r=Rl[('a',l)]
        else: eff=S(k,l); r=Rl[('f',l)]
        opts=[]
        for nt in ('iq3_k','iq2_kt'):
            if D(nt)<=D(cur): continue
            if nt not in rb[k]: continue
            sv=rb[k][cur]-rb[k][nt]
            opts.append((nt,sv,eff*(D(nt)-D(cur)),eff*(D(nt)-D(cur))/r))
        items.append((f'blk.{l}.{k}',k,l,cur,opts))
print(len(items),'upgraded block tensors; total saving if all->iq2_kt', sum(max(o[1] for o in it[4]) for it in items))
json.dump(items,open('research/work/r59kt/items.json','w'))
T=1300786816-1245915872; W=4096
def solve(col, lam=0.0):
    # DP over saved bytes (unit 256) min penalty
    U=256; cap=(T+W)//U+2000
    INF=1e300
    best=[INF]*(cap+1); best[0]=0.0; choice=[None]*(cap+1)
    back=[]
    cur=best[:]; hist=[]
    for it in items:
        new=cur[:]; ch=[None]*(cap+1)
        for s in range(cap+1):
            if cur[s]>=INF: continue
            for oi,o in enumerate(it[4]):
                ns=s+o[1]//U
                if ns>cap: continue
                v=cur[s]+o[col]+lam
                if v<new[ns]: new[ns]=v; ch[ns]=(s,oi)
        hist.append(ch); cur=new
    lo=-(-T//U); hi=(T+W)//U
    s=min(range(lo,hi+1),key=lambda s:cur[s])
    pen=cur[s]; sel=[]
    for i in range(len(items)-1,-1,-1):
        c=hist[i][s]
        if c is not None:
            ps,oi=c
            # verify transition changed from previous? hist[i][s] set only if improved at stage i
            sel.append((items[i][0],items[i][4][oi][0],items[i][4][oi][1])); s=ps
    return pen,sel[::-1]
for name,col,lam in (('rel',3,0.0),):
    pen,sel=solve(col,lam)
    sv=sum(x[2] for x in sel)
    from collections import Counter
    print(name,'n=',len(sel),'saved',sv,'delta_vs_T',sv-T,'(%.3f%%)'%(100*(sv-T)/1241608288),'pen %.4g'%pen)
    print('  ',Counter((x[0].split('.')[2],x[1]) for x in sel))
    
    import json as J; J.dump(sel,open('research/work/r59kt/sel_%s.json'%name,'w'))
    for kind in ('attn_output','attn_v','ffn_down'):
        for nt in ('iq2_kt','iq3_k'):
            L=sorted(int(x[0].split('.')[1]) for x in sel if x[0].split('.')[2]==kind and x[1]==nt)
            if L: print('   ',kind,nt,L)
    print('   file bytes',1300786816-sv,'vs ModelB',1300786816-sv-1241608288)
