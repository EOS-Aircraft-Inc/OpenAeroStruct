"""Rank real airfoils on Cl_max as well as the 7 in spar and drag.

The first sweep ranked on spar depth then drag, and the winners came back with
clean Cl_max of 1.22-1.53. At MTOW the wing runs at CL ~= 0.93, and 3D CL_max is
typically ~0.9x the 2D section value, so those sections leave almost no stall
margin. This re-ranks with Cl_max as a first-class criterion and evaluates drag
at the *operating* lift coefficient rather than at Cl = 0.5.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
import aerosandbox as asb
from studies.vsp_planform import config

SPAR_IN=7.0; IN=0.0254; CHORD_MAX_IN=68.0; TOC=(0.12,0.30)
CL_OP=0.93                      # wing CL at MTOW
CLMAX_MIN=1.70                  # 2D floor -> ~1.5 in 3D
AL=np.arange(-6,22.05,0.5)

def measure(n):
    try:
        af=asb.Airfoil(n)
        if af.coordinates is None or len(af.coordinates)<20: return None
        xs=np.linspace(0.02,0.98,97); t=np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
        if not np.all(np.isfinite(t)) or t.max()<=0: return None
        t75=float(af.local_thickness(x_over_c=0.75))
        if t75<=0: return None
        return dict(name=n,af=af,tmax=float(t.max()),xt=float(xs[t.argmax()]),t75=t75,
                    chord=SPAR_IN/t75)
    except Exception: return None

root=Path(asb.__file__).parent/"geometry"/"airfoil"/"airfoil_database"
cands=[]
for p in sorted(root.glob("*.dat")):
    m=measure(p.stem)
    if m and TOC[0]<=m["tmax"]<=TOC[1] and m["chord"]<=CHORD_MAX_IN: cands.append(m)
print(f"{len(cands)} real sections meet the 7 in spar within {CHORD_MAX_IN:.0f} in of chord\n")

rows=[]
for m in cands:
    re=config.RE_PER_M*m["chord"]*IN
    try:
        a=m["af"].get_aero_from_neuralfoil(alpha=AL,Re=re,mach=config.MACH,model_size="medium")
        g=lambda k: np.atleast_1d(np.asarray(a[k],float)).ravel()
        cl,cd,conf=g("CL"),g("CD"),g("analysis_confidence")
        k=int(np.argmax(cl)); pre=slice(0,k+1)
        if cl[k]<CLMAX_MIN: continue
        if not (cl[pre].min()<=CL_OP<=cl[pre].max()): continue
        cdop=float(np.interp(CL_OP,cl[pre],cd[pre]))
        rows.append((cdop,m["name"],m["tmax"],m["xt"],m["chord"],re,float(cl[k]),float(AL[k]),float(conf.min())))
    except Exception: continue

rows.sort()
print(f"{'airfoil':>12} {'t/c':>6} {'x_t':>5} {'chord in':>9} {'Re':>9} "
      f"{'Cd@Cl.93':>9} {'L/D':>7} {'Cl_max':>7} {'a_st':>6} {'conf':>6}")
print("-"*92)
for cdop,n,tm,xt,ch,re,clm,ast,cf in rows[:16]:
    print(f"{n:>12} {tm:6.3f} {xt:5.2f} {ch:9.1f} {re:9.2e} {cdop:9.5f} {CL_OP/cdop:7.1f} "
          f"{clm:7.3f} {ast:6.1f} {cf:6.3f}")
print(f"\n(filtered to Cl_max >= {CLMAX_MIN}; drag at the wing's operating Cl = {CL_OP})")
