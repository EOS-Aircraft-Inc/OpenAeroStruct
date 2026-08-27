"""Airfoil DOE v2 under the corrected screen.

v1 ranked on thickness at 0.75c, which selected blunt aft-loaded sections and
produced picks that were later overturned. With a kinking spar the rear spar sits
wherever the section is deepest inside 0.12-0.75c, so the right screen is MAX
thickness and where it sits. Objective is L/D at the wing's operating Cl.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0,"/home/alex/repos/OpenAeroStruct")
import aerosandbox as asb
from studies.vsp_planform import config
from studies.vsp_planform.degen_csv import read_degen_csv
AL=np.arange(-6,20.05,0.5); CL_OP=0.93; DEPTH=7.0; BOX=25.0; FMIN=0.12; AMAX=0.75
XS=np.linspace(FMIN,AMAX,120)
def fp(cl):
    for i in range(2,len(cl)-1):
        if cl[i]>=cl[i-1] and cl[i]>cl[i+1]: return i
    return int(np.argmax(cl))
def selig(p):
    U,L,cur=[],[],None
    for ln in open(p):
        s=ln.strip()
        if s.startswith("#"): cur=U if "Upper" in s else L; continue
        t=s.split()
        if len(t)==2 and cur is not None:
            try: cur.append((float(t[0]),float(t[1])))
            except ValueError: pass
    U,L=np.array(U),np.array(L)
    return asb.Airfoil(name=Path(p).stem,coordinates=np.vstack([U[::-1],L[1:]])).repanel()
def asbuilt():
    c=[x for x in read_degen_csv(config.BASELINES["const_chord"]) if x.surf_index==0][0]
    p=c.plate;j=90
    xc=p.x[j]+p.nCamber_x[j]*p.zCamber[j]; zc=p.z[j]+p.nCamber_z[j]*p.zCamber[j]
    t=p.t[j];nx,nz=p.nCamber_x[j],p.nCamber_z[j]
    up=np.column_stack([xc+.5*t*nx,zc+.5*t*nz]);lo=np.column_stack([xc-.5*t*nx,zc-.5*t*nz])
    le=np.array([xc[-1],zc[-1]]);te=np.array([xc[0],zc[0]]);ch=np.linalg.norm(te-le);ct,st=(te-le)/ch
    loc=lambda P:(lambda d:np.column_stack([(d[:,0]*ct+d[:,1]*st)/ch,(-d[:,0]*st+d[:,1]*ct)/ch]))(P-le)
    U,L=loc(up),loc(lo)
    return asb.Airfoil(name="as-built",coordinates=np.vstack([U,L[::-1][1:]])).repanel()
def evaluate(af):
    try:
        T=np.array([float(af.local_thickness(x_over_c=x)) for x in XS])
        if not np.all(np.isfinite(T)) or T.max()<=0.05 or T.max()>0.32: return None
        k=int(np.argmax(T)); xt=float(XS[k]); tm=float(T[k])
        c_depth=DEPTH/tm
        c_box=BOX/(xt-FMIN) if xt>FMIN+1e-3 else 1e9
        cj=max(c_depth,c_box)
        if cj>120.0: return None
        re=config.RE_PER_M*cj*0.0254
        a=af.get_aero_from_neuralfoil(alpha=AL,Re=re,mach=config.MACH,model_size="medium")
        g=lambda n: np.atleast_1d(np.asarray(a[n],float)).ravel()
        cl,cd,cf=g("CL"),g("CD"),g("analysis_confidence")
        i=fp(cl)
        if cl[:i+1].max()<CL_OP: return None
        cdop=float(np.interp(CL_OP,cl[:i+1],cd[:i+1]))
        return dict(name=af.name,tmax=tm,xt=xt,cj=cj,re=re,ld=CL_OP/cdop,cd=cdop,
                    clmax=float(cl[i]),ast=float(AL[i]),conf=float(cf.min()))
    except Exception: return None
rows=[]
manual={}
for nm,p in (("as-built",None),("S6","S6_airfoil"),("S7","S7_airfoil"),("S9","S9_airfoil"),("S11","S11_airfoil")):
    af=asbuilt() if p is None else selig(f"/mnt/c/Users/AlexanderAmos/Downloads/{p}.dat")
    af.name=nm; r=evaluate(af)
    if r: manual[nm]=r; rows.append(r)
root=Path(asb.__file__).parent/"geometry"/"airfoil"/"airfoil_database"
names=sorted(x.stem for x in root.glob("*.dat"))
print(f"screening {len(names)} database sections (7 in depth, 25 in box, spar free 0.12-0.75c)...")
for n in names:
    try: r=evaluate(asb.Airfoil(n))
    except Exception: r=None
    if r: rows.append(r)
rows.sort(key=lambda r:-r["ld"])
print(f"{len(rows)} feasible\n")
print(f"{'rank':>4} {'airfoil':>14} {'t_max':>7} {'x_t':>5} {'chord in':>9} {'L/D@.93':>8} {'Clmax1':>7} {'a':>5} {'conf':>6}")
print("-"*74)
for i,r in enumerate(rows[:15],1):
    tag="  <-- manual" if r["name"] in manual else ""
    print(f"{i:4d} {r['name']:>14} {r['tmax']:7.4f} {r['xt']:5.2f} {r['cj']:9.1f} {r['ld']:8.1f} "
          f"{r['clmax']:7.3f} {r['ast']:5.1f} {r['conf']:6.3f}{tag}")
print("\nmanual selections:")
for nm,r in manual.items():
    rk=[i for i,x in enumerate(rows,1) if x is r][0]
    print(f"   {nm:>9}  rank {rk:4d} of {len(rows)}   L/D {r['ld']:6.1f}  chord {r['cj']:6.1f} in  "
          f"Clmax {r['clmax']:.3f}  conf {r['conf']:.3f}")
