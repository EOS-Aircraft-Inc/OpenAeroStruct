"""Perturb the ORIGINAL as-built section for aft thickness, instead of replacing it.

The as-built airfoil is recovered exactly from the DegenGeom plate (camber line
plus the plate's own thickness distribution), not approximated by a NACA. It is
then CST-fitted and its aft weights scaled to buy spar depth, keeping the nose
and camber that the DOE showed were already good.
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/home/alex/repos/OpenAeroStruct")
import aerosandbox as asb
from studies.vsp_planform import config
from studies.vsp_planform.degen_csv import read_degen_csv

AL=np.arange(-8,22.05,0.5); CL_OP=0.93; SPAR=7.0
def first_peak(cl):
    for i in range(2,len(cl)-1):
        if cl[i]>=cl[i-1] and cl[i]>cl[i+1]: return i
    return int(np.argmax(cl))

c=[x for x in read_degen_csv(config.BASELINES["const_chord"]) if x.surf_index==0][0]
p=c.plate; j=90
xc=p.x[j]+p.nCamber_x[j]*p.zCamber[j]; zc=p.z[j]+p.nCamber_z[j]*p.zCamber[j]
t=p.t[j]; nx,nz=p.nCamber_x[j],p.nCamber_z[j]
up=np.column_stack([xc+.5*t*nx, zc+.5*t*nz]); lo=np.column_stack([xc-.5*t*nx, zc-.5*t*nz])
le=np.array([xc[-1],zc[-1]]); te=np.array([xc[0],zc[0]]); ch=np.linalg.norm(te-le); ct,st=(te-le)/ch
loc=lambda P:(lambda d:np.column_stack([(d[:,0]*ct+d[:,1]*st)/ch,(-d[:,0]*st+d[:,1]*ct)/ch]))(P-le)
U,L=loc(up),loc(lo)
orig=asb.Airfoil(name="as-built", coordinates=np.vstack([U, L[::-1][1:]])).repanel()
K0=orig.to_kulfan_airfoil()
print(f"as-built section {j}: chord {ch:.1f} in, t_max {float(orig.max_thickness()):.4f}, "
      f"t@0.75c {float(orig.local_thickness(x_over_c=0.75)):.4f}")

def perturb(scale, n_aft=4):
    uw=np.array(K0.upper_weights,float).copy(); lw=np.array(K0.lower_weights,float).copy()
    w=np.linspace(0,1,len(uw))**2                      # weight the aft end
    uw=uw*(1+(scale-1)*w); lw=lw*(1+(scale-1)*w)
    return asb.KulfanAirfoil(name=f"as-built x{scale:.2f}", upper_weights=uw, lower_weights=lw,
                             leading_edge_weight=K0.leading_edge_weight, TE_thickness=K0.TE_thickness)

rows=[]
for s in (1.0,1.4,1.8,2.2,2.6):
    af=perturb(s); t75=float(af.local_thickness(x_over_c=0.75)); tm=float(af.max_thickness())
    chord_req=SPAR/t75
    re=config.RE_PER_M*chord_req*0.0254
    a=af.get_aero_from_neuralfoil(alpha=AL,Re=re,mach=config.MACH,model_size="large")
    g=lambda k: np.atleast_1d(np.asarray(a[k],float)).ravel()
    cl,cd,cf=g("CL"),g("CD"),g("analysis_confidence")
    i=first_peak(cl); pre=slice(0,i+1)
    cdop=float(np.interp(CL_OP,cl[pre],cd[pre])) if cl[pre].max()>=CL_OP else np.nan
    rows.append((s,af,tm,t75,chord_req,re,cdop,cl[i],AL[i],cf.min()))
print(f"\n{'scale':>6} {'t_max':>7} {'t@.75c':>8} {'chord in':>9} {'Cd@Cl.93':>9} {'L/D':>7} {'Clmax1st':>9} {'a':>6} {'conf':>6}")
print("-"*76)
for s,af,tm,t75,cr,re,cdop,clm,al,cf in rows:
    print(f"{s:6.2f} {tm:7.4f} {t75:8.4f} {cr:9.1f} {cdop:9.5f} {CL_OP/cdop:7.1f} {clm:9.3f} {al:6.1f} {cf:6.3f}")

fig,ax=plt.subplots(2,2,figsize=(13.5,10))
fig.suptitle("Perturbing the as-built section for aft thickness (CST weights scaled toward the trailing edge)",fontsize=13)
cols=plt.cm.viridis(np.linspace(.15,.85,len(rows)))
for (s,af,tm,t75,cr,re,cdop,clm,al,cf),col in zip(rows,cols):
    a=af.get_aero_from_neuralfoil(alpha=AL,Re=config.RE_PER_M*cr*0.0254,mach=config.MACH,model_size="large")
    g=lambda k: np.atleast_1d(np.asarray(a[k],float)).ravel()
    cl,cd=g("CL"),g("CD"); i=first_peak(cl); pre=slice(0,i+1)
    lb=f"x{s:.1f}  t/c {tm:.3f}  c={cr:.0f}in"
    ax[0,0].plot(AL,cl,color=col,lw=1.5,label=lb); ax[0,0].plot(AL[i],cl[i],'o',ms=6,mfc='white',color=col)
    ax[0,1].plot(cd[pre]*1e4,cl[pre],color=col,lw=1.5,label=lb)
    ax[1,0].plot(cl[pre],(cl/cd)[pre],color=col,lw=1.5,label=lb)
    co=af.coordinates; ax[1,1].plot(co[:,0],co[:,1],color=col,lw=1.3,label=lb)
for n,cl_,lb in (("fx77w270s","#C44E52","fx77w270s"),):
    af=asb.Airfoil(n); co=af.coordinates; ax[1,1].plot(co[:,0],co[:,1],color=cl_,lw=1.5,ls="--",label=lb)
co=orig.coordinates; ax[1,1].plot(co[:,0],co[:,1],color="k",lw=1.8,ls=":",label="as-built (original)")
ax[0,0].axhline(CL_OP,color='0.4',ls='--',lw=1); ax[0,0].set(title="lift curve (first peak marked)",xlabel=r"$\alpha$",ylabel="$C_l$")
ax[0,1].axhline(CL_OP,color='0.4',ls='--',lw=1); ax[0,1].set(title="drag polar",xlabel="$C_d$ [counts]",ylabel="$C_l$"); ax[0,1].set_xlim(0,250)
ax[1,0].axvline(CL_OP,color='0.4',ls='--',lw=1); ax[1,0].set(title="section L/D",xlabel="$C_l$",ylabel="$C_l/C_d$")
ax[1,1].axvline(.75,color='#2a9d8f',ls='--',lw=1.2); ax[1,1].axvline(.125,color='#2a9d8f',ls='--',lw=1.2)
ax[1,1].set(title="sections",xlabel="x/c",ylabel="y/c"); ax[1,1].set_aspect("equal")
for a_ in ax.ravel(): a_.grid(alpha=.25); a_.legend(fontsize=7)
fig.tight_layout(rect=(0,0,1,.95))
p="studies/vsp_planform/out/figures/perturb_original.png"; fig.savefig(p,dpi=130); print("\n"+p)
