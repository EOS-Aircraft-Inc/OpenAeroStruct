"""2D results for the high-Cl_max candidates. Cl_max = FIRST peak only."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
import aerosandbox as asb
from studies.vsp_planform import config
AL=np.arange(-8,22.05,0.5); CL_OP=0.93
C=[("goe625",67.6,"#C44E52"),("fx77w270s",56.8,"#2a9d8f"),("fx69274",51.6,"#7d4b9a"),
   ("naca664221",56.0,"#DD8452"),("goe264",None,"#8c8c8c"),("naca6512",40.0,"#4C72B0")]
def first_peak(cl):
    for i in range(2,len(cl)-1):
        if cl[i]>=cl[i-1] and cl[i]>cl[i+1]: return i
    return int(np.argmax(cl))
fig,ax=plt.subplots(2,2,figsize=(13.5,10))
fig.suptitle(f"2D candidate sections — NeuralFoil, M={config.MACH:.3f}, Re at each section's required chord\n"
             f"Cl_max marked at the FIRST peak; wing operates at Cl ~ {CL_OP}",fontsize=13)
print(f"{'airfoil':>12} {'t/c':>6} {'t@.75c':>7} {'chord':>7} {'Re':>9} {'Cd@Cl.93':>9} {'L/D':>7} "
      f"{'Clmax1st':>9} {'a1st':>6} {'margin':>7} {'conf':>6}")
print("-"*102)
for n,ch,col in C:
    af=asb.Airfoil(n); t75=float(af.local_thickness(x_over_c=0.75))
    if ch is None: ch=7.0/t75
    re=config.RE_PER_M*ch*0.0254
    a=af.get_aero_from_neuralfoil(alpha=AL,Re=re,mach=config.MACH,model_size="large")
    g=lambda k: np.atleast_1d(np.asarray(a[k],float)).ravel()
    cl,cd,cm,cf=g("CL"),g("CD"),g("CM"),g("analysis_confidence")
    i=first_peak(cl); pre=slice(0,i+1)
    cdop=float(np.interp(CL_OP,cl[pre],cd[pre])) if cl[pre].max()>=CL_OP else float("nan")
    print(f"{n:>12} {float(af.max_thickness()):6.3f} {t75:7.4f} {ch:7.1f} {re:9.2e} "
          f"{cdop:9.5f} {CL_OP/cdop if cdop==cdop else float('nan'):7.1f} {cl[i]:9.3f} {AL[i]:6.1f} "
          f"{cl[i]-CL_OP:+7.3f} {cf.min():6.3f}")
    ax[0,0].plot(AL,cl,color=col,lw=1.5,label=n); ax[0,0].plot(AL[i],cl[i],'o',ms=6,mfc='white',color=col)
    ax[0,1].plot(cd[pre]*1e4,cl[pre],color=col,lw=1.5,label=n)
    ax[1,0].plot(cl[pre],(cl/cd)[pre],color=col,lw=1.5,label=n)
    xy=af.coordinates; ax[1,1].plot(xy[:,0],xy[:,1],color=col,lw=1.4,label=n)
ax[0,0].axhline(CL_OP,color='0.4',ls='--',lw=1); ax[0,0].set(title="lift curve (first peak marked)",xlabel=r"$\alpha$ [deg]",ylabel="$C_l$")
ax[0,1].axhline(CL_OP,color='0.4',ls='--',lw=1); ax[0,1].set(title="drag polar to first peak",xlabel="$C_d$ [counts]",ylabel="$C_l$"); ax[0,1].set_xlim(0,300)
ax[1,0].axvline(CL_OP,color='0.4',ls='--',lw=1); ax[1,0].set(title="section L/D",xlabel="$C_l$",ylabel="$C_l/C_d$")
ax[1,1].axvline(0.75,color='#2a9d8f',ls='--',lw=1.2); ax[1,1].axvline(0.125,color='#2a9d8f',ls='--',lw=1.2)
ax[1,1].set(title="sections (spar stations dashed)",xlabel="x/c",ylabel="y/c"); ax[1,1].set_aspect("equal")
for a_ in ax.ravel(): a_.grid(alpha=.25); a_.legend(fontsize=8)
fig.tight_layout(rect=(0,0,1,.94))
p="studies/vsp_planform/out/figures/candidates_2d.png"; fig.savefig(p,dpi=130); print("\n"+p)
