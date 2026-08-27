"""2D section results (NeuralFoil) for the candidate airfoils."""
import os
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
import aerosandbox as asb
from studies.vsp_planform import config

RE = 6.93e6          # junction chord 57.2 in at cruise
AL = np.arange(-8, 20.05, 0.5)
FOILS = [("fx2","#C44E52","chosen (c=57.2 in)"),
         ("naca664221","#7d4b9a","NACA 66(4)-221 (c=56 in)"),
         ("naca6512","#4C72B0","as-built equivalent"),
         ("naca2412","0.55","NACA 2412 (reference)")]

def polar(n, re):
    a = asb.Airfoil(n).get_aero_from_neuralfoil(alpha=AL, Re=re, mach=config.MACH, model_size="large")
    g = lambda k: np.atleast_1d(np.asarray(a[k], dtype=float)).ravel()
    return g("CL"), g("CD"), g("CM"), g("analysis_confidence")

fig, ax = plt.subplots(2,2, figsize=(13.5,10))
fig.suptitle(f"2D section results — NeuralFoil, Re = {RE:.2e}, M = {config.MACH:.3f}", fontsize=14)
print(f"{'airfoil':>12} {'t/c':>6} {'Cl_max':>7} {'a_stall':>8} {'Cd_min':>8} {'Cd@Cl.5':>8} "
      f"{'(L/D)max':>9} {'Cl@LDmax':>9} {'Cm0':>8} {'minconf':>8}")
print("-"*100)
for n,c,lb in FOILS:
    cl,cd,cm,conf = polar(n,RE); k=int(np.argmax(cl)); pre=slice(0,k+1)
    ld = cl/cd; kl=int(np.argmax(ld[pre]))
    tc = float(asb.Airfoil(n).max_thickness())
    cd05 = float(np.interp(0.5, cl[pre], cd[pre]))
    cm0 = float(np.interp(0.0, cl[pre], cm[pre]))
    print(f"{n:>12} {tc:6.3f} {cl[k]:7.3f} {AL[k]:8.1f} {cd.min():8.5f} {cd05:8.5f} "
          f"{ld[kl]:9.1f} {cl[kl]:9.3f} {cm0:+8.4f} {conf.min():8.3f}")
    ax[0,0].plot(AL,cl,color=c,lw=1.6,label=lb); ax[0,0].plot(AL[k],cl[k],'o',ms=5,mfc='white',color=c)
    ax[0,1].plot(cd[pre]*1e4,cl[pre],color=c,lw=1.6,label=lb)
    ax[1,0].plot(cl[pre],ld[pre],color=c,lw=1.6,label=lb)
    ax[1,1].plot(cl[pre],cm[pre],color=c,lw=1.6,label=lb)
ax[0,0].set(title="lift curve",xlabel=r"$\alpha$ [deg]",ylabel="$C_l$"); ax[0,0].axhline(0,color='0.85',lw=.8)
ax[0,1].set(title="drag polar (pre-stall)",xlabel="$C_d$ [counts]",ylabel="$C_l$"); ax[0,1].set_xlim(0,250)
ax[1,0].set(title="section L/D",xlabel="$C_l$",ylabel="$C_l/C_d$")
ax[1,1].set(title="pitching moment",xlabel="$C_l$",ylabel="$C_m$"); ax[1,1].axhline(0,color='0.85',lw=.8)
for a in ax.ravel(): a.grid(alpha=.25); a.legend(fontsize=8)
fig.tight_layout(rect=(0,0,1,.95))
p=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"out","figures","section_2d_results.png")
fig.savefig(p,dpi=130); print("\n"+p)
