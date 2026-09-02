"""Best ConstChord case: planform, t/c and spar depth, plus the 2D section."""
import os
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
import aerosandbox as asb
from studies.vsp_planform import config, param
param.REGION_A_RULE["const_chord"]="preserved"
import studies.vsp_planform.run_opt as ro
from studies.vsp_planform.run_opt import POINT, load_baseline, trim_alpha
from scipy.optimize import lsq_linear

W=382547.0; IN=0.0254; RATIO_FX2=0.621; RATIO_BASE=0.527; CJ=57.2; ROOT=105.0; SPAR=7.0
q=0.5*config.RHO*config.V_MS**2
mesh,stick,regions,p0,_,_ = load_baseline("const_chord", config.N_SPANWISE_HALF, 9)
ys=np.abs(stick.le[:,1])*config.SCALE
orig=ro.build_surface
def mk(c):
    def b(m,s,r):
        d=orig(m,s,r); d["c_max_t"]=c; d["t_over_c_cp"]=np.full(35,float(np.mean(d["t_over_c_cp"]))); return d
    return b

def run(c_max_t, c_junc, ratio, spar_rule):
    ro.build_surface = mk(c_max_t) if c_max_t else orig
    prob,_ = ro.build_problem("const_chord",mesh,stick,regions,p0); ro.build_surface=orig
    if c_junc: prob.set_val("wing.taper_B", c_junc/ROOT)
    prob.run_model()
    g=prob.get_val("wing.mesh",units="m"); y=np.abs(g[0,:,1]); c=g[-1,:,0]-g[0,:,0]
    yj=regions.y_c_start*config.SCALE; jj=int(np.argmin(abs(y-yj)))
    toc=np.interp(y,ys,stick.toc)
    if spar_rule:
        need=SPAR*IN/(ratio*np.maximum(c,1e-6))
        toc=np.where(np.arange(y.size)<=jj, np.clip(need,0.08,0.30), toc)
    ncp=prob.get_val("wing.t_over_c_cp").size
    cols=[]
    for i in range(ncp):
        e=np.zeros(ncp); e[i]=1.0; prob.set_val("wing.t_over_c_cp",e); prob.run_model()
        cols.append(np.asarray(prob.get_val("wing.t_over_c")).ravel())
    M=np.column_stack(cols)
    yp=0.5*(y[:-1]+y[1:])
    prob.set_val("wing.t_over_c_cp", lsq_linear(M, np.interp(yp,y,toc), bounds=(0.08,0.30)).x)
    prob.run_model()
    trim_alpha(prob, W/(q*float(prob.get_val(f"{POINT}.wing.S_ref")[0])))
    S=float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    tocA=np.asarray(prob.get_val("wing.t_over_c")).ravel()
    cp=0.5*(c[:-1]+c[1:])
    return dict(y=y,c=c,le=g[0,:,0],te=g[-1,:,0],yp=yp,toc=tocA,cp=cp,S=S,
                D=q*S*sum(float(prob.get_val(f"{POINT}.wing_perf.{k}")[0]) for k in ("CDi","CDv","CDw")),
                Di=q*S*float(prob.get_val(f"{POINT}.wing_perf.CDi")[0]), ratio=ratio, yj=yj,
                ya=regions.y_a_end*config.SCALE)
base=run(None,None,RATIO_BASE,False); best=run(0.50,CJ,RATIO_FX2,True)

fig=plt.figure(figsize=(14,10.5))
fig.suptitle("ConstChord — best 7 in spar case: fx2 section, 57.2 in junction chord\n"
             f"drag {base['D']:.0f} -> {best['D']:.0f} N ({best['D']/base['D']-1:+.2%}) at MTOW, span 118 ft",fontsize=13)
gs=fig.add_gridspec(3,2,hspace=0.42,wspace=0.24,top=0.90,bottom=0.06,left=0.08,right=0.97)
ax=fig.add_subplot(gs[0,:])
for d,cl,lb in ((base,"#4C72B0","as-built"),(best,"#C44E52","fx2, c=57.2 in")):
    k=d["y"]>=0
    ax.plot(d["y"][k],d["le"][k],color=cl,lw=1.7,label=lb); ax.plot(d["y"][k],d["te"][k],color=cl,lw=1.7)
    ax.fill_between(d["y"][k],d["le"][k],d["te"][k],color=cl,alpha=.10)
ax.set_title("planform, half span");ax.set_xlabel("y [m]");ax.set_ylabel("x [m]");ax.invert_yaxis();ax.set_aspect("equal");ax.grid(alpha=.25);ax.legend(fontsize=9)
ax=fig.add_subplot(gs[1,0])
for d,cl,lb in ((base,"#4C72B0","as-built"),(best,"#C44E52","fx2")):
    ax.plot(d["yp"]/d["y"].max(),d["toc"],color=cl,lw=1.7,label=lb)
for x in (best["ya"]/best["y"].max(),best["yj"]/best["y"].max()): ax.axvline(x,color="0.75",ls="--",lw=1)
ax.set_title("t/c distribution");ax.set_xlabel(r"$\eta$");ax.set_ylabel("t/c");ax.grid(alpha=.25);ax.legend(fontsize=9)
ax=fig.add_subplot(gs[1,1])
for d,cl,lb in ((base,"#4C72B0","as-built"),(best,"#C44E52","fx2")):
    ax.plot(d["yp"]/d["y"].max(), d["ratio"]*d["toc"]*d["cp"]/IN, color=cl,lw=1.7,label=lb)
ax.axhline(7.0,color="#2a9d8f",ls="--",lw=1.4,label="7 in target")
ax.axhline(6.5,color="#e07a5f",ls=":",lw=1.4,label="6.5 in acceptable")
for x in (best["ya"]/best["y"].max(),best["yj"]/best["y"].max()): ax.axvline(x,color="0.75",ls="--",lw=1)
ax.set_title("aft spar depth at 0.75c");ax.set_xlabel(r"$\eta$");ax.set_ylabel("depth [in]");ax.grid(alpha=.25);ax.legend(fontsize=8)
ax=fig.add_subplot(gs[2,:])
for nm,cl in (("fx2","#C44E52"),("naca664221","#7d4b9a")):
    xy=asb.Airfoil(nm).coordinates; ax.plot(xy[:,0],xy[:,1],color=cl,lw=1.6,label=nm)
xy=asb.Airfoil("naca2412").coordinates
ax.plot(xy[:,0],xy[:,1],color="0.6",lw=1.2,ls="--",label="naca2412 (reference)")
ax.axvline(0.75,color="#2a9d8f",ls="--",lw=1.3); ax.axvline(0.125,color="#2a9d8f",ls="--",lw=1.3)
ax.annotate("front spar 0.125c",xy=(0.125,-0.13),fontsize=8,color="#2a9d8f",ha="center")
ax.annotate("aft spar 0.75c",xy=(0.75,-0.13),fontsize=8,color="#2a9d8f",ha="center")
ax.set_title("2D sections, normalized");ax.set_xlabel("x/c");ax.set_ylabel("y/c");ax.set_aspect("equal");ax.grid(alpha=.25);ax.legend(fontsize=9,loc="upper right")
p=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"out","figures","best_case_constchord.png")
fig.savefig(p,dpi=130); print(p)
for nm in ("fx2","naca664221"):
    a=asb.Airfoil(nm); print(f"  {nm:12s} t_max {float(a.max_thickness()):.4f}  t@0.75c {float(a.local_thickness(x_over_c=0.75)):.4f}")
print(f"  min spar in region B: {min(best['ratio']*best['toc'][ (best['yp']>=best['ya'])&(best['yp']<=best['yj']) ]*best['cp'][ (best['yp']>=best['ya'])&(best['yp']<=best['yj']) ]/IN):.2f} in")
