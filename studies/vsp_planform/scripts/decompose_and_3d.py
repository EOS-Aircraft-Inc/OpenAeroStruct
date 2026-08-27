"""Split the -4.7% into planform vs airfoil, and render the 3D wing + twist."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from scipy.optimize import lsq_linear
from studies.vsp_planform import config, param
param.REGION_A_RULE["const_chord"]="preserved"
import studies.vsp_planform.run_opt as ro
from studies.vsp_planform.run_opt import POINT, load_baseline, trim_alpha

W=382547.0; IN=0.0254; ROOT=105.0; SPAR=7.0; RAT_NEW=0.621; RAT_OLD=0.527
q=0.5*config.RHO*config.V_MS**2
mesh,stick,regions,p0,_,_=load_baseline("const_chord",config.N_SPANWISE_HALF,9)
ys=np.abs(stick.le[:,1])*config.SCALE; orig=ro.build_surface
def mk(c):
    def b(m,s,r):
        d=orig(m,s,r)
        if c: d["c_max_t"]=c
        d["t_over_c_cp"]=np.full(35,float(np.mean(d["t_over_c_cp"]))); return d
    return b
def run(c_max_t,c_junc,ratio,spar):
    ro.build_surface=mk(c_max_t); pr,_=ro.build_problem("const_chord",mesh,stick,regions,p0); ro.build_surface=orig
    if c_junc: pr.set_val("wing.taper_B",c_junc/ROOT)
    pr.run_model()
    g=pr.get_val("wing.mesh",units="m"); y=np.abs(g[0,:,1]); c=g[-1,:,0]-g[0,:,0]
    yj=regions.y_c_start*config.SCALE; jj=int(np.argmin(abs(y-yj))); toc=np.interp(y,ys,stick.toc)
    if spar:
        need=SPAR*IN/(ratio*np.maximum(c,1e-6))
        toc=np.where(np.arange(y.size)<=jj,np.clip(need,0.08,0.30),toc)
    ncp=pr.get_val("wing.t_over_c_cp").size; cols=[]
    for i in range(ncp):
        e=np.zeros(ncp); e[i]=1.0; pr.set_val("wing.t_over_c_cp",e); pr.run_model()
        cols.append(np.asarray(pr.get_val("wing.t_over_c")).ravel())
    yp=0.5*(y[:-1]+y[1:])
    pr.set_val("wing.t_over_c_cp",lsq_linear(np.column_stack(cols),np.interp(yp,y,toc),bounds=(0.08,0.30)).x)
    pr.run_model(); trim_alpha(pr,W/(q*float(pr.get_val(f"{POINT}.wing.S_ref")[0])))
    S=float(pr.get_val(f"{POINT}.wing.S_ref")[0])
    D=q*S*sum(float(pr.get_val(f"{POINT}.wing_perf.{k}")[0]) for k in ("CDi","CDv","CDw"))
    return dict(D=D,S=S,mesh=pr.get_val("wing.mesh",units="m").copy(),
                twist=pr.get_val("twist_abs",units="deg").copy(),
                Di=q*S*float(pr.get_val(f"{POINT}.wing_perf.CDi")[0]))
A=run(None,None,RAT_OLD,False)                  # baseline
B=run(None,57.2,RAT_OLD,False)                  # planform only
C=run(0.50,None,RAT_NEW,True)                   # airfoil only
D=run(0.50,57.2,RAT_NEW,True)                   # both
print(f"{'case':>34} {'S m2':>7} {'D N':>9} {'vs base':>9}")
print("-"*64)
for lb,r in (("A baseline",A),("B planform only (c=57.2)",B),("C airfoil only (c=40)",C),("D both",D)):
    print(f"{lb:>34} {r['S']:7.2f} {r['D']:9.1f} {r['D']/A['D']-1:+8.2%}")
print(f"\n  planform alone : {B['D']-A['D']:+8.1f} N")
print(f"  airfoil alone  : {C['D']-A['D']:+8.1f} N")
print(f"  sum of parts   : {(B['D']-A['D'])+(C['D']-A['D']):+8.1f} N")
print(f"  actual together: {D['D']-A['D']:+8.1f} N   (interaction {(D['D']-A['D'])-((B['D']-A['D'])+(C['D']-A['D'])):+.1f} N)")

# ---- 3D figure
def full(m):
    l=m[:,::-1,:].copy(); l[:,:,1]*=-1; return np.hstack((l[:,:-1,:],m))
fig=plt.figure(figsize=(14,11))
fig.suptitle(f"ConstChord: as-built vs best (fx2 section, 57.2 in junction chord)\n"
             f"drag {A['D']:.0f} -> {D['D']:.0f} N ({D['D']/A['D']-1:+.2%}) at MTOW",fontsize=13)
gs=fig.add_gridspec(3,2,hspace=0.35,wspace=0.2,top=0.90,bottom=0.06)
sets=(("as-built","#4C72B0",A),("best","#C44E52",D))
def draw(ax,i,j,eq=True,zex=1.0):
    for lb,cl,r in sets:
        m=full(r["mesh"])
        for k in range(m.shape[1]): ax.plot(m[:,k,i],m[:,k,j]*zex,color=cl,lw=.35,alpha=.85)
        for k in range(m.shape[0]): ax.plot(m[k,:,i],m[k,:,j]*zex,color=cl,lw=.35,alpha=.85)
    if eq: ax.set_aspect("equal")
    ax.grid(alpha=.25)
ax=fig.add_subplot(gs[0,:]); draw(ax,1,0); ax.invert_yaxis()
ax.set(title="top view",xlabel="y [m]",ylabel="x [m]")
ax.legend(handles=[plt.Line2D([],[],color=c,lw=2,label=l) for l,c,_ in sets],fontsize=9)
ax=fig.add_subplot(gs[1,0]); draw(ax,1,2,eq=False)
ax.set_aspect(5.0); ax.set(title="front view (z exaggerated 5x)",xlabel="y [m]",ylabel="z [m]")
ax=fig.add_subplot(gs[1,1])
for lb,cl,r in sets:
    m=r["mesh"]; ax.plot(m[:,0,0],m[:,0,2],color=cl,lw=1.6,label=f"{lb} root")
    ax.plot(m[:,-1,0],m[:,-1,2],color=cl,lw=1.2,ls="--",label=f"{lb} tip")
ax.set_aspect("equal"); ax.set(title="side view: root and tip sections",xlabel="x [m]",ylabel="z [m]")
ax.grid(alpha=.25); ax.legend(fontsize=8)
ax=fig.add_subplot(gs[2,0],projection="3d")
for lb,cl,r in sets:
    m=full(r["mesh"]); ax.plot_wireframe(m[:,:,0],m[:,:,1],m[:,:,2],color=cl,lw=.35,rstride=1,cstride=1)
sp=[float(np.ptp(full(A["mesh"])[:,:,i])) for i in range(3)]
ax.set_box_aspect([s/max(sp) for s in sp],zoom=1.3); ax.view_init(elev=52,azim=-70)
ax.set_title("isometric, true scale",y=1.06,loc="left",fontsize=11)
ax.set_xlabel("x [m]",labelpad=-4); ax.set_ylabel("y [m]",labelpad=6); ax.set_zlabel(""); ax.set_zticklabels([])
for a in (ax.xaxis,ax.yaxis,ax.zaxis): a.set_major_locator(plt.MaxNLocator(4))
ax.tick_params(labelsize=7,pad=-1)
ax=fig.add_subplot(gs[2,1])
for lb,cl,r in sets:
    y=np.abs(r["mesh"][0,:,1]); ax.plot(y/y.max(),r["twist"],color=cl,lw=1.7,marker="o",ms=3,label=lb)
for x in (regions.y_a_end*config.SCALE,regions.y_c_start*config.SCALE):
    ax.axvline(x/np.abs(A["mesh"][0,:,1]).max(),color="0.75",ls="--",lw=1)
ax.set(title="twist distribution",xlabel=r"$\eta$",ylabel="twist [deg]"); ax.grid(alpha=.25); ax.legend(fontsize=9)
p="studies/vsp_planform/out/figures/decompose_3d.png"; fig.savefig(p,dpi=130); print("\n"+p)
