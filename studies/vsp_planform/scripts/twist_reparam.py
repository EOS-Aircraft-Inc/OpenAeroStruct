"""Twist rerun: 15 control points, and the spline terminated at the winglet root.

Reports the distribution, not endpoints: full spanwise twist, sign changes in the
gradient, and where peak incidence sits.
"""
import sys, numpy as np, openmdao.api as om
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from scipy.optimize import lsq_linear
from studies.vsp_planform import config, param
param.REGION_A_RULE["const_chord"]="preserved"
import studies.vsp_planform.run_opt as ro
from studies.vsp_planform.run_opt import POINT, load_baseline, trim_alpha

W=382547.0; IN=0.0254; ROOT=105.0; SPAR=7.0
q=0.5*config.RHO*config.V_MS**2
mesh,stick,regions,p0,_,_=load_baseline("const_chord",config.N_SPANWISE_HALF,9)
ys=np.abs(stick.le[:,1])*config.SCALE; orig=ro.build_surface
CASES={"as-built":(None,None,0.527),"fx77w270s":(0.34,56.8,0.676)}

def mk(cmt,ntw):
    def b(m,s,r):
        d=orig(m,s,r,n_twist_cp=ntw) if "n_twist_cp" in orig.__code__.co_varnames else orig(m,s,r)
        if cmt: d["c_max_t"]=cmt
        d["t_over_c_cp"]=np.full(35,float(np.mean(d["t_over_c_cp"])))
        d["twist_cp"]=np.zeros(ntw)
        return d
    return b

def build(cmt,cj,ratio,spar,ntw):
    ro.build_surface=mk(cmt,ntw); pr,_=ro.build_problem("const_chord",mesh,stick,regions,p0); ro.build_surface=orig
    if cj: pr.set_val("wing.taper_B",cj/ROOT)
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
    pr.run_model(); return pr,y,jj

def drag(pr):
    S=float(pr.get_val(f"{POINT}.wing.S_ref")[0])
    return q*S*sum(float(pr.get_val(f"{POINT}.wing_perf.{k}")[0]) for k in ("CDi","CDv","CDw"))

def describe(tag,y,tw):
    e=y/y.max(); d=np.diff(tw); sc=int(np.sum(np.diff(np.sign(d))!=0))
    mono=bool(np.all(d<=1e-9))
    print(f"  {tag}: sign changes {sc}   monotonic decreasing: {mono}   "
          f"peak {tw.max():+.2f} at eta {e[int(np.argmax(tw))]:.2f}   tip {tw[-1]:+.2f}")
    print("     eta :", "".join(f"{v:6.2f}" for v in e[::4]))
    print("     tw  :", "".join(f"{v:6.2f}" for v in tw[::4]))
    return sc,mono

for ntw in (5,15):
    print(f"\n{'='*86}\ntwist control points = {ntw}\n{'='*86}")
    for nm,(cmt,cj,rt) in CASES.items():
        pr,y,jj=build(cmt,cj,rt,cmt is not None,ntw)
        trim_alpha(pr,W/(q*float(pr.get_val(f"{POINT}.wing.S_ref")[0])))
        d0=drag(pr); tw0=pr.get_val("twist_abs",units="deg").copy()
        pr2,y2,_=build(cmt,cj,rt,cmt is not None,ntw)
        m=pr2.model
        m.add_design_var("wing.twist_cp",lower=-8.,upper=8.,units="deg")
        m.add_design_var("alpha",lower=-5.,upper=12.,units="deg")
        m.add_constraint("lift",equals=W,ref=W); m.add_objective("drag",ref=1e4)
        pr2.driver=om.ScipyOptimizeDriver(optimizer="SLSQP",tol=1e-7,maxiter=300,disp=False)
        pr2.setup(); pr2.run_model()
        pr2.set_val("wing.t_over_c_cp",pr.get_val("wing.t_over_c_cp"))
        if cj: pr2.set_val("wing.taper_B",cj/ROOT)
        trim_alpha(pr2,W/(q*float(pr2.get_val(f"{POINT}.wing.S_ref")[0])))
        pr2.run_driver()
        d1=drag(pr2); tw1=pr2.get_val("twist_abs",units="deg").copy()
        ok=bool(pr2.driver.result.success)
        print(f"\n{nm}: frozen {d0:.1f} N  ->  optimised {d1:.1f} N  ({d1/d0-1:+.2%})  "
              f"{'converged' if ok else 'FAILED'}")
        describe("frozen   ",y,tw0); describe("optimised",y2,tw1)
        e=y2/y2.max(); k=np.argmin(abs(e-0.48))
        print(f"     eta 0.48 (A|B junction) twist {tw1[k]:+.2f}, neighbours "
              f"{tw1[k-1]:+.2f} / {tw1[k+1]:+.2f}")
