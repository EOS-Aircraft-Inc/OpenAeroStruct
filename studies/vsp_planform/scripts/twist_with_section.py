"""Does twist still buy anything once the section is fixed? MTOW, span 118 ft."""
import sys, numpy as np, openmdao.api as om
sys.path.insert(0,"/home/alex/repos/OpenAeroStruct")
from scipy.optimize import lsq_linear
from studies.vsp_planform import config, param
param.REGION_A_RULE["const_chord"]="preserved"
import studies.vsp_planform.run_opt as ro
from studies.vsp_planform.run_opt import POINT, load_baseline, trim_alpha

W=382547.0; IN=0.0254; ROOT=105.0; SPAR=7.0
SECT={"as-built":(None,None,0.527),"fx77w270s":(0.34,56.8,0.676)}
q=0.5*config.RHO*config.V_MS**2
mesh,stick,regions,p0,_,_=load_baseline("const_chord",config.N_SPANWISE_HALF,9)
ys=np.abs(stick.le[:,1])*config.SCALE; orig=ro.build_surface
def mk(c):
    def b(m,s,r):
        d=orig(m,s,r)
        if c: d["c_max_t"]=c
        d["t_over_c_cp"]=np.full(35,float(np.mean(d["t_over_c_cp"]))); return d
    return b

def setup(cmt,cj,ratio,spar):
    ro.build_surface=mk(cmt); pr,_=ro.build_problem("const_chord",mesh,stick,regions,p0); ro.build_surface=orig
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
    pr.run_model(); return pr,y

def drag(pr):
    S=float(pr.get_val(f"{POINT}.wing.S_ref")[0])
    return q*S*sum(float(pr.get_val(f"{POINT}.wing_perf.{k}")[0]) for k in ("CDi","CDv","CDw"))

print(f"{'case':>40} {'twist':>8} {'drag N':>9} {'root':>7} {'tip':>7}")
print("-"*78)
res={}
for nm,(cmt,cj,rt) in SECT.items():
    pr,y=setup(cmt,cj,rt,cmt is not None)
    trim_alpha(pr,W/(q*float(pr.get_val(f"{POINT}.wing.S_ref")[0])))
    tw=pr.get_val("twist_abs",units="deg")
    d0=drag(pr); res[nm]=dict(fixed=d0,tw0=tw.copy(),y=y.copy())
    print(f"{nm+' , twist frozen':>40} {'frozen':>8} {d0:9.1f} {tw[0]:+7.2f} {tw[-1]:+7.2f}")

    # now optimise twist + alpha at fixed planform and section
    pr2,_=None,None
    pr2,y2=setup(cmt,cj,rt,cmt is not None)
    m=pr2.model
    m.add_design_var("wing.twist_cp",lower=-8.0,upper=8.0,units="deg")
    m.add_design_var("alpha",lower=-5.0,upper=12.0,units="deg")
    m.add_constraint("lift",equals=W,ref=W)
    m.add_objective("drag",ref=1e4)
    pr2.driver=om.ScipyOptimizeDriver(optimizer="SLSQP",tol=1e-7,maxiter=200,disp=False)
    pr2.setup(); pr2.run_model()
    # restore the t/c we solved for (setup() resets)
    pr2.set_val("wing.t_over_c_cp",pr.get_val("wing.t_over_c_cp"))
    if cj: pr2.set_val("wing.taper_B",cj/ROOT)
    trim_alpha(pr2,W/(q*float(pr2.get_val(f"{POINT}.wing.S_ref")[0])))
    pr2.run_driver()
    tw2=pr2.get_val("twist_abs",units="deg"); d1=drag(pr2)
    ok=bool(pr2.driver.result.success)
    print(f"{nm+' , twist optimised':>40} {'opt':>8} {d1:9.1f} {tw2[0]:+7.2f} {tw2[-1]:+7.2f}"
          f"   {'converged' if ok else 'FAILED'}   delta {d1-d0:+.1f} N ({d1/d0-1:+.2%})")
    res[nm]["opt"]=d1; res[nm]["tw1"]=tw2.copy()
np.save("studies/vsp_planform/out/logs/twist_result.npy",res,allow_pickle=True)
