import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax
from jax import jit,grad,vmap
from jax.lax import reduce
from jax.tree_util import Partial
import jaxopt as jopt
from jaxopt.projection import projection_box, projection_non_negative
from itertools import chain

jax.config.update("jax_enable_x64", True)
N_DIAMOND = 2.417
N_SUB = 1.4633
N_GLASS = 1.482
N_AIR = 1

@jit
def mat_prod(mats):
    out = mats[0]
    for mat in mats[1:]:
        out = jnp.matmul(mat,out)
    return out

@jit
def intersperse_mat_prod(As,Bs):
    out = As[0]
    mats = jnp.array(list(chain(*zip(Bs,As[1:],strict=True))))
    for mat in mats:
        out = jnp.matmul(mat, out)
    return out

@jit 
def distance_mat(d,n,wl):
    k = 2 * jnp.pi * n / wl
    phi = k * d
    phasor = jnp.exp(-1j * phi)
    M = jnp.array([[phasor,0],[0,phasor.conjugate()]])
    return M

@jit
def interface_mat(ni,nj):
    ni = ni.real
    nj = nj.real
    a = (ni+nj)/(2*nj)
    b = (-ni+nj)/(2*nj)
    M = jnp.array([[a,b],[b,a]])
    return M

@jit
def curved_diamond_prop(td,L,R0,k,wl):
    """
    Propagation matrix through the diamond in a curved hybrid cavity.
    """
    nd = 2.417
    na = 1.0
    z1 = td 
    ta = L - td
    w0d = jnp.sqrt(wl/jnp.pi) * ((ta+td/nd)*(R0-(ta+td/nd)))**(1/4)
    phasor = jnp.exp(-1j*(2*nd*jnp.pi*z1/wl - (1+k) * jnp.arctan(z1*wl/(nd*jnp.pi*w0d**2))))
    M = jnp.array([[phasor,0],[0,phasor.conjugate()]])
    return M

@jit 
def curved_air_prop(td,L,R0,k,wl):
    """
    Propagation matrix through the air in a curved hybrid cavity.
    """
    nd = 2.417
    na = 1.0
    z1 = td 
    z2 = td*(1-1/nd)
    ta = L - td
    w0a = jnp.sqrt(wl/jnp.pi)*((ta+td/nd)*(R0-(ta+td/nd)))**(1/4)
    delta_gouy = jnp.arctan((L-z2)*wl/(na*jnp.pi*w0a**2)) - jnp.arctan((-z2+z1)*wl/(na*jnp.pi*w0a**2))
    phasor = jnp.exp(-1j*(2*na*jnp.pi*(L-z1)/wl - (1+k) * delta_gouy))
    M = jnp.array([[phasor,0],[0,phasor.conjugate()]])
    return M

@jit
def make_mirror(ds,ns,n_left,n_right,wl):
    """
    Generate the full matrix for a sequence of dielectric layers. 
    Computes the interface and propagation matrices and multiplies them together.
    """
    fn = vmap(Partial(distance_mat,wl=wl))
    dist_mats = fn(ds,ns)
    n_tot = jnp.concatenate((jnp.array([n_left]),
                             ns,
                             jnp.array([n_right])))
    fn = vmap(interface_mat)
    int_mats = fn(n_tot[:-1],n_tot[1:])

    return intersperse_mat_prod(int_mats,dist_mats)

@jit
def rij(ni,nj,wl,sigma=None):
    """
    Modified reflection amplitude coefficient for a rough interface.
    """
    # Reflection coefficient
    ni = ni.real
    nj = nj.real
    r0 = (ni-nj)/(ni+nj)
    if sigma is None:
        return r0
    return r0*jnp.exp(-2*(2*jnp.pi*sigma*nj/wl)**2)

@jit
def tij(ni,nj,wl,sigma):
    """
    Modified transmission amplitude coefficient for a rough interface.
    """
    ni = ni.real
    nj = nj.real
    t0 = 2*ni/(ni+nj)
    if sigma is None:
        return t0
    return t0*jnp.exp(-1/2*(2*jnp.pi*sigma*(ni-nj)/wl)**2)

@jit
def interface_air_diamond(wl,sigma):
    nd = N_DIAMOND
    na = 1.0

    # Modified Fresnel Coefficients
    # for rough surfaces
    rad = rij(na,nd,wl,sigma)
    rda = rij(nd,na,wl,sigma)
    tad = tij(na,nd,wl,sigma)
    tda = tij(nd,na,wl,sigma)

    # Compute interface matrix
    a = tad - rad*rda/tda
    c = rda/tda
    b = -rad/tda
    d = 1/tda

    M = jnp.array([[a,b],[c,d]])
    return M

@jit 
def interface_diamond_air(wl,sigma):
    nd = N_DIAMOND
    na = 1.0
    
    # Modified Fresenel Coefficients
    # for rough surfaces
    rda = rij(nd,na,wl,sigma)
    rad = rij(na,nd,wl,sigma)
    tda = tij(nd,na,wl,sigma)
    tad = tij(na,nd,wl,sigma)
    
    # Compute interface matrix
    a = tda - rda*rad/tad
    c = rad/tad
    b = -rda/tad
    d = 1/tad

    M = jnp.array([[a,b],[c,d]])
    return M

@jit
def make_cavity(l_mirror,r_mirror,td,L,R0,sigma,wl,k=0):
    La = curved_air_prop(td,L,R0,k,wl)
    Dad  = interface_air_diamond(wl,sigma)
    Ld = curved_diamond_prop(td,L,R0,k,wl)
    Dda  = interface_diamond_air(wl,sigma)
    total_mats = jnp.array([l_mirror,Dad,Ld,Dda,La,r_mirror])
    cavity = mat_prod(total_mats)
    return cavity

@jit
def make_mirrors_and_cavity(l_ns,l_ds,r_ns,r_ds,td,L,R0,sigma,wl,k=0):
    l_mirror = make_mirror(l_ds,l_ns,N_SUB,N_AIR,wl)
    r_mirror = jnp.conj(jla.inv(make_mirror(r_ds,r_ns,N_GLASS,N_AIR,wl)))
    return make_cavity(l_mirror,r_mirror,td,L,R0,sigma,wl,k)
@jit
def trans_refl_from_mat(matrix,n1=1.0,n2=1.0):
    n1 = n1.real
    n2 = n2.real

    term1 = matrix[1,0]/matrix[1,1]
    term2 = matrix[0,0]-matrix[0,1]*term1

    T = n2/n1 * jnp.abs(term2)**2 
    R = jnp.abs(term1)**2

    return T,R

@jit
def trans_from_mat(matrix,n1=1.0,n2=1.0):
    n1 = n1.real
    n2 = n2.real

    term1 = matrix[1,0]/matrix[1,1]
    term2 = matrix[0,0]-matrix[0,1]*term1

    T = n2/n1 * jnp.abs(term2)**2 

    return T

@jit
def trans_full(l_ns,l_ds,r_ns,r_ds,td,L,R0,sigma,wl,k=0):
    return trans_from_mat(make_mirrors_and_cavity(l_ns,l_ds,r_ns,r_ds,td,L,R0,sigma,wl,k))

@jit
def obj_trans_from_mat(matrix):
    term1 = matrix[1,0]/matrix[1,1]
    term2 = matrix[0,0]-matrix[0,1]*term1

    return -jnp.abs(term2)**2

@jit
def obj_refl_from_mat(matrix):
    term1 = matrix[1,0]/matrix[1,1]
    return jnp.abs(term1)**2

@jit
def objective_td(td,l_mirror,r_mirror,L,R0,sigma,wl,k=0):
    cav = make_cavity(l_mirror,r_mirror,td,L,R0,sigma,wl,k=0)
    return obj_refl_from_mat(cav)
@jit
def objective_L(L,l_mirror,r_mirror,td,R0,sigma,wl,k=0):
    cav = make_cavity(l_mirror,r_mirror,td,L,R0,sigma,wl,k=0)
    return obj_refl_from_mat(cav)
@jit
def objective_wl(wl,l_ns,l_ds,r_ns,r_ds,td,L,R0,sigma,k=0):
    cav = make_mirrors_and_cavity(l_ns,l_ds,r_ns,r_ds,td,L,R0,sigma,wl,k=0)
    return obj_refl_from_mat(cav)

# # Bisetion kind of works, but because of gradient sign and bounds
# # problems, we can't quite nail every mode.
# def find_resonance_in_L(l_ns,l_ds,r_ns,r_ds,td,R0,sigma,wl,k,
#                         Lmin=0,Lmax=jnp.inf):
#     dRdL = grad(objective_L)
#     l_mirror = make_mirror(l_ds,l_ns,N_SUB,N_AIR,wl)
#     r_mirror = jnp.conj(jla.inv(make_mirror(r_ds,r_ns,N_GLASS,N_AIR,wl)))
#     solver = jopt.Bisection(dRdL, lower=Lmin,upper=Lmax,tol=1E-3,maxiter=1000,
#                             check_bracket=False)
#     sol = solver.run(l_mirror = l_mirror,
#                      r_mirror = r_mirror,
#                      td = td,
#                      R0 = R0,
#                      sigma = sigma,
#                      wl = wl,
#                      k = k)
#     return sol.params

# @jit
# def find_resonance_in_L(l_ns,l_ds,r_ns,r_ds,td,R0,sigma,wl,k,
#                         Linit,Lmin=0,Lmax=jnp.inf):
#     solver = jopt.ProjectedGradient(objective_L,projection_non_negative,#projection_box,
#                                     stepsize=1e-9,
#                                     tol=1E-10,
#                                     acceleration=True,
#                                     verbose=True,
#                                     maxiter=10000,
#                                     maxls=100)
#     l_mirror = make_mirror(l_ds,l_ns,N_SUB,N_AIR,wl)
#     r_mirror = jnp.conj(jla.inv(make_mirror(r_ds,r_ns,N_GLASS,N_AIR,wl)))
#     res = solver.run(init_params = Linit,
#                      hyperparams_proj=(Lmin,Lmax),
#                      l_mirror = l_mirror,
#                      r_mirror = r_mirror,
#                      td=td,
#                      R0 = R0,
#                      sigma = sigma,
#                      wl = wl,
#                      k = k)
#     return res.params, res

# def find_resonance_in_L(l_ns,l_ds,r_ns,r_ds,td,R0,sigma,wl,k,
#                         Linit,Lmin=0,Lmax=jnp.inf):
#     solver = jopt.GradientDescent(objective_L,
#                                   maxiter=10000,tol=1E-8,
#                                   verbose=True)
#     l_mirror = make_mirror(l_ds,l_ns,N_SUB,N_AIR,wl)
#     r_mirror = jnp.conj(jla.inv(make_mirror(r_ds,r_ns,N_GLASS,N_AIR,wl)))
#     res = solver.run(init_params = jnp.array(Linit),
#                      l_mirror = l_mirror,
#                      r_mirror = r_mirror,
#                      td=td,
#                      R0 = R0,
#                      sigma = sigma,
#                      wl = wl,
#                      k = k)
#     return res.params, res

def find_resonance_in_L(l_ns,l_ds,r_ns,r_ds,td,R0,sigma,wl,k,
                        Linit,Lmin=0,Lmax=jnp.inf):
    l_mirror = make_mirror(l_ds,l_ns,N_SUB,N_AIR,wl)
    r_mirror = jnp.conj(jla.inv(make_mirror(r_ds,r_ns,N_GLASS,N_AIR,wl)))
    solver = jopt.ScipyBoundedMinimize(fun=objective_L,
                                       method="l-bfgs-b",
                                       tol=1E-8)
    res = solver.run(init_params = Linit,
                     bounds = (Lmin,Lmax),
                     l_mirror = l_mirror,
                     r_mirror = r_mirror,
                     td=td,
                     R0 = R0,
                     sigma = sigma,
                     wl = wl,
                     k = k)
    return res.params

def Ls_from_wls(l_ns,l_ds,r_ns,r_ds,td,R0,sigma,wls,k,m):
    Linits = wls/2*m
    Lmins  = wls/2*(m-0.5)
    Lmaxs  = wls/2*(m+0.5)

    output = jnp.array([find_resonance_in_L(l_ns,l_ds,r_ns,r_ds,
                                        td,R0,sigma,
                                        wls[i],k,
                                        Linits[i],Lmins[i],Lmaxs[i]) 
                        for i in range(len(wls))])
    return output

    # It'd be great if we could get this to work
    # But the bounds conversion seems to go wrong...
    # This is due to internal JaxOpt problems that they hope to fix.
    # find_L = vmap(find_resonance_in_L,
    #               in_axes = [None,None,None,None,None,None,None,0,None,0,0])
    # return find_L(l_ns,l_ds,r_ns,r_ds,td,R0,sigma,wls,k,Lmins,Lmaxs)

# I'm not sure if we ever actually want to do this?
# @jit
# def find_resonance_in_td(l_ns,l_ds,r_ns,r_ds,L,R0,sigma,wl,k,
#                          tdinit,tdmin=0,tdmax=jnp.inf):
#     solver = jopt.ProjectedGradient(objective_td,projection_box,
#                                     stepsize=1E-9)
#     res = solver.run(tdinit,
#                      l_ns = l_ns,
#                      l_ds = l_ds,
#                      r_ns = r_ns,
#                      r_ds = r_ds,
#                      L = L,
#                      R0 = R0,
#                      sigma = sigma,
#                      wl = wl,
#                      k = k,
#                      hyperparams_proj=(tdmin,tdmax))
#     return res.params,res

# Old version of wl fit using ProjectionGradient,
# Never seemed to actually work... Would be nice if it did
# @jit
# def find_resonance_in_wl(l_ns,l_ds,r_ns,r_ds,td,L,R0,sigma,k,
#                          wlinit,wlmin=0,wlmax=jnp.inf):
#     solver = jopt.ProjectedGradient(objective_wl,projection_box,
#                                     stepsize=1E-9)
#     res = solver.run(init_params=wlinit,
#                      l_ns = l_ns,
#                      l_ds = l_ds,
#                      r_ns = r_ns,
#                      r_ds = r_ds,
#                      td=td,
#                      L = L,
#                      R0 = R0,
#                      sigma = sigma,
#                      k = k,
#                      hyperparams_proj=(wlmin,wlmax))
#     return res.params,res
def find_resonance_in_wl(l_ns,l_ds,r_ns,r_ds,td,L,R0,sigma,k,
                         wlinit,wlmin=0,wlmax=jnp.inf):
    solver = jopt.ScipyBoundedMinimize(fun=objective_wl,
                                       method="l-bfgs-b",
                                       tol=1E-13)
    res = solver.run(init_params = wlinit,
                     bounds=(wlmin,wlmax),
                     l_ns = l_ns,
                     l_ds = l_ds,
                     r_ns = r_ns,
                     r_ds = r_ds,
                     td=td,
                     L = L,
                     R0 = R0,
                     sigma = sigma,
                     k = k)
    return res.params

def wls_from_Ls(l_ns,l_ds,r_ns,r_ds,td,R0,sigma,Ls,k,m):
    wlinits = Ls/m * 2
    wlmaxs  = Ls/(m-0.5) * 2
    wlmins  = Ls/(m+0.5) * 2

    output = jnp.array([find_resonance_in_wl(l_ns,l_ds,r_ns,r_ds,
                                        td,R0,sigma,
                                        Ls[i],k,
                                        wlinits[i],wlmin=wlmins[i],wlmax=wlmaxs[i]) 
                        for i in range(len(Ls))])
    return output

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    indices = {"substrate" : 1.4633,
               "Ta2O5"     : 2.125,
               "SiO2"      : 1.482,
               "diamond"   : 2.417,
               "air"       : 1.00}
    lambda_ref = 615E-9

    mirror_df = pd.read_csv("flatMC3.csv",index_col=0)
    fiber_df = pd.read_csv("fiberC3.csv",index_col=0)
    for df in [mirror_df,fiber_df]:
        df['n'] = list(map(indices.get,df.Material))
        df['d'] = df['Length [lambda/4n]'] * lambda_ref /  (4 * df['n'])

    mirror_ns = jnp.array(mirror_df.n.to_numpy(dtype=float))
    mirror_ds = jnp.array(mirror_df.d.to_numpy(dtype=float))

    fiber_ns = jnp.array(fiber_df.n.to_numpy(dtype=float))
    fiber_ds = jnp.array(fiber_df.d.to_numpy(dtype=float))

    lambda_test = 602E-9
    td_test = 800E-9
    L_test = 10E-6
    R0_test = 19.8E-6
    sigma_test = 0E-9

    # Testing Mirrors
    mirror = make_mirror(mirror_ds,mirror_ns,
                         indices['substrate'],
                         indices['air'],
                         lambda_test)
    fiber = make_mirror(fiber_ds,fiber_ns,
                        indices['substrate'],
                        indices['air'],
                        lambda_test)
    fiber = jnp.conjugate(jla.inv(fiber))

    # Testing Cavity
    # cav = make_cavity(mirror,fiber,
    #                   td_test,L_test,R0_test,
    #                   sigma_test,lambda_test,k=0)

    # T,R = trans_refl_from_mat(cav)

    # Mode Resonances
    # wls = jnp.linspace(500E-9,700E-9,100000)
    # def cav_T(wl):
    #     c = make_mirrors_and_cavity(mirror_ns,mirror_ds,
    #                                 fiber_ns,fiber_ds,
    #                                 td_test,L_test,R0_test,sigma_test,
    #                                 wl,k=0)
    #     T,R = trans_refl_from_mat(c)
    #     return T
    # ct = vmap(cav_T)
    # Ts = ct(wls)

    # print(mirror)
    # print(jla.det(mirror))
    # print(fiber)
    # print(jla.det(fiber))
    # print(cav)
    # print(jla.det(cav))
    # print(T,R)
    
    # Testing single Resonances
    # plt.plot(wls,Ts)
    # print(find_resonance_in_wl(mirror_ns,mirror_ds,fiber_ns,fiber_ds,td_test,10E-6,R0_test,0E-9,0,
    #                            wlinit=620E-9,wlmin=609E-9,wlmax=629E-9))
    # print(find_resonance_in_L(mirror_ns,mirror_ds,fiber_ns,fiber_ds,td_test,R0_test,0E-9,602.3E-9,0,
    #                           Linit=lambda_test/2*35,Lmin=lambda_test/2*34.6,Lmax=lambda_test/2*35.4))

    # Testing Bisection
    # m = 35
    # Lmin = lambda_test/2*(m-0.18)*1E9
    # Lmax = lambda_test/2*(m+0.05)*1E9
    # Ls = jnp.linspace(Lmin,Lmax,10000)
    # vR = vmap(objective_L,[0,None,None,None,None,None,None,None])
    # Rs = vR(Ls,mirror,fiber,td_test,R0_test,sigma_test,lambda_test,0)
    # dR = grad(objective_L)
    # vdR = vmap(dR,[0,None,None,None,None,None,None,None])
    # plt.plot(Ls,Rs-1)
    # Linit = lambda_test/2*m
    # plt.scatter(Linit,objective_L(Linit,mirror,fiber,td_test,R0_test,sigma_test,lambda_test,0))
    # dRdL = vdR(Ls,mirror,fiber,td_test,R0_test,sigma_test,lambda_test,0)
    # plt.scatter(Ls[dRdL>=0],dRdL[dRdL>=0])
    # plt.scatter(Ls[dRdL<0],dRdL[dRdL<0])

    # solver = jopt.Bisection(dR,lower=Lmin,upper=Lmax,tol=1E-11,maxiter=1000)
    # sol = solver.run(l_mirror = mirror,
    #                  r_mirror = fiber,
    #                  td=td_test,
    #                  R0 = R0_test,
    #                  sigma = sigma_test,
    #                  wl = lambda_test,
    #                  k = 0)
    # print(sol)
    # plt.scatter(sol.params,objective_L(sol.params,mirror,fiber,td_test,R0_test,sigma_test,lambda_test,0))
    # plt.figure()
    # Ls = jnp.linspace(sol.params*0.99,sol.params*1.01,10000)
    # Rs = vR(Ls,mirror,fiber,td_test,R0_test,sigma_test,lambda_test,0)
    # plt.plot(Ls,Rs)
    # plt.scatter(sol.params,objective_L(sol.params,mirror,fiber,td_test,R0_test,sigma_test,lambda_test,0))
    
    # # Testing Optimization Routines
    # m = 35
    # Lmin = lambda_test/2*(m-0.5)*1E9
    # Lmax = lambda_test/2*(m+0.5)*1E9
    # Ls = jnp.linspace(Lmin,Lmax,10000)
    # # solver = jopt.ProjectedGradient(objective_L,projection_box,
    # #                                 stepsize=100,
    # #                                 tol=1E-11,
    # #                                 acceleration=False,
    # #                                 verbose=True,
    # #                                 maxiter=100,
    # #                                 maxls=10000)
    # print(sol.params)
    # solver = jopt.LBFGS(objective_L,
    #                     # max_stepsize=5,
    #                     # min_stepsize=1E-3,
    #                     maxiter=1000,
    #                     maxls=100,
    #                     condition = 'armijo',
    #                     tol=1E-12,
    #                     stop_if_linesearch_fails=False)
    # solver = jopt.ProjectedGradient(objective_L,projection_box,
    #                                 verbose=True,tol=1E-13,
    #                                 acceleration=True,
    #                                 decrease_factor=0.1,
    #                                 stepsize=1E-3)
    # sol2 = solver.run(init_params = jnp.array(Linit*1E9),
    #                                 hyperparams_proj=(Lmin,Lmax),
    #                                 l_mirror = mirror,
    #                                 r_mirror = fiber,
    #                                 td=td_test,
    #                                 R0 = R0_test,
    #                                 sigma = sigma_test,
    #                                 wl = lambda_test,
    #                                 k = 0)
    # print(sol2.params)
    # print(sol2.state)
    # plt.scatter(sol2.params,objective_L(sol2.params,mirror,fiber,td_test,R0_test,sigma_test,lambda_test,0))
    # state = solver.init_state(init_params = jnp.array(Linit*1E9),#sol.params,
    #                           hyperparams_proj=(Lmin,Lmax),
    #                           l_mirror = mirror,
    #                           r_mirror = fiber,
    #                           td=td_test,
    #                           R0 = R0_test,
    #                           sigma = sigma_test,
    #                           wl = lambda_test,
    #                           k = 0)
    # ps = [Linit*1E9]
    # states = [state]
    # for i in range(50):
    #     print(i,end='\r')
    #     p,s = solver.update(params=ps[-1],state=states[-1],
    #                         hyperparams_proj=(Lmin,Lmax),
    #                         l_mirror = mirror,
    #                         r_mirror = fiber,
    #                         td=td_test,
    #                         R0 = R0_test,
    #                         sigma = sigma_test,
    #                         wl = lambda_test,
    #                         k = 0)
    #     ps.append(p)
    #     states.append(s)
    # plt.figure()
    # Rs = vR(Ls,mirror,fiber,td_test,R0_test,sigma_test,lambda_test,0)
    # dRs = vdR(Ls,mirror,fiber,td_test,R0_test,sigma_test,lambda_test,0)
    # plt.scatter(Ls[dRs>0],Rs[dRs>0])
    # plt.scatter(Ls[dRs==0],Rs[dRs==0])
    # plt.scatter(Ls[dRs<0],Rs[dRs<0])
    # plt.scatter(jnp.array(ps),vR(jnp.array(ps),mirror,fiber,td_test,R0_test,sigma_test,lambda_test,0),
    #             c=range(len(ps)),cmap='viridis')
    # plt.xlim(Lmin,Lmax)
    # plt.show()

    # # Testing Mode Results
    m = 35
    wlss = jnp.linspace(600E-9,650E-9,100)
    Ls = Ls_from_wls(mirror_ns,mirror_ds,fiber_ns,fiber_ds,td_test,R0_test,sigma_test,wlss,0,m)
    Lis = jnp.linspace(10.4E-6,11.5E-6,100)
    wlis = wls_from_Ls(mirror_ns,mirror_ds,fiber_ns,fiber_ds,td_test,R0_test,sigma_test,Lis,0,m)
    plt.plot(wlss,Ls)
    plt.plot(wlis,Lis,linestyle='--')

    plt.plot(wlss,wlss/2*(m-0.5))
    plt.plot(wlss,wlss/2*(m+0.5))