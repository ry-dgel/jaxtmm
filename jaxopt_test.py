from jax import jit,grad,vmap
import jax.numpy as jnp
import jaxopt as jopt
from jaxopt.projection import projection_box
import matplotlib.pyplot as plt
@jit
def lorentz(x,amp,mean,gamma,offset):
    delta = x-mean
    return amp * gamma**2/(gamma**2+4*delta**2) + offset

@jit
def lor(x):
    return lorentz(x,-0.8,10.0E-6,10E-10,1.0)

x_init = 10.1E-6
x_min = 9.5E-6
x_max = 10.5E-6
tol = 1E-8

xs = jnp.linspace(x_min,x_max,10000)
plt.plot(xs,lor(xs))
plt.scatter(x_init,lor(x_init),label='initial')

# Projected Gradient
solver = jopt.ProjectedGradient(lor,projection_box,tol=tol)
params = solver.run(x_init,
                    hyperparams_proj=(x_min,x_max))
plt.scatter(params.params,lor(params.params),
            label='projected',zorder=10)

# BFGS
# Stepsize avoids pinging off to large values
solver = jopt.BFGS(lor,tol=tol,stepsize=1E-9)
params = solver.run(x_init)
plt.scatter(params.params,lor(params.params),label='bfgs',zorder=10)

# Scipy Bounded LBFGSB
solver = jopt.ScipyBoundedMinimize(fun=lor,
                                   tol=tol,
                            method="l-bfgs-b")
params = solver.run(x_init,bounds=(x_min,x_max))
plt.scatter(params.params,lor(params.params),label='scipy - lbfgsb')
plt.legend()

# Scipy Bounded BFGS
solver = jopt.ScipyMinimize(fun=lor,
                            tol=tol,
                            method="bfgs")
params = solver.run(x_init)
plt.scatter(params.params,
            lor(params.params),
            label='scipy - bfgs',
            marker='x')
plt.legend()

# Scipy Newton
solver = jopt.ScipyMinimize(fun=lor,
                            tol=tol,
                            method="newton-cg")
params = solver.run(x_init)
plt.scatter(params.params,
            lor(params.params),
            label='scipy - newton',
            marker='+')
plt.legend()