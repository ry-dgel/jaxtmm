import jaxtmm as tmm
import pandas as pd
import numpy as np

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

mirror_ns = tmm.jnp.array(mirror_df.n.to_numpy(dtype=np.float32))
mirror_ds = tmm.jnp.array(mirror_df.d.to_numpy(dtype=np.float32))

fiber_ns = tmm.jnp.array(fiber_df.n.to_numpy(dtype=np.float32))
fiber_ds = tmm.jnp.array(fiber_df.d.to_numpy(dtype=np.float32))

lambda_test = 602E-9
td_test = 800E-9
L_test = 10E-6
R0_test = 19.8E-6
sigma_test = 0E-9

mirror = tmm.make_mirror(mirror_ds,mirror_ns,
                        indices['substrate'],
                        indices['air'],
                        lambda_test)
fiber = tmm.make_mirror(fiber_ds,fiber_ns,
                        indices['substrate'],
                        indices['air'],
                        lambda_test)
fiber = tmm.jnp.conjugate(tmm.jla.inv(fiber))

cav = tmm.make_cavity(mirror,fiber,
                      td_test,L_test,R0_test,
                      sigma_test,lambda_test,k=0)

print(mirror)
print(fiber)
print(cav)