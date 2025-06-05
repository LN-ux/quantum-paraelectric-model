import jax
import jax.numpy as jnp
import numpy as np

import netket as nk
nk.config.netket_experimental_fft_autocorrelation = True

from hamiltonian import TiO2Hamiltonian
from hilbert_space import Clock, TiO2Constraint, TiO2LocalRule
from ansatz import ResNetTransInvJastrow
from jax.nn.initializers import normal

from jax import config
config.update("jax_enable_x64", True)
print(jax.default_backend())

import flax

import optax

import os
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--jobid', dest='jobid', help="Job id")
parser.add_argument('--parameters', dest='parameters', help="Python namespace containing simulation parameters")
args = parser.parse_args()

##
jobid = args.jobid
pars = json.load(open(args.parameters))
ld = os.path.dirname(args.parameters)

N = pars['N']
n_dim = pars['n_dim']
extent = pars['extent']
n_sites = pars['n_sites']
pbc = pars['pbc']
J = pars['J']

features = pars['features']
depth = pars['depth']

n_samples = pars['n_samples']
n_chains = pars['n_chains']
sweep_factor = pars['sweep_factor']
n_sweeps = pars['n_sweeps']
n_discard_per_chain = pars['n_discard_per_chain']
n_burnin = pars['n_burnin']
chunk_size = pars['chunk_size']

n_iter = pars['n_iter']
lrate = pars['lrate']
dshift = pars['dshift']

ham_dtype = pars['ham_dtype']
sampler_dtype = pars['sampler_dtype']
model_dtype = pars['model_dtype']

##

g = nk.graph.Square(N, pbc=True)
constraint = TiO2Constraint(g)
hi = Clock(4, n_sites, constraint=constraint)
ha = TiO2Hamiltonian(hi, g, J, dtype=ham_dtype)

rule = TiO2LocalRule(graph=g)
sampler = nk.sampler.MetropolisSampler(hi, rule=rule, dtype=sampler_dtype, n_chains=n_chains, sweep_size=sweep_factor*n_sweeps)
model = ResNetTransInvJastrow(g, depth * (features,), (3, 3), jastrow_kernel_init=normal(1e-1/g.n_nodes))
vs = nk.vqs.MCState(model=model, sampler=sampler, n_samples=n_samples, chunk_size=chunk_size)

print('Number of parameters = ', vs.n_parameters)

prefix = 'ResNetTransInvJastrow'
suffix = f'.{jobid}'
log = nk.logging.JsonLog(os.path.join(ld, prefix+suffix), save_params_every=1)
model_parameters_fname = os.path.join(ld, 'vqs-'+prefix+suffix+'.mpack')
burnin = True

optimizer = optax.sgd(learning_rate=lrate)
solver = nk.optimizer.solver.svd
preconditioner=nk.optimizer.SR(diag_shift=dshift, solver=solver)
gs = nk.driver.VMC(ha, optimizer, variational_state=vs, preconditioner=preconditioner)

def get_callback(graph):
    edges = jnp.array(graph.edges())
    def _cb(step, logged_data, driver):
        acceptance = float(driver.state.sampler_state.acceptance)
        logged_data["acceptance"] = acceptance

        samples = driver.state.samples
        phis = 0.5 * np.pi * samples

        ps = jnp.abs(jnp.sum(jnp.exp(1j * phis), axis=-1)) / phis.shape[-1]
        logged_data["polarization"] = nk.stats.statistics(ps).to_dict()

        phi_ij = phis[..., edges]
        phi_i, phi_j = phi_ij[..., 0], phi_ij[..., 1]
        l_order = nk.stats.statistics(jnp.mean(jnp.cos(phi_i - phi_j), axis=-1))
        logged_data["local order"] = l_order.to_dict()
        if step % 10 == 0:
            with open(model_parameters_fname, 'wb') as file:
                file.write(flax.serialization.to_bytes(driver.state))
        return True
    return _cb


if burnin:
    print('Burn-in in progress...')
    for _ in range(n_burnin):
        vs.sample()
    print('Thermalised!')

print('Run the optimisation problem.\n Logger: '
      f'ResNetTransInvJastrow.{jobid}'
)

gs.run(n_iter=n_iter, out=log, callback=get_callback(g))

print('ResNetTransInvJastrow: ')

e_stats = vs.expect(ha)
print('Energy: ', e_stats.mean, e_stats.error_of_mean)

samples = vs.samples
phis = 0.5 * np.pi * samples

ps = jnp.abs(jnp.sum(jnp.exp(1j * phis), axis=-1)) / phis.shape[-1]
p_stats = nk.stats.statistics(ps)

print('Polarisation: ', p_stats.mean, p_stats.error_of_mean)
