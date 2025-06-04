import netket as nk
from netket.hilbert import HomogeneousHilbert, DiscreteHilbert
from netket.hilbert.constraint import DiscreteHilbertConstraint
from netket.utils import struct, HashableArray
from netket.utils.types import DType
from netket.graph import AbstractGraph
from netket.sampler import MetropolisRule
from netket.utils import dispatch
from netket.jax.sharding import sharding_decorator

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

import numpy as np
from functools import partial

def min_image_distance(dist, extent):
    return dist - extent * np.rint(dist / extent)


class TiO2Constraint(DiscreteHilbertConstraint):
    edges: HashableArray = struct.field(pytree_node=False)
    forbidden_configurations: HashableArray = struct.field(pytree_node=False)

    def __init__(self, g: AbstractGraph):
        edges = np.array(g.edges())
        x = g.positions[edges]
        dist = min_image_distance(x[:, 1, :] - x[:, 0, :], g.extent)
        forb_confs = jnp.array([3, 1]) * jnp.all(dist == [0, -1], axis=-1, keepdims=True)
        forb_confs += jnp.array([1, 3]) * jnp.all(dist == [0, 1], axis=-1, keepdims=True)
        forb_confs += jnp.array([2, 0]) * jnp.all(dist == [-1, 0], axis=-1, keepdims=True)
        forb_confs += jnp.array([0, 2]) * jnp.all(dist == [1, 0], axis=-1, keepdims=True)
        self.edges = HashableArray(edges)
        self.forbidden_configurations = HashableArray(forb_confs)

    def __call__(self, x):
        edges = np.asarray(self.edges)
        forb_confs = np.asarray(self.forbidden_configurations)
        
        return jnp.all(jnp.any(x[..., edges] != forb_confs, axis=-1), axis=-1)
    
    def __hash__(self):
        return hash(("TiO2Constraint", self.edges, self.forbidden_configurations))

    def __eq__(self, other):
        if isinstance(other, TiO2Constraint):
            # In principle, only edges matter
            return (self.edges == other.edges and 
                    self.forbidden_configurations == other.forbidden_configurations
                    )
        return False


class Clock(HomogeneousHilbert):
    def __init__(
        self,
        n_clock_states: int,
        N: int,
        *,
        constraint: DiscreteHilbertConstraint | None = None,
    ):
        assert isinstance(N, int)

        local_states = StaticRange(start=0, step=1, length=n_clock_states)
        super().__init__(local_states, N=N, constraint=constraint)


@dispatch.dispatch
def random_state(hilb: Clock,
                 constraint: TiO2Constraint,
                 key,
                 batches: int,
                 *,
                 dtype=None
                ):
    x = jax.random.choice(key, np.array(hilb.local_states, dtype=dtype), shape=(batches, 1))
    return jnp.repeat(x, hilb.size, axis=-1)

def f(x, y):
    if (x, y) == (-1, 0):
        return 0
    elif (x, y) == (1, 0):
        return 2
    elif (x, y) == (0, -1):
        return 1
    elif (x, y) == (0, 1):
        return 3
    else:
        raise ValueError(f"Invalid input: ({x}, {y}) is not a valid pair.")

def comp_forbidden_conf(i, j, graph):
    xi, xj = graph.positions[[i, j]]
    dist = min_image_distance(xj - xi, graph.extent)
    return f(*dist)

class TiO2LocalRule(MetropolisRule):
    neighbours: jax.Array
    forbidden_configurations: jax.Array
    directions: jax.Array

    def __init__(
        self,
        *,
        graph: AbstractGraph | None = None,
    ):
        # For every site, enumerate all neighbours
        neigh = [[] for _ in range(graph.n_nodes)]
        # For every neighbour, specify the polar coordinate
        forb = [[] for _ in range(graph.n_nodes)]
        for (i, j) in graph.edges():
            neigh[i].append(j)
            forb[i].append(comp_forbidden_conf(i, j, graph))
            neigh[j].append(i)
            forb[j].append(comp_forbidden_conf(j, i, graph))
        self.neighbours = jnp.array(neigh)
        self.forbidden_configurations = jnp.array(forb)
        self.directions = jnp.mod(self.forbidden_configurations + 2, 4)

    def transition(self, sampler, machine, params, sampler_state, key, xs):
        n_chains = xs.shape[0]
        N = xs.shape[-1]
        k1, k2 = jax.random.split(key)
        cells = jax.random.randint(k1, shape=n_chains, minval=0, maxval=N)
        keys = jax.random.split(k2, num=n_chains)

        @partial(sharding_decorator, sharded_args_tree=(True, True, True))
        @jax.vmap
        def _update_samples(key, x, cell):
            mask = x[self.neighbours[cell]] != self.forbidden_configurations[cell]
            mask *= self.directions[cell] != x[cell]
            # jax.debug.print('Initial configuration {x}, cell {y}', x=x, y=cell)
            # jax.debug.print('mask {x}, xi {y}, xj {z}', x=mask, y=x[cell], z=x[self.neighbours[cell]])

            new_cell_x = jax.random.choice(
                key,
                a=self.directions[cell],
                p=mask,
                replace=True,
            )
            prop_x = x.at[cell].set(new_cell_x)
            any_mask = jnp.any(mask)
            prop_x = prop_x * any_mask
            
            # jax.debug.print('Proposed configuration {x}', x=x)
            log_prob_corr = jnp.log(any_mask)
            return prop_x.astype(sampler.dtype), log_prob_corr
        
        return _update_samples(keys, xs, cells)

