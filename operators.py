from netket.operator import DiscreteJaxOperator
from netket.graph import AbstractGraph
from netket.utils.types import DType
from netket.jax import canonicalize_dtypes
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class TiO2MomentumMomentum(DiscreteJaxOperator):
    def __init__(
            self, 
            hilbert: Clock, 
            graph: AbstractGraph,
            t: float = 1.0,
            dtype: DType | None = None
            ):
        # TODO: add checks 
        super().__init__(hilbert)

        t = jnp.asarray(t)
        dtype = canonicalize_dtypes(float, t, dtype=dtype)
        self._dtype = dtype

        self._t = t.astype(dtype=dtype)

        self._n_sites = hilbert.size
        self._max_conn = (2 * self._n_sites) ** 2

        self._edges = jnp.asarray(graph.edges(), dtype=np.intp)
        # TODO: I should consider also the case in which the Hilbert
        # space is not constrained.
        self._forbidden_configurations = jnp.asarray(
            hilbert.constraint.forbidden_configurations, 
            dtype=np.intp
            )
    
    @property
    def is_hermitian(self):
        return True

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return self._max_conn
    
    @property
    def edges(self) -> np.ndarray:
        """The edges of the underlying matrix."""
        return self._edges
    
    @property
    def n_sites(self):
        """The number of lattice sites."""
        return self._n_sites
    
    @property
    def t(self):
        """The bond hopping amplitude."""
        return self._t
    
    @property
    def dtype(self):
        """The dtype of the matrix elements."""
        return self._dtype
    
    def tree_flatten(self):
        data = (self.t, self.edges, self._forbidden_configurations)
        metadata = {"hilbert": self.hilbert}
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        t, edges, forb = data
        hi = metadata["hilbert"]
        res = cls(hi, graph=nk.graph.Graph([(0, 0)]))
        res._t = t
        res._edges = edges
        res._forbidden_configurations = forb
        return res
    
    def conjugate(self, *, concrete=True):
        # if real
        if isinstance(self.dtype, float):
            return self
        else:
            raise NotImplementedError
    
    @jax.jit
    def get_conn_padded(self, x):
        # Kinetic term (clockwise turn)

        xp1 = jnp.mod(x[..., None, :] - np.eye(self.n_sites), 4)
        mel1 = -1j * jnp.ones((*x.shape[:-1], self.n_sites))

        # Kinetic term (counter-clockwise turn)
        xp2 = jnp.mod(x[..., None, :] + np.eye(self.n_sites), 4)
        mel2 = 1j * jnp.ones((*x.shape[:-1], self.n_sites))

        # Concatenate everything
        mels = jnp.concatenate([mel1, mel2], axis=-1)
        xps = jnp.concatenate([xp1, xp2], axis=-2)

        # Kinetic term (clockwise turn)
        xp1 = jnp.mod(xps[..., None, :] - np.eye(self.n_sites), 4)
        mel1 = -1j * jnp.ones((*xps.shape[:-1], self.n_sites)) * mels[..., :, None]

        # Kinetic term (counter-clockwise turn)
        xp2 = jnp.mod(xps[..., None, :] + np.eye(self.n_sites), 4)
        mel2 = 1j * jnp.ones((*xps.shape[:-1], self.n_sites)) * mels[..., :, None]
        
        # Concatenate everything

        mels = jnp.concatenate([mel1, mel2], axis=-1).reshape(*x.shape[:-1], -1)
        xps = jnp.concatenate([xp1, xp2], axis=-2).reshape(*x.shape[:-1], -1, x.shape[-1])

        # Check whether for any connected element the new configuration on an edge is compatible 
        # with the constraint (we check that the configuration (x_i, x_j) on edge <i,j> is not forbidden)
        mask = jnp.all(jnp.any(xps[..., self.edges] != self._forbidden_configurations, axis=-1), axis=-1)
        
        # Zero out all matrix elements with unphysical configurations, zero out those configurations as well
        # since (0, 0, ..., 0) is a valid configuration that we use a placeholder.
        return jnp.astype(xps * mask[..., None], x.dtype), self.t * mels * mask


@register_pytree_node_class
class TiO2KineticEnergy(DiscreteJaxOperator):
    def __init__(
            self, 
            hilbert: Clock, 
            graph: AbstractGraph,
            J: float,
            t: float = 1.0,
            dtype: DType | None = None
            ):
        # TODO: add checks 
        super().__init__(hilbert)

        J, t = jax.tree_util.tree_map(jnp.asarray, (J, t))
        dtype = canonicalize_dtypes(float, J, t, dtype=dtype)
        self._dtype = dtype

        self._J = J.astype(dtype=dtype)
        self._t = t.astype(dtype=dtype)

        self._n_sites = hilbert.size
        self._max_conn = 2 * self._n_sites

        self._edges = jnp.asarray(graph.edges(), dtype=np.intp)
        # TODO: I should consider also the case in which the Hilbert
        # space is not constrained.
        self._forbidden_configurations = jnp.asarray(
            hilbert.constraint.forbidden_configurations, 
            dtype=np.intp
            )
    
    @property
    def is_hermitian(self):
        return True

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return self._max_conn
    
    @property
    def edges(self) -> np.ndarray:
        """The edges of the underlying matrix."""
        return self._edges
    
    @property
    def n_sites(self):
        """The number of lattice sites."""
        return self._n_sites

    @property
    def J(self):
        """The interaction strength between bonds."""
        return self._J
    
    @property
    def t(self):
        """The bond hopping amplitude."""
        return self._t
    
    @property
    def dtype(self):
        """The dtype of the matrix elements."""
        return self._dtype
    
    def tree_flatten(self):
        data = (self.J, self.t, self.edges, self._forbidden_configurations)
        metadata = {"hilbert": self.hilbert}
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        J, t, edges, forb = data
        hi = metadata["hilbert"]
        res = cls(hi, graph=nk.graph.Graph([(0, 0)]), J=1.0)
        res._J = J
        res._t = t
        res._edges = edges
        res._forbidden_configurations = forb
        return res
    
    def conjugate(self, *, concrete=True):
        # if real
        if isinstance(self.dtype, float):
            return self
        else:
            raise NotImplementedError
    
    @jax.jit
    def get_conn_padded(self, x):
        # Indices i and j of all edges <i,j>
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        # The clock value at those sites
        x_i = jnp.vectorize(lambda x: x[i], signature="(n)->(m)")(x)
        x_j = jnp.vectorize(lambda x: x[j], signature="(n)->(m)")(x)
        
        # Kinetic term (clockwise turn)
        xp1 = jnp.mod(x[..., None, :] - np.eye(self.n_sites), 4)
        mel1 = -self.t * jnp.ones((*x.shape[:-1], self.n_sites))
        
        # Kinetic term (counter-clockwise turn)
        xp2 = jnp.mod(x[..., None, :] + np.eye(self.n_sites), 4)
        mel2 = -self.t * jnp.ones((*x.shape[:-1], self.n_sites))
        
        # Concatenate everything
        mels = jnp.concatenate([mel1, mel2], axis=-1)
        xps = jnp.concatenate([xp1, xp2], axis=-2)

        # Check whether for any connected element the new configuration on an edge is compatible 
        # with the constraint (we check that the configuration (x_i, x_j) on edge <i,j> is not forbidden)
        mask = jnp.all(jnp.any(xps[..., self.edges] != self._forbidden_configurations, axis=-1), axis=-1)
        
        # Zero out all matrix elements with unphysical configurations, zero out those configurations as well
        # since (0, 0, ..., 0) is a valid configuration that we use a placeholder.
        return jnp.astype(xps * mask[..., None], x.dtype), mels * mask
