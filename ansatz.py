import netket as nk
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Sequence, Callable, Union, Optional

from flax.linen import Module, Conv, Dense, LayerNorm, gelu
from flax.linen.dtypes import promote_dtype
from jax.nn.initializers import zeros, normal
from jax.tree_util import register_pytree_node_class

from netket.graph import AbstractGraph
from netket.utils.types import DType, NNInitFunc
from netket.nn.masked_linear import default_kernel_init

def _min_image_distance(x, extent):
    dis = -x[np.newaxis, :, :] + x[:, np.newaxis, :]
    dis = dis - extent * np.rint(dis / extent)
    return dis


def _relative_position_indices(x, extent):
    n = len(x)
    dis = _min_image_distance(x, extent)
    dis_flat = dis.reshape(-1, 2)
    unique_dis, idcs = np.unique(dis_flat, axis=0, return_inverse=True)
    return idcs.reshape(n, n)

def _min_image_distance(x, extent):
    dis = -x[np.newaxis, :, :] + x[:, np.newaxis, :]
    dis = dis - extent * np.rint(dis / extent)
    return dis


def _distance_indices(x, extent):
    n = len(x)
    rel_pos = _min_image_distance(x, extent)
    dis = np.linalg.norm(rel_pos, ord=2, axis=-1)
    dis_flat = dis.reshape(-1)
    unique_dis, idcs = np.unique(dis_flat, axis=0, return_inverse=True)
    return idcs.reshape(n, n)


class ResNetTransInvJastrow(nn.Module):
    """ResNet-based neural backflow Jastrow."""
    graph: AbstractGraph
    """Graph corresponding to the lattice."""
    features: Sequence[int]
    """Number of channels at each layer."""
    kernel_size: Union[Sequence[int]]
    """Size of the convolutional filters."""
    padding: str = 'CIRCULAR'
    """Type of padding. Must be set to 'CIRCULAR' for OBC and 'SAME' for OBC."""
    use_bias: bool = True
    """if True uses a bias in all layers."""
    backflow_param_dtype: DType = float
    """The dtype of all backflow parameters."""
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""
    activation: Callable = nn.gelu
    """Hidden activation function."""
    output_activation: Optional[Callable] = None
    """Output activation function."""

    mixing_kernel_init: NNInitFunc = normal(1e-4)
    """Initializer for the backflow mixing weights."""
    jastrow_kernel_init: NNInitFunc = normal(5e-1)
    """Initializer for the Jastrow weights."""
    out_param_dtype: DType = float
    """The dtype of the output."""

    @nn.compact
    def __call__(self, x):
        shape = x.shape
        depth = len(self.features)
        kernel_sizes = depth * (self.kernel_size,)  # if isinstance(self.kernel_size, int) else self.kernel_size
        assert len(kernel_sizes) == depth
        cnn = partial(nn.Conv, padding=self.padding, use_bias=self.use_bias, param_dtype=self.backflow_param_dtype,
                      kernel_init=self.kernel_init, bias_init=self.bias_init)

        a = nn.Dense(2, use_bias=False, kernel_init=self.kernel_init, param_dtype=self.backflow_param_dtype)

        x = jnp.stack([jnp.cos(0.5 * np.pi * x), jnp.sin(0.5 * np.pi * x)], axis=-1)
        x0 = x.copy()
        x = x.reshape(*shape[:-1], *self.graph.extent, 2)

        residual = x.copy() # This does nothing
        for i, (feature, kernel_size) in enumerate(zip(self.features, kernel_sizes)):
            if i:
                x = nn.LayerNorm(use_scale=False, use_bias=False)(x)
                x = self.activation(x)
            # else:
                # x = x / self.density - 1.0
            x = cnn(name=f"CNN_{i}", features=feature, kernel_size=kernel_size)(x)
            if i % 2:
                if i > 1:
                    x += residual
                residual = x.copy()
        if self.output_activation:
            x = self.output_activation(x)
        x_tilde = x0 + a(x).reshape(*shape, 2)

        d_max = np.floor_divide(self.graph.extent, 2).sum()
        sd = _min_image_distance(self.graph.positions, self.graph.extent)
        d = np.linalg.norm(sd, ord=1, axis=-1).astype(int)
        kernel = self.param('Jastrow', self.jastrow_kernel_init, (2, 2, d_max+1), self.out_param_dtype)
        out = jnp.einsum('...im,mnij,...jn->...', x_tilde, kernel[..., d], x_tilde)

        # d = _distance_indices(self.graph.positions, self.graph.extent).astype(int)
        # jastrow = self.param('Jastrow', self.jastrow_kernel_init, (2, 2, d.max()+1), self.out_param_dtype)
        # out = jnp.einsum('...im,mnij,...jn->...', x_tilde, jastrow[..., d], x_tilde)

        return out


