from math import pi

import torch
import torch_geometric.transforms as T

from eman.nn.eman_conv import EmanAttLayer
from gem_cnn.tests.utils import random_geometry
from gem_cnn.transform.gauge_transformer import GaugeTransformer
from gem_cnn.transform.gem_precomp import GemPrecomp
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.utils.rep_act import rep_act


def test_eman_layer_gauge_invariance():
    n_rings = 2
    num_v = 5
    max_order = 4
    gem_precomp = GemPrecomp(n_rings=n_rings, max_order=max_order)
    dtype = torch.float64

    transform_angle = torch.rand(num_v, dtype=dtype) * 2 * pi
    transform = T.Compose(
        (
            SimpleGeometry(),
            gem_precomp,
        )
    )

    transform_t = T.Compose(
        (
            SimpleGeometry(),
            GaugeTransformer(transform_angle),
            gem_precomp,
        )
    )

    data_raw = random_geometry(num_v, edge_p=0.6, dtype=dtype)
    data = transform(data_raw)
    data_t = transform_t(data_raw)

    in_order = 2
    out_order = 2
    channels = (16, 16)
    emanatt = EmanAttLayer(
        *channels, in_order, out_order, n_rings=n_rings, n_heads=4, batch=100_000
    ).to(dtype)
    x = torch.randn(num_v, channels[0], 2 * in_order + 1, dtype=dtype)

    x_a = emanatt(
        x,
        data.edge_index,
        data.precomp.double(),
        data.precomp_self.double(),
        data.connection,
    )
    # x_a_t = rep_act(x_a, -transform_angle)

    x_t = rep_act(x, -transform_angle)
    x_t_a = emanatt(
        x_t,
        data_t.edge_index,
        data_t.precomp.double(),
        data_t.precomp_self.double(),
        data_t.connection,
    )

    assert torch.allclose(x_a, x_t_a, atol=1e-14)


if __name__ == "__main__":
    test_eman_layer_gauge_invariance()
