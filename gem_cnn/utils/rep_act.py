# This file was modified by the EMAN authors to include additional code comments

# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import torch


def rep_dim(l):
    return 1 if l == 0 else 2


def _rep_act(x, theta):
    """
    action of the representation on the feature x by an angle theta

    :param x: [N, C, 2*order+1]
    :param theta: [N]
    :return: [N, C, 2*order+1]

    N = number of edges
    C = number of channels
    order = maximal order n of the representation rho0 + ... + rhon
    recall that rho0 has dimension 1, rhol dimension 2 for l>0
    """
    y = torch.zeros_like(x)  # [N, C, 2*order+1]
    order = x.shape[2] // 2
    # do not modify scalar features relative to rho0
    y[:, :, 0] = x[:, :, 0]  # [N, C, 1]

    for l in range(1, order + 1):
        # rotate feature of type rhol by the angle -l*theta (notice -sin before sin)
        cos = torch.cos(l * theta)[:, None, None]  # [N, 1, 1]
        sin = torch.sin(l * theta)[:, None, None]  # [N, 1, 1]
        offset = l * 2 - 1
        y[..., offset : offset + 2 : 2] = (
            cos * x[..., offset : offset + 2 : 2]
            + -sin * x[..., offset + 1 : offset + 2 : 2]
        )  # [N, C, 1]
        y[..., offset + 1 : offset + 2 : 2] = (
            sin * x[..., offset : offset + 2 : 2]
            + cos * x[..., offset + 1 : offset + 2 : 2]
        )  # [N, C, 1]
    return y.view(*y.shape)


class RepAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta):
        ctx.args = theta
        return _rep_act(x, theta)

    @staticmethod
    def backward(ctx, grad_y):
        theta = ctx.args
        grad_x = _rep_act(grad_y, -theta)
        return grad_x, None


rep_act = RepAct.apply


def act_so2_vector(th, v):
    """
    :param th: transform angle [N]
    :param v: [N, 2]
    :return: rotate vector by angle
    """
    cos, sin = torch.cos(th), torch.sin(th)
    rotator = torch.stack([cos, -sin, sin, cos], 1).view(-1, 2, 2)
    return torch.einsum("nij,nj->ni", rotator, v)
