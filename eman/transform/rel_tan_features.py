import torch
from torch_scatter import scatter_add


class RelTanTransform(object):
    def __init__(self, args):
        self.rel_power_list = args.rel_power_list
        self.deg_power = args.deg_power
        self.rel_power_const = args.rel_power_const

    def initial_3D_vectors(
        self, data, rel_power=0.7, deg_power=1.5, rel_power_const=2.45
    ):
        # given data, returns a [num_v, 3] tensor
        # as p varies, the returned vector is (1/N_p) * ( \sum_(q \in N_p) \pi_p(q-p) / |q-p|^2 ) * ( \sum_(q \in N_p) |q-p| )
        edge_index = data.edge_index
        normal = (data.normal / data.normal.norm(p=2, dim=1, keepdim=True)).to(
            data.pos.device
        )
        num_v = data.normal.shape[0]
        degree = scatter_add(torch.ones_like(edge_index[0]), edge_index[0])

        # compute projector
        projector = torch.eye(3, device=data.pos.device).reshape(1, 3, 3).repeat(
            num_v, 1, 1
        ) - torch.einsum("ni,nj->nij", normal, normal).to(
            data.pos.device
        )  # [num_edges, 3, 3]
        position = data.pos
        # compute relative position (q-p)
        relative_position = (position[edge_index[0]] - position[edge_index[1]]).to(
            data.pos.device
        )

        # compute relative norm |q-p|_2
        relative_norm = relative_position.norm(
            p=2, dim=1, keepdim=True
        )  # num_e norms |q-p|

        position_src = position[edge_index[0]]  # dim [num_edges, 3]
        position_dst = position[edge_index[1]]  # dim [num_edges, 3]
        position_norm = (position_dst - position_src) / (
            torch.norm(position_dst - position_src, dim=1, p=2) ** rel_power
        ).unsqueeze(
            dim=1
        )  # dim [num_edges, 1]

        # compute projection \pi_p(q-p)
        projection = torch.einsum(
            "nij, nj -> ni", projector[edge_index[0]], position_norm
        )  # n=num_e, i=j=3

        # compute the second term in the equation ( \sum_(q \in N_p) |q-p| )
        neighboring_norm = scatter_add(
            relative_norm ** (rel_power - 1), edge_index[0], dim=0
        ).squeeze(dim=1)

        # compute the first term ( \sum_(q \in N_p) \pi_p(q-p) / |q-p|^2 )
        neighboring_projection = scatter_add(projection, edge_index[0], dim=0)

        # compute the final equation
        initial_3D_v = torch.einsum(
            "i,ij,i->ij",
            1 / (rel_power_const * degree**deg_power),
            neighboring_projection,
            neighboring_norm
            # compacting to (1/N_p) * ( \sum_(q \in N_p) \pi_p(q-p) / |q-p|^2 ) * ( \sum_(q \in N_p) |q-p| )
        ).to(data.pos.device)

        return initial_3D_v

    def initial_tangent_features(self, data, feature):
        # given data and feature, returns a [num_v, 2] tensor
        # as p varies, the feature is a 2D coordinate vector representing the original feature 3D vector with respect to the frame
        frame = data.frame  # gauge_x, gauge_y, normal are stored by rows
        tangent_coordinates = torch.einsum("nij,nj->ni", frame, feature).to(
            data.pos.device
        )  # num_v vectors of 3D coordinates, the last one of which is zero
        return tangent_coordinates[:, :-1]

    def initial_rho0_rho1_features(self, data, tangent_coordinates):
        # add zeros to rho0
        num_v = tangent_coordinates.shape[0]
        return torch.cat(
            (torch.zeros(num_v, 1, device=data.pos.device), tangent_coordinates), dim=1
        )

    def __call__(self, data):
        rho0_rho1_features_list = []
        for rel_power in self.rel_power_list:
            feature = self.initial_3D_vectors(
                data, rel_power, self.deg_power, self.rel_power_const
            )
            tangent_coords = self.initial_tangent_features(data, feature)
            rho0_rho1_features = self.initial_rho0_rho1_features(data, tangent_coords)
            rho0_rho1_features_list.append(rho0_rho1_features)
        data.rel_tang_feat = torch.stack(rho0_rho1_features_list, dim=1)
