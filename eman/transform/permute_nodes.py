import numpy as np
import torch
from torch_geometric.data import Data


class PermuteNodes:
    def __init__(self, shuffle):
        self.shuffle = shuffle

    def shuffle_backwards(self):
        a = torch.arange(len(self.shuffle))
        a[self.shuffle] = torch.arange(len(self.shuffle))
        return a

    def __call__(self, data):
        pos = data.pos
        face = data.face

        node_numbers = torch.arange(len(pos))  # target node numbers
        face_indices = torch.arange(len(pos))  # shuffling used for computing faces

        backward_shuffle = self.shuffle_backwards()

        pos_shuffle = pos.clone()[backward_shuffle]
        node_numbers_shuffle = node_numbers.clone()[backward_shuffle]
        face_indices_shuffle = face_indices.clone()[self.shuffle]

        face_shuffle = face.clone()
        face_shuffle[0], face_shuffle[1], face_shuffle[2] = (
            face_indices_shuffle.clone()[face_shuffle[0]],
            face_indices_shuffle.clone()[face_shuffle[1]],
            face_indices_shuffle.clone()[face_shuffle[2]],
        )
        data_shuffle = data.clone()
        data_shuffle.target = node_numbers_shuffle
        data_shuffle.pos = pos_shuffle
        data_shuffle.face = face_shuffle
        return data_shuffle
