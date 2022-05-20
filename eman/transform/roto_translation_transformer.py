import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def rotate_frame(frame, rot_direction, rot_angle):
    rot_direction = np.array(rot_direction) / np.linalg.norm(rot_direction)
    # get rotation matrices
    rot_vec = R.from_rotvec(rot_angle * rot_direction)
    rot_matrix = rot_vec.as_matrix()
    # compute rotated positions
    frame_rot = torch.einsum(
        "ij, jk -> ik", frame, torch.tensor(rot_matrix, dtype=torch.float)
    )
    return frame_rot


class RotoTranslationTransformer:
    """
    Transform data.pos by SO(3) rotation matrix and translations
    """

    def __init__(self, rot_angle=np.pi, translation_mag=20):
        self.rot_angle = rot_angle  # magnitude of rotation about some chosen axis
        self.rot_direction = np.random.rand(3)
        self.translation_mag = translation_mag

    def __call__(self, data):
        # data dim [N, 3], where 551200 for FAUST traindataset, i.e. 6980 * 80, similarly for testset
        # compute num of data points
        num_data_points = data.pos.shape[0]
        # generate random axis of rotation for each datapoint
        rot_direction = self.rot_direction
        # normalize the direction of axis so that rot_angle remains same
        rot_direction = np.array(rot_direction) / np.linalg.norm(rot_direction)
        # get rotation matrices
        rot_vec = R.from_rotvec(self.rot_angle * rot_direction)
        rot_matrix = rot_vec.as_matrix()
        # compute rotated positions
        data.pos = torch.einsum(
            "ij, jk -> ik",
            data.pos,
            torch.tensor(rot_matrix, dtype=torch.float, device=data.pos.device),
        )

        # translate positions
        translation_direction = np.random.rand(3)
        translation_direction = np.array(translation_direction) / np.linalg.norm(
            translation_direction
        )
        translation = self.translation_mag * torch.tensor(
            translation_direction, dtype=torch.float, device=data.pos.device
        )
        data.pos = data.pos + translation[None, :]
        return data
