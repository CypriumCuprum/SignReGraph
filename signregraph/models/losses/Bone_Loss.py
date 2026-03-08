import torch
import torch.nn as nn


class Bone_Loss(nn.Module):
    def __init__(self, graph: list, num_node: int):
        super().__init__()
        self.num_node = num_node

        self.register_buffer("graph", torch.tensor(graph, dtype=torch.long))
        self.register_buffer("bone_matrix", None)
        self.make_bone_matrix()

    def make_bone_matrix(self):
        device = self.graph.device
        num_bone = self.graph.shape[0]
        idx = torch.arange(num_bone, device=device)

        j1 = self.graph[:, 0]
        j2 = self.graph[:, 1]

        bone_matrix = torch.zeros((num_bone, self.num_node), device=device)
        bone_matrix[idx, j1] = 1
        bone_matrix[idx, j2] = -1

        self.bone_matrix = bone_matrix.T

    def forward(self, features, label):
        # features: (batch, channels, num nodes) channels is the dimension of a joint
        # label: sameeeee features
        features = torch.matmul(features, self.bone_matrix)
        label  = torch.matmul(label, self.bone_matrix)
        return torch.nn.L1Loss()(features, label)
        