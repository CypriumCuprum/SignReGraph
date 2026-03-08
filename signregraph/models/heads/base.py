import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import linalg as LA
from abc import ABCMeta, abstractmethod
from ...core import top_k_accuracy
from ..builder import build_loss

from ..losses.Class_Specific_Contrastive_Loss import Class_Specific_Contrastive_Loss
from ..losses.Bone_Loss import Bone_Loss

    
class BaseHead(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 joint_cfg,
                 num_classes,
                 in_channels,
                 weight,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.weight = weight
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        if joint_cfg == 'sign_language':
            num_node = 77
            graph = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (7, 8), 
                    (8, 9), (9, 10), (6, 11), (11, 12), (12, 13), (13, 14), 
                    (6, 15), (15, 16), (16, 17), (17, 18), (6, 19), (19, 20), 
                    (20, 21), (21, 22), (6, 23), (23, 24), (24, 25), (25, 26), 
                    (27, 28), (28, 29), (29, 30), (30, 31), (27, 32), (32, 33), 
                    (33, 34), (34, 35), (27, 36), (36, 37), (37, 38), (38, 39), 
                    (27, 40), (40, 41), (41, 42), (42, 43), (27, 44), (44, 45), 
                    (45, 46), (46, 47)] + \
                    [(i, i+1) for i in range(48, 48 + 17 - 1)] + \
                    [(i, i+1) for i in range(48+17, 48+17+12 - 1)] + [(48+17+12 -1, 48+17)]
        self.bone_loss = Bone_Loss(graph=graph, num_node=num_node)
        self.l1_loss = torch.nn.L1Loss()
        
    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def loss(self, cls_score, hand_pred, label, drop_area,**kwargs):

        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            label = label.unsqueeze(0)

        if not self.multi_class and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            label = ((1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes)

        """
        ************
        *** Loss ***
        ************
        """  
        loss_cls_1 = self.loss_cls(cls_score, label, **kwargs)
        # print("drop area shape", drop_area.shape)
        loss_reconstruct_hand = self.l1_loss(hand_pred, drop_area) 
        loss_bone_loss = self.bone_loss(hand_pred, drop_area)
        loss_cls = {
            'loss_cls': loss_cls_1.mean(),
            'loss_reconstruct_hand': loss_reconstruct_hand,
            'loss_bone_loss': 6*loss_bone_loss
        }
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses