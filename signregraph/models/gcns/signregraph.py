import copy as cp
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import load_checkpoint
from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .utils import unit_gcn, mstcn, unit_tcn

EPS = 1e-4


class GCN_Block(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        common_args = ['act', 'norm', 'g1x1']
        for arg in common_args:
            if arg in kwargs:
                value = kwargs.pop(arg)
                kwargs['tcn_' + arg] = value
                kwargs['gcn_' + arg] = value
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[1:4] != 'cn_'}
        assert len(kwargs) == 0

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)
        self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x, gcl_graph = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), gcl_graph


class Recontruction_Drop_Area_Network(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        self.time_pool = nn.AdaptiveAvgPool2d((1, num_nodes))
        # self.decoder = nn.Conv1d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=1
        # )
        
        mid_channels = in_channels // 4  # e.g., 384 // 4 = 96
        
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(mid_channels, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        ## x shape: [batch_size, feature_dim, ?times?, num_keypoint/node]
        ## [batch_size, 384, 25, 77]
        x = self.time_pool(x).squeeze(2)
        x = self.decoder(x)
        # x shape: [batch_size, 2, num_keypoint/node]
        return x

@BACKBONES.register_module()
class SignReGraph(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=96,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 data_bn_type='VC',
                 num_person=2,
                 pretrained=None,
                 num_drop_node=21,
                 **kwargs):
        super().__init__()
        
        kwargs.pop('init_cfg', None)
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # print("A shape:", A.shape)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs
        

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        lw_kwargs[0].pop('g1x1', None)
        lw_kwargs[0].pop('gcn_g1x1', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            modules = [GCN_Block(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        down_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(GCN_Block(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
            down_times += (i in down_stages)

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained
        
        out_channels = base_channels
        norm = 'BN'
        norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        
        self.post = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU()
        
        dim = 384   # base_channels * 4
        self.num_drop_nodes = num_drop_node  
        self.hand_reconstruction = Recontruction_Drop_Area_Network(
            in_channels=dim,
            out_channels=2, # Kênh gốc (2 hoặc 3)
            num_nodes=self.num_drop_nodes
        )
        
    def init_weights(self):
        # super().init_weights()
        if isinstance(self.pretrained, str):
            print("Loading checkpoint")
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        N, M, T, V, C = x.size()
        # print("X shape:", x.size())
        ''' Cup note out from issue #1
        N: number of training samples in the current batch (e.g. N=16)
        M: number of people in the sample (e.g. M=2)
        T: number of frames (e.g. T=100)
        V: number of keypoints (e.g. V=25 for NTU RGB+D 3D skeleton)
        C: number of dimensions for skeleton representions (e.g. the initial C=3 for 3D keypoint)
        '''
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        ### x shape: [batch_size, 2 (coordinates), ?times? (100 = num frames), num_keypoint/node]
        get_graph = []
        for i in range(self.num_stages):
            x, gcl_graph = self.gcn[i](x)
            # N*M C V V
            get_graph.append(gcl_graph)
        
        x = x.reshape((N, M) + x.shape[1:])
        
        n, m, d, t, k = x.shape
        ## x shape: [batch_size, feature_dim, ?times?, num_keypoint/node]
        ## get features from hand, propagate to recontruction network
        # x_hand_area = x[: , : , : , : , 6 : 27].view(n*m, d, -1)
        x_hand_area = x[: , : , : , : , :].view(n*m, d, t, self.num_drop_nodes)
        hand_area_recontruct = self.hand_reconstruction(x_hand_area)
        
        return x, hand_area_recontruct