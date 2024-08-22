# Copyright (c) OpenBII
# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

import os, time

import torch as pt
import torch.nn as nn
from torch import ops
from torch.nn import Flatten
import numpy

from ..builder import BACKBONES
from ..layers.lif import Conv2dLif, FcLif
from ..layers.lifplus import Conv2dLifPlus, FcLifPlus
from ..layers.warp_load_save import load,save,load_kernel,save_kernel
from . import base_backbone


@BACKBONES.register_module()
class SeqClif3Fc3DmItout_test(nn.Module):
    """For NEVACS."""

    def __init__(self,
                 timestep=20, c0=2, h0=40, w0=40, nclass=10, cmode='spike', amode='mean', soma_params='all_share',
                 noise=0, neuron='lif', neuron_config=None):
        super(SeqClif3Fc3DmItout_test, self).__init__()
        neuron=neuron.lower()
        assert neuron in ['lif']
        #self.clif1 = Conv2dLif(c0, 32, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.mp1 = nn.MaxPool2d(2, stride=2)
        self.clif2 = Conv2dLif(c0, 8, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.mp2 = nn.MaxPool2d(2, stride=2)
        self.clif3 = Conv2dLif(8, 8, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.mp3 = nn.MaxPool2d(2, stride=2)
        assert amode == 'mean'
        self.flat = Flatten(1, -1)
        self.head = nn.Sequential(
            nn.Linear(h0 // 8 * w0 // 8 * 8, nclass),
            #nn.ReLU(),
            #nn.Linear(512, 128),
            #nn.ReLU(),
            #nn.Linear(256, nclass)
        )
        self.tempAdd = None
        self.timestep = timestep

    def reset(self, xi):
        self.tempAdd = pt.zeros_like(xi)

    def forward(self, xis: pt.Tensor) -> pt.Tensor:
        if base_backbone.ON_APU:
            assert len(xis.shape) == 4
            x0 = xis
            #self.clif1.reset(xis)
            #x1 = self.mp1(self.clif1(x0))
            x1 = self.mp1(x0)
           
            self.clif2.reset(x1)
            # x2 = self.clif2(x1)
            x2 = self.mp2(self.clif2(x1))
            self.clif3.reset(x2)
            x3 = self.mp3(self.clif3(x2))
            # xt = ops.custom.tempAdd(x3, 0)
            # xt = xt*0.05
            # xo = self.flat(x3)
            # xo = self.head(xo)
            # return xo
            x4 = self.flat(x3)
            x5 = self.head(x4)
            x5 = x5.unsqueeze(2).unsqueeze(3)
            self.reset(x5)
            self.tempAdd = load_kernel(self.tempAdd, f'tempAdd')
            self.tempAdd = self.tempAdd + x5 / self.timestep
            output = self.tempAdd.clone()
            save_kernel(self.tempAdd, f'tempAdd')
            return output.squeeze(-1).squeeze(-1)

        else:
            #print(xis.shape)
            t = xis.size(1)
            xo_list = []
            xo = 0
            for i in range(t):
                x0 = xis[:, i, ...]
                #if i == 0: self.clif1.reset(x0)
                x1 = self.mp1(x0)
                #print(x1.size())
                if i == 0: self.clif2.reset(x1)
                x2 = self.mp2(self.clif2(x1))
                #x2 = self.clif2(x1)
                if i == 0: self.clif3.reset(x2)
                x3 = self.mp3(self.clif3(x2))
                # xo_list.append(x3)
                x4 = self.flat(x3)
                x5 = self.head(x4)
                xo = xo + x5 / self.timestep
            # xo = sum(xo_list) / t
            # xo = self.flat(xo)
            # xo = self.head(xo)
            return xo

@BACKBONES.register_module()
class SeqClif2Fc1CeItout(nn.Module):
    """For NEVACS."""

    def __init__(self,
                 timestep=20, c0=1, h0=120, w0=120, nclass=2, cmode='spike', amode='mean', soma_params='all_share',
                 noise=0, neuron='lif', neuron_config=None):
        super(SeqClif2Fc1CeItout, self).__init__()
        neuron=neuron.lower()
        assert neuron in ['lif']
        #self.clif1 = Conv2dLif(c0, 32, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.mp1 = nn.MaxPool2d(6, stride=6)
        self.clif2 = Conv2dLif(c0, 8, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.mp2 = nn.MaxPool2d(2, stride=2)
        self.clif3 = Conv2dLif(8, 8, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.mp3 = nn.MaxPool2d(2, stride=2)
        assert amode == 'mean'
        self.flat = Flatten(1, -1)
        self.head = nn.Sequential(
            nn.Linear(h0 // 24 * w0 // 24 * 8, nclass),
            #nn.ReLU(),
            #nn.Linear(512, 128),
            #nn.ReLU(),
            #nn.Linear(256, nclass)
        )
        self.tempAdd = None
        self.timestep = timestep

    def reset(self, xi):
        self.tempAdd = pt.zeros_like(xi)

    def forward(self, xis: pt.Tensor) -> pt.Tensor:
        if base_backbone.ON_APU:
            assert len(xis.shape) == 4
            x0 = xis
            #self.clif1.reset(xis)
            #x1 = self.mp1(self.clif1(x0))
            x1 = self.mp1(x0)
           
            self.clif2.reset(x1)
            # x2 = self.clif2(x1)
            x2 = self.mp2(self.clif2(x1))
            self.clif3.reset(x2)
            x3 = self.mp3(self.clif3(x2))
            # xt = ops.custom.tempAdd(x3, 0)
            # xt = xt*0.05
            # xo = self.flat(x3)
            # xo = self.head(xo)
            # return xo
            x4 = self.flat(x3)
            x5 = self.head(x4)
            x5 = x5.unsqueeze(2).unsqueeze(3)
            self.reset(x5)
            self.tempAdd = load_kernel(self.tempAdd, f'tempAdd')
            self.tempAdd = self.tempAdd + x5 / self.timestep
            output = self.tempAdd.clone()
            save_kernel(self.tempAdd, f'tempAdd')
            return output.squeeze(-1).squeeze(-1)

        else:
            t = xis.size(1)
            #print(xis.shape)
            xo_list = []
            xo = 0
            for i in range(t):
                gpuStartTime = time.time()  
                x0 = xis[:, i, ...]
                #if i == 0: self.clif1.reset(x0)
                x1 = self.mp1(x0)
                #print(x1.size())
                if i == 0: self.clif2.reset(x1)
                x2 = self.mp2(self.clif2(x1))
                #x2 = self.clif2(x1)
                if i == 0: self.clif3.reset(x2)
                x3 = self.mp3(self.clif3(x2))
                # xo_list.append(x3)
                x4 = self.flat(x3)
                x5 = self.head(x4)
                xo = xo + x5 / self.timestep
                gpuEndTime = time.time() 
                print(gpuEndTime-gpuStartTime)
            # xo = sum(xo_list) / t
            # xo = self.flat(xo)
            # xo = self.head(xo)
            return xo


