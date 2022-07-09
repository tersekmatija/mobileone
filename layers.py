#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    '''Normal Conv with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result
            

class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, k = 1, use_1x1 = True, nonlinearity=nn.ReLU):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        assert kernel_size == 3 or kernel_size == 1
        #assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nonlinearity()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = nn.ModuleList([conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups) for _ in range(k)])
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups) if use_1x1 else None

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        dense = torch.stack([rbr_d(inputs) for rbr_d in self.rbr_dense]).sum(dim=0)
        if self.rbr_1x1 is not None:
            dense = dense + self.rbr_1x1(inputs)

        return self.nonlinearity(self.se(dense + id_out))

    def get_equivalent_kernel_bias(self):
        kb3x3 =[self._fuse_bn_tensor(rbr_d) for rbr_d in self.rbr_dense]
        kernel3x3 = torch.stack([x[0] for x in kb3x3]).sum(dim=0)
        bias3x3 = torch.stack([x[1] for x in kb3x3]).sum(dim=0)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        if self.rbr_1x1 is None:
            return kernel3x3 + kernelid, bias3x3 + biasid
        
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                #kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                #for i in range(self.in_channels):
                #    kernel_value[i, i % input_dim, 1, 1] = 1
                kernel_value = np.zeros((self.in_channels, input_dim, self.kernel_size, self.kernel_size), dtype=np.float32)
                j = 1 if self.kernel_size == 3 else 0
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, j, j] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense[0].conv.in_channels, out_channels=self.rbr_dense[0].conv.out_channels,
                                     kernel_size=self.rbr_dense[0].conv.kernel_size, stride=self.rbr_dense[0].conv.stride,
                                     padding=self.rbr_dense[0].conv.padding, dilation=self.rbr_dense[0].conv.dilation, groups=self.rbr_dense[0].conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


def conv_bn_v2(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
               padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


class MobileOneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, stride=1):
        super(MobileOneBlock, self).__init__()
        self.block_3x3 = RepVGGBlock(in_channels, in_channels, kernel_size = 3, padding = 1, dilation=1, groups=in_channels, k=k, use_1x1=True, stride=stride)
        self.block_1x1 = RepVGGBlock(in_channels, out_channels, kernel_size = 1, padding = 0, dilation=1, groups=1, k=k, use_1x1=False)
        
    def forward(self, x):
        return self.block_1x1(self.block_3x3(x))
        
    def switch_to_deploy(self):
        self.block_3x3.switch_to_deploy()
        self.block_1x1.switch_to_deploy()