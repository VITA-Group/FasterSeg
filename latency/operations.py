from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
import sys
import os.path as osp
from easydict import EasyDict as edict
C = edict()
"""please config ROOT_dir and user when u first using"""
C.repo_name = 'FasterSeg'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'tools'))
try:
    from utils.darts_utils import compute_latency_ms_tensorrt as compute_latency
    print("use TensorRT for latency test")
except:
    from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
    print("use PyTorch for latency test")
from slimmable_ops import USConv2d, USBatchNorm2d

__all__ = ['ConvNorm', 'BasicResidual1x', 'BasicResidual_downup_1x', 'BasicResidual2x', 'BasicResidual_downup_2x', 'FactorizedReduce', 'OPS', 'OPS_name', 'OPS_Class']

latency_lookup_table = {}
table_file_name = "latency_lookup_table.npy"
if osp.isfile(table_file_name):
    latency_lookup_table = np.load(table_file_name).item()


class ConvNorm(nn.Module):
    '''
    conv => norm => activation
    use native nn.Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, slimmable=True, width_mult_list=[1.]):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)

        if slimmable:
            self.conv = nn.Sequential(
                USConv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list),
                USBatchNorm2d(C_out, width_mult_list),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias),
                nn.BatchNorm2d(C_out),
                nn.ReLU(inplace=True),
            )
    
    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv[0].set_ratio(ratio)
        self.conv[1].set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = ConvNorm._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        assert x.size()[1] == self.C_in, "{} {}".format(x.size()[1], self.C_in)
        x = self.conv(x)
        return x


class BasicResidual1x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual1x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn1 = nn.BatchNorm2d(C_out)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), "c_in %d, int(self.C_in * self.ratio[0]) %d"%(c_in, int(self.C_in * self.ratio[0]))
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual1x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual1x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class BasicResidual_downup_1x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual_downup_1x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn1 = nn.BatchNorm2d(C_out)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_1x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), "c_in %d, int(self.C_in * self.ratio[0]) %d"%(c_in, int(self.C_in * self.ratio[0]))
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual_downup_1x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual_downup_1x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='nearest')
        out = self.conv1(out)
        out = self.bn1(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='nearest')
        out = self.relu(out)
        return out


class BasicResidual2x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual2x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if self.slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
            self.conv2 = USConv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn2 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn1 = nn.BatchNorm2d(C_out)
            self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn2 = nn.BatchNorm2d(C_out)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])
        self.conv2.set_ratio((ratio[1], ratio[1]))
        self.bn2.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in%d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual2x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class BasicResidual_downup_2x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True, width_mult_list=[1.]):
        super(BasicResidual_downup_2x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
        if self.slimmable:
            self.conv1 = USConv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn1 = USBatchNorm2d(C_out, width_mult_list)
            self.conv2 = USConv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
            self.bn2 = USBatchNorm2d(C_out, width_mult_list)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn1 = nn.BatchNorm2d(C_out)
            self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
            self.bn2 = nn.BatchNorm2d(C_out)
    
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])
        self.conv2.set_ratio((ratio[1], ratio[1]))
        self.bn2.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        assert stride in [1, 2]
        layer = BasicResidual_downup_2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in%d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = BasicResidual2x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='nearest')
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='nearest')
        out = self.relu(out)
        return out


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride=1, slimmable=True, width_mult_list=[1.]):
        super(FactorizedReduce, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)
        if stride == 1 and slimmable:
            self.conv1 = USConv2d(C_in, C_out, 1, stride=1, padding=0, bias=False, width_mult_list=width_mult_list)
            self.bn = USBatchNorm2d(C_out, width_mult_list)
            self.relu = nn.ReLU(inplace=True)
        elif stride == 2:
            self.relu = nn.ReLU(inplace=True)
            if slimmable:
                self.conv1 = USConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False, width_mult_list=width_mult_list)
                self.conv2 = USConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False, width_mult_list=width_mult_list)
                self.bn = USBatchNorm2d(C_out, width_mult_list)
            else:
                # TensorRT: [Slice]: slice is out of input range
                self.conv1 = nn.Conv2d(C_in, C_out, 1, stride=2, padding=0, bias=False)
                self.bn = nn.BatchNorm2d(C_out)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        if self.stride == 1:
            self.ratio = ratio
            self.conv1.set_ratio(ratio)
            self.bn.set_ratio(ratio[1])
        elif self.stride == 2:
            self.ratio = ratio
            self.conv1.set_ratio(ratio)
            self.conv2.set_ratio(ratio)
            self.bn.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, stride=1):
        layer = FactorizedReduce(C_in, C_out, stride, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, stride=1):
        layer = FactorizedReduce(C_in, C_out, stride, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "FactorizedReduce_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = FactorizedReduce._latency(h_in, w_in, c_in, c_out, self.stride)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        if self.stride == 2:
            # TensorRT: [Slice]: slice is out of input range
            out = self.conv1(x)
            out = self.bn(out)
            out = self.relu(out)
            return out
        else:
            if self.slimmable:
                out = self.conv1(x)
                out = self.bn(out)
                out = self.relu(out)
                return out
            else:
                return x



from collections import OrderedDict
OPS = {
    'skip' : lambda C_in, C_out, stride, slimmable, width_mult_list: FactorizedReduce(C_in, C_out, stride, slimmable, width_mult_list),
    'conv' : lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual1x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'conv_downup' : lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual_downup_1x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'conv_2x' : lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual2x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'conv_2x_downup' : lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual_downup_2x(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
}
OPS_name = ["FactorizedReduce", "BasicResidual1x", "BasicResidual_downup_1x", "BasicResidual2x", "BasicResidual_downup_2x"]
OPS_Class = OrderedDict()
OPS_Class['skip'] = FactorizedReduce
OPS_Class['conv'] = BasicResidual1x
OPS_Class['conv_downup'] = BasicResidual_downup_1x
OPS_Class['conv_2x'] = BasicResidual2x
OPS_Class['conv_2x_downup'] = BasicResidual_downup_2x
