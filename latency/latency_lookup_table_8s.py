import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from pdb import set_trace as bp
from operations import *
from seg_oprs import FeatureFusion, BiSeNetHead

H = 1024
W = 2048
scale_range = [8, 16, 32]
widths_range = [4./12, 6./12, 8./12, 10./12, 1.]
file_name = "latency_lookup_table_8s.npy"
if os.path.isfile(file_name):
    lookup_table = np.load(file_name).item()
else:
    lookup_table = {}


print("cells......")
Fch_range = [12]
for Fch in Fch_range:
    for scale in scale_range:
        print("Fch", Fch, "scale", scale)
        for w_in in widths_range:
            for w_out in widths_range:
                C_in = int(Fch*scale*w_in); C_out = int(Fch*scale*w_out)
                latency = BasicResidual1x._latency(H//scale, W//scale, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1)
                lookup_table["BasicResidual1x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(H//scale, W//scale, C_in, C_out, 1, 1)] = latency
                if scale < 32:
                    C_in = int(Fch*scale*w_in); C_out = int(Fch*scale*2*w_out)
                    latency = BasicResidual1x._latency(H//scale, W//scale, C_in, C_out, kernel_size=3, stride=2, dilation=1, groups=1)
                    lookup_table["BasicResidual1x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(H//scale, W//scale, C_in, C_out, 2, 1)] = latency

                C_in = int(Fch*scale*w_in); C_out = int(Fch*scale*w_out)
                latency = BasicResidual_downup_1x._latency(H//scale, W//scale, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1)
                lookup_table["BasicResidual_downup_1x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(H//scale, W//scale, C_in, C_out, 1, 1)] = latency
                if scale < 32:
                    C_in = int(Fch*scale*w_in); C_out = int(Fch*scale*2*w_out)
                    latency = BasicResidual_downup_1x._latency(H//scale, W//scale, C_in, C_out, kernel_size=3, stride=2, dilation=1, groups=1)
                    lookup_table["BasicResidual_downup_1x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(H//scale, W//scale, C_in, C_out, 2, 1)] = latency

                C_in = int(Fch*scale*w_in); C_out = int(Fch*scale*w_out)
                latency = BasicResidual2x._latency(H//scale, W//scale, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1)
                lookup_table["BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(H//scale, W//scale, C_in, C_out, 1, 1)] = latency
                if scale < 32:
                    C_in = int(Fch*scale*w_in); C_out = int(Fch*scale*2*w_out)
                    latency = BasicResidual2x._latency(H//scale, W//scale, C_in, C_out, kernel_size=3, stride=2, dilation=1, groups=1)
                    lookup_table["BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(H//scale, W//scale, C_in, C_out, 2, 1)] = latency

                C_in = int(Fch*scale*w_in); C_out = int(Fch*scale*w_out)
                latency = BasicResidual_downup_2x._latency(H//scale, W//scale, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1)
                lookup_table["BasicResidual_downup_2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(H//scale, W//scale, C_in, C_out, 1, 1)] = latency
                if scale < 32:
                    C_in = int(Fch*scale*w_in); C_out = int(Fch*scale*2*w_out)
                    latency = BasicResidual_downup_2x._latency(H//scale, W//scale, C_in, C_out, kernel_size=3, stride=2, dilation=1, groups=1)
                    lookup_table["BasicResidual_downup_2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(H//scale, W//scale, C_in, C_out, 2, 1)] = latency

                C_in = int(Fch*scale*w_in); C_out = int(Fch*scale*w_out)
                latency = FactorizedReduce._latency(H//scale, W//scale, C_in, C_out, stride=1)
                lookup_table["FactorizedReduce_H%d_W%d_Cin%d_Cout%d_stride%d"%(H//scale, W//scale, C_in, C_out, 1)] = latency
                if scale < 32:
                    C_in = int(Fch*scale*w_in); C_out = int(Fch*scale*2*w_out)
                    latency = FactorizedReduce._latency(H//scale, W//scale, C_in, C_out, stride=2)
                    lookup_table["FactorizedReduce_H%d_W%d_Cin%d_Cout%d_stride%d"%(H//scale, W//scale, C_in, C_out, 2)] = latency
                np.save(file_name, lookup_table)



print("stem...")
Fch_range = [8, 12]
for Fch in Fch_range:
    print("Fch", Fch)
    latency = ConvNorm._latency(H, W, 3, 2*Fch*2, kernel_size=3, stride=2, padding=1, bias=False, groups=1)
    lookup_table["ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(H, W, 3, 2*Fch*2, 3, 2)] = latency
    latency = ConvNorm._latency(H//2, W//2, 2*Fch*2, 4*Fch*2, kernel_size=3, stride=2, padding=1, bias=False, groups=1)
    lookup_table["ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(H//2, W//2, 2*Fch*2, 4*Fch*2, 3, 2)] = latency
    latency = BasicResidual2x._latency(H//2, W//2, 2*Fch*2, 4*Fch*2, kernel_size=3, stride=2, dilation=1, groups=1)
    lookup_table["BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(H//2, W//2, 2*Fch*2, 4*Fch*2, 2, 1)] = latency
    latency = BasicResidual2x._latency(H//4, W//4, 4*Fch*2, 8*Fch, kernel_size=3, stride=2, dilation=1, groups=1)
    lookup_table["BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d"%(H//4, W//4, 4*Fch*2, 8*Fch, 2, 1)] = latency
    np.save(file_name, lookup_table)

print("FFM......")
Fch_range = [8, 12]
Fch_max = 12
for Fch in Fch_range:
    print("Fch", Fch)
    latency = ConvNorm._latency(H//32, W//32, 32*Fch, 16*Fch, kernel_size=1, stride=1)
    lookup_table["ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(H//32, W//32, 32*Fch, 16*Fch, 1, 1)] = latency
    for w_in in widths_range:
        latency = ConvNorm._latency(H//16, W//16, int(16*Fch+16*Fch_max*w_in), 16*Fch, kernel_size=3, stride=1, padding=1)
        lookup_table["ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(H//16, W//16, 16*Fch+16*Fch_max*w_in, 16*Fch, 3, 1)] = latency
    latency = ConvNorm._latency(H//16, W//16, 16*Fch, 8*Fch, kernel_size=1, stride=1)
    lookup_table["ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(H//16, W//16, 16*Fch, 8*Fch, 1, 1)] = latency
    for w_in in widths_range:
        latency = ConvNorm._latency(H//8, W//8, int(8*Fch+8*Fch_max*w_in), 8*Fch, kernel_size=3, stride=1, padding=1)
        lookup_table["ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(H//8, W//8, 8*Fch+8*Fch_max*w_in, 8*Fch, 3, 1)] = latency
    latency = ConvNorm._latency(H//16, W//16, 16*Fch, 8*Fch, kernel_size=1, stride=1)
    lookup_table["ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(H//16, W//16, 16*Fch, 8*Fch, 1, 1)] = latency
    for w_in in widths_range:
        latency = ConvNorm._latency(H//8, W//8, int(8*Fch+8*Fch_max*w_in), 8*Fch, kernel_size=3, stride=1, padding=1)
        lookup_table["ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(H//8, W//8, 8*Fch+8*Fch_max*w_in, 8*Fch, 3, 1)] = latency
    for branch in range(1, 4):
        lookup_table["ff_H%d_W%d_C%d"%(H//8, W//8, 8*Fch*branch)] = FeatureFusion._latency(H//8, W//8, 8*Fch*branch, 8*Fch*branch)
    np.save(file_name, lookup_table)


print("head......")
Fch_range = [8, 12]
for Fch in Fch_range:
    for branch in range(1,4):
        print("Fch", Fch, "branch", branch)
        lookup_table["head_H%d_W%d_Cin%d_Cout%d"%(H//8, W//8, 8*Fch*branch, 19)] = BiSeNetHead._latency(H//8, W//8, 8*Fch*branch, 19)
        np.save(file_name, lookup_table)



def find_latency(name, info, lookup_table, H=1024, W=2048):
    if name == "stem":
        latency = lookup_table["stem_H%d_W%d_F%d"%(H, W, info["F"])]
    elif name == "head":
        latency = lookup_table["head_H%d_W%d_F%d_branch%d_19"%(H, W, info["F"], info["branch"])]
    elif name == "refines":
        latency = lookup_table["refines%d_H%d_W%d_F%d"%(info["scale"], H, W, info["F"])]
    elif name == "arms":
        latency = lookup_table["arms%d_H%d_W%d_F%d"%(info["scale"], H, W, info["F"])]
    elif name == "ff":
        latency = lookup_table["ff_H%d_W%d_F%d_branch%d"%(H, W, info["F"], info["branch"])]
    else:
        latency = lookup_table["%s_H%d_W%d_Cin%d_Cout%d_scale%d_stride%d_dilation%d"%(name, H, W, info["C_in"], info["C_out"], info["scale"], info["stride"], info["dilation"])]
    return latency


def profile_estimated(model, lookup_table):
    latency_stem = find_latency("stem", {"F": model._Fch}, lookup_table)
    latency_head = find_latency("head", {"F": model._Fch, "branch": model._branch}, lookup_table)
    latency_refines = find_latency("refines", {"F": model._Fch, "branch": model._branch}, lookup_table)
    latency_arms = find_latency("arms", {"F": model._Fch, "branch": model._branch}, lookup_table)
    if 3 in model.lasts:
        latency_refines += find_latency("refines", {"scale": 32, "F": model._Fch}, lookup_table)
        latency_arms += find_latency("arms", {"scale": 32, "F": model._Fch}, lookup_table)
    if 2 in model.lasts:
        latency_refines += find_latency("refines", {"scale": 16, "F": model._Fch}, lookup_table)
        latency_arms += find_latency("arms", {"scale": 16, "F": model._Fch}, lookup_table)
    if 0 in model.lasts:
        latency_arms += find_latency("arms", {"scale": 4, "F": model._Fch}, lookup_table)
    latency_ff = find_latency("ff", {"F": model._Fch, "branch": model._branch}, lookup_table)
    latency_cells = 0
    for layer in range(len(model.branch_groups)):
        for group in model.branch_groups[layer]:
            op, scale, down, C_in, C_out = model.cells_code[str(layer)+"-"+str(group[0])].split(".")
            op = int(op); scale = int(scale); down = int(down)
            if op == 2 or op == 4: dilation = 2
            else: dilation = 1
            if down: dilation = 1
            latency_cells += find_latency(OPS_name[op], {"C_in": C_in, "C_out": C_out, "scale": scale, "stride": down+1, "dilation": dilation}, lookup_table)
    return latency_stem + latency_cells + latency_refines + latency_arms + latency_ff + latency_head
