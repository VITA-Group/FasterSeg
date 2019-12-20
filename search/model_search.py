import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
# from utils.darts_utils import drop_path, compute_speed, compute_speed_tensorrt
from pdb import set_trace as bp
from seg_oprs import Head
import numpy as np


# https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, stride=1, width_mult_list=[1.]):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._width_mult_list = width_mult_list
        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, stride, True, width_mult_list=width_mult_list)
            self._ops.append(op)

    def set_prun_ratio(self, ratio):
        for op in self._ops:
            op.set_ratio(ratio)

    def forward(self, x, weights, ratios):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratios[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratios[0].argmax()]
            r_score0 = ratios[0][ratios[0].argmax()]
        else:
            ratio0 = ratios[0]
            r_score0 = 1.
        if isinstance(ratios[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratios[1].argmax()]
            r_score1 = ratios[1][ratios[1].argmax()]
        else:
            ratio1 = ratios[1]
            r_score1 = 1.
        self.set_prun_ratio((ratio0, ratio1))
        for w, op in zip(weights, self._ops):
            result = result + op(x) * w * r_score0 * r_score1
        return result

    def forward_latency(self, size, weights, ratios):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratios[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratios[0].argmax()]
            r_score0 = ratios[0][ratios[0].argmax()]
        else:
            ratio0 = ratios[0]
            r_score0 = 1.
        if isinstance(ratios[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratios[1].argmax()]
            r_score1 = ratios[1][ratios[1].argmax()]
        else:
            ratio1 = ratios[1]
            r_score1 = 1.
        self.set_prun_ratio((ratio0, ratio1))
        for w, op in zip(weights, self._ops):
            latency, size_out = op.forward_latency(size)
            result = result + latency * w * r_score0 * r_score1
        return result, size_out


class Cell(nn.Module):
    def __init__(self, C_in, C_out=None, down=True, width_mult_list=[1.]):
        super(Cell, self).__init__()
        self._C_in = C_in
        if C_out is None: C_out = C_in
        self._C_out = C_out
        self._down = down
        self._width_mult_list = width_mult_list

        self._op = MixedOp(C_in, C_out, width_mult_list=width_mult_list)

        if self._down:
            self.downsample = MixedOp(C_in, C_in*2, stride=2, width_mult_list=width_mult_list)
    
    def forward(self, input, alphas, ratios):
        # ratios: (in, out, down)
        out = self._op(input, alphas, (ratios[0], ratios[1]))
        assert (self._down and (ratios[2] is not None)) or ((not self._down) and (ratios[2] is None))
        down = self.downsample(input, alphas, (ratios[0], ratios[2])) if self._down else None
        return out, down

    def forward_latency(self, size, alphas, ratios):
        # ratios: (in, out, down)
        out = self._op.forward_latency(size, alphas, (ratios[0], ratios[1]))
        assert (self._down and (ratios[2] is not None)) or ((not self._down) and (ratios[2] is None))
        down = self.downsample.forward_latency(size, alphas, (ratios[0], ratios[2])) if self._down else None
        return out, down


class Network_Multi_Path(nn.Module):
    def __init__(self, num_classes=19, layers=16, criterion=nn.CrossEntropyLoss(ignore_index=-1), Fch=12, width_mult_list=[1.,], prun_modes=['arch_ratio',], stem_head_width=[(1., 1.),]):
        super(Network_Multi_Path, self).__init__()
        self._num_classes = num_classes
        assert layers >= 3
        self._layers = layers
        self._criterion = criterion
        self._Fch = Fch
        self._width_mult_list = width_mult_list
        self._prun_modes = prun_modes
        self.prun_mode = None # prun_mode is higher priority than _prun_modes
        self._stem_head_width = stem_head_width
        self._flops = 0
        self._params = 0

        self.stem = nn.ModuleList([
            nn.Sequential(
                ConvNorm(3, self.num_filters(2, stem_ratio)*2, kernel_size=3, stride=2, padding=1, bias=False, groups=1, slimmable=False),
                BasicResidual2x(self.num_filters(2, stem_ratio)*2, self.num_filters(4, stem_ratio)*2, kernel_size=3, stride=2, groups=1, slimmable=False),
                BasicResidual2x(self.num_filters(4, stem_ratio)*2, self.num_filters(8, stem_ratio), kernel_size=3, stride=2, groups=1, slimmable=False)
            ) for stem_ratio, _ in self._stem_head_width ])

        self.cells = nn.ModuleList()
        for l in range(layers):
            cells = nn.ModuleList()
            if l == 0:
                # first node has only one input (prev cell's output)
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list))
            elif l == 1:
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list))
            elif l < layers - 1:
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(32), down=False, width_mult_list=width_mult_list))
            else:
                cells.append(Cell(self.num_filters(8), down=False, width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(16), down=False, width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(32), down=False, width_mult_list=width_mult_list))
            self.cells.append(cells)

        self.refine32 = nn.ModuleList([
            nn.ModuleList([
                ConvNorm(self.num_filters(32, head_ratio), self.num_filters(16, head_ratio), kernel_size=1, bias=False, groups=1, slimmable=False),
                ConvNorm(self.num_filters(32, head_ratio), self.num_filters(16, head_ratio), kernel_size=3, padding=1, bias=False, groups=1, slimmable=False),
                ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=1, bias=False, groups=1, slimmable=False),
                ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=3, padding=1, bias=False, groups=1, slimmable=False)]) for _, head_ratio in self._stem_head_width ])
        self.refine16 = nn.ModuleList([
            nn.ModuleList([
                ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=1, bias=False, groups=1, slimmable=False),
                ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=3, padding=1, bias=False, groups=1, slimmable=False)]) for _, head_ratio in self._stem_head_width ])

        self.head0 = nn.ModuleList([ Head(self.num_filters(8, head_ratio), num_classes, False) for _, head_ratio in self._stem_head_width ])
        self.head1 = nn.ModuleList([ Head(self.num_filters(8, head_ratio), num_classes, False) for _, head_ratio in self._stem_head_width ])
        self.head2 = nn.ModuleList([ Head(self.num_filters(8, head_ratio), num_classes, False) for _, head_ratio in self._stem_head_width ])
        self.head02 = nn.ModuleList([ Head(self.num_filters(8, head_ratio)*2, num_classes, False) for _, head_ratio in self._stem_head_width ])
        self.head12 = nn.ModuleList([ Head(self.num_filters(8, head_ratio)*2, num_classes, False) for _, head_ratio in self._stem_head_width ])

        # contains arch_param names: {"alphas": alphas, "betas": betas, "ratios": ratios}
        self._arch_names = []
        self._arch_parameters = []
        for i in range(len(self._prun_modes)):
            arch_name, arch_param = self._build_arch_parameters(i)
            self._arch_names.append(arch_name)
            self._arch_parameters.append(arch_param)
            self._reset_arch_parameters(i)
        # switch set of arch if we have more than 1 arch
        self.arch_idx = 0
    
    def num_filters(self, scale, width=1.0):
        return int(np.round(scale * self._Fch * width))

    def new(self):
        model_new = Network(self._num_classes, self._layers, self._criterion, self._Fch).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
                x.data.copy_(y.data)
        return model_new

    def sample_prun_ratio(self, mode="arch_ratio"):
        '''
        mode: "min"|"max"|"random"|"arch_ratio"(default)
        '''
        assert mode in ["min", "max", "random", "arch_ratio"]
        if mode == "arch_ratio":
            ratios = self._arch_names[self.arch_idx]["ratios"]
            ratios0 = getattr(self, ratios[0])
            ratios0_sampled = []
            for layer in range(self._layers - 1):
                ratios0_sampled.append(gumbel_softmax(F.log_softmax(ratios0[layer], dim=-1), hard=True))
            ratios1 = getattr(self, ratios[1])
            ratios1_sampled = []
            for layer in range(self._layers - 1):
                ratios1_sampled.append(gumbel_softmax(F.log_softmax(ratios1[layer], dim=-1), hard=True))
            ratios2 = getattr(self, ratios[2])
            ratios2_sampled = []
            for layer in range(self._layers - 2):
                ratios2_sampled.append(gumbel_softmax(F.log_softmax(ratios2[layer], dim=-1), hard=True))
            return [ratios0_sampled, ratios1_sampled, ratios2_sampled]
        elif mode == "min":
            ratios0_sampled = []
            for layer in range(self._layers - 1):
                ratios0_sampled.append(self._width_mult_list[0])
            ratios1_sampled = []
            for layer in range(self._layers - 1):
                ratios1_sampled.append(self._width_mult_list[0])
            ratios2_sampled = []
            for layer in range(self._layers - 2):
                ratios2_sampled.append(self._width_mult_list[0])
            return [ratios0_sampled, ratios1_sampled, ratios2_sampled]
        elif mode == "max":
            ratios0_sampled = []
            for layer in range(self._layers - 1):
                ratios0_sampled.append(self._width_mult_list[-1])
            ratios1_sampled = []
            for layer in range(self._layers - 1):
                ratios1_sampled.append(self._width_mult_list[-1])
            ratios2_sampled = []
            for layer in range(self._layers - 2):
                ratios2_sampled.append(self._width_mult_list[-1])
            return [ratios0_sampled, ratios1_sampled, ratios2_sampled]
        elif mode == "random":
            ratios0_sampled = []
            for layer in range(self._layers - 1):
                ratios0_sampled.append(np.random.choice(self._width_mult_list))
            ratios1_sampled = []
            for layer in range(self._layers - 1):
                ratios1_sampled.append(np.random.choice(self._width_mult_list))
            ratios2_sampled = []
            for layer in range(self._layers - 2):
                ratios2_sampled.append(np.random.choice(self._width_mult_list))
            return [ratios0_sampled, ratios1_sampled, ratios2_sampled]

    def forward(self, input):
        # out_prev: cell-state
        # index 0: keep; index 1: down
        stem = self.stem[self.arch_idx]
        refine16 = self.refine16[self.arch_idx]
        refine32 = self.refine32[self.arch_idx]
        head0 = self.head0[self.arch_idx]
        head1 = self.head1[self.arch_idx]
        head2 = self.head2[self.arch_idx]
        head02 = self.head02[self.arch_idx]
        head12 = self.head12[self.arch_idx]

        alphas0 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["alphas"][0]), dim=-1)
        alphas1 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["alphas"][1]), dim=-1)
        alphas2 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["alphas"][2]), dim=-1)
        alphas = [alphas0, alphas1, alphas2]
        betas1 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["betas"][0]), dim=-1)
        betas2 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["betas"][1]), dim=-1)
        betas = [None, betas1, betas2]
        if self.prun_mode is not None:
            ratios = self.sample_prun_ratio(mode=self.prun_mode)
        else:
            ratios = self.sample_prun_ratio(mode=self._prun_modes[self.arch_idx])

        out_prev = [[stem(input), None]] # stem: one cell
        # i: layer | j: scale
        for i, cells in enumerate(self.cells):
            # layers
            out = []
            for j, cell in enumerate(cells):
                # scales
                # out,down -- 0: from down; 1: from keep
                out0 = None; out1 = None
                down0 = None; down1 = None
                alpha = alphas[j][i-j]
                # ratio: (in, out, down)
                # int: force #channel; tensor: arch_ratio; float(<=1): force width
                if i == 0 and j == 0:
                    # first cell
                    ratio = (self._stem_head_width[self.arch_idx][0], ratios[j][i-j], ratios[j+1][i-j])
                elif i == self._layers - 1:
                    # cell in last layer
                    if j == 0:
                        ratio = (ratios[j][i-j-1], self._stem_head_width[self.arch_idx][1], None)
                    else:
                        ratio = (ratios[j][i-j], self._stem_head_width[self.arch_idx][1], None)
                elif j == 2:
                    # cell in last scale: no down ratio "None"
                    ratio = (ratios[j][i-j], ratios[j][i-j+1], None)
                else:
                    if j == 0:
                        ratio = (ratios[j][i-j-1], ratios[j][i-j], ratios[j+1][i-j])
                    else:
                        ratio = (ratios[j][i-j], ratios[j][i-j+1], ratios[j+1][i-j])
                # out,down -- 0: from down; 1: from keep
                if j == 0:
                    out1, down1 = cell(out_prev[0][0], alpha, ratio)
                    out.append((out1, down1))
                else:
                    if i == j:
                        out0, down0 = cell(out_prev[j-1][1], alpha, ratio)
                        out.append((out0, down0))
                    else:
                        if betas[j][i-j-1][0] > 0:
                            out0, down0 = cell(out_prev[j-1][1], alpha, ratio)
                        if betas[j][i-j-1][1] > 0:
                            out1, down1 = cell(out_prev[j][0], alpha, ratio)
                        out.append((
                            sum(w * out for w, out in zip(betas[j][i-j-1], [out0, out1])),
                            sum(w * down if down is not None else 0 for w, down in zip(betas[j][i-j-1], [down0, down1])),
                            ))
            out_prev = out
        ###################################
        out0 = None; out1 = None; out2 = None

        out0 = out[0][0]
        out1 = F.interpolate(refine16[0](out[1][0]), scale_factor=2, mode='bilinear', align_corners=True)
        out1 = refine16[1](torch.cat([out1, out[0][0]], dim=1))
        out2 = F.interpolate(refine32[0](out[2][0]), scale_factor=2, mode='bilinear', align_corners=True)
        out2 = refine32[1](torch.cat([out2, out[1][0]], dim=1))
        out2 = F.interpolate(refine32[2](out2), scale_factor=2, mode='bilinear', align_corners=True)
        out2 = refine32[3](torch.cat([out2, out[0][0]], dim=1))

        pred0 = head0(out0)
        pred1 = head1(out1)
        pred2 = head2(out2)
        pred02 = head02(torch.cat([out0, out2], dim=1))
        pred12 = head12(torch.cat([out1, out2], dim=1))

        if not self.training:
            pred0 = F.interpolate(pred0, scale_factor=8, mode='bilinear', align_corners=True)
            pred1 = F.interpolate(pred1, scale_factor=8, mode='bilinear', align_corners=True)
            pred2 = F.interpolate(pred2, scale_factor=8, mode='bilinear', align_corners=True)
            pred02 = F.interpolate(pred02, scale_factor=8, mode='bilinear', align_corners=True)
            pred12 = F.interpolate(pred12, scale_factor=8, mode='bilinear', align_corners=True)
        return pred0, pred1, pred2, pred02, pred12
        ###################################
    
    def forward_latency(self, size, alpha=True, beta=True, ratio=True, scale_latency_weights=[3./12, 4./12, 5./12]):
        # out_prev: cell-state
        # index 0: keep; index 1: down
        stem = self.stem[self.arch_idx]

        if alpha:
            alphas0 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["alphas"][0]), dim=-1)
            alphas1 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["alphas"][1]), dim=-1)
            alphas2 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["alphas"][2]), dim=-1)
            alphas = [alphas0, alphas1, alphas2]
        else:
            alphas = [
                torch.ones_like(getattr(self, self._arch_names[self.arch_idx]["alphas"][0])).cuda() * 1./len(PRIMITIVES),
                torch.ones_like(getattr(self, self._arch_names[self.arch_idx]["alphas"][1])).cuda() * 1./len(PRIMITIVES),
                torch.ones_like(getattr(self, self._arch_names[self.arch_idx]["alphas"][2])).cuda() * 1./len(PRIMITIVES)]
        if beta:
            betas1 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["betas"][0]), dim=-1)
            betas2 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["betas"][1]), dim=-1)
            betas = [None, betas1, betas2]
        else:
            betas = [
                None,
                torch.ones_like(getattr(self, self._arch_names[self.arch_idx]["betas"][0])).cuda() * 1./2,
                torch.ones_like(getattr(self, self._arch_names[self.arch_idx]["betas"][1])).cuda() * 1./2]
        if ratio:
            # ratios = self.sample_prun_ratio(mode='arch_ratio')
            if self.prun_mode is not None:
                ratios = self.sample_prun_ratio(mode=self.prun_mode)
            else:
                ratios = self.sample_prun_ratio(mode=self._prun_modes[self.arch_idx])
        else:
            ratios = self.sample_prun_ratio(mode='max')

        stem_latency = 0
        latency, size = stem[0].forward_latency(size); stem_latency = stem_latency + latency
        latency, size = stem[1].forward_latency(size); stem_latency = stem_latency + latency
        latency, size = stem[2].forward_latency(size); stem_latency = stem_latency + latency
        out_prev = [[size, None]] # stem: one cell
        latency_total = [[stem_latency, 0], [0, 0], [0, 0]] # (out, down)

        # i: layer | j: scale
        for i, cells in enumerate(self.cells):
            # layers
            out = []
            latency = []
            for j, cell in enumerate(cells):
                # scales
                # out,down -- 0: from down; 1: from keep
                out0 = None; out1 = None
                down0 = None; down1 = None
                alpha = alphas[j][i-j]
                # ratio: (in, out, down)
                # int: force #channel; tensor: arch_ratio; float(<=1): force width
                if i == 0 and j == 0:
                    # first cell
                    ratio = (self._stem_head_width[self.arch_idx][0], ratios[j][i-j], ratios[j+1][i-j])
                elif i == self._layers - 1:
                    # cell in last layer
                    if j == 0:
                        ratio = (ratios[j][i-j-1], self._stem_head_width[self.arch_idx][1], None)
                    else:
                        ratio = (ratios[j][i-j], self._stem_head_width[self.arch_idx][1], None)
                elif j == 2:
                    # cell in last scale
                    ratio = (ratios[j][i-j], ratios[j][i-j+1], None)
                else:
                    if j == 0:
                        ratio = (ratios[j][i-j-1], ratios[j][i-j], ratios[j+1][i-j])
                    else:
                        ratio = (ratios[j][i-j], ratios[j][i-j+1], ratios[j+1][i-j])
                # out,down -- 0: from down; 1: from keep
                if j == 0:
                    out1, down1 = cell.forward_latency(out_prev[0][0], alpha, ratio)
                    out.append((out1[1], down1[1] if down1 is not None else None))
                    latency.append([out1[0], down1[0] if down1 is not None else None])
                else:
                    if i == j:
                        out0, down0 = cell.forward_latency(out_prev[j-1][1], alpha, ratio)
                        out.append((out0[1], down0[1] if down0 is not None else None))
                        latency.append([out0[0], down0[0] if down0 is not None else None])
                    else:
                        if betas[j][i-j-1][0] > 0:
                            # from down
                            out0, down0 = cell.forward_latency(out_prev[j-1][1], alpha, ratio)
                        if betas[j][i-j-1][1] > 0:
                            # from keep
                            out1, down1 = cell.forward_latency(out_prev[j][0], alpha, ratio)
                        assert (out0 is None and out1 is None) or out0[1] == out1[1]
                        assert (down0 is None and down1 is None) or down0[1] == down1[1]
                        out.append((out0[1], down0[1] if down0 is not None else None))
                        latency.append([
                            sum(w * out for w, out in zip(betas[j][i-j-1], [out0[0], out1[0]])),
                            sum(w * down if down is not None else 0 for w, down in zip(betas[j][i-j-1], [down0[0] if down0 is not None else None, down1[0] if down1 is not None else None])),
                        ])
            out_prev = out
            for ii, lat in enumerate(latency):
                # layer: i | scale: ii
                if ii == 0:
                    # only from keep
                    if lat[0] is not None: latency_total[ii][0] = latency_total[ii][0] + lat[0]
                    if lat[1] is not None: latency_total[ii][1] = latency_total[ii][0] + lat[1]
                else:
                    if i == ii:
                        # only from down
                        if lat[0] is not None: latency_total[ii][0] = latency_total[ii-1][1] + lat[0]
                        if lat[1] is not None: latency_total[ii][1] = latency_total[ii-1][1] + lat[1]
                    else:
                        if lat[0] is not None: latency_total[ii][0] = betas[j][i-j-1][1] * latency_total[ii][0] + betas[j][i-j-1][0] * latency_total[ii-1][1] + lat[0]
                        if lat[1] is not None: latency_total[ii][1] = betas[j][i-j-1][1] * latency_total[ii][0] + betas[j][i-j-1][0] * latency_total[ii-1][1] + lat[1]
        ###################################
        latency0 = latency_total[0][0]
        latency1 = latency_total[1][0]
        latency2 = latency_total[2][0]
        latency = sum(lat * w for lat, w in zip([latency0, latency1, latency2], scale_latency_weights))
        return latency
        ###################################

    def _loss(self, input, target, pretrain=False):
        loss = 0
        if pretrain is not True:
            # "random width": sampled by gambel softmax
            self.prun_mode = None
            for idx in range(len(self._arch_names)):
                self.arch_idx = idx
                logits = self(input)
                loss = loss + sum(self._criterion(logit, target) for logit in logits)
        if len(self._width_mult_list) > 1:
            self.prun_mode = "max"
            logits = self(input)
            loss = loss + sum(self._criterion(logit, target) for logit in logits)
            self.prun_mode = "min"
            logits = self(input)
            loss = loss + sum(self._criterion(logit, target) for logit in logits)
            if pretrain == True:
                self.prun_mode = "random"
                logits = self(input)
                loss = loss + sum(self._criterion(logit, target) for logit in logits)
                self.prun_mode = "random"
                logits = self(input)
                loss = loss + sum(self._criterion(logit, target) for logit in logits)
        elif pretrain == True and len(self._width_mult_list) == 1:
            self.prun_mode = "max"
            logits = self(input)
            loss = loss + sum(self._criterion(logit, target) for logit in logits)
        return loss

    def _build_arch_parameters(self, idx):
        num_ops = len(PRIMITIVES)

        # define names
        alphas = [ "alpha_"+str(idx)+"_"+str(scale) for scale in [0, 1, 2] ]
        betas = [ "beta_"+str(idx)+"_"+str(scale) for scale in [1, 2] ]

        setattr(self, alphas[0], nn.Parameter(Variable(1e-3*torch.ones(self._layers, num_ops).cuda(), requires_grad=True)))
        setattr(self, alphas[1], nn.Parameter(Variable(1e-3*torch.ones(self._layers-1, num_ops).cuda(), requires_grad=True)))
        setattr(self, alphas[2], nn.Parameter(Variable(1e-3*torch.ones(self._layers-2, num_ops).cuda(), requires_grad=True)))
        # betas are now in-degree probs
        # 0: from down; 1: from keep
        setattr(self, betas[0], nn.Parameter(Variable(1e-3*torch.ones(self._layers-2, 2).cuda(), requires_grad=True)))
        setattr(self, betas[1], nn.Parameter(Variable(1e-3*torch.ones(self._layers-3, 2).cuda(), requires_grad=True)))

        ratios = [ "ratio_"+str(idx)+"_"+str(scale) for scale in [0, 1, 2] ]
        if self._prun_modes[idx] == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1
        setattr(self, ratios[0], nn.Parameter(Variable(1e-3*torch.ones(self._layers-1, num_widths).cuda(), requires_grad=True)))
        setattr(self, ratios[1], nn.Parameter(Variable(1e-3*torch.ones(self._layers-1, num_widths).cuda(), requires_grad=True)))
        setattr(self, ratios[2], nn.Parameter(Variable(1e-3*torch.ones(self._layers-2, num_widths).cuda(), requires_grad=True)))
        return {"alphas": alphas, "betas": betas, "ratios": ratios}, [getattr(self, name) for name in alphas] + [getattr(self, name) for name in betas] + [getattr(self, name) for name in ratios]

    def _reset_arch_parameters(self, idx):
        num_ops = len(PRIMITIVES)
        if self._prun_modes[idx] == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1

        getattr(self, self._arch_names[idx]["alphas"][0]).data = Variable(1e-3*torch.ones(self._layers, num_ops).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["alphas"][1]).data = Variable(1e-3*torch.ones(self._layers-1, num_ops).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["alphas"][2]).data = Variable(1e-3*torch.ones(self._layers-2, num_ops).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["betas"][0]).data = Variable(1e-3*torch.ones(self._layers-2, 2).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["betas"][1]).data = Variable(1e-3*torch.ones(self._layers-3, 2).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["ratios"][0]).data = Variable(1e-3*torch.ones(self._layers-1, num_widths).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["ratios"][1]).data = Variable(1e-3*torch.ones(self._layers-1, num_widths).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["ratios"][2]).data = Variable(1e-3*torch.ones(self._layers-2, num_widths).cuda(), requires_grad=True)
