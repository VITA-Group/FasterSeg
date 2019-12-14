from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from thop import profile

from config_search import config
from dataloader import get_train_loader
from datasets import Cityscapes

from utils.init_func import init_weight
from seg_opr.loss_opr import ProbOhemCrossEntropy2d
from eval import SegEvaluator

from architect import Architect
from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_search import Network_Multi_Path as Network
from model_seg import Network_Multi_Path_Infer
import seg_metrics


def main(pretrain=True):
    config.save = 'search-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    assert type(pretrain) == bool or type(pretrain) == str
    update_arch = True
    if pretrain == True:
        update_arch = False
    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # config network and criterion ################
    min_kept = int(config.batch_size * config.image_height * config.image_width // (16 * config.gt_down_sampling ** 2))
    ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=min_kept, use_weight=False)

    # Model #######################################
    model = Network(config.num_classes, config.layers, ohem_criterion, Fch=config.Fch, width_mult_list=config.width_mult_list, prun_modes=config.prun_modes, stem_head_width=config.stem_head_width)
    flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 2048),))
    logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
    model = model.cuda()
    if type(pretrain) == str:
        partial = torch.load(pretrain + "/weights.pt", map_location='cuda:0')
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)
    else:
        init_weight(model, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    architect = Architect(model, config)

    # Optimizer ###################################
    base_lr = config.lr
    parameters = []
    parameters += list(model.stem.parameters())
    parameters += list(model.cells.parameters())
    parameters += list(model.refine32.parameters())
    parameters += list(model.refine16.parameters())
    parameters += list(model.head0.parameters())
    parameters += list(model.head1.parameters())
    parameters += list(model.head2.parameters())
    parameters += list(model.head02.parameters())
    parameters += list(model.head12.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        lr=base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    # lr policy ##############################
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.978)

    # data loader ###########################
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'down_sampling': config.down_sampling}
    train_loader_model, train_sampler_model = get_train_loader(config, Cityscapes, portion=config.train_portion)
    train_loader_arch, train_sampler_arch = get_train_loader(config, Cityscapes, portion=config.train_portion-1)#, worker=0)

    evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), config.num_classes, config.image_mean,
                             config.image_std, model, config.eval_scale_array, config.eval_flip, 0, config=config,
                             verbose=False, save_path=None, show_image=False)

    if update_arch:
        for idx in range(len(config.latency_weight)):
            logger.add_scalar("arch/latency_weight%d"%idx, config.latency_weight[idx], 0)
            logging.info("arch_latency_weight%d = "%idx + str(config.latency_weight[idx]))

    tbar = tqdm(range(config.nepochs), ncols=80)
    valid_mIoU_history = []; FPSs_history = [];
    latency_supernet_history = []; latency_weight_history = [];
    valid_names = ["8s", "16s", "32s", "8s_32s", "16s_32s"]
    arch_names = {0: "teacher", 1: "student"}
    for epoch in tbar:
        logging.info(pretrain)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        logging.info("update arch: " + str(update_arch))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(pretrain, train_loader_model, train_loader_arch, model, architect, ohem_criterion, optimizer, lr_policy, logger, epoch, update_arch=update_arch)
        torch.cuda.empty_cache()
        lr_policy.step()

        # validation
        tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
        with torch.no_grad():
            if pretrain == True:
                model.prun_mode = "min"
                valid_mIoUs = infer(epoch, model, evaluator, logger, FPS=False)
                for i in range(5):
                    logger.add_scalar('mIoU/val_min_%s'%valid_names[i], valid_mIoUs[i], epoch)
                    logging.info("Epoch %d: valid_mIoU_min_%s %.3f"%(epoch, valid_names[i], valid_mIoUs[i]))
                if len(model._width_mult_list) > 1:
                    model.prun_mode = "max"
                    valid_mIoUs = infer(epoch, model, evaluator, logger, FPS=False)
                    for i in range(5):
                        logger.add_scalar('mIoU/val_max_%s'%valid_names[i], valid_mIoUs[i], epoch)
                        logging.info("Epoch %d: valid_mIoU_max_%s %.3f"%(epoch, valid_names[i], valid_mIoUs[i]))
                    model.prun_mode = "random"
                    valid_mIoUs = infer(epoch, model, evaluator, logger, FPS=False)
                    for i in range(5):
                        logger.add_scalar('mIoU/val_random_%s'%valid_names[i], valid_mIoUs[i], epoch)
                        logging.info("Epoch %d: valid_mIoU_random_%s %.3f"%(epoch, valid_names[i], valid_mIoUs[i]))
            else:
                valid_mIoUss = []; FPSs = []
                model.prun_mode = None
                for idx in range(len(model._arch_names)):
                    # arch_idx
                    model.arch_idx = idx
                    valid_mIoUs, fps0, fps1 = infer(epoch, model, evaluator, logger)
                    valid_mIoUss.append(valid_mIoUs)
                    FPSs.append([fps0, fps1])
                    for i in range(5):
                        # preds
                        logger.add_scalar('mIoU/val_%s_%s'%(arch_names[idx], valid_names[i]), valid_mIoUs[i], epoch)
                        logging.info("Epoch %d: valid_mIoU_%s_%s %.3f"%(epoch, arch_names[idx], valid_names[i], valid_mIoUs[i]))
                    if config.latency_weight[idx] > 0:
                        logger.add_scalar('Objective/val_%s_8s_32s'%arch_names[idx], objective_acc_lat(valid_mIoUs[3], 1000./fps0), epoch)
                        logging.info("Epoch %d: Objective_%s_8s_32s %.3f"%(epoch, arch_names[idx], objective_acc_lat(valid_mIoUs[3], 1000./fps0)))
                        logger.add_scalar('Objective/val_%s_16s_32s'%arch_names[idx], objective_acc_lat(valid_mIoUs[4], 1000./fps1), epoch)
                        logging.info("Epoch %d: Objective_%s_16s_32s %.3f"%(epoch, arch_names[idx], objective_acc_lat(valid_mIoUs[4], 1000./fps1)))
                valid_mIoU_history.append(valid_mIoUss)
                FPSs_history.append(FPSs)
                if update_arch:
                    latency_supernet_history.append(architect.latency_supernet)
                latency_weight_history.append(architect.latency_weight)

        save(model, os.path.join(config.save, 'weights.pt'))
        if type(pretrain) == str:
            # contains arch_param names: {"alphas": alphas, "betas": betas, "gammas": gammas, "ratios": ratios}
            for idx, arch_name in enumerate(model._arch_names):
                state = {}
                for name in arch_name['alphas']:
                    state[name] = getattr(model, name)
                for name in arch_name['betas']:
                    state[name] = getattr(model, name)
                for name in arch_name['ratios']:
                    state[name] = getattr(model, name)
                state["mIoU02"] = valid_mIoUs[3]
                state["mIoU12"] = valid_mIoUs[4]
                if pretrain is not True:
                    state["latency02"] = 1000. / fps0
                    state["latency12"] = 1000. / fps1
                torch.save(state, os.path.join(config.save, "arch_%d_%d.pt"%(idx, epoch)))
                torch.save(state, os.path.join(config.save, "arch_%d.pt"%(idx)))

        if update_arch:
            for idx in range(len(config.latency_weight)):
                if config.latency_weight[idx] > 0:
                    if (int(FPSs[idx][0] >= config.FPS_max[idx]) + int(FPSs[idx][1] >= config.FPS_max[idx])) >= 1:
                        architect.latency_weight[idx] /= 2
                    elif (int(FPSs[idx][0] <= config.FPS_min[idx]) + int(FPSs[idx][1] <= config.FPS_min[idx])) > 0:
                        architect.latency_weight[idx] *= 2
                    logger.add_scalar("arch/latency_weight_%s"%arch_names[idx], architect.latency_weight[idx], epoch+1)
                    logging.info("arch_latency_weight_%d = "%arch_names[idx] + str(architect.latency_weight[idx]))


def train(pretrain, train_loader_model, train_loader_arch, model, architect, criterion, optimizer, lr_policy, logger, epoch, update_arch=True):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)

    for step in pbar:
        lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        minibatch = dataloader_model.next()
        imgs = minibatch['data']
        target = minibatch['label']
        imgs = imgs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if update_arch:
            # get a random minibatch from the search queue with replacement
            pbar.set_description("[Arch Step %d/%d]" % (step + 1, len(train_loader_model)))
            minibatch = dataloader_arch.next()
            imgs_search = minibatch['data']
            target_search = minibatch['label']
            imgs_search = imgs_search.cuda(non_blocking=True)
            target_search = target_search.cuda(non_blocking=True)
            loss_arch = architect.step(imgs, target, imgs_search, target_search)
            if (step+1) % 10 == 0:
                logger.add_scalar('loss_arch/train', loss_arch, epoch*len(pbar)+step)
                logger.add_scalar('arch/latency_supernet', architect.latency_supernet, epoch*len(pbar)+step)

        loss = model._loss(imgs, target, pretrain)
        logger.add_scalar('loss/train', loss, epoch*len(pbar)+step)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))
    torch.cuda.empty_cache()
    del loss
    if update_arch: del loss_arch


def infer(epoch, model, evaluator, logger, FPS=True):
    model.eval()
    mIoUs = []
    for idx in range(5):
        evaluator.out_idx = idx
        # _, mIoU = evaluator.run_online()
        _, mIoU = evaluator.run_online_multiprocess()
        mIoUs.append(mIoU)
    if FPS:
        fps0, fps1 = arch_logging(model, config, logger, epoch)
        return mIoUs, fps0, fps1
    else:
        return mIoUs


def arch_logging(model, args, logger, epoch):
    input_size = (1, 3, 1024, 2048)
    net = Network_Multi_Path_Infer(
        [getattr(model, model._arch_names[model.arch_idx]["alphas"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["alphas"][1]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["alphas"][2]).clone().detach()],
        [None, getattr(model, model._arch_names[model.arch_idx]["betas"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["betas"][1]).clone().detach()],
        [getattr(model, model._arch_names[model.arch_idx]["ratios"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["ratios"][1]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["ratios"][2]).clone().detach()],
        num_classes=model._num_classes, layers=model._layers, Fch=model._Fch, width_mult_list=model._width_mult_list, stem_head_width=model._stem_head_width[model.arch_idx])

    plot_op(net.ops0, net.path0, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops0_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)
    plot_op(net.ops1, net.path1, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops1_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)
    plot_op(net.ops2, net.path2, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops2_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)

    net.build_structure([2, 0])
    net = net.cuda()
    net.eval()
    latency0, _ = net.forward_latency(input_size[1:])
    logger.add_scalar("arch/fps0_arch%d"%model.arch_idx, 1000./latency0, epoch)
    logger.add_figure("arch/path_width_arch%d_02"%model.arch_idx, plot_path_width([2, 0], [net.path2, net.path0], [net.widths2, net.widths0]), epoch)

    net.build_structure([2, 1])
    net = net.cuda()
    net.eval()
    latency1, _ = net.forward_latency(input_size[1:])
    logger.add_scalar("arch/fps1_arch%d"%model.arch_idx, 1000./latency1, epoch)
    logger.add_figure("arch/path_width_arch%d_12"%model.arch_idx, plot_path_width([2, 1], [net.path2, net.path1], [net.widths2, net.widths1]), epoch)

    return 1000./latency0, 1000./latency1


if __name__ == '__main__':
    main(pretrain=config.pretrain) 
