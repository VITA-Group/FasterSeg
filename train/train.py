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
from thop import profile

from config_train_student import config
config.save = 'train-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
from dataloader import get_train_loader
from datasets import Cityscapes

from utils.init_func import init_weight
from seg_opr.loss_opr import ProbOhemCrossEntropy2d
from eval import SegEvaluator
from test import SegTester

from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_seg_8s import Network_Multi_Path_Infer as Network
import seg_metrics



def adjust_learning_rate(base_lr, power, optimizer, epoch, total_epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * power


def main():
    create_exp_dir(config.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
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
    distill_criterion = nn.KLDivLoss()

    # data loader ###########################
    if config.is_test:
        data_setting = {'img_root': config.img_root_folder,
                        'gt_root': config.gt_root_folder,
                        'train_source': config.train_eval_source,
                        'eval_source': config.eval_source,
                        'test_source': config.test_source,
                        'down_sampling': config.down_sampling}
    else:
        data_setting = {'img_root': config.img_root_folder,
                        'gt_root': config.gt_root_folder,
                        'train_source': config.train_source,
                        'eval_source': config.eval_source,
                        'test_source': config.test_source,
                        'down_sampling': config.down_sampling}

    train_loader, train_sampler = get_train_loader(config, Cityscapes, test=config.is_test)


    # Model #######################################
    models = []
    optimizers = []
    evaluators = []
    testers = []
    lasts = []
    for idx, arch_idx in enumerate(config.arch_idx):
        if config.load_epoch == "last":
            state = torch.load(os.path.join(config.load_path, "arch_%d.pt"%arch_idx))
        else:
            state = torch.load(os.path.join(config.load_path, "arch_%d_%d.pt"%(arch_idx, int(config.load_epoch))))

        model = Network(
            [state["alpha_%d_0"%arch_idx].detach(), state["alpha_%d_1"%arch_idx].detach(), state["alpha_%d_2"%arch_idx].detach()],
            [None, state["beta_%d_1"%arch_idx].detach(), state["beta_%d_2"%arch_idx].detach()],
            [state["ratio_%d_0"%arch_idx].detach(), state["ratio_%d_1"%arch_idx].detach(), state["ratio_%d_2"%arch_idx].detach()],
            num_classes=config.num_classes, layers=config.layers, Fch=config.Fch, width_mult_list=config.width_mult_list, stem_head_width=config.stem_head_width[idx], ignore_skip=arch_idx==0)

        mIoU02 = state["mIoU02"]; latency02 = state["latency02"]; obj02 = objective_acc_lat(mIoU02, latency02)
        mIoU12 = state["mIoU12"]; latency12 = state["latency12"]; obj12 = objective_acc_lat(mIoU12, latency12)
        if obj02 > obj12: last = [2, 0]
        else: last = [2, 1]
        lasts.append(last)
        model.build_structure(last)
        logging.info("net: " + str(model))
        for b in last:
            if len(config.width_mult_list) > 1:
                plot_op(getattr(model, "ops%d"%b), getattr(model, "path%d"%b), width=getattr(model, "widths%d"%b), head_width=config.stem_head_width[idx][1], F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png"%(arch_idx,b)), bbox_inches="tight")
            else:
                plot_op(getattr(model, "ops%d"%b), getattr(model, "path%d"%b), F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png"%(arch_idx,b)), bbox_inches="tight")
        plot_path_width(model.lasts, model.paths, model.widths).savefig(os.path.join(config.save, "path_width%d.png"%arch_idx))
        plot_path_width([2, 1, 0], [model.path2, model.path1, model.path0], [model.widths2, model.widths1, model.widths0]).savefig(os.path.join(config.save, "path_width_all%d.png"%arch_idx))
        flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 2048),))
        logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
        logging.info("ops:" + str(model.ops))
        logging.info("path:" + str(model.paths))
        logging.info("last:" + str(model.lasts))
        model = model.cuda()
        init_weight(model, nn.init.kaiming_normal_, torch.nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')

        if config.is_eval:
            partial = torch.load(os.path.join(config.eval_path, "/weights%d.pt"%arch_idx))
            state = model.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state}
            state.update(pretrained_dict)
            model.load_state_dict(state)
        else:
            if arch_idx == 0 and len(config.arch_idx) > 1:
                partial = torch.load(os.path.join(config.teacher_path, "/weights%d.pt"%arch_idx))
                state = model.state_dict()
                pretrained_dict = {k: v for k, v in partial.items() if k in state}
                state.update(pretrained_dict)
                model.load_state_dict(state)

        evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), config.num_classes, config.image_mean,
                                 config.image_std, model, config.eval_scale_array, config.eval_flip, 0, out_idx=0, config=config,
                                 verbose=False, save_path=None, show_image=False)
        evaluators.append(evaluator)
        tester = SegTester(Cityscapes(data_setting, 'test', None), config.num_classes, config.image_mean,
                                 config.image_std, model, config.eval_scale_array, config.eval_flip, 0, out_idx=0, config=config,
                                 verbose=False, save_path=None, show_image=False)
        testers.append(tester)

        # Optimizer ###################################
        base_lr = config.lr
        if arch_idx == 1 or len(config.arch_idx) == 1:
            # optimize teacher solo OR student (w. distill from teacher)
            optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        models.append(model)


    # Cityscapes ###########################################
    if config.is_eval:
        logging.info(config.load_path)
        logging.info(config.eval_path)
        logging.info(config.save)
        # validation
        tbar.set_description("[validation...]")
        with torch.no_grad():
            valid_mIoUs = infer(models, evaluators, logger)
            for idx, arch_idx in enumerate(config.arch_idx):
                logger.add_scalar("mIoU/val%d"%arch_idx, valid_mIoUs[idx], epoch)
                logging.info("valid_mIoU%d %.3f"%(arch_idx, valid_mIoUs[idx]))

    tbar = tqdm(range(config.nepochs), ncols=80)
    for epoch in tbar:
        logging.info(config.load_path)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))
        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train_mIoUs = train(train_loader, models, ohem_criterion, distill_criterion, optimizer, logger, epoch)
        torch.cuda.empty_cache()
        for idx, arch_idx in enumerate(config.arch_idx):
            logger.add_scalar("mIoU/train%d"%arch_idx, train_mIoUs[idx], epoch)
            logging.info("train_mIoU%d %.3f"%(arch_idx, train_mIoUs[idx]))
        adjust_learning_rate(base_lr, 0.992, optimizer, epoch+1, config.nepochs)

        # validation
        if not config.is_test and ((epoch+1) % 10 == 0 or epoch == 0):
            tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
            with torch.no_grad():
                valid_mIoUs = infer(models, evaluators, logger)
                for idx, arch_idx in enumerate(config.arch_idx):
                    logger.add_scalar("mIoU/val%d"%arch_idx, valid_mIoUs[idx], epoch)
                    logging.info("Epoch %d: valid_mIoU%d %.3f"%(epoch, arch_idx, valid_mIoUs[idx]))
                    save(models[idx], os.path.join(config.save, "weights%d.pt"%arch_idx))
        # test
        if config.is_test and (epoch+1) >= 250 and (epoch+1) % 10 == 0:
            tbar.set_description("[Epoch %d/%d][test...]" % (epoch + 1, config.nepochs))
            with torch.no_grad():
                test(epoch, models, testers, logger)

        for idx, arch_idx in enumerate(config.arch_idx):
            save(models[idx], os.path.join(config.save, "weights%d.pt"%arch_idx))


def train(train_loader, models, criterion, distill_criterion, optimizer, logger, epoch):
    if len(models) == 1:
        # train teacher solo
        models[0].train()
    else:
        # train student (w. distill from teacher)
        models[0].eval()
        models[1].train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader = iter(train_loader)

    metrics = [ seg_metrics.Seg_Metrics(n_classes=config.num_classes) for _ in range(len(models)) ]
    lamb = 0.2
    for step in pbar:
        current_idx = epoch * config.niters_per_epoch + step
        optimizer.zero_grad()

        minibatch = dataloader.next()
        imgs = minibatch['data']
        target = minibatch['label']
        imgs = imgs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits_list = []
        loss = 0
        loss_kl = 0
        description = ""
        for idx, arch_idx in enumerate(config.arch_idx):
            model = models[idx]
            if arch_idx == 0 and len(models) > 1:
                with torch.no_grad():
                    logits8 = model(imgs)
                    logits_list.append(logits8)
            else:
                logits8, logits16, logits32 = model(imgs)
                logits_list.append(logits8)
                loss = loss + criterion(logits8, target)
                loss = loss + lamb * criterion(logits16, target)
                loss = loss + lamb * criterion(logits32, target)
                if len(logits_list) > 1:
                    loss = loss + distill_criterion(F.softmax(logits_list[1], dim=1).log(), F.softmax(logits_list[0], dim=1))

            metrics[idx].update(logits8.data, target)
            description += "[mIoU%d: %.3f]"%(arch_idx, metrics[idx].get_scores())

        pbar.set_description("[Step %d/%d]"%(step + 1, len(train_loader)) + description)
        logger.add_scalar('loss/train', loss+loss_kl, epoch*len(pbar)+step)

        loss.backward()
        optimizer.step()

    return [ metric.get_scores() for metric in metrics ]


def infer(models, evaluators, logger):
    mIoUs = []
    for model, evaluator in zip(models, evaluators):
        model.eval()
        # _, mIoU = evaluator.run_online()
        _, mIoU = evaluator.run_online_multiprocess()
        mIoUs.append(mIoU)
    return mIoUs

def test(epoch, models, testers, logger):
    for idx, arch_idx in enumerate(config.arch_idx):
        model = models[idx]
        tester = testers[idx]
        if arch_idx == 0: continue
        os.system("mkdir %s"%os.path.join(os.path.join(os.path.realpath('.'), config.save, "test")))
        model.eval()
        tester.run_online()
        os.system("mv %s %s"%(os.path.join(os.path.realpath('.'), config.save, "test"), os.path.join(os.path.realpath('.'), config.save, "test_%d_%d"%(arch_idx, epoch))))


if __name__ == '__main__':
    main() 
