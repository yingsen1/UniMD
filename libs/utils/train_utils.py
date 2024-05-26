import argparse
import copy
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

import shutil
import time
import pickle
from pathlib import Path

import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import ipdb
import math
import json
from prettytable import PrettyTable as pt
from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm
from ..modeling.BiFPN import BiFPNBLOCK, BiFPN1D, BiFPN1D_ConvNeXt
from ..modeling.query_cls import QueryScaleClassifier
from ..modeling.convnext.convnext import LayerNorm2, MaskConvNextBlock
from .charades_sta_utils import MREvaluator
from .charades_utils import convert_result_to_charades, Charades_v1_localize
from .ego4d_utils import evaluate_nlq_performance, display_results, convert_result_to_ego4d
from .anet_utils import eval_result as anet_mr_eval
from .anet_utils import convert_result_to_anet_cap


################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (
        torch.nn.Linear,
        torch.nn.Conv1d,
        MaskedConv1D,
        torch.nn.Parameter,
    )
    # blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)
    blacklist_weight_modules = (
        LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        LayerNorm2,  # from convnext,
        QueryScaleClassifier,
    )
    bifpn_list = (
        BiFPN1D, BiFPN1D_ConvNeXt,
    )  # for _att
    convnext_list = (
        MaskConvNextBlock
    )  # for gamma

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            # print(fpn)
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            ### no decay ###
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)
            elif pn.endswith("_att") and isinstance(m, bifpn_list):
                no_decay.add(fpn)  # from bi-fpn
            elif pn.endswith("gamma") and isinstance(m, convnext_list):
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # print(list(param_dict.keys()))
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
        optimizer,
        optimizer_config,
        num_iters_per_epoch,
        last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def load_complete(checkpoint, model, model_ema):
    print("#" * 10, " loading complete checkpoint ", "#" * 10)
    # args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
    return True


def load_wo_head(checkpoint, model, model_ema):
    print("#" * 10, " loading checkpoint wo head", "#" * 10)
    model_state_dict = checkpoint["state_dict"]
    ema_state_dict = checkpoint["state_dict_ema"]
    state_dict_needed = {
        k: v for k, v in model_state_dict.items() \
        if not k.startswith("module.cls_head") and \
           not k.startswith("module.reg_head")
    }
    ema_state_dict_needed = {
        k: v for k, v in ema_state_dict.items() \
        if not k.startswith("module.cls_head") and \
           not k.startswith("module.reg_head")
    }
    print("before load state dict")
    print(len(model.state_dict().keys()))

    hook1 = state_dict_needed["module.backbone.embd.0.conv.weight"]

    # update model
    ori_model_st = model.state_dict()
    ori_model_st.update(state_dict_needed)
    model.load_state_dict(ori_model_st)
    ori_ema_st = model_ema.module.state_dict()
    ori_ema_st.update(ema_state_dict_needed)
    model_ema.module.load_state_dict(ori_ema_st)
    # model.state_dict().update(state_dict_needed)
    # model_ema.state_dict().update(ema_state_dict_needed)

    # validate
    print("after load state dict")
    print(len(model.state_dict().keys()))
    hook2 = model.state_dict()["module.backbone.embd.0.conv.weight"]

    print("success or not", torch.equal(hook1, hook2))
    print(hook1 is hook2)

    return True


def cotrain_random_one_epoch(
        train_loader,
        model,
        optimizer,
        scheduler,
        curr_epoch,
        max_epoch,
        model_ema=None,
        clip_grad_l2norm=-1,
        tb_writer=None,
        print_freq=20,
        tad_loss_weight=1.0,
        mr_loss_weight=1.0,

):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, data in enumerate(train_loader, 0):
        video_list = []
        tasks = []
        valid_data_idx = []
        for data_i in range(len(data)):
            one_video = dict()
            one_tasks = []
            tad_data = data[data_i]["tad"]
            mr_data = data[data_i]["mr"]

            if tad_data is not None and tad_data["segments"] is not None:
                one_video["tad"] = tad_data
                one_tasks.append("tad")
            if mr_data is not None and mr_data["segments"] is not None:
                one_video["mr"] = mr_data
                one_tasks.append("mr")
            if len(one_video):
                valid_data_idx.append(data_i)
                video_list.append(one_video)
                tasks.append(one_tasks)
            else:
                pass

        if len(video_list) <= 0:
            print("warning: get no data, skip")
            continue

        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        losses, fpn_cls_val, _, fpn_mask = model(video_list, tasks, curr_epoch, max_epoch)

        tad_loss, mr_loss = 0, 0
        if losses["tad"]:
            tad_loss = losses["tad"]["final_loss"]
        if losses["mr"]:
            mr_loss = losses["mr"]["final_loss"]
        backward_loss = tad_loss * tad_loss_weight + mr_loss * mr_loss_weight
        backward_loss.backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            if losses["tad"]:
                for key, value in losses["tad"].items():
                    # init meter if necessary
                    key = key + "_tad"
                    if key not in losses_tracker:
                        losses_tracker[key] = AverageMeter()
                    # update
                    losses_tracker[key].update(value.item())
            if losses["mr"]:
                for key, value in losses["mr"].items():
                    # init meter if necessary
                    key = key + "_mr"
                    if key not in losses_tracker:
                        losses_tracker[key] = AverageMeter()
                    # update
                    losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if "final_loss" not in key:
                        # if key != "final_loss":
                        tag_dict[key] = value.val
                # tb_writer.add_scalars(
                #     'train/all_losses',
                #     tag_dict,
                #     global_step
                # )
                if losses["tad"]:
                    tb_writer.add_scalar(
                        "train/cls_loss_tad",
                        losses_tracker["cls_loss_tad"].val,
                        global_step
                    )
                    tb_writer.add_scalar(
                        "train/reg_loss_tad",
                        losses_tracker["reg_loss_tad"].val,
                        global_step
                    )
                    # final loss
                    tb_writer.add_scalar(
                        'train/final_loss_tad',
                        losses_tracker['final_loss_tad'].val,
                        global_step
                    )
                if losses["mr"]:
                    tb_writer.add_scalar(
                        "train/cls_loss_mr",
                        losses_tracker["cls_loss_mr"].val,
                        global_step
                    )
                    tb_writer.add_scalar(
                        "train/reg_loss_mr",
                        losses_tracker["reg_loss_mr"].val,
                        global_step
                    )
                    # final loss
                    tb_writer.add_scalar(
                        'train/final_loss_mr',
                        losses_tracker['final_loss_mr'].val,
                        global_step
                    )

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = ""
            for k, v in losses_tracker.items():
                if "final_loss" not in k:
                    continue
                else:
                    block3 += 'Loss {}{:.2f} ({:.2f})\n'.format(
                        k,
                        losses_tracker[k].val,
                        losses_tracker[k].avg
                    )
            block4 = ''
            for key, value in losses_tracker.items():
                if "final_loss" in key:
                    continue
                block4 += '\t{:s} {:.2f} ({:.2f})'.format(
                    key, value.val, value.avg
                )

            print('\t'.join([block1, block2, block3, block4]))

        # clean calculate map
        if losses["tad"]:
            del losses["tad"]["final_loss"]
        if losses["mr"]:
            del losses["mr"]["final_loss"]

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return


def cotrain_synchronized_one_epoch(
        loaders,
        model,
        optimizer,
        scheduler,
        curr_epoch,
        max_epoch,
        model_ema=None,
        clip_grad_l2norm=-1,
        tb_writer=None,
        print_freq=20,
        tad_loss_weight=1.0,
        mr_loss_weight=1.0,

):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    assert isinstance(loaders, list)  # include tad_loader, mr_loader, both_loader
    assert len(loaders) == 3
    tad_train_loader = loaders[0]
    mr_train_loader = loaders[1]
    both_train_loader = loaders[2]
    # number of iterations per epoch
    diff_iters = len(tad_train_loader)  # tad/mr loader only include the non-intersection part
    both_iters = len(both_train_loader)  # both loader include the intersection part
    num_iters = diff_iters + both_iters
    print("num iters each epoch: %d" % num_iters)
    # switch to train mode
    model.train()

    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    iter_tad_train, iter_mr_train = iter(tad_train_loader), iter(mr_train_loader)
    iter_both_train = iter(both_train_loader)

    # shuffle the loader(=tad_loader+both_loader)
    random_list = list(range(num_iters))
    random.shuffle(random_list)

    for iter_idx, random_num in enumerate(random_list):
        if random_num < both_iters:  # intersection part
            batch_data = next(iter_both_train)
        else:  # non-intersection part
            batch_tad = next(iter_tad_train)
            batch_mr = next(iter_mr_train)
            batch_data = batch_tad + batch_mr
        # combine together
        video_list = []
        tasks = []  # record task each video
        valid_data_idx = []
        for data_i in range(len(batch_data)):
            one_video = dict()
            one_tasks = []
            tad_data = batch_data[data_i]["tad"]
            mr_data = batch_data[data_i]["mr"]

            if tad_data is not None and tad_data["segments"] is not None:
                one_video["tad"] = tad_data
                one_tasks.append("tad")
            if mr_data is not None and mr_data["segments"] is not None:
                one_video["mr"] = mr_data
                one_tasks.append("mr")
            if len(one_video):
                valid_data_idx.append(data_i)
                video_list.append(one_video)
                tasks.append(one_tasks)
            else:
                pass

        if len(video_list) <= 0:
            print("warning: get no data, skip")
            continue

        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        losses, fpn_cls_val, _, fpn_mask = model(video_list, tasks, curr_epoch, max_epoch)

        tad_loss, mr_loss = 0, 0
        if losses["tad"]:
            tad_loss = losses["tad"]["final_loss"]
        if losses["mr"]:
            mr_loss = losses["mr"]["final_loss"]
        backward_loss = tad_loss * tad_loss_weight + mr_loss * mr_loss_weight
        backward_loss.backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            if losses["tad"]:
                for key, value in losses["tad"].items():
                    # init meter if necessary
                    key = key + "_tad"
                    if key not in losses_tracker:
                        losses_tracker[key] = AverageMeter()
                    # update
                    losses_tracker[key].update(value.item())
            if losses["mr"]:
                for key, value in losses["mr"].items():
                    # init meter if necessary
                    key = key + "_mr"
                    if key not in losses_tracker:
                        losses_tracker[key] = AverageMeter()
                    # update
                    losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if "final_loss" not in key:
                        # if key != "final_loss":
                        tag_dict[key] = value.val
                # tb_writer.add_scalars(
                #     'train/all_losses',
                #     tag_dict,
                #     global_step
                # )
                if losses["tad"]:
                    tb_writer.add_scalar(
                        "train/cls_loss_tad",
                        losses_tracker["cls_loss_tad"].val,
                        global_step
                    )
                    tb_writer.add_scalar(
                        "train/reg_loss_tad",
                        losses_tracker["reg_loss_tad"].val,
                        global_step
                    )
                    # final loss
                    tb_writer.add_scalar(
                        'train/final_loss_tad',
                        losses_tracker['final_loss_tad'].val,
                        global_step
                    )
                if losses["mr"]:
                    tb_writer.add_scalar(
                        "train/cls_loss_mr",
                        losses_tracker["cls_loss_mr"].val,
                        global_step
                    )
                    tb_writer.add_scalar(
                        "train/reg_loss_mr",
                        losses_tracker["reg_loss_mr"].val,
                        global_step
                    )
                    # final loss
                    tb_writer.add_scalar(
                        'train/final_loss_mr',
                        losses_tracker['final_loss_mr'].val,
                        global_step
                    )

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = ""
            for k, v in losses_tracker.items():
                if "final_loss" not in k:
                    continue
                else:
                    block3 += 'Loss {}{:.2f} ({:.2f})\n'.format(
                        k,
                        losses_tracker[k].val,
                        losses_tracker[k].avg
                    )
            block4 = ''
            for key, value in losses_tracker.items():
                if "final_loss" in key:
                    continue
                block4 += '\t{:s} {:.2f} ({:.2f})'.format(
                    key, value.val, value.avg
                )

            print('\t'.join([block1, block2, block3, block4]))

        # clean calculate map
        if losses["tad"]:
            del losses["tad"]["final_loss"]
        if losses["mr"]:
            del losses["mr"]["final_loss"]

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return


def valid_one_epoch_charades(val_loader, model, curr_epoch, ext_score_file=None, evaluator=None, output_file=None,
                             tb_writer=None, print_freq=20, MR_GT_JSON="./data/charades/Charades_v1_test.csv"):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    mr_evaluator = MREvaluator(detail=False)
    model.eval()  # switch to evaluate mode
    # dict for results (for our evaluation code)
    results = {'video-id': [], 't-start': [], 't-end': [], 'label': [], 'score': []}
    results_charades_format = []
    mr_results = {'video-id': [], 't-start': [], 't-end': [], 'label': [], 'score': []}
    mr_raw_data = val_loader.dataset.data_list["mr"]  # get MR groundtruth

    # loop over validation set
    start = time.time()
    for iter_idx, data in tqdm(enumerate(val_loader, 0)):
        video_list = []  # record video id
        tasks = []  # record tasks corresponding to specific video id

        for data_i in range(len(data)):
            one_video = dict()
            one_tasks = []
            tad_data = data[data_i]["tad"]
            mr_data = data[data_i]["mr"]
            if tad_data is not None:  # get tad task
                one_video["tad"] = tad_data
                one_tasks.append("tad")
            if mr_data is not None:  # get mr task
                one_video["mr"] = mr_data
                one_tasks.append("mr")
            if len(one_video):  # if get task
                video_list.append(one_video)
                tasks.append(one_tasks)
        if not video_list:
            print("warning: get no data, skip")
            continue

        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list, tasks)
            # ----- TAD ----
            # unpack the results into ANet format
            tad_output = output["tad"]
            if tad_output:
                num_vids = len(output["tad"])
                for vid_idx in range(num_vids):
                    results_charades_format += convert_result_to_charades(tad_output[vid_idx],
                                                                          video_list[vid_idx]["tad"])
                for vid_idx in range(num_vids):
                    if tad_output[vid_idx]['segments'].shape[0] > 0:
                        results['video-id'].extend(
                            [tad_output[vid_idx]['video_id']] * tad_output[vid_idx]['segments'].shape[0])
                        results['t-start'].append(tad_output[vid_idx]['segments'][:, 0])
                        results['t-end'].append(tad_output[vid_idx]['segments'][:, 1])
                        results['label'].append(tad_output[vid_idx]['labels'])
                        results['score'].append(tad_output[vid_idx]['scores'])

            # ----- MR ----
            # unpack the results into ANet format [MR]
            mr_output = output["mr"]
            if mr_output:
                num_vids = len(output["mr"])
                for vid_idx in range(num_vids):
                    if mr_output[vid_idx]['segments'].shape[0] > 0:
                        mr_results['video-id'].extend(
                            [mr_output[vid_idx]['video_id']] * mr_output[vid_idx]['segments'].shape[0])
                        mr_results['t-start'].append(mr_output[vid_idx]['segments'][:, 0])
                        mr_results['t-end'].append(mr_output[vid_idx]['segments'][:, 1])
                        mr_results['label'].append(mr_output[vid_idx]['labels'])
                        mr_results['score'].append(mr_output[vid_idx]['scores'])
            mr_evaluator.evalute(
                results=mr_output if mr_output is not None else [],
                gts=[mr_raw_data[video_list[0]["mr"]["video_id"]]] if "mr" in video_list[0] else [],
            )

        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()
            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    if results["label"]:
        results['t-start'] = torch.cat(results['t-start']).numpy()
        results['t-end'] = torch.cat(results['t-end']).numpy()
        results['label'] = torch.cat(results['label']).numpy()
        results['score'] = torch.cat(results['score']).numpy()
    if mr_results["label"]:
        mr_results['t-start'] = torch.cat(mr_results['t-start']).numpy()
        mr_results['t-end'] = torch.cat(mr_results['t-end']).numpy()
        mr_results['label'] = torch.cat(mr_results['label']).numpy()
        mr_results['score'] = torch.cat(mr_results['score']).numpy()

    if evaluator is not None:
        # whether use external score file
        if ext_score_file is not None and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        _, mAP, _ = evaluator.evaluate(results, verbose=True)  # get tad-map in anet standrad format
    else:
        # dump to a pickle file that can be directly used for evaluation
        mAP = 0.0
    # save tad result
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    # save MR results
    with open(output_file.replace("eval_results.pkl", "eval_results_mr.pkl"), "wb") as f:
        pickle.dump(mr_results, f)
    # save MR result in charades standrad format
    tad_charades_out_txt = output_file.replace("eval_results.pkl", "eval_results_tad.txt")
    with open(tad_charades_out_txt, "w") as o_f:
        for o in results_charades_format:
            o_f.write("%s\n" % " ".join(list(map(str, o))))
    # ---- tad perform in charades standrad ----
    if len(results_charades_format) == 0:
        print("==results_charades_format empty!")
        tad_map = 0
    else:
        _, _, _, tad_map = Charades_v1_localize(tad_charades_out_txt, gtpath=MR_GT_JSON)
    print("final TAD mAP:", tad_map)
    # ---- mr perform ----
    mr_perform = mr_evaluator.summary()
    print("final MR performance:", mr_perform)

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)
        tb_writer.add_scalar("validation/tad-mAp", tad_map * 100, curr_epoch)
        tb_writer.add_scalar("validation/mr-R1@50", mr_perform["1-0.50"] * 100, curr_epoch)
        tb_writer.add_scalar("validation/mr-R1@70", mr_perform["1-0.70"] * 100, curr_epoch)
        tb_writer.add_scalar("validation/mr-R5@50", mr_perform["5-0.50"] * 100, curr_epoch)
        tb_writer.add_scalar("validation/mr-R5@70", mr_perform["5-0.70"] * 100, curr_epoch)

    return mAP, tad_map, mr_perform  # anet map, charades performance, mr performance


def valid_one_epoch_ego4d(val_loader, model, curr_epoch, ext_score_file=None, evaluator=None, output_file=None,
                          tb_writer=None, print_freq=20, GT_JSON="./data/ego4d/nlq/nlq_val.json"):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    model.eval()  # switch to evaluate mode

    # dict for results (for our evaluation code)
    results = {'video-id': [], 't-start': [], 't-end': [], 'label': [], 'score': []}
    mr_ego_results = {"version": "1.0", "challenge": "ego4d_nlq_challenge", "results": [], }

    # loop over validation set
    start = time.time()
    for iter_idx, data in tqdm(enumerate(val_loader, 0)):
        video_list = []  # record video id
        tasks = []  # record tasks corresponding to specific video id
        valid_data_idx = []  # valid idx
        mr_valid_idx = []  # mr

        for data_i in range(len(data)):
            one_video = dict()
            one_tasks = []
            tad_data = data[data_i]["tad"]
            mr_data = data[data_i]["mr"]
            if tad_data is not None:  # get tad task
                one_video["tad"] = tad_data
                one_tasks.append("tad")
            if mr_data is not None and mr_data["queries"] is not None:  # get mr task
                one_video["mr"] = mr_data
                one_tasks.append("mr")
                mr_valid_idx.append(data_i)
            if len(one_video):  # if get task
                valid_data_idx.append(data_i)
                video_list.append(one_video)
                tasks.append(one_tasks)
        if not video_list:
            print("warning: get no data, skip")
            continue

        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list, tasks)
            # ----- TAD ----
            # unpack the results into ANet format
            tad_output = output["tad"]
            if tad_output:
                num_vids = len(output["tad"])
                for vid_idx in range(num_vids):
                    if tad_output[vid_idx]['segments'].shape[0] > 0:
                        results['video-id'].extend(
                            [tad_output[vid_idx]['video_id']] * tad_output[vid_idx]['segments'].shape[0])
                        results['t-start'].append(tad_output[vid_idx]['segments'][:, 0])
                        results['t-end'].append(tad_output[vid_idx]['segments'][:, 1])
                        results['label'].append(tad_output[vid_idx]['labels'])
                        results['score'].append(tad_output[vid_idx]['scores'])

            # ----- MR ----
            # unpack the results into ANet format [MR]
            mr_output = output["mr"]
            mr_vid_qid_res = dict()
            for mr_idx in mr_valid_idx:
                mr_raw_data = data[mr_idx]["mr"]
                vid = mr_raw_data["video_id"]
                assert len(mr_raw_data["queries"]) == len(mr_raw_data["anno_uids"])
                for q_id in range(len(mr_raw_data["queries"])):
                    mr_vid_qid_res["%s_%d" % (vid, q_id)] = {
                        "clip_uid": vid,
                        "annotation_uid": mr_raw_data["anno_uids"][q_id],
                        "query_idx": int(mr_raw_data["anno_idxs"][q_id]),
                        "predicted_times": [], }

            if mr_output:
                mr_vid_qid_res = convert_result_to_ego4d(mr_output, mr_vid_qid_res)
                for k, v in mr_vid_qid_res.items():
                    mr_ego_results["results"].append(v)

        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()
            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    if results["label"]:
        results['t-start'] = torch.cat(results['t-start']).numpy()
        results['t-end'] = torch.cat(results['t-end']).numpy()
        results['label'] = torch.cat(results['label']).numpy()
        results['score'] = torch.cat(results['score']).numpy()

    if evaluator is not None:
        if ext_score_file is not None and isinstance(ext_score_file, str):  # whether use external score file
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        _, mAP, m_recall = evaluator.evaluate(results, verbose=True)
        recall_top1_50 = m_recall[-1][0] * 100  # R1@50
        print('R1@50: {:>4.2f} (%)'.format(recall_top1_50))
    else:
        # dump to a pickle file that can be directly used for evaluation
        mAP = 0.0
        recall_top1_50 = 0.0
    # save tad result
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    # save mr result
    with open(os.path.join(os.path.dirname(output_file), "mr_eval_results.json"), "w") as o_f:
        json.dump(mr_ego_results, o_f)
    with open(GT_JSON) as in_f:
        mr_gt = json.load(in_f)
    thresholds = [0.3, 0.5, 0.01]
    topK = [1, 3, 5]
    mr_perf_res, mr_mIoU = evaluate_nlq_performance(mr_ego_results["results"], mr_gt, thresholds, topK)
    score_str, score_dict = display_results(mr_perf_res, mr_mIoU, thresholds, topK, title="Ego4d_NLQ_MR")
    print("final MR performance:")
    print(score_str)
    mr_r1_30 = score_dict['Rank@1\nmIoU@0.3']
    mr_r1_50 = score_dict['Rank@1\nmIoU@0.5']
    mr_r5_30 = score_dict['Rank@5\nmIoU@0.3']
    mr_r5_50 = score_dict['Rank@5\nmIoU@0.5']
    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/tad-mAP', mAP, curr_epoch)
        tb_writer.add_scalar("validation/tad-R1@50", recall_top1_50, curr_epoch)
        # tb_writer.add_scalar("validation/tad-mAp", tad_map * 100, curr_epoch)
        tb_writer.add_scalar("validation/mr-R1@30", mr_r1_30, curr_epoch)
        tb_writer.add_scalar("validation/mr-R1@50", mr_r1_50, curr_epoch)
        tb_writer.add_scalar("validation/mr-R5@30", mr_r5_30, curr_epoch)
        tb_writer.add_scalar("validation/mr-R5@50", mr_r5_50, curr_epoch)

    return {"tad-map": mAP, "r1_50": recall_top1_50, }, mAP, {
        "r1_30": mr_r1_30,
        "r1_50": mr_r1_50,
        "r5_30": mr_r5_30,
        "r5_50": mr_r5_50, }  # tad_performance, tad-mAP, mr_performance


def valid_one_epoch_anet(val_loader, model, curr_epoch, ext_score_file=None, evaluator=None, output_file=None,
                         tb_writer=None, print_freq=20,
                         MR_GT_JSON_DIR="./data/anet/captiondata/"):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    model.eval()  # switch to evaluate mode
    # dict for results
    results = {'video-id': [], 't-start': [], 't-end': [], 'label': [], 'score': []}  # for tad
    mr_results = {'video-id': [], 't-start': [], 't-end': [], 'label': [], 'score': []}  # for mr
    mr_anet_results = dict()

    # loop over validation set
    start = time.time()
    for iter_idx, data in tqdm(enumerate(val_loader, 0)):
        video_list = []  # record video id
        tasks = []  # record tasks corresponding to specific video id

        for data_i in range(len(data)):
            one_video = dict()
            one_tasks = []
            tad_data = data[data_i]["tad"]
            mr_data = data[data_i]["mr"]
            if tad_data is not None:  # get tad task
                one_video["tad"] = tad_data
                one_tasks.append("tad")
            if mr_data is not None and mr_data["queries"] is not None:  # get mr task
                one_video["mr"] = mr_data
                one_tasks.append("mr")
            if len(one_video):  # if get task
                video_list.append(one_video)
                tasks.append(one_tasks)
        if not video_list:
            print("warning: get no data, skip")
            continue

        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list, tasks)
            # ----- TAD ----
            # unpack the results into ANet format
            tad_output = output["tad"]
            if tad_output:
                num_vids = len(output["tad"])
                for vid_idx in range(num_vids):
                    if tad_output[vid_idx]['segments'].shape[0] > 0:
                        results['video-id'].extend(
                            [tad_output[vid_idx]['video_id']] * tad_output[vid_idx]['segments'].shape[0])
                        results['t-start'].append(tad_output[vid_idx]['segments'][:, 0])
                        results['t-end'].append(tad_output[vid_idx]['segments'][:, 1])
                        results['label'].append(tad_output[vid_idx]['labels'])
                        results['score'].append(tad_output[vid_idx]['scores'])

            # ----- MR ----
            # unpack the results into ANet format [MR]
            mr_output = output["mr"]
            mr_vid_qid_res = dict()
            if mr_output:
                num_vids = len(output["mr"])
                for vid_idx in range(num_vids):
                    if mr_output[vid_idx]['segments'].shape[0] > 0:
                        mr_results['video-id'].extend(
                            [mr_output[vid_idx]['video_id']] * mr_output[vid_idx]['segments'].shape[0])
                        mr_results['t-start'].append(mr_output[vid_idx]['segments'][:, 0])
                        mr_results['t-end'].append(mr_output[vid_idx]['segments'][:, 1])
                        mr_results['label'].append(mr_output[vid_idx]['labels'])
                        mr_results['score'].append(mr_output[vid_idx]['scores'])
                mr_vid_qid_res = convert_result_to_anet_cap(mr_output, mr_vid_qid_res)
                for k, v in mr_vid_qid_res.items():
                    mr_anet_results[k] = v

        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()
            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    if results["label"]:
        results['t-start'] = torch.cat(results['t-start']).numpy()
        results['t-end'] = torch.cat(results['t-end']).numpy()
        results['label'] = torch.cat(results['label']).numpy()
        results['score'] = torch.cat(results['score']).numpy()
    if mr_results["label"]:
        mr_results['t-start'] = torch.cat(mr_results['t-start']).numpy()
        mr_results['t-end'] = torch.cat(mr_results['t-end']).numpy()
        mr_results['label'] = torch.cat(mr_results['label']).numpy()
        mr_results['score'] = torch.cat(mr_results['score']).numpy()

    if evaluator is not None:
        # convert vidoe name, match video feature name to that in groundturth
        new_vid_ids = []
        for v_id in results["video-id"]:
            if v_id.startswith("v_"):
                v_id = v_id[2:]
            new_vid_ids.append(v_id)
        results["video-id"] = new_vid_ids
        # whether use external score file
        if ext_score_file is not None and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator function
        mAP_under_ious, mAP, m_recall = evaluator.evaluate(results, verbose=True)  # mAP
        map_50 = mAP_under_ious[0] * 100  # mAP@50
        print('mAP@50 = {:>4.2f} (%) '.format(map_50))
    else:
        # dump to a pickle file that can be directly used for evaluation
        mAP = 0.0
        map_50 = 0.0
    # save tad result
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    # save mr result
    with open(output_file.replace("eval_results.pkl", "eval_results_mr.pkl"), "wb") as f:
        pickle.dump(mr_results, f)
    mr_output_file = os.path.join(os.path.dirname(output_file), "anet_mr_result.json")
    with open(mr_output_file, "w") as o_f:
        json.dump({"results": mr_anet_results}, o_f)

    MR_GT_VAL = os.path.join(MR_GT_JSON_DIR, "val_1.json")
    MR_GT_TEST = os.path.join(MR_GT_JSON_DIR, "val_2.json")
    mr_scores = anet_mr_eval(mr_output_file, MR_GT_TEST)
    # r1_50, r1_70, r5_50, r5_70
    r1_50 = mr_scores["R@{}IOU{}".format(1, 0.5)] * 100
    r1_70 = mr_scores["R@{}IOU{}".format(1, 0.7)] * 100
    r5_50 = mr_scores["R@{}IOU{}".format(5, 0.5)] * 100
    r5_70 = mr_scores["R@{}IOU{}".format(5, 0.7)] * 100
    print("final MR performance:")
    print("R1@50, %.2f\nR1@70, %.2f\nR5@50, %.2f\nR5@70, %.2f" % (
        r1_50, r1_70, r5_50, r5_70
    ))

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/tad-mAP', mAP, curr_epoch)
        tb_writer.add_scalar("validation/tad-mAP@50", map_50, curr_epoch)
        # tb_writer.add_scalar("validation/tad-mAp", tad_map * 100, curr_epoch)
        tb_writer.add_scalar("validation/mr-R1@50", r1_50, curr_epoch)
        tb_writer.add_scalar("validation/mr-R1@70", r1_70, curr_epoch)
        tb_writer.add_scalar("validation/mr-R5@50", r5_50, curr_epoch)
        tb_writer.add_scalar("validation/mr-R5@70", r5_70, curr_epoch)

    return {"mAP": mAP, "ap-50": map_50}, mAP, {
        "r1_50": r1_50,
        "r1_70": r1_70,
        "r5_50": r5_50,
        "r5_70": r5_70, }  # tad_performance, tad-mAP, mr_performance
