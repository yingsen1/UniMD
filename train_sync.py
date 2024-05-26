# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

import ipdb
# torch imports
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.sampler import Sampler
import yaml
# for visualization
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader, SUPPORT_DATASET
from libs.datasets.data_utils import trivial_batch_collator, worker_init_reset_seed
from libs.modeling import make_meta_arch
from libs.utils import (cotrain_random_one_epoch, cotrain_synchronized_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler, fix_random_seed, ModelEma)
from libs.utils.train_utils import (valid_one_epoch_charades, valid_one_epoch_ego4d, valid_one_epoch_anet)


def print_highlight(message):
    print("#" * 10, " ", message, " ", "#" * 10)


def save_config(d, path):
    with open(path, "w") as o_f:
        yaml.safe_dump(d, o_f)
    return


class FixLenSampler(Sampler):
    def __init__(self, data_source, num_samples):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):

        n = len(self.data_source)
        if self.num_samples <= n:  # > num_sample, abandon
            for i in torch.randperm(len(self.data_source))[:self.num_samples].tolist():
                yield i
        else:  # < num_sample, repeat sample
            for i in torch.randperm(n):
                yield i.item()
            for i in torch.randint(high=n, size=(self.num_samples - n,)):
                yield i.item()

    def __len__(self):
        return self.num_samples


def make_dataloader_with_fix_len(dataset, is_training, generator, batch_size, num_workers, fix_size):
    sampler = FixLenSampler(dataset, fix_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=False,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True,
        sampler=sampler
    )
    return loader


################################################################################
def main(args):
    """main function that handles training / inference
    """

    """0. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    print_highlight(args.config)
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    print_highlight("config")

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])
    save_config(cfg, os.path.join(ckpt_folder, "config.yaml"))
    tad_loss_weight = cfg["train_cfg"]["tad_loss_weight"] if "tad_loss_weight" in cfg[
        "train_cfg"] else 1.0  # loss weight
    mr_loss_weight = cfg["train_cfg"]["mr_loss_weight"] if "mr_loss_weight" in cfg["train_cfg"] else 1.0
    print_highlight("tad loss weight, %.2f, mr loss weight, %.2f" % (tad_loss_weight, mr_loss_weight))

    """1. create dataset / dataloader"""
    dataset = cfg_filename.split("_")[0]
    print_highlight(dataset)
    assert dataset in SUPPORT_DATASET, f"{dataset} is not supported currently"
    print_highlight("dataset_name: " + cfg["dataset_name"])

    # specific dataset
    assert args.data_type == "all", "co-training only support training all tasks together."
    print_highlight("data_type " + args.data_type)

    tad_train_dataset = make_dataset(cfg['dataset_name'], True, cfg['train_split'], **cfg[dataset])
    tad_train_dataset.get_type("tad")
    fetch_mode = tad_train_dataset.set_fetch_mode("diff")
    print_highlight("tad fetch mode :%s" % fetch_mode)
    mr_train_dataset = make_dataset(cfg['dataset_name'], True, cfg['train_split'], **cfg[dataset])
    mr_train_dataset.get_type("mr")
    fetch_mode = mr_train_dataset.set_fetch_mode("diff")
    print_highlight("mr fetch mode :%s" % fetch_mode)

    # intersection
    both_train_dataset = make_dataset(cfg['dataset_name'], True, cfg['train_split'], **cfg[dataset])
    fetch_mode = both_train_dataset.set_fetch_mode("both")
    print_highlight("both_part fetch mode :%s" % fetch_mode)

    tad_mr_min_len = min(len(tad_train_dataset), len(mr_train_dataset))
    print_highlight(
        "tad len: %d, mr len: %d, min len: %d" % (len(tad_train_dataset), len(mr_train_dataset), tad_mr_min_len))
    print_highlight("tad-mr both len: %d" % len(both_train_dataset))

    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = tad_train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    tad_train_loader = make_dataloader_with_fix_len(
        tad_train_dataset, True, rng_generator, fix_size=tad_mr_min_len, **cfg['loader'])
    mr_train_loader = make_dataloader_with_fix_len(
        mr_train_dataset, True, rng_generator, fix_size=tad_mr_min_len, **cfg['loader'])
    both_train_loader = make_data_loader(
        both_train_dataset, True, rng_generator, **cfg['loader'])

    # validation dataset, default None
    val_dataset = None
    val_loader = None
    if args.val_freq > 0:
        assert len(cfg['val_split']) > 0, "Test set must be specified!"
        val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg[dataset])
        val_dataset.get_type(args.data_type)
        # set bs = 1, and disable shuffle
        val_loader = make_data_loader(val_dataset, False, None, 1, cfg['loader']['num_workers'])
        # mkdir for raw val_data
        os.makedirs(os.path.join(ckpt_folder, "submission_data"), exist_ok=True)
        # validation setting
        if cfg["valid_type"] in ["charades", "ego4d"]:
            val_db_vars = val_dataset.get_attributes()
            det_eval = ANETdetection(
                val_dataset.json_file,
                val_dataset.split[0],
                tiou_thresholds=val_db_vars['tiou_thresholds']
            )
        elif cfg["valid_type"] == "anet":
            val_split = "validation" if "validation" in val_dataset.split else val_dataset.split[0]
            val_db_vars = val_dataset.get_attributes()
            det_eval = ANETdetection(
                val_dataset.json_file,
                val_split,
                tiou_thresholds=val_db_vars['tiou_thresholds']
            )
            if val_dataset.num_classes == 1:
                assert cfg['test_cfg']['ext_score_file'] is not None
                assert os.path.exists(cfg['test_cfg']['ext_score_file'])
        else:
            raise NotImplemented(f"{cfg['valid_type']} not implemented yet.")

    """2. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(tad_train_loader) + len(both_train_loader)  # tad(mr) + both
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print_highlight("Using model EMA ...")
    # print("Using model EMA ...")
    model_ema = ModelEma(model)

    """3. Resume from model / Misc"""
    best_map = 0
    assert not (bool(args.resume) and bool(args.checkpoint))
    if args.checkpoint:
        print_highlight("loading checkpoint, %s " % args.checkpoint)
        if os.path.isfile(args.checkpoint):
            # load ckpt
            checkpoint = torch.load(
                args.checkpoint,
                map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
            )
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.checkpoint, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best resume
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(
                                        cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_map = checkpoint["best_map"] if "best_map" in checkpoint else 0
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            model.module.set_iter_count(args.start_epoch * num_iters_per_epoch)
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print_highlight("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    model.module.set_tensorboard(tb_writer)

    for epoch in range(args.start_epoch, max_epochs):
        # train individually or cotrain w. random sampling
        cotrain_synchronized_one_epoch(
            [tad_train_loader, mr_train_loader, both_train_loader],
            model,
            optimizer,
            scheduler,
            epoch,
            max_epochs,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq,
            tad_loss_weight=tad_loss_weight,
            mr_loss_weight=mr_loss_weight,
        )
        best_model = False

        # validation skip some epoch
        if epoch < args.skip_val_epoch:
            continue

        # validation
        if args.val_freq > 0 and (epoch + 1) % args.val_freq == 0:
            if cfg["valid_type"] == "charades":
                # anet_map, tad_map, mr_perform = valid_one_epoch_charades(
                tad_map, charades_perform, mr_performan = valid_one_epoch_charades(
                    val_loader,
                    model=model_ema.module,
                    curr_epoch=epoch,
                    evaluator=det_eval,  # det_eval
                    output_file=os.path.join(ckpt_folder, "eval_results.pkl"),
                    ext_score_file=cfg['test_cfg']['ext_score_file'],
                    tb_writer=tb_writer,
                    print_freq=100,
                )
                mAP = charades_perform
            elif cfg["valid_type"] == "ego4d":
                ego4d_perform, tad_map, mr_perform = valid_one_epoch_ego4d(
                    val_loader,
                    model=model_ema.module,
                    curr_epoch=epoch,
                    evaluator=det_eval if args.data_type != "mr" else None,  # det_eval
                    output_file=os.path.join(ckpt_folder, "eval_results.pkl"),
                    ext_score_file=cfg['test_cfg']['ext_score_file'],
                    tb_writer=tb_writer,
                    print_freq=100,
                )
                mAP = ego4d_perform["tad-map"]
            elif cfg["valid_type"] == "anet":
                anet_perform, tad_map, mr_perform = valid_one_epoch_anet(
                    val_loader,
                    model=model_ema.module,
                    curr_epoch=epoch,
                    evaluator=det_eval if args.data_type != "mr" else None,  # det_eval
                    output_file=os.path.join(ckpt_folder, "eval_results.pkl"),
                    ext_score_file=cfg['test_cfg']['ext_score_file'],
                    tb_writer=tb_writer,
                    print_freq=100,
                )
                mAP = anet_perform["mAP"]
            best_map = max(mAP, best_map)
            # save best model
            if best_map == mAP:  best_model = True

        # save ckpt once in a while
        if (((epoch + 1) == max_epochs) or (
                (args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0)) or best_model):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_map': best_map,
            }
            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                best_model,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )

    # wrap up
    tb_writer.close()
    print("All done!")
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('--config', default="", type=str,
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--val-freq', default=1, type=int,
                        help='validation frequency (default: every 1 epochs), <0 means no val')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none), load ckpt and schedule')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none), only load ckpt')
    parser.add_argument('--skip-val-epoch', default=-1, type=int,
                        help='first epoch not to save checkpoint and validation')
    parser.add_argument("--data_type", type=str, default="all")
    args = parser.parse_args()

    main(args)
