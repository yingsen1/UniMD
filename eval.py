# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader, SUPPORT_DATASET
from libs.modeling import make_meta_arch
from libs.utils import ANETdetection, fix_random_seed
from libs.utils.train_utils import valid_one_epoch_charades, valid_one_epoch_ego4d, valid_one_epoch_anet


def print_highlight(message):
    print("#" * 10, " ", message, " ", "#" * 10)


################################################################################
def main(args):
    '''0. load config & check'''
    print_highlight(args.config)
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    '''1. fix all randomness'''
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    dataset = cfg_filename.split("_")[0]
    if dataset not in SUPPORT_DATASET:
        dataset = cfg["dataset_name"].split("_")[0]
    print_highlight(dataset)

    assert dataset in SUPPORT_DATASET, f"{dataset} is not supported currently"

    # specific dataset tasks, all: tad+mr
    assert args.data_type in ["all", "tad", "mr"]
    print_highlight("data_type " + args.data_type)
    data_split = args.eval_split_name + "_split"  # val_split, test_split
    print_highlight("data_split, %s" % data_split)
    val_dataset = make_dataset(cfg['dataset_name'], False, cfg[data_split], **cfg[dataset])
    val_dataset.get_type(args.data_type)  # set task for dataset

    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(val_dataset, False, None, 1, cfg['loader']['num_workers'])
    if args.eval_split_name == "test": val_dataset.no_gt = True

    '''3. create model and evaluator'''
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    '''4. load ckpt'''
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    '''5. set validation tool'''
    # anet for activitynet and activitynet-caption
    # charades for charades and charades-sta
    # ego4d for ego4d-mq and ego4d nlq
    if cfg["valid_type"] in ["charades", "ego4d"]:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds']
        )
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')
    elif cfg["valid_type"] == "anet":
        val_split = "validation" if "validation" in val_dataset.split else val_dataset.split[0]
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_split,
            tiou_thresholds=val_db_vars['tiou_thresholds']
        )
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')
    else:
        raise NotImplemented(f"{cfg['valid_type']} not implemented yet.")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    """5. inference"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    if cfg["valid_type"] == "anet":
        anet_perform, tad_map, mr_perform = valid_one_epoch_anet(
            val_loader,
            model=model,
            curr_epoch=-1,
            evaluator=det_eval if args.data_type != "mr" else None,  # det_eval
            output_file=output_file,
            ext_score_file=cfg['test_cfg']['ext_score_file'],
            tb_writer=None,
            print_freq=100,
        )
        mAP = tad_map
    elif cfg["valid_type"] == "charades":
        tad_map, charades_perform, mr_performan = valid_one_epoch_charades(
            val_loader,
            model=model,
            curr_epoch=-1,
            evaluator=det_eval,  # det_eval
            output_file=output_file,
            ext_score_file=cfg['test_cfg']['ext_score_file'],
            tb_writer=None,
            print_freq=100,
        )
        mAP = charades_perform
    elif cfg["valid_type"] == "ego4d":
        ego4d_perform, tad_map, mr_perform = valid_one_epoch_ego4d(
            val_loader,
            model=model,
            curr_epoch=-1,
            evaluator=det_eval if args.data_type != "mr" else None,  # det_eval
            output_file=output_file,
            ext_score_file=cfg['test_cfg']['ext_score_file'],
            tb_writer=None,
            print_freq=100,
        )
        mAP = tad_map

    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('--ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('--eval_split_name', type=str, default="val", )
    parser.add_argument('--save_model_output', action="store_true", help="save model output")
    parser.add_argument("--data_type", type=str, default="all")
    args = parser.parse_args()

    main(args)
