import os
import copy
import random
import numpy as np
import random
import torch


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch


def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def truncate_feats(
        data_dict,
        max_seq_len,
        trunc_thresh,
        offset,
        crop_ratio=None,
        max_num_trials=200,
        has_action=True,
        no_trunc=False
):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {'video_id'        : str
                 'feats'           : Tensor C x T
                 'segments'        : Tensor N x 2 (in feature grid)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : int
                 'feat_num_frames' : in

    """
    # get the meta info
    feat_len = data_dict['feats'].shape[1]
    num_segs = data_dict['segments'].shape[0]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return data_dict
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return data_dict

    # otherwise, deep copy the dict
    data_dict = copy.deepcopy(data_dict)

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):

        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0] - offset, data_dict['segments'][:, 0])
        right = torch.minimum(window[:, 1] + offset, data_dict['segments'][:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs

        # only select those segments over the thresh
        seg_idx = (inter_ratio >= trunc_thresh)

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0:
                break
        else:
            # without any constraints
            break

    # feats: C x T
    data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    # segments: N x 2 in feature grids
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # shift the time stamps due to truncation
    data_dict['segments'] = data_dict['segments'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()
    # if "queries" in data_dict:  # mr
    #     # queies: N
    #     data_dict["queries"] = data_dict["queries"][seg_idx.numpy()].copy()
    #     # query_feats N x C
    #     data_dict["q_feats"] = data_dict["q_feats"][seg_idx, :].clone()

    return data_dict


def truncate_feats_tad_mr(
        tad_data_dict,
        mr_data_dict,
        max_seq_len,
        trunc_thresh,
        offset,
        crop_ratio=None,
        max_num_trials=200,
        has_action=True,
        no_trunc=False
):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {'video_id'        : str
                 'feats'           : Tensor C x T
                 'segments'        : Tensor N x 2 (in feature grid)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : int
                 'feat_num_frames' : in

    """
    # get the meta info
    feat_len = tad_data_dict['feats'].shape[1]
    tad_num_segs = tad_data_dict['segments'].shape[0]
    mr_num_segs = mr_data_dict["segments"].shape[0]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return tad_data_dict, mr_data_dict
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return tad_data_dict, mr_data_dict

    # otherwise, deep copy the dict
    tad_data_dict = copy.deepcopy(tad_data_dict)
    mr_data_dict = copy.deepcopy(mr_data_dict)

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):
        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(tad_num_segs + mr_num_segs, 1)
        total_segs = torch.concat([
            tad_data_dict["segments"], mr_data_dict["segments"]
        ], dim=0)
        left = torch.maximum(window[:, 0] - offset, total_segs[:, 0])
        right = torch.minimum(window[:, 1] + offset, total_segs[:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            total_segs[:, 1] - total_segs[:, 0])
        inter_ratio = inter / area_segs

        # only select those segments over the thresh
        seg_idx = (inter_ratio >= trunc_thresh)  # bool

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0:
                break
        else:
            # without any constraints
            break

    # feats: C x T
    # tad
    tad_data_dict["feats"] = tad_data_dict["feats"][:, st:ed].clone()
    tad_seg_idx = seg_idx[:tad_num_segs]
    tad_left = left[:tad_num_segs]
    tad_right = right[:tad_num_segs]
    # segments: N x 2 in feature grids
    tad_data_dict['segments'] = torch.stack((tad_left[tad_seg_idx], tad_right[tad_seg_idx]), dim=1)
    # shift the time stamps due to truncation
    tad_data_dict['segments'] = tad_data_dict['segments'] - st
    # labels: N
    tad_data_dict['labels'] = tad_data_dict['labels'][tad_seg_idx].clone()

    # mr，多了queries, query_feats
    mr_data_dict["feats"] = tad_data_dict["feats"]
    mr_seg_idx = seg_idx[tad_num_segs:]
    mr_left = left[tad_num_segs:]
    mr_right = right[tad_num_segs:]
    # segments: N x 2 in feature grids
    mr_data_dict['segments'] = torch.stack((mr_left[mr_seg_idx], mr_right[mr_seg_idx]), dim=1)
    # shift the time stamps due to truncation
    mr_data_dict['segments'] = mr_data_dict['segments'] - st
    # labels: N
    mr_data_dict['labels'] = mr_data_dict['labels'][mr_seg_idx].clone()
    # # queies: N
    # mr_data_dict["queries"] = mr_data_dict["queries"][mr_seg_idx.numpy()].copy()
    # # query_feats N x C
    # mr_data_dict["q_feats"] = mr_data_dict["q_feats"][mr_seg_idx, :].clone()

    return tad_data_dict, mr_data_dict
