import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
modify from https://github.com/TencentARC/UMT/tree/main/tools
"""


def temporal_intersection(windows1, windows2, aligned=False):
    """
    Compute the intersections among temporal windows.

    Args:
        windows1 (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        windows2 (:obj:`nn.Tensor[M, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        aligned (bool, optional): Whether to only compute the intersections
            among aligned temporal windows. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N]` | :obj:`nn.Tensor[N, M]`: The computed \
            intersection values.
    """
    if aligned:
        s = torch.max(windows1[:, 0], windows2[:, 0])
        e = torch.min(windows1[:, 1], windows2[:, 1])
    else:
        s = torch.max(windows1[:, None, 0], windows2[:, 0])
        e = torch.min(windows1[:, None, 1], windows2[:, 1])

    inter = (e - s).clamp(0)
    return inter


def temporal_area(windows):
    """
    Compute the areas of temporal windows.

    Args:
        windows (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed. They
            are expected to be in ``(start, end)`` format.

    Returns:
        :obj:`nn.Tensor[N]`: The computed areas.
    """
    return windows[:, 1] - windows[:, 0]


def temporal_iou(windows1, windows2, aligned=False):
    """
    Compute the intersection-over-unions (IoUs) among temporal windows.

    Args:
        windows1 (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        windows2 (:obj:`nn.Tensor[M, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        aligned (bool, optional): Whether to only compute the IoU among
            aligned temporal windows. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N]` | :obj:`nn.Tensor[N, M]`: The computed pairwise \
            IoU values.
    """
    area1 = temporal_area(windows1)
    area2 = temporal_area(windows2)

    inter = temporal_intersection(windows1, windows2, aligned=aligned)

    if aligned:
        iou = inter / (area1 + area2 - inter)
    else:
        iou = inter / (area1[:, None] + area2 - inter)

    return iou


class MREvaluator:
    def __init__(self, rank=[1, 5], iou_thr=[0.5, 0.7], detail=False, **kwargs):
        self.rank = rank
        self.iou_thr = iou_thr
        self.kwargs = kwargs
        self.labels = 0
        # self.hits_top1 = 0
        # self.hits_top5 = 0
        self.hits_dict = dict()
        for r in self.rank:
            for iou in iou_thr:
                self.hits_dict["%d-%.2f" % (r, iou)] = 0
        self.detail = detail
        self.vid_perform = dict()

    def evalute(self, results, gts):

        # num_res = len(results["video-id"])
        # segs = results["segments"][0].cpu().numpy()
        # scores = results["scores"][0].cpu().numpy()
        # labels = results["label"][0].cpu().numpy()

        # results: from model
        sorted_res = dict()
        for one_vid in results:
            vid = one_vid["video_id"]
            sorted_res[vid] = dict()

            for i in range(len(one_vid["segments"])):
                label = one_vid["labels"][i].cpu().item()
                seg = one_vid["segments"][i].cpu().tolist()
                score = one_vid["scores"][i].cpu().item()
                sorted_res[vid].setdefault(label, []).append([seg[0], seg[1], score])

        # gts: from dataset
        for one_gt in gts:
            vid = one_gt["id"]
            num_gt = one_gt["segments"].shape[0]
            # self.labels += num_gt
            r1_50 = 0
            r1_70 = 0

            for i in range(num_gt):
                self.labels += 1
                label = one_gt["labels"][i].item()
                one_res = []
                if vid in sorted_res and label in sorted_res[vid]:
                    one_res = sorted_res[vid][label]
                else:
                    continue
                one_res = torch.Tensor(one_res)
                for k in self.rank:
                    for thr in self.iou_thr:
                        inds = torch.argsort(one_res[:, -1], descending=True)
                        keep = inds[:k]
                        bnd = one_res[:, :-1][keep]
                        gt = torch.from_numpy(one_gt["segments"][i])
                        iou = temporal_iou(gt[None], bnd)
                        if iou.max() >= thr:
                            self.hits_dict["%d-%.2f" % (k, thr)] += 1
                            if k == 1 and thr == 0.7:
                                r1_70 += 1
                            elif k == 1 and thr == 0.5:
                                r1_50 += 1
            if self.detail:
                self.vid_perform[vid] = {
                    "r1_50": r1_50 / num_gt * 100,
                    "r1_70": r1_70 / num_gt * 100,
                }

    def summary(self):
        for k, v in self.hits_dict.items():
            self.hits_dict[k] = v / (self.labels + 1e-5)
        # print("### AP in charades_sta : ###")
        # print("R1@70: %.2f" % self.hits_dict["1-0.70"])
        # print("R1@50: %.2f" % self.hits_dict["1-0.50"])
        # print("R5@70: %.2f" % self.hits_dict["5-0.70"])
        # print("R5@50: %.2f" % self.hits_dict["5-0.50"])
        return self.hits_dict

    def get_detail(self):
        return self.vid_perform
