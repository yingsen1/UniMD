import torch
import numpy as np


def iou_batch(bb1, bb2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]

    """

    bb1 = bb1[None, :]
    bb2 = bb2[:, None]

    xx1 = torch.maximum(bb1[..., 0], bb2[..., 0])
    xx2 = torch.minimum(bb1[..., 1], bb2[..., 1])

    overlap = torch.maximum(torch.Tensor(0), xx2 - xx1)
    return overlap / (bb1[..., 1] - bb1[..., 0] + bb2[..., 1] - bb2[..., 0])

    #
    #
    # bb_gt = np.expand_dims(bb_gt, 0)
    # bb_test = np.expand_dims(bb_test, 1)
    #
    # xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    # yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    # xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    # yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    # w = np.maximum(0., xx2 - xx1)
    # h = np.maximum(0., yy2 - yy1)
    # wh = w * h
    # o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    #           + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    # return (o)
