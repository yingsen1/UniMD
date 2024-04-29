import functools
import torch
import torch.nn as nn
import torch.nn.functional as F




class Project(nn.Module):
    """
    A fixed project layer for distribution
    """

    def __init__(self, reg_max=16, interval=1):
        super(Project, self).__init__()
        self.reg_max = reg_max
        # self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))
        self.register_buffer('project', torch.linspace(0, self.reg_max * interval, self.reg_max + 1))
        self.interval = interval

    def forward(self, x):
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project).reshape(-1, 2)
        return x


def t_target(input_offsets, target_offsets, input_score, alpha=1.0, beta=6.0, iou_fn=None):
    """ t = score ** alpha + offset ** beta"""

    # input_offsets = input_offsets.float()
    # target_offsets = target_offsets.float()
    # input_score = input_score.float()
    # # check all 1D events are valid
    # assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    # assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"
    #
    # lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    # lg, rg = target_offsets[:, 0], target_offsets[:, 1]
    #
    # # intersection key points
    # lkis = torch.min(lp, lg)
    # rkis = torch.min(rp, rg)
    #
    # # iou
    # intsctk = rkis + lkis
    # unionk = (lp + rp) + (lg + rg) - intsctk
    # iouk = intsctk / unionk.clamp(min=eps)
    if iou_fn is None:
        raise Exception

    iouk = iou_fn(input_offsets, target_offsets)

    t_k = (input_score ** alpha) * (iouk ** beta)
    return t_k, iouk


def iou_target(input_offsets, target_offsets, eps=1e-7):
    """ iou for 1d"""
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    # assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    # assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    return iouk


def diou_target(input_offsets, target_offsets, eps=1e-7):
    """ iou for 1d"""
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    # assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    # assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    return iouk - torch.square(rho / len_c.clamp(min=eps))


def iou_target_ori(input_offsets, target_offsets, eps=1e-7):
    """ iou for 1d"""
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    # assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    # assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.max(lp, lg)
    rkis = torch.min(rp, rg)

    overlap = torch.maximum(torch.Tensor([0]).to(input_offsets.device), rkis - lkis)

    # iou
    # intsctk = rkis + lkis

    # unionk = (lp + rp) + (lg + rg) - intsctk
    # iouk = intsctk / unionk.clamp(min=eps)
    iouk = overlap / ((rp - lp + rg - lg - overlap).clamp(min=eps))

    return iouk


def giou_target(input_offsets, target_offsets, eps=1e-7):
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    # assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    # assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)
    return iouk


def distance2bbox(points, distance, max_shape=False):
    """Decode distance prediction to bounding box for 1d.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    left = points[:, 0] - distance[:, 0]
    right = points[:, 0] + distance[:, 1]

    if max_shape:
        max_pts = points[:, -1] - 1
        min_pts = torch.zeros_like(max_pts, dtype=points.dtype)
        left = left.clamp(min=min_pts, max=max_pts)
        right = right.clamp(min=min_pts, max=max_pts)
    else:
        pass
    #
    # if max_shape is not None:
    #     left = left.clamp(min=0, max=max_shape - 1)
    #     right = right.clamp(min=0, max=max_shape - 1)

    return torch.stack([left, right], -1)


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper




def quality_focal_loss(
        pred,  # (B, FT, n_cls)
        label,  # (B, FT, n_cls)
        score,  # (#pos, 1)
        valid_mask,  # (B, FT)
        pos_mask,  # (B, FT)
        weight=None,
        beta=2.0,
        reduction='mean',
        alpha=None,
        # avg_factor=None
):
    # neg_loss:  all goes to 0
    pred_sigmoid = pred.sigmoid()
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pred.shape)
    # loss: B * FT * 1
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * pt.pow(beta)

    # positive goes to bbox quality
    # pt = score[a] - pred_sigmoid[a, b]
    assert score.size(0) == pred_sigmoid[pos_mask].size(0), \
        "t_score size, %s, cls_score size, %s" % (str(score.size()), str(pred_sigmoid[pos_mask].size()))
    pt = score - pred_sigmoid[pos_mask]

    # loss[a, b] = F.binary_cross_entropy_with_logits(
    #     pred[a, b], score[a], reduction='none') * pt.pow(beta)
    loss[pos_mask] = F.binary_cross_entropy_with_logits(
        pred[pos_mask], label[pos_mask] * score, reduction="none"
    ) * pt.pow(beta)

    # GFL是soft label，相当于能动态分配正负样本
    device = pos_mask.device
    if isinstance(alpha, (int, float)):
        loss = loss * (1 - alpha)
        loss[pos_mask] = F.binary_cross_entropy_with_logits(
            pred[pos_mask], score, reduction="none", pos_weight=torch.Tensor([alpha]).to(device)
        ) * pt.pow(beta)

    loss = loss[valid_mask]

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def quality_focal_loss_wilson(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        scores: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none",
) -> torch.Tensor:
    """根据actionFormer官方的FL进行修改"""

    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    # ce_loss = - (y logx + (1-y) log (1-x))
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # p_t = p * targets + (1 - p) * (1 - targets)
    # loss = ce_loss * ((1 - p_t) ** gamma)
    loss = ce_loss * ((scores - p) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def distribution_focal_loss(
        pred,
        label,
        weight=None,
        reduction='mean',
        avg_factor=None,
        interval=1,
        reg_max=4,
):
    # label: (#pos, 2)
    max_reg = interval * reg_max
    label = label.clamp(max=(max_reg - (1e-4)))
    disl = torch.div(label, interval, rounding_mode='floor')
    disl = disl.int().long()
    disr = disl + 1

    wl = (disr.float() * interval - label) / interval
    wr = (label - disl.float() * interval) / interval

    loss = F.cross_entropy(pred, disl, reduction='none') * wl \
           + F.cross_entropy(pred, disr, reduction='none') * wr
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class QualityFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                score,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                score,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


class DistributionFocalLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls
