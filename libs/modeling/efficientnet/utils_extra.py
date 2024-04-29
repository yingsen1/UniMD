# Author: Zylo117

import math

from torch import nn
import torch.nn.functional as F


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x


class Conv1dStaticSamePadding(nn.Module):
    # """
    # created by Zylo117
    # The real keras/tensorflow conv2d with same padding
    # """
    pass
    #
    # def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
    #     super().__init__()
    #     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
    #                           bias=bias, groups=groups)
    #     self.stride = self.conv.stride
    #     self.kernel_size = self.conv.kernel_size
    #     self.dilation = self.conv.dilation
    #
    #     if isinstance(self.stride, int):
    #         self.stride = [self.stride] * 2
    #     elif len(self.stride) == 1:
    #         self.stride = [self.stride[0]] * 2
    #
    #     if isinstance(self.kernel_size, int):
    #         self.kernel_size = [self.kernel_size] * 2
    #     elif len(self.kernel_size) == 1:
    #         self.kernel_size = [self.kernel_size[0]] * 2
    #
    # def forward(self, x):
    #     h, w = x.shape[-2:]
    #
    #     extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
    #     extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
    #
    #     left = extra_h // 2
    #     right = extra_h - left
    #     top = extra_v // 2
    #     bottom = extra_v - top
    #
    #     x = F.pad(x, [left, right, top, bottom])
    #
    #     x = self.conv(x)
    #     return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


class MaskMaxPool1d(nn.Module):
    """带有mask的pooling"""

    def __init__(
            self,
            kernel_size,
            stride,
            padding=0,
            dilation=1,
            out_mask=True
    ):
        super(MaskMaxPool1d, self).__init__()
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=1)
        self.stride = stride
        self.ks = kernel_size
        self.out_mask = out_mask

    def forward(self, x, mask):

        B, C, T = x.size()
        assert T % self.stride == 0  # 保证不抛弃任何时间端，以免造成误差

        # pool
        out_pool = self.pool(x)
        # compute mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_pool.size(-1), mode='nearest'
            )
        else:
            out_mask = mask.to(x.dtype)

        # mask the output
        out_pool = out_pool * out_mask.detach()
        out_mask = out_mask.bool()
        if self.out_mask:
            return out_pool, out_mask
        else:
            return out_pool

# class MaxPool1dStaticSamePadding(nn.Module):
#     """
#     created by Zylo117
#     The real keras/tensorflow MaxPool2d with same padding
#     输入输出的尺寸保持一致
#     """
#
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         # self.pool = nn.MaxPool2d(*args, **kwargs)
#         self.pool = nn.MaxPool1d(*args, **kwargs)
#         self.stride = self.pool.stride
#         self.kernel_size = self.pool.kernel_size
#
#         if isinstance(self.stride, int):
#             self.stride = [self.stride] * 2
#         elif len(self.stride) == 1:
#             self.stride = [self.stride[0]] * 2
#
#         if isinstance(self.kernel_size, int):
#             self.kernel_size = [self.kernel_size] * 2
#         elif len(self.kernel_size) == 1:
#             self.kernel_size = [self.kernel_size[0]] * 2
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#
#         extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
#         extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
#
#         left = extra_h // 2
#         right = extra_h - left
#         top = extra_v // 2
#         bottom = extra_v - top
#
#         x = F.pad(x, [left, right, top, bottom])
#
#         x = self.pool(x)
#         return x
