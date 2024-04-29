import torch.nn as nn
import torch
from torchvision.ops.boxes import nms as nms_torch

from .efficientnet import EfficientNet as EffNet
from .efficientnet.utils import MemoryEfficientSwish, Swish
from .efficientnet.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding
# for 1d
from .efficientnet.utils_extra import Conv1dStaticSamePadding, MaskMaxPool1d
from .blocks import MaskedConv1D, LayerNorm
from .models import register_neck
from collections import OrderedDict
from .convnext.convnext import MaskConvNextBlock


def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class SeparableConv1DBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3,
                 norm=True, activation=False, onnx_export=False):
        super(SeparableConv1DBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        # self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
        #                                               kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.depthwise_conv = MaskedConv1D(in_channels, in_channels,
                                           kernel_size=kernel_size, stride=1, groups=in_channels,
                                           padding=kernel_size // 2, bias=False)
        # self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)
        self.pointwise_conv = MaskedConv1D(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm1d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x, mask):
        x, _ = self.depthwise_conv(x, mask)
        x, _ = self.pointwise_conv(x, mask)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x, mask


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    make suit for conv_1d
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True,
                 use_p8=False):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.use_p8 = use_p8

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        if use_p8:
            self.conv7_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
            self.conv8_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
        if use_p8:
            self.p7_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.p8_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )
            if use_p8:
                self.p7_to_p8 = nn.Sequential(
                    MaxPool2dStaticSamePadding(3, 2)
                )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)

        return outs

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            if self.use_p8:
                p8_in = self.p7_to_p8(p7_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            if self.use_p8:
                # P3_0, P4_0, P5_0, P6_0, P7_0 and P8_0
                p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            else:
                # P3_0, P4_0, P5_0, P6_0 and P7_0
                p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        if self.use_p8:
            # P8_0 to P8_2

            # Connections for P7_0 and P8_0 to P7_1 respectively
            p7_up = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in)))

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)))
        else:
            # P7_0 to P7_2

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        if self.use_p8:
            # Connections for P7_0, P7_1 and P6_2 to P7_2 respectively
            p7_out = self.conv7_down(
                self.swish(p7_in + p7_up + self.p7_downsample(p6_out)))

            # Connections for P8_0 and P7_2 to P8_2
            p8_out = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out, p8_out
        else:
            # Connections for P7_0 and P6_2 to P7_2
            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out


@register_neck("bifpn")
class BiFPNBLOCK(nn.Module):
    "封装"

    def __init__(self, num_channels, conv_channels, num_repeats, kernel_size=3, attention=True):
        super(BiFPNBLOCK, self).__init__()
        self.num_channels = num_channels
        self.conv_channels = conv_channels
        self.num_repeats = num_repeats
        self.attention = attention
        print("#" * 10, " using bifpn with %d layers " % num_repeats, "#" * 10)
        self.bifpn_list = nn.ModuleList()
        self.bifpn_list.append(BiFPN1D(
            num_channels, conv_channels,
            kernel_size=kernel_size, first_time=True, attention=attention
        ))
        for _ in range(num_repeats - 1):
            self.bifpn_list.append(
                BiFPN1D(num_channels, num_channels,
                        kernel_size=kernel_size, first_time=False, attention=attention)
            )

    def forward(self, fpn_feats, fpn_masks):
        # inputs must be a list / tuple
        # assert len(inputs) == len(self.in_channels)
        # assert len(fpn_masks) == len(self.in_channels)
        assert len(fpn_feats) == len(fpn_masks)

        for i in range(self.num_repeats):
            fpn_feats, fpn_masks = self.bifpn_list[i](
                fpn_feats, fpn_masks
            )
        return fpn_feats, fpn_masks


class BiFPN1D(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels,
                 kernel_size=3,
                 first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
            注意：内部卷积的输入输出尺寸均保持一致
        """
        super(BiFPN1D, self).__init__()
        self.epsilon = epsilon

        # Conv layers
        self.conv7_up = SeparableConv1DBlock(num_channels, kernel_size=kernel_size, onnx_export=onnx_export)
        self.conv6_up = SeparableConv1DBlock(num_channels, kernel_size=kernel_size, onnx_export=onnx_export)
        self.conv5_up = SeparableConv1DBlock(num_channels, kernel_size=kernel_size, onnx_export=onnx_export)
        self.conv4_up = SeparableConv1DBlock(num_channels, kernel_size=kernel_size, onnx_export=onnx_export)
        self.conv3_up = SeparableConv1DBlock(num_channels, kernel_size=kernel_size, onnx_export=onnx_export)
        self.conv4_down = SeparableConv1DBlock(num_channels, kernel_size=kernel_size, onnx_export=onnx_export)
        self.conv5_down = SeparableConv1DBlock(num_channels, kernel_size=kernel_size, onnx_export=onnx_export)
        self.conv6_down = SeparableConv1DBlock(num_channels, kernel_size=kernel_size, onnx_export=onnx_export)
        self.conv7_down = SeparableConv1DBlock(num_channels, kernel_size=kernel_size, onnx_export=onnx_export)
        self.conv8_down = SeparableConv1DBlock(num_channels, kernel_size=kernel_size, onnx_export=onnx_export)

        # Feature scaling layers
        self.p7_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        # self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        # self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        # self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
        # kernel_size, stride, padding = \
        #                 n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
        self.p4_downsample = MaskMaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2, out_mask=False)
        self.p5_downsample = MaskMaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2, out_mask=False)
        self.p6_downsample = MaskMaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2, out_mask=False)
        self.p7_downsample = MaskMaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2, out_mask=False)
        self.p8_downsample = MaskMaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2, out_mask=False)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        self.first_time = first_time
        # first_time = 直接获取backbone的feats
        self.ft_modules = OrderedDict()
        self.ft_bns = OrderedDict()
        if self.first_time:
            self.ft_modules["p8_down_channel"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p8_down_channel"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p7_down_channel"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p7_down_channel"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p6_down_channel"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p6_down_channel"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p5_down_channel"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p5_down_channel"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p4_down_channel"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p4_down_channel"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p3_down_channel"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p3_down_channel"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p4_down_channel_2"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p4_down_channel_2"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p5_down_channel_2"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p5_down_channel_2"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p6_down_channel_2"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p6_down_channel_2"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p7_down_channel_2"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p7_down_channel_2"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules = nn.ModuleDict(self.ft_modules)
            self.ft_bns = nn.ModuleDict(self.ft_bns)

        # Weight
        # todo: 增加p7 p8
        self.p7_w1_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w1_relu = nn.ReLU()
        self.p6_w1_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2_att = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2_att = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2_att = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()
        self.p8_w2_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p8_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs, masks):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            # outs, masks = self._forward_fast_attention(inputs, masks)
            outs, masks = self._forward_fast_attention(inputs, masks)
            # raise NotImplementedError
        else:
            outs, masks = self._forward(inputs, masks)

        return outs, masks

    def _forward_fast_attention(self, inputs, masks):
        if self.first_time:
            p3, p4, p5, p6, p7, p8 = inputs
            m3, m4, m5, m6, m7, m8 = masks
            p3_in = self.ft_bns.p3_down_channel(self.ft_modules.p3_down_channel(p3, m3))
            p4_in = self.ft_bns.p4_down_channel(self.ft_modules.p4_down_channel(p4, m4))
            p5_in = self.ft_bns.p5_down_channel(self.ft_modules.p5_down_channel(p5, m5))
            p6_in = self.ft_bns.p6_down_channel(self.ft_modules.p6_down_channel(p6, m6))
            p7_in = self.ft_bns.p7_down_channel(self.ft_modules.p7_down_channel(p7, m7))
            p8_in = self.ft_bns.p8_down_channel(self.ft_modules.p8_down_channel(p8, m8))

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            m3, m4, m5, m6, m7, m8 = masks

        # p8_0 to P8_2

        # Weights for P7_0 and P8_0 to P7_1
        p7_w1 = self.p7_w1_relu(self.p7_w1_att)
        weight = p7_w1 / (torch.sum(p7_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p7_up, _ = self.conv7_up(self.swish(weight[0] * p7_in + weight[1] * self.p7_upsample(p8_in)), m7)

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1_att)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up, _ = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_up)), m6)

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1_att)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up, _ = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)), m5)

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1_att)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up, _ = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)), m4)

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1_att)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out, _ = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)), m3)

        if self.first_time:
            p4_in = self.ft_bns.p4_down_channel_2(self.ft_modules.p4_down_channel_2(p4, m4))
            p5_in = self.ft_bns.p5_down_channel_2(self.ft_modules.p5_down_channel_2(p5, m5))
            p6_in = self.ft_bns.p6_down_channel_2(self.ft_modules.p6_down_channel_2(p6, m6))
            p7_in = self.ft_bns.p7_down_channel_2(self.ft_modules.p7_down_channel_2(p7, m7))

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2_att)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out, _ = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out, m3)), m4)

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2_att)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out, _ = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out, m4)), m5)

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2_att)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out, _ = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out, m5)), m6)

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2_att)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out, _ = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out, m6)), m7)

        # Weights for P8_0 and P7_2 to P8_2
        p8_w2 = self.p8_w2_relu(self.p8_w2_att)
        weight = p8_w2 / (torch.sum(p8_w2, dim=0) + self.epsilon)
        # Connections for P8_0 and P7_2 to P8_2
        p8_out, _ = self.conv8_down(self.swish(weight[0] * p8_in + weight[1] * self.p8_downsample(p7_out, m7)), m8)

        return [p3_out, p4_out, p5_out, p6_out, p7_out, p8_out], \
               [m3, m4, m5, m6, m7, m8]

    def _forward(self, inputs, masks):
        # 原来为5层: p3:p7, 当前为p3:p8
        if self.first_time:
            p3, p4, p5, p6, p7, p8 = inputs
            m3, m4, m5, m6, m7, m8 = masks
            p3_in = self.ft_bns.p3_down_channel(self.ft_modules.p3_down_channel(p3, m3))
            p4_in = self.ft_bns.p4_down_channel(self.ft_modules.p4_down_channel(p4, m4))
            p5_in = self.ft_bns.p5_down_channel(self.ft_modules.p5_down_channel(p5, m5))
            p6_in = self.ft_bns.p6_down_channel(self.ft_modules.p6_down_channel(p6, m6))
            p7_in = self.ft_bns.p7_down_channel(self.ft_modules.p7_down_channel(p7, m7))
            p8_in = self.ft_bns.p8_down_channel(self.ft_modules.p8_down_channel(p8, m8))

            #
            #
            # # p3, p4, p5 = inputs
            # #
            # # p6_in = self.p5_to_p6(p5)
            # # p7_in = self.p6_to_p7(p6_in)
            # if self.use_p8:
            #     p8_in = self.p7_to_p8(p7_in)
            #
            # p3_in = self.p3_down_channel(p3)
            # p4_in = self.p4_down_channel(p4)
            # p5_in = self.p5_down_channel(p5)

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            m3, m4, m5, m6, m7, m8 = masks

        # if self.use_p8:
        # P8_0 to P8_2
        # pass

        # Connections for P7_0 and P8_0 to P7_1 respectively
        p7_up, _ = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in)), m7)
        # # Connections for P7_0 and P8_0 to P7_1 respectively
        # p7_up = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in)))

        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up, _ = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)), m6)
        # p6_up, _ = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up, _ = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)), m5)
        # p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up, _ = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)), m4)
        # p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out, _ = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)), m3)
        # p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.ft_bns.p4_down_channel_2(self.ft_modules.p4_down_channel_2(p4, m4))
            p5_in = self.ft_bns.p5_down_channel_2(self.ft_modules.p5_down_channel_2(p5, m5))
            p6_in = self.ft_bns.p6_down_channel_2(self.ft_modules.p6_down_channel_2(p6, m6))
            p7_in = self.ft_bns.p7_down_channel_2(self.ft_modules.p7_down_channel_2(p7, m7))

            # p8_in = self.p8_down_channel_2(p8, m8)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out, _ = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out, m3)), m4)
        # p4_out = self.conv4_down(
        #     self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out, _ = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out, m4)), m5)
        # p5_out = self.conv5_down(
        #     self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out, _ = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out, m5)), m6)
        # p6_out = self.conv6_down(
        #     self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        # if self.use_p8:
        # Connections for P7_0, P7_1 and P6_2 to P7_2 respectively
        p7_out, _ = self.conv7_down(
            self.swish(p7_in + p7_up + self.p7_downsample(p6_out, m6)), m7)

        # Connections for P8_0 and P7_2 to P8_2
        p8_out, _ = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out, m7)), m8)
        # p8_out = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out)))

        return [p3_out, p4_out, p5_out, p6_out, p7_out, p8_out], \
               [m3, m4, m5, m6, m7, m8]


class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers, pyramid_levels=5, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats


class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, pyramid_levels=5, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()

        return feats


class EfficientNet(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, compound_coef, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:]


class BiFPN1D_ConvNeXt(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels,
                 kernel_size=3,
                 first_time=False, epsilon=1e-4, onnx_export=False, attention=True,
                 ):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
            注意：内部卷积的输入输出尺寸均保持一致
        """
        super(BiFPN1D_ConvNeXt, self).__init__()
        self.epsilon = epsilon

        # Conv layers
        self.conv7_up = MaskConvNextBlock(num_channels, kernel_size=kernel_size)
        self.conv6_up = MaskConvNextBlock(num_channels, kernel_size=kernel_size)
        self.conv5_up = MaskConvNextBlock(num_channels, kernel_size=kernel_size)
        self.conv4_up = MaskConvNextBlock(num_channels, kernel_size=kernel_size)
        self.conv3_up = MaskConvNextBlock(num_channels, kernel_size=kernel_size)
        self.conv4_down = MaskConvNextBlock(num_channels, kernel_size=kernel_size)
        self.conv5_down = MaskConvNextBlock(num_channels, kernel_size=kernel_size)
        self.conv6_down = MaskConvNextBlock(num_channels, kernel_size=kernel_size)
        self.conv7_down = MaskConvNextBlock(num_channels, kernel_size=kernel_size)
        self.conv8_down = MaskConvNextBlock(num_channels, kernel_size=kernel_size)

        # Feature scaling layers
        self.p7_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        # self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        # self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        # self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
        # kernel_size, stride, padding = \
        #                 n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
        self.p4_downsample = MaskMaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2, out_mask=False)
        self.p5_downsample = MaskMaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2, out_mask=False)
        self.p6_downsample = MaskMaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2, out_mask=False)
        self.p7_downsample = MaskMaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2, out_mask=False)
        self.p8_downsample = MaskMaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2, out_mask=False)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        self.first_time = first_time
        # first_time = 直接获取backbone的feats
        self.ft_modules = OrderedDict()
        self.ft_bns = OrderedDict()
        if self.first_time:
            self.ft_modules["p8_down_channel"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p8_down_channel"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p7_down_channel"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p7_down_channel"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p6_down_channel"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p6_down_channel"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p5_down_channel"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p5_down_channel"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p4_down_channel"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p4_down_channel"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p3_down_channel"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p3_down_channel"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p4_down_channel_2"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p4_down_channel_2"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p5_down_channel_2"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p5_down_channel_2"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p6_down_channel_2"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p6_down_channel_2"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules["p7_down_channel_2"] = MaskedConv1D(conv_channels, num_channels, 1, out_mask=False)
            self.ft_bns["p7_down_channel_2"] = nn.BatchNorm1d(num_channels, momentum=0.01, eps=1e-3)

            self.ft_modules = nn.ModuleDict(self.ft_modules)
            self.ft_bns = nn.ModuleDict(self.ft_bns)

        # Weight
        # todo: 增加p7 p8
        self.p7_w1_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w1_relu = nn.ReLU()
        self.p6_w1_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2_att = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2_att = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2_att = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()
        self.p8_w2_att = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p8_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs, masks):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            # outs, masks = self._forward_fast_attention(inputs, masks)
            outs, masks = self._forward_fast_attention(inputs, masks)
            # raise NotImplementedError
        else:
            outs, masks = self._forward(inputs, masks)

        return outs, masks

    def _forward_fast_attention(self, inputs, masks):
        if self.first_time:
            p3, p4, p5, p6, p7, p8 = inputs
            m3, m4, m5, m6, m7, m8 = masks
            p3_in = self.ft_bns.p3_down_channel(self.ft_modules.p3_down_channel(p3, m3))
            p4_in = self.ft_bns.p4_down_channel(self.ft_modules.p4_down_channel(p4, m4))
            p5_in = self.ft_bns.p5_down_channel(self.ft_modules.p5_down_channel(p5, m5))
            p6_in = self.ft_bns.p6_down_channel(self.ft_modules.p6_down_channel(p6, m6))
            p7_in = self.ft_bns.p7_down_channel(self.ft_modules.p7_down_channel(p7, m7))
            p8_in = self.ft_bns.p8_down_channel(self.ft_modules.p8_down_channel(p8, m8))

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            m3, m4, m5, m6, m7, m8 = masks

        # p8_0 to P8_2

        # Weights for P7_0 and P8_0 to P7_1
        p7_w1 = self.p7_w1_relu(self.p7_w1_att)
        weight = p7_w1 / (torch.sum(p7_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p7_up, _ = self.conv7_up(self.swish(weight[0] * p7_in + weight[1] * self.p7_upsample(p8_in)), m7)

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1_att)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up, _ = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_up)), m6)

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1_att)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up, _ = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)), m5)

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1_att)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up, _ = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)), m4)

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1_att)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out, _ = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)), m3)

        if self.first_time:
            p4_in = self.ft_bns.p4_down_channel_2(self.ft_modules.p4_down_channel_2(p4, m4))
            p5_in = self.ft_bns.p5_down_channel_2(self.ft_modules.p5_down_channel_2(p5, m5))
            p6_in = self.ft_bns.p6_down_channel_2(self.ft_modules.p6_down_channel_2(p6, m6))
            p7_in = self.ft_bns.p7_down_channel_2(self.ft_modules.p7_down_channel_2(p7, m7))

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2_att)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out, _ = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out, m3)), m4)

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2_att)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out, _ = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out, m4)), m5)

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2_att)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out, _ = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out, m5)), m6)

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2_att)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out, _ = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out, m6)), m7)

        # Weights for P8_0 and P7_2 to P8_2
        p8_w2 = self.p8_w2_relu(self.p8_w2_att)
        weight = p8_w2 / (torch.sum(p8_w2, dim=0) + self.epsilon)
        # Connections for P8_0 and P7_2 to P8_2
        p8_out, _ = self.conv8_down(self.swish(weight[0] * p8_in + weight[1] * self.p8_downsample(p7_out, m7)), m8)

        return [p3_out, p4_out, p5_out, p6_out, p7_out, p8_out], \
               [m3, m4, m5, m6, m7, m8]

    def _forward(self, inputs, masks):
        # 原来为5层: p3:p7, 当前为p3:p8
        if self.first_time:
            p3, p4, p5, p6, p7, p8 = inputs
            m3, m4, m5, m6, m7, m8 = masks
            p3_in = self.ft_bns.p3_down_channel(self.ft_modules.p3_down_channel(p3, m3))
            p4_in = self.ft_bns.p4_down_channel(self.ft_modules.p4_down_channel(p4, m4))
            p5_in = self.ft_bns.p5_down_channel(self.ft_modules.p5_down_channel(p5, m5))
            p6_in = self.ft_bns.p6_down_channel(self.ft_modules.p6_down_channel(p6, m6))
            p7_in = self.ft_bns.p7_down_channel(self.ft_modules.p7_down_channel(p7, m7))
            p8_in = self.ft_bns.p8_down_channel(self.ft_modules.p8_down_channel(p8, m8))

            #
            #
            # # p3, p4, p5 = inputs
            # #
            # # p6_in = self.p5_to_p6(p5)
            # # p7_in = self.p6_to_p7(p6_in)
            # if self.use_p8:
            #     p8_in = self.p7_to_p8(p7_in)
            #
            # p3_in = self.p3_down_channel(p3)
            # p4_in = self.p4_down_channel(p4)
            # p5_in = self.p5_down_channel(p5)

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            m3, m4, m5, m6, m7, m8 = masks

        # if self.use_p8:
        # P8_0 to P8_2
        # pass

        # Connections for P7_0 and P8_0 to P7_1 respectively
        p7_up, _ = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in)), m7)
        # # Connections for P7_0 and P8_0 to P7_1 respectively
        # p7_up = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in)))

        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up, _ = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)), m6)
        # p6_up, _ = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up, _ = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)), m5)
        # p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up, _ = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)), m4)
        # p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out, _ = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)), m3)
        # p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.ft_bns.p4_down_channel_2(self.ft_modules.p4_down_channel_2(p4, m4))
            p5_in = self.ft_bns.p5_down_channel_2(self.ft_modules.p5_down_channel_2(p5, m5))
            p6_in = self.ft_bns.p6_down_channel_2(self.ft_modules.p6_down_channel_2(p6, m6))
            p7_in = self.ft_bns.p7_down_channel_2(self.ft_modules.p7_down_channel_2(p7, m7))

            # p8_in = self.p8_down_channel_2(p8, m8)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out, _ = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out, m3)), m4)
        # p4_out = self.conv4_down(
        #     self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out, _ = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out, m4)), m5)
        # p5_out = self.conv5_down(
        #     self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out, _ = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out, m5)), m6)
        # p6_out = self.conv6_down(
        #     self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        # if self.use_p8:
        # Connections for P7_0, P7_1 and P6_2 to P7_2 respectively
        p7_out, _ = self.conv7_down(
            self.swish(p7_in + p7_up + self.p7_downsample(p6_out, m6)), m7)

        # Connections for P8_0 and P7_2 to P8_2
        p8_out, _ = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out, m7)), m8)
        # p8_out = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out)))

        return [p3_out, p4_out, p5_out, p6_out, p7_out, p8_out], \
               [m3, m4, m5, m6, m7, m8]


@register_neck("bifpn_convnext")
class BiFPNBLOCK_CONVNEXT(nn.Module):

    def __init__(self, num_channels, conv_channels, num_repeats, kernel_size=3, attention=True):
        super(BiFPNBLOCK_CONVNEXT, self).__init__()
        self.num_channels = num_channels
        self.conv_channels = conv_channels
        self.num_repeats = num_repeats
        self.attention = attention
        print("#" * 10, " using bifpn with %d layers " % num_repeats, "#" * 10)
        self.bifpn_list = nn.ModuleList()
        self.bifpn_list.append(BiFPN1D_ConvNeXt(
            num_channels, conv_channels,
            kernel_size=kernel_size, first_time=True, attention=attention
        ))
        for _ in range(num_repeats - 1):
            self.bifpn_list.append(
                BiFPN1D_ConvNeXt(num_channels, num_channels,
                                 kernel_size=kernel_size, first_time=False, attention=attention)
            )

    def forward(self, fpn_feats, fpn_masks):
        # inputs must be a list / tuple
        # assert len(inputs) == len(self.in_channels)
        # assert len(fpn_masks) == len(self.in_channels)
        assert len(fpn_feats) == len(fpn_masks)

        for i in range(self.num_repeats):
            fpn_feats, fpn_masks = self.bifpn_list[i](
                fpn_feats, fpn_masks
            )
        return fpn_feats, fpn_masks


if __name__ == '__main__':
    # from tensorboardX import SummaryWriter

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
