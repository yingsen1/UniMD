import os
import sys
import torch
import math
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.conv import _ConvNd
from typing import Optional, List, Tuple, Union

import numpy as np
from .blocks import MaskedConv1D, Scale, LayerNorm




class QueryRegClsWiseConv1d(_ConvNd):
    """1d conv with input kernel"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            padding: Union[str, _size_1_t] = 0,
            dilation: _size_1_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super(QueryRegClsWiseConv1d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _single(0), groups, bias, padding_mode, **factory_kwargs)

        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # conv setting
        self.s = stride
        self.ks = kernel_size
        self.in_dims = in_channels
        self.out_dims = out_channels

        # if using bn/ln, no need to set bias
        self.use_bias = bias  # default no use
        # overwrite weights + bias
        self.weight = None
        self.bias = None

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input, mask, conv_weight, conv_bias=None, task=None, cls_wise=False):
        # input: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)

        B, C, T = input.size()
        # input length must be divisible by stride
        assert T % self.s == 0
        device = input.device

        if task == "mr":
            # B x (#n_query, ind_cw, outd_cw , ks_cw)
            B_cw = len(conv_weight)
            assert B_cw == B
            out_feats = []  # B x (#n_query, out_dim, T)
            for i, one_conv_weight in enumerate(conv_weight):
                n_query, ind_cw, outd_cw, ks_cw = one_conv_weight.size()
                assert ind_cw == self.in_dims
                assert outd_cw == self.out_dims
                assert ks_cw == self.ks

                one_conv_weight = one_conv_weight.permute(0, 2, 1, 3)  # n_query, out, in, k
                one_conv_weight = one_conv_weight.reshape(n_query * outd_cw, ind_cw, ks_cw)  # n_query*out, in, k
                if self.use_bias:
                    one_conv_bias = conv_bias[i].reshape(n_query * outd_cw)
                else:
                    one_conv_bias = None
                one_feat = input[i][None, ...]
                one_out_feat = self._conv_forward(one_feat, one_conv_weight, one_conv_bias)
                one_out_feat = one_out_feat.view(n_query, outd_cw, one_out_feat.size(-1))
                out_feats.append(one_out_feat)

        elif task == "tad" and not cls_wise:  # tad + not_cls_wise
            one, ind_cw, outd_cw, ks_cw = conv_weight.size()
            assert one == 1
            assert ind_cw == self.in_dims
            assert outd_cw == self.out_dims
            assert ks_cw == self.ks
            conv_weight = conv_weight.permute(0, 2, 1, 3)  # b, out, in, k
            # output
            out_feats = torch.zeros((B, outd_cw, T), dtype=torch.float32, device=device)
            for b_i in range(B):
                one_feat = input[b_i][None, :]
                one_weight = conv_weight[0]
                if self.use_bias:
                    one_bias = conv_bias[0]
                else:
                    one_bias = None
                out_feats[b_i] = self._conv_forward(one_feat, one_weight, one_bias)
            # output (b, 1, out_dim, T)
            out_feats = out_feats[:, None, :, :]

        elif task == "tad" and cls_wise:  # tad + cls_wise
            # ensure conv_weights
            ncls_cw, ind_cw, outd_cw, ks_cw = conv_weight.size()
            assert ind_cw == self.in_dims
            assert outd_cw == self.out_dims
            assert ks_cw == self.ks
            conv_weight = conv_weight.permute(0, 2, 1, 3)  # n_cls, out, in, k
            # output (b, #n_cls, out_dim, T)
            out_feats = torch.zeros((B, ncls_cw, outd_cw, T), dtype=torch.float32, device=device)

            # per-batch
            for batch_i in range(B):
                # per-cls
                for cls_i in range(ncls_cw):
                    one_feat = input[batch_i][None, :]
                    one_weight = conv_weight[cls_i]
                    if self.use_bias:
                        one_bias = conv_bias[cls_i]
                    else:
                        one_bias = None
                    out_feats[batch_i, cls_i, :, :] = self._conv_forward(one_feat, one_weight, one_bias)
        else:
            raise NotImplementedError

        if isinstance(out_feats, list):
            t = out_feats[0].size(-1)
            if self.s > 1:
                # downsample the mask using nearest neighbor
                out_mask = F.interpolate(
                    mask.to(input.dtype), size=t, mode='nearest'
                )
            else:
                # masking out the features
                out_mask = mask.to(input.dtype)
            # masking the output, stop grad to mask ()
            out_feats_new = []
            for k, one_out_feat in enumerate(out_feats):
                one_out_feat = one_out_feat * out_mask[k][None, :, :].detach()  # (n_query, out_dim , T)
                out_feats_new.append(one_out_feat)
            out_feats = out_feats_new  # B x (n_query, out_dim, T)
            out_mask = out_mask.bool()
        else:
            # compute mask
            if self.s > 1:
                # downsample the mask using nearest neighbor
                out_mask = F.interpolate(
                    mask.to(input.dtype), size=out_feats.size(-1), mode='nearest'
                )
            else:
                # masking out the features
                out_mask = mask.to(input.dtype)
            # masking the output, stop grad to mask ()
            out_feats = out_feats * out_mask[:, None, :, :].detach()  # (b, n_cls, out_dim, T)
            if (not cls_wise) or task == "mr":  # (b, out_dim, t)
                # eliminate n_cls
                out_feats = torch.squeeze(out_feats, dim=1)

            out_mask = out_mask.bool()

        return out_feats, out_mask


class QueryRegressionLast_NEW(nn.Module):
    """query independent regression head"""

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            query_dim,
            final_out_dim=2,  # default 2 for query_reg
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            pooling="max",  # avg / max
            prepooling=False,  # true: first avg pooling
            head_bias=True,  #
            num_query_layer=3,  # fc layers
            dropout=0.1,  # default 0.5
            class_wise=True,
            dfl=False,  # using dfl or not
            query_fc_chanel=512,
    ):
        super(QueryRegressionLast_NEW, self).__init__()
        self.fpn_levels = fpn_levels
        self.prepooling = prepooling
        self.pooling = pooling
        self.act = act_layer()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.query_dim = query_dim
        self.ks = kernel_size
        self.final_out_dim = final_out_dim
        self.num_q_layers = num_query_layer
        self.class_wise = class_wise
        self.dfl = dfl

        ### main brach
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):  # except for last layer
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                ))
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()  # according to fpn_scale
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_ks = kernel_size
        self.offset_head = QueryRegClsWiseConv1d(
            feat_dim, final_out_dim, self.offset_ks, stride=1, padding=self.offset_ks // 2,
            bias=head_bias,
        )  # final_out_dim=2

        ### query branch
        self.query_fcs = nn.ModuleList()
        self.query_norm = nn.ModuleList()

        if num_query_layer > 1:
            print("### query regress using %d fc layers" % num_query_layer)
        in_fc = query_dim  # query_dim
        for i in range(num_query_layer):
            # out_fc = in_fc
            out_fc = query_fc_chanel
            if i == num_query_layer - 1:
                out_fc = (final_out_dim * (feat_dim * self.offset_ks + 1))
            print("%d layer, in: %d, out: %d" % (i, in_fc, out_fc))
            self.query_fcs.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_fc, out_fc, bias=True)
            ))
            self.query_norm.append(nn.LayerNorm(out_fc))
            in_fc = out_fc

    def _draw_query_weights(self, query_feats, device, task):
        # query_feats:
        # tad: (#n_cls, 512)
        # mr: B x (#n_query, 512)

        assert task in ["tad", "mr"]
        if task == "tad" and not self.class_wise:
            # (#n_cls, 512) -> (1, 512)
            query_feats = torch.mean(query_feats, dim=0, keepdim=True)
        else:
            pass

        if isinstance(query_feats, list):
            query_pool_res = []
            for batch_i, query_i in enumerate(query_feats):
                query_i = query_i.to(device)
                for i in range(self.num_q_layers):
                    if i == self.num_q_layers - 1:
                        query_i = self.query_fcs[i](query_i)
                        query_i = self.act(query_i)
                        continue
                    query_i = self.query_fcs[i](query_i)
                    query_i = self.query_norm[i](query_i)
                    query_i = self.act(query_i)
                query_pool_res.append(query_i[:, :, None])
            return query_pool_res
        else:
            query_pool_res = query_feats.to(device)
            for i in range(self.num_q_layers):
                if i == self.num_q_layers - 1:
                    query_pool_res = self.query_fcs[i](query_pool_res)
                    query_pool_res = self.act(query_pool_res)
                    continue
                query_pool_res = self.query_fcs[i](query_pool_res)
                query_pool_res = self.query_norm[i](query_pool_res)
                query_pool_res = self.act(query_pool_res)
            return query_pool_res[:, :, None]

    def forward(self, fpn_feats, fpn_masks, query_feats, task=None):
        """

        Args:
            fpn_feats:
            fpn_masks:
            query_feats:  tad: (#n_cls, 512); mr: (b, 512)  /B x [#n_query, 512]
            task: tad, mr

        Returns:

        """
        # - qvhigh: 1query/vid， (b, 512)
        # - charades: multi query/vid, B x [#n_query, 512]
        if task == "mr":
            # B x [#n_query, 512]
            if isinstance(query_feats, list):
                pass
            else:  # -> List
                query_feats_new = []
                for i in range(query_feats.size(0)):
                    query_feats_new.append(query_feats[i][None])
                query_feats = query_feats_new

        # query_weights,
        # tad: (n_cls, final_out_dim *feat_dim * (ks+1), 1)
        # mr: B x (n_query, final_out_dim *feat_dim * (ks+1), 1)
        query_weights = self._draw_query_weights(query_feats, device=fpn_feats[0].device, task=task)

        if isinstance(query_weights, list):  # new
            # query_weights -> B x (#n_query, c, 2, k)
            query_weights_new = []
            query_bias_new = []
            for one_query_weight in query_weights:
                assert one_query_weight.size(1) == self.final_out_dim * (self.feat_dim * self.offset_ks + 1)
                one_query_weight, one_query_bias = torch.split(
                    one_query_weight,
                    [self.final_out_dim * self.feat_dim * self.offset_ks, self.final_out_dim * 1],
                    dim=1
                )
                one_query_weight = one_query_weight.reshape(-1, self.feat_dim, self.final_out_dim, self.offset_ks)
                one_query_bias = one_query_bias.reshape(-1, self.final_out_dim)
                query_weights_new.append(one_query_weight)
                query_bias_new.append(one_query_bias)
            query_weights = query_weights_new
            query_bias = query_bias_new
        else:  # old
            assert query_weights.size(1) == self.final_out_dim * (self.feat_dim * self.offset_ks + 1)
            # B*2C*1 -> B*C*2*k  (k为offset的kernel_size)
            query_weights, query_bias = torch.split(
                query_weights,
                [self.final_out_dim * self.feat_dim * self.offset_ks, self.final_out_dim * 1],
                dim=1
            )
            query_weights = query_weights.reshape(-1, self.feat_dim, self.final_out_dim, self.offset_ks)
            # assert query_weights.size(0) == len(query_feats)
            query_bias = query_bias.reshape(-1, self.final_out_dim)

        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels
        # per stage of fpn
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            # head conv
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                if idx != len(self.head) - 1:  # last layer without LN
                    cur_out = self.act(self.norm[idx](cur_out))
                else:  # last layer
                    cur_out = self.act(self.norm[idx](cur_out))
                    # cur_out = self.act(cur_out)
                    # cur_out = self.norm[idx](cur_out)
            # regress 1d conv
            cur_offsets, _ = self.offset_head(cur_out, cur_mask, query_weights, query_bias,
                                              task=task, cls_wise=self.class_wise)
            # no relu
            if self.dfl:
                if isinstance(cur_offsets, list):
                    cur_out_offsets = []
                    for batch_i in range(len(cur_offsets)):  # dim-0: b
                        cur_out_offsets.append(self.scale[l](cur_offsets[batch_i]))
                        # cur_out_offsets += self.scale[l]
                    out_offsets += (cur_out_offsets,)
                    # out_offsets[batch_i] += (self.scale[l](cur_offsets[batch_i]), )
                else:
                    out_offsets += (self.scale[l](cur_offsets),)
            else:
                if isinstance(cur_offsets, list):
                    cur_out_offsets = []
                    for batch_i in range(len(cur_offsets)):  # dim-0: b
                        cur_out_offsets.append(
                            F.relu(self.scale[l](cur_offsets[batch_i])))
                    out_offsets += (cur_out_offsets,)
                else:
                    out_offsets += (F.relu(self.scale[l](cur_offsets)),)

        return out_offsets
