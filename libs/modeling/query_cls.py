# coding: utf-8
import os
import sys
import torch
import math
from torch import nn
from torch.nn import functional as F

import numpy as np
from .blocks import MaskedConv1D, Scale, LayerNorm


class QueryScaleClassifier(nn.Module):
    """make query feats squeeze with pooling, and work as cls weights"""

    def __init__(self, act_layer=nn.ReLU):
        super(QueryScaleClassifier, self).__init__()
        self.act = act_layer()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def forward(self, input, mask, query_weights, task="mr"):
        # input: B, 512, T
        # mask: B, 1, T (bool)
        # query_weights:  tad: (#n_cls, 512), mr: (b, 512)

        B, C, T = input.size()
        input = input.permute(0, 2, 1)  # (b, 512, t) -> (b, t, 512)
        if task == "mr":
            # new, for charades
            if isinstance(query_weights, list):
                out_conv = []  # B x List(#n_query, T)
                for i, one_query_weights in enumerate(query_weights):
                    # one_query_weights (#n_query, 512)
                    n_q, q_c = one_query_weights.size()
                    assert q_c == C
                    out = input[i, ...] @ one_query_weights.t()  # (T, #n_query)
                    out_conv.append(out.permute(1, 0))  # (#n_query, T)
            # old, for qvhigh
            else:
                q_b, q_c = query_weights.size()
                assert q_b == B
                assert q_c == C
                query_weights = query_weights[:, :, None]  # (b, 512, 1)
                out_conv = torch.bmm(input, query_weights)  # (b, t, 1)
                out_conv = out_conv.permute(0, 2, 1)  # (b, 1, t)
                # out_conv = [o_c for o_c in out_conv]  # B x (1 x T)
                out_conv = [out_conv[o_c] for o_c in range(out_conv.size(0))]  # B x List(1 x T)
        elif task == "tad":
            n_cls, q_c = query_weights.size()
            assert q_c == C
            out_conv = input @ query_weights.t()  # (b, t, #n_cls)
            out_conv = out_conv.permute(0, 2, 1)  # (b, #n_cls, t)
        else:
            raise NotImplemented()

        # input = input[:, None, :, :]  # (b, 1, c, t)
        # # out_conv = input * query_weights.detach()  # (b, n_cls, c, t)
        # out_conv = input * query_weights
        # out_conv = torch.sum(out_conv, dim=2)  # (b, n_cls, t)
        # scale and bias
        if isinstance(out_conv, list):
            # out_conv: B x List(#n_query, T)
            out_conv_new = []
            for i, one_out_conv in enumerate(out_conv):
                # one_out_conv: (#n_query, T)
                # mask:  (B, 1, T)
                one_out_conv_new = (one_out_conv * self.weight + self.bias)
                one_out_conv_new = one_out_conv_new * mask[i].detach()
                out_conv_new.append(one_out_conv_new)
            out_conv = out_conv_new
        else:
            out_conv = out_conv * self.weight + self.bias
            out_conv = out_conv * mask.detach()
        return out_conv, mask


class QueryIdentiedClassifier(nn.Module):
    """make query feats squeeze with pooling, and work as cls weights
        without scale function
    """

    def __init__(self, act_layer=nn.ReLU):
        super(QueryIdentiedClassifier, self).__init__()
        self.act = act_layer()
        # self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        # self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
        self.weight = 1
        self.bias = 0
        # self.feat_dim = feat_dim

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        # bias_value = 0
        # if  prior_prob > 0:
        #     bias_value = -(math.log((1 - prior_prob) / prior_prob))

        # assert num_of_output is 1, no need to makes empty_cls
        # pass

    def forward(self, input, mask, query_weights, task="mr"):
        # input: B, 512, T
        # mask: B, 1, T (bool)
        # query_weights:  tad: (#n_cls, 512), mr: (b, 512)

        B, C, T = input.size()
        input = input.permute(0, 2, 1)  # (b, 512, t) -> (b, t, 512)
        if task == "mr":
            # new, for charades
            if isinstance(query_weights, list):
                out_conv = []  # B x List(#n_query, T)
                for i, one_query_weights in enumerate(query_weights):
                    # one_query_weights (#n_query, 512)
                    n_q, q_c = one_query_weights.size()
                    assert q_c == C
                    out = input[i, ...] @ one_query_weights.t()  # (T, #n_query)
                    out_conv.append(out.permute(1, 0))  # (#n_query, T)
            # old, for qvhigh
            else:
                q_b, q_c = query_weights.size()
                assert q_b == B
                assert q_c == C
                query_weights = query_weights[:, :, None]  # (b, 512, 1)
                out_conv = torch.bmm(input, query_weights)  # (b, t, 1)
                out_conv = out_conv.permute(0, 2, 1)  # (b, 1, t)
                # out_conv = [o_c for o_c in out_conv]  # B x (1 x T)
                out_conv = [out_conv[o_c] for o_c in range(out_conv.size(0))]  # B x List(1 x T)
        elif task == "tad":
            n_cls, q_c = query_weights.size()
            assert q_c == C
            out_conv = input @ query_weights.t()  # (b, t, #n_cls)
            out_conv = out_conv.permute(0, 2, 1)  # (b, #n_cls, t)
        else:
            raise NotImplemented()

        # input = input[:, None, :, :]  # (b, 1, c, t)
        # # out_conv = input * query_weights.detach()  # (b, n_cls, c, t)
        # out_conv = input * query_weights
        # out_conv = torch.sum(out_conv, dim=2)  # (b, n_cls, t)
        # scale and bias
        if isinstance(out_conv, list):
            # out_conv: B x List(#n_query, T)
            out_conv_new = []
            for i, one_out_conv in enumerate(out_conv):
                # one_out_conv: (#n_query, T)
                # mask:  (B, 1, T)
                one_out_conv_new = (one_out_conv * self.weight + self.bias)
                one_out_conv_new = one_out_conv_new * mask[i].detach()
                out_conv_new.append(one_out_conv_new)
            out_conv = out_conv_new
        else:
            out_conv = out_conv * self.weight + self.bias
            out_conv = out_conv * mask.detach()
        return out_conv, mask


class QueryClassification(nn.Module):
    """
        1D Conv heads for classification
        make query feats pooling, work as conv weights
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            cls_fn,
            num_classes=1,  # just match and background
            pooling="max",  # with max or avg pooling
            prior_prob=0.01,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            empty_cls=[]
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
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
                )
            )
            # self.norm
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # # classifier
        # self.cls_head = MaskedConv1D(
        #     feat_dim, num_classes, kernel_size,
        #     stride=1, padding=kernel_size // 2
        # )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        # if prior_prob > 0:
        #     bias_value = -(math.log((1 - prior_prob) / prior_prob))
        #     torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)
        # self.bias_value = bias_value

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        # if len(empty_cls) > 0:
        #     bias_value = -(math.log((1 - 1e-6) / 1e-6))
        #     for idx in empty_cls:
        #         torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

        # make query weights
        self.pool_mode = pooling
        self.cls_fn = cls_fn()

    def forward(self, fpn_feats, fpn_masks, query_weights, task="mr"):
        """

        Args:
            fpn_feats:
            fpn_masks:
            query_weights: tad: (#ncls, 512); mr: (b, 512)
            cls_fn:

        Returns:

        """
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()

        device = fpn_feats[0].device

        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            # head conv
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                if idx != len(self.head) - 1:  # last layer without LN
                    cur_out = self.act(self.norm[idx](cur_out))
                else:  # last layer
                    # cur_out = self.act(cur_out)
                    # cur_out = self.norm[idx](cur_out)
                    pass
            # cls 1d conv
            cur_logits, _ = self.cls_fn(cur_out, cur_mask, query_weights, task=task)
            # cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits,)

        # fpn_masks remains the same
        return out_logits
