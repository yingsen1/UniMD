import os.path
import time
import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .query_cls import QueryClassification, QueryScaleClassifier, QueryIdentiedClassifier
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss

from .query_reg_tad_mr import QueryRegressionLast_NEW

from ..utils import batched_nms
import random


def highlight(message):
    print("#" * 10, " ", message, " ", "#" * 10)


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            num_classes,
            prior_prob=0.01,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            empty_cls=[],
            init_bias=None,
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
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size // 2
        )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)
        if init_bias:
            init_bias_tensor = torch.Tensor(init_bias)
            self.cls_head.conv.bias.data = init_bias_tensor
            self.cls_head.conv.bias.data = -(
                torch.log((1 - self.cls_head.conv.bias.data) / self.cls_head.conv.bias.data))
            # self.cls_head.conv.bias.data
        print("init_bias", self.cls_head.conv.bias)

        # print("self.cls_head.conv.bias", self.cls_head.conv.bias.size(), self.cls_head.conv.bias)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits,)

        # fpn_masks remains the same
        return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
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
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)),)

        # fpn_masks remains the same
        return out_offsets


@register_meta_arch("LocPointTransformerSyntheses")
class PtTransformerSyntheses(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            backbone_type,  # a string defines which backbone we use
            fpn_type,  # a string defines which fpn we use
            backbone_arch,  # a tuple defines #layers in embed / stem / branch
            scale_factor,  # scale factor between branch layers
            input_dim,  # input feat dim
            max_seq_len,  # max sequence length (used for training)
            max_buffer_len_factor,  # max buffer size (defined a factor of max_seq_len)
            n_head,  # number of heads for self-attention in transformer
            n_mha_win_size,  # window size for self attention; -1 to use full seq
            embd_kernel_size,  # kernel size of the embedding network
            embd_dim,  # output feat channel of the embedding network
            embd_with_ln,  # attach layernorm to embedding network
            fpn_dim,  # feature dim on FPN
            fpn_with_ln,  # if to apply layer norm at the end of fpn
            fpn_start_level,  # start level of fpn
            head_dim,  # feature dim for head
            regression_range,  # regression range on each level of FPN
            head_num_layers,  # number of layers in the head (including the classifier)
            head_kernel_size,  # kernel size for reg/cls heads
            head_with_ln,  # attache layernorm to reg/cls heads
            use_abs_pe,  # if to use abs position encoding
            use_rel_pe,  # if to use rel position encoding
            num_classes,  # number of action classes
            train_cfg,  # other cfg for training
            test_cfg,  # other cfg for testing
            task,  # use for moment
            reg_head_type="ori",  # one_head, three_head
            cls_head_type="ori",  # dot, one_head, three_head
            fpn_layer=3,
            cls_head_fn="scale",  # function: w*x+b
            cls_weight_list=None,  # in loss func, make different weights for diff cls
            init_bias=None,  # cls_head set initbias
            reg_head_fc_num=3,  # fc num of reg head transform branch
            query_fc_chanel=512,  # clip_query_dim
            fpn_kernel_size=3,
            convnext_depths=[1, 1, 1, 1, 1],
            # condconv
            n_conv_group=4,
    ):
        super().__init__()
        self.task = task
        highlight(task)
        self.iter_count = 0
        self.err_gt_count = 0
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor ** i for i in range(
            fpn_start_level, backbone_arch[-1] + 1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        self.cls_weight_list = cls_weight_list
        if self.cls_weight_list:
            if isinstance(self.cls_weight_list, list):
                assert len(self.cls_weight_list) == self.num_classes
            elif isinstance(self.cls_weight_list, (int, float)):
                self.cls_weight_list = [self.cls_weight_list] * self.num_classes
            else:
                raise
        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        self.n_mha_win_size = n_mha_win_size
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            print(s, stride)
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # load train_cfg, test_cfg
        self.load_config(train_cfg, test_cfg)

        # model setting
        self.backbone_arch = backbone_arch
        self.input_dim = input_dim
        self.max_buffer_len_factor = max_buffer_len_factor
        self.n_head = n_head
        self.embd_kernel_size = embd_kernel_size
        self.embd_dim = embd_dim
        self.embd_with_ln = embd_with_ln
        self.fpn_dim = fpn_dim
        self.fpn_with_ln = fpn_with_ln
        self.fpn_start_level = fpn_start_level
        self.head_dim = head_dim
        self.head_num_layers = head_num_layers
        self.head_kernel_size = head_kernel_size
        self.head_with_ln = head_with_ln
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        self.fpn_layer = fpn_layer
        self.cls_head_fn = cls_head_fn
        self.fpn_kernel_size = fpn_kernel_size
        self.convnext_depths = convnext_depths
        # for reg head
        self.reg_head_fc_num = reg_head_fc_num
        self.query_fc_chanel = query_fc_chanel

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # self.input_noise = input_noise
        # if self.input_noise > 0:
        #     highlight("using input_noise, %f" % self.input_noise)
        self.init_bias = init_bias
        if self.init_bias is not None:
            assert len(self.init_bias) == num_classes

        # backbone network
        self.make_backbone(backbone_type)
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)
        self.embd_dim = embd_dim

        # neck network
        self.make_neck(fpn_type)

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len': max_seq_len * max_buffer_len_factor,
                'fpn_strides': self.fpn_strides,
                'regression_range': self.reg_range
            }
        )

        # classification head
        self.make_cls_head(cls_head_type, cls_head_fn)

        # regress head
        self.reg_head_type = reg_head_type
        self.make_reg_head(reg_head_type)

        # make loss
        self.make_loss("")

        # make tensorboard
        self.tb_writer = None

        # gt_assign_type
        self.assign_type = "class-wise"  # for class-wise regression if not none, else for unique regression

    def reset_task(self, task):
        self.task = task

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def load_config(self, train_cfg, test_cfg):
        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.reg_loss_weight = train_cfg["reg_loss_weight"]
        self.cls_loss_weight = train_cfg["cls_loss_weight"]
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        # self.test_pre_nms_topk = 5000
        self.test_iou_threshold = test_cfg['iou_threshold']
        # self.test_iou_threshold = 0.3
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        # self.test_max_seg_num = 5000
        print("==",
              self.test_pre_nms_topk,
              self.test_iou_threshold,
              self.test_max_seg_num,
              )
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

    def make_backbone(self, backbone_type):
        assert backbone_type in ['convTransformer', 'conv', "convnext", "convnext_stage", ]
        highlight("backbone kernel size %d " % self.embd_kernel_size)
        self.backbone_type = backbone_type
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in': self.input_dim,
                    'n_embd': self.embd_dim,
                    'n_head': self.n_head,
                    'n_embd_ks': self.embd_kernel_size,
                    'max_len': self.max_seq_len,
                    'arch': self.backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor': self.scale_factor,
                    'with_ln': self.embd_with_ln,
                    'attn_pdrop': 0.0,
                    'proj_pdrop': self.train_dropout,
                    'path_pdrop': self.train_droppath,
                    'use_abs_pe': self.use_abs_pe,
                    'use_rel_pe': self.use_rel_pe
                }
            )
        elif backbone_type == "conv":
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': self.input_dim,
                    'n_embd': self.embd_dim,
                    'n_embd_ks': self.embd_kernel_size,
                    'arch': self.backbone_arch,
                    'scale_factor': self.scale_factor,
                    'with_ln': self.embd_with_ln
                }
            )
        elif backbone_type == "convnext":
            # convnext
            self.backbone = make_backbone(
                "convnext",
                **{
                    'n_in': self.input_dim,
                    'n_embd': self.embd_dim,
                    'n_embd_ks': self.embd_kernel_size,
                    'arch': self.backbone_arch,
                    'scale_factor': self.scale_factor,
                    'with_ln': self.embd_with_ln,
                })
        elif backbone_type == "convnext_stage":
            self.backbone = make_backbone(
                "convnext_stage",
                **{
                    'n_in': self.input_dim,
                    'n_embd': self.embd_dim,
                    'n_embd_ks': self.embd_kernel_size,
                    'arch': self.backbone_arch,
                    'scale_factor': self.scale_factor,
                    'with_ln': self.embd_with_ln,
                    "drop_path_rate": 0.4,
                    "depths": self.convnext_depths,
                })
        else:
            raise NotImplementedError("not support %s yet" % backbone_type)

    def make_neck(self, fpn_type):
        assert fpn_type in [
            'fpn', 'identity', "bifpn", "bifpn_convnext",
        ]
        highlight("fpn kernel size %d " % self.fpn_kernel_size)
        self.neck_type = fpn_type

        if fpn_type in ['fpn', 'identity']:
            self.neck = make_neck(
                fpn_type,
                **{
                    'in_channels': [self.embd_dim] * (self.backbone_arch[-1] + 1),
                    'out_channel': self.fpn_dim,
                    'scale_factor': self.scale_factor,
                    'start_level': self.fpn_start_level,
                    'with_ln': self.fpn_with_ln
                }
            )
        elif fpn_type in ["bifpn"]:
            self.neck = make_neck(
                fpn_type,
                **{
                    "num_channels": self.embd_dim,
                    "conv_channels": self.embd_dim,
                    "attention": True,
                    "num_repeats": self.fpn_layer,
                    "kernel_size": self.fpn_kernel_size,
                }
            )
        elif fpn_type in ["bifpn_convnext"]:
            self.neck = make_neck(
                "bifpn_convnext",
                **{
                    "num_channels": self.embd_dim,
                    "conv_channels": self.embd_dim,
                    "attention": True,
                    "num_repeats": self.fpn_layer,
                    "kernel_size": self.fpn_kernel_size,
                }
            )
        else:
            raise NotImplementedError(fpn_type)

    def make_cls_head(self, cls_head_type, cls_head_fn):
        if cls_head_fn == "identity":
            self.query_fn = QueryIdentiedClassifier
        elif cls_head_fn == "scale":  # w*x+b, having scale and bias
            self.query_fn = QueryScaleClassifier
        else:
            raise NotImplementedError("not support cls_head_fn, %s" % cls_head_fn)

        highlight("head kernel size %d " % self.head_kernel_size)
        assert cls_head_type in ["dot", "ori"]
        self.cls_head_type = cls_head_type
        if cls_head_type == "dot":
            self.cls_head = QueryClassification(
                input_dim=self.fpn_dim,
                feat_dim=self.head_dim,
                cls_fn=self.query_fn,
                pooling="max",
                with_ln=self.head_with_ln,
                num_layers=self.head_num_layers,
                empty_cls=self.train_cfg["head_empty_cls"]
            )
        elif cls_head_type == "ori":
            self.cls_head = PtTransformerClsHead(
                self.fpn_dim, self.head_dim, self.num_classes,
                kernel_size=self.head_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                with_ln=self.head_with_ln,
                num_layers=self.head_num_layers,
                empty_cls=self.train_cfg['head_empty_cls'],
                init_bias=self.init_bias
            )

    def make_reg_head(self, reg_head_type):
        self.use_dfl = False
        self.reg_multi = False
        print("==reg fc num", self.reg_head_fc_num)
        if reg_head_type == "one_head":  # not class-wise
            self.reg_head = QueryRegressionLast_NEW(
                input_dim=self.fpn_dim,
                feat_dim=self.head_dim,
                fpn_levels=len(self.fpn_strides),
                kernel_size=self.head_kernel_size,
                num_layers=self.head_num_layers,
                with_ln=self.head_with_ln,
                query_dim=512,  # clip text dim
                class_wise=False,
                num_query_layer=self.reg_head_fc_num,
            )
        elif reg_head_type == "one_head_multi":  # class wise
            self.reg_head = QueryRegressionLast_NEW(
                input_dim=self.fpn_dim,
                feat_dim=self.head_dim,
                fpn_levels=len(self.fpn_strides),
                kernel_size=self.head_kernel_size,
                num_layers=self.head_num_layers,
                with_ln=self.head_with_ln,
                query_dim=512,  # clip text dim
                class_wise=True,
                num_query_layer=self.reg_head_fc_num,
                query_fc_chanel=self.query_fc_chanel,
            )
            self.reg_multi = True
        elif reg_head_type == "ori":
            highlight("reg_head_type %s" % reg_head_type)
            self.reg_head = PtTransformerRegHead(
                self.fpn_dim, self.head_dim, len(self.fpn_strides),
                kernel_size=self.head_kernel_size,
                num_layers=self.head_num_layers,
                with_ln=self.head_with_ln
            )
        else:
            raise NotImplementedError("not support reg_head_type, %s" % reg_head_type)

    def make_loss(self, head_loss):
        assert not head_loss, "not support loss %s" % head_loss
        if not head_loss:
            highlight("using default loss")
        else:
            highlight("using loss: %s" % head_loss)
        self.head_loss = head_loss

        self.loss_normalizer = self.train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

        self.projection = None
        
    def set_tensorboard(self, tb_writer):
        self.tb_writer = tb_writer

    def set_iter_count(self, gc):
        self.iter_count = gc

    def batch_query(self, video_list):
        q_feats = []  # B x [#n_query, 512]
        self.n_queries = []
        for x in video_list:
            q_feats.append(x["q_feats"])
            self.n_queries.append(len(x["queries"]))
        return q_feats

    def _forward_tad(self, video_list, fpn_feats, fpn_masks, points, cur_epoch=None, max_epoch=None,
                     extra_txt_feats=None,
                     save_mid_file=False):
        # 1. batch the video list into feats (B, C, T) and masks (B, 1, T)
        data_list = [video[0] for video in video_list]
        idx_list = [video[1] for video in video_list]
        self.reset_task("tad")
        task = self.task

        batched_queries = data_list[0]["q_feats"]  # (#ncls, 512)

        # 2. backbone & neck
        # share with tad and mr, only forward once in self.forward()

        # 3. cls_head -> cls score, reg_head -> reg_distance
        # out_cls: List[B, #cls + 1, FT] * n_stage
        fpn_feats_sort = [f[idx_list] for f in fpn_feats]
        fpn_masks_sort = [f[idx_list] for f in fpn_masks]

        time_head = time.time()
        if self.cls_head_type in ["ori"]:
            out_cls_logits = self.cls_head(fpn_feats_sort, fpn_masks_sort)
        else:
            # old-mr;new-mr -> out_cls_logits `List` B x (#n_query, T)
            # tad -> out_cls_logits (B, #cls, T)
            out_cls_logits = self.cls_head(fpn_feats_sort, fpn_masks_sort, batched_queries, task="tad")
        # permute the outputs
        if isinstance(out_cls_logits[0], list):
            out_cls_logits_new = []
            for fpn_i in range(len(out_cls_logits)):
                out_cls_logits_new.append(
                    [x.permute(1, 0) for x in out_cls_logits[fpn_i]]
                )
            # F List [B x List[ #n_query, T ] ] -> F List [B x List[ T, #n_query ]]
            out_cls_logits = out_cls_logits_new
        else:
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # print("-[tad] time for cls_head, %.2f" % (time.time() - time_head))
        # 4. reg_head
        # out_offset: List[B, 2, T_i] * n_stage
        time_reg_head = time.time()
        if self.reg_head_type in ["ori"]:
            out_offsets = self.reg_head(fpn_feats_sort, fpn_masks_sort)
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        else:
            out_offsets = self.reg_head(fpn_feats_sort, fpn_masks_sort, batched_queries, task=task)
            # old-mr;new-mr -> List[out_offset `List` B x (#n_query, out_dim, T)]
            # tad-not multi-cls:  List[b, out_dim, t]
            # tad multi_cls: List[b, n_cls, out_dim, t]
            if isinstance(out_offsets[0], list):
                out_offsets_new = []
                for fpn_i in range(len(out_offsets)):
                    out_offsets_new.append(
                        [x.permute(2, 0, 1) for x in out_offsets[fpn_i]])  # B x (T, #n_query, out_dim)
                out_offsets = out_offsets_new
            elif len(out_offsets[0].size()) == 4:
                # out_offset: F List[B, n_cls, 2 (xC), T_i] -> F List[B, T_i, n_cls, 2 (xC)]
                out_offsets = [x.permute(0, 3, 1, 2) for x in out_offsets]  # F List[B, T_i, n_cls, out_dim]
            elif len(out_offsets[0].size()) == 3:
                # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
                out_offsets = [x.permute(0, 2, 1) for x in out_offsets]  # F List[B, T_i, out_dim]

        fpn_masks_sort = [x.squeeze(1) for x in fpn_masks_sort]
        # print("-[tad] time for reg_head, %.2f" % (time.time() - time_reg_head))

        # 5. train loss
        if self.training:
            self.iter_count += 1
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert data_list[0]['segments'] is not None, "GT action labels does not exist"
            assert data_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in data_list]
            gt_labels = [x['labels'].to(self.device) for x in data_list]

            # assign gt and compute the gt labels for cls & reg
            # gt_cls_labels: [Tensor(n_points * n_class)] x batch_size with one-hot label but multi-label
            # gt_offsets: [Tensor(n_points, #ncls/#nquery,  2)]  offset with feat grid
            time_label_assign = time.time()
            gt_cls_labels, gt_offsets, align_metrics = self.label_points(
                points, gt_segments, gt_labels,
                out_cls_logits, out_offsets, fpn_masks_sort, cur_epoch, max_epoch
            )
            # print("-[tad] time for laebl assign, %.2f" % (time.time() - time_label_assign))
            time_loss = time.time()
            losses = self.cal_losses(
                fpn_masks_sort,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets,
                points,
                align_metrics=align_metrics,
                cur_epoch=cur_epoch,
                max_epoch=max_epoch,
            )
            # print("-[tad] time for loss, %.2f" % (time.time() - time_loss))

            # output last layer value  # wilson
            fpn_out_cls_val = []
            # for i, one_lay_cls in enumerate(out_cls_logits):
            #     out_cls_logits_avg = torch.mean(one_lay_cls, dim=1, keepdim=False)
            #     out_cls_logits_min = torch.min(one_lay_cls, dim=1, keepdim=False)
            #     out_cls_logits_max = torch.max(one_lay_cls, dim=1, keepdim=False)
            #     fpn_out_cls_val.append([
            #         out_cls_logits_avg,
            #         out_cls_logits_min,
            #         out_cls_logits_max
            #     ])
            return losses, fpn_out_cls_val, None, fpn_masks
        else:
            # decode the actions (sigmoid / stride, etc)
            if save_mid_file:
                results, mid_file = self.inference(
                    data_list, points, fpn_masks_sort,
                    out_cls_logits, out_offsets, save_mid_file
                )
                return results, mid_file
            else:
                time_infer = time.time()
                results = self.inference(
                    data_list, points, fpn_masks_sort,
                    out_cls_logits, out_offsets, save_mid_file
                )
                # print("-[tad] time for infer, %.2f" % (time.time() - time_infer))
                return results

    def _forward_mr(self, video_list, fpn_feats, fpn_masks, points, cur_epoch=None, max_epoch=None,
                    extra_txt_feats=None,
                    save_mid_file=False):
        # 1. batch the video list into feats (B, C, T) and masks (B, 1, T)
        data_list = [video[0] for video in video_list]
        idx_list = [video[1] for video in video_list]
        self.reset_task("mr")
        task = self.task
        assert extra_txt_feats is None
        batched_queries = self.batch_query(data_list)  # (#nqry, 512)  qry means query

        # 2. backbone & neck
        # share with tad and mr, only forward once in self.forward()

        # 3. cls_head -> cls score, reg_head -> reg_distance
        # out_cls: List[B, #cls + 1, FT] * n_stage
        fpn_feats_sort = [f[idx_list] for f in fpn_feats]
        fpn_masks_sort = [f[idx_list] for f in fpn_masks]

        time_head = time.time()
        if self.cls_head_type in ["ori"]:
            out_cls_logits = self.cls_head(fpn_feats_sort, fpn_masks_sort)
        else:
            # old-mr;new-mr -> out_cls_logits `List` B x (#n_query, T)
            # tad -> out_cls_logits (B, #cls, T)
            out_cls_logits = self.cls_head(fpn_feats_sort, fpn_masks_sort, batched_queries, task=task)
        # permute the outputs
        if isinstance(out_cls_logits[0], list):
            out_cls_logits_new = []
            for fpn_i in range(len(out_cls_logits)):
                out_cls_logits_new.append(
                    [x.permute(1, 0) for x in out_cls_logits[fpn_i]]
                )
            # F List [B x List[ #n_query, T ] ] -> F List [B x List[ T, #n_query ]]
            out_cls_logits = out_cls_logits_new
        else:
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # print("-[mr] time for cls_head, %.2f" % (time.time() - time_head))

        # 4. reg_head
        # out_offset: List[B, 2, T_i] * n_stage
        time_reg_head = time.time()
        if self.reg_head_type in ["ori", "ori_dfl"]:
            out_offsets = self.reg_head(fpn_feats_sort, fpn_masks_sort)
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        else:
            out_offsets = self.reg_head(fpn_feats_sort, fpn_masks_sort, batched_queries, task=task)
            # old-mr;new-mr -> List[out_offset `List` B x (#n_query, out_dim, T)]
            # tad-not multi-cls:  List[b, out_dim, t]
            # tad multi_cls: List[b, n_cls, out_dim, t]
            if isinstance(out_offsets[0], list):
                out_offsets_new = []
                for fpn_i in range(len(out_offsets)):
                    out_offsets_new.append(
                        [x.permute(2, 0, 1) for x in out_offsets[fpn_i]])  # B x (T, #n_query, out_dim)
                out_offsets = out_offsets_new  # F List[B x List[(T, #n_query, out_dim)]]
            elif len(out_offsets[0].size()) == 4:
                # out_offset: F List[B, n_cls, 2 (xC), T_i] -> F List[B, T_i, n_cls, 2 (xC)]
                out_offsets = [x.permute(0, 3, 1, 2) for x in out_offsets]  # F List[B, T_i, n_cls, out_dim]
            elif len(out_offsets[0].size()) == 3:
                # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
                out_offsets = [x.permute(0, 2, 1) for x in out_offsets]  # F List[B, T_i, out_dim]

        fpn_masks_sort = [x.squeeze(1) for x in fpn_masks_sort]
        # print("-[mr] time for reg_head, %.2f" % (time.time() - time_reg_head))

        # 5. train loss
        if self.training:
            self.iter_count += 1
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert data_list[0]['segments'] is not None, "GT action labels does not exist"
            assert data_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in data_list]
            gt_labels = [x['labels'].to(self.device) for x in data_list]

            # assign gt and compute the gt labels for cls & reg
            # gt_cls_labels: [Tensor(n_points * n_class)] x batch_size with one-hot label but multi-label
            # gt_offsets: [Tensor(n_points, #ncls/#nquery,  2)]  offset with feat grid
            time_label_assign = time.time()
            gt_cls_labels, gt_offsets, align_metrics = self.label_points(
                points, gt_segments, gt_labels,
                out_cls_logits, out_offsets, fpn_masks_sort, cur_epoch, max_epoch
            )
            # print("-[mr] time for laebl assign, %.2f" % (time.time() - time_label_assign))
            time_loss = time.time()
            losses = self.cal_losses(
                fpn_masks_sort,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets,
                points,
                align_metrics=align_metrics,
                cur_epoch=cur_epoch,
                max_epoch=max_epoch,
            )
            # print("-[mr] time for loss, %.2f" % (time.time() - time_loss))
            # output last layer value  # wilson
            fpn_out_cls_val = []
            # for i, one_lay_cls in enumerate(out_cls_logits):
            #     out_cls_logits_avg = torch.mean(one_lay_cls, dim=1, keepdim=False)
            #     out_cls_logits_min = torch.min(one_lay_cls, dim=1, keepdim=False)
            #     out_cls_logits_max = torch.max(one_lay_cls, dim=1, keepdim=False)
            #     fpn_out_cls_val.append([
            #         out_cls_logits_avg,
            #         out_cls_logits_min,
            #         out_cls_logits_max
            #     ])
            return losses, fpn_out_cls_val, None, fpn_masks
        else:
            # decode the actions (sigmoid / stride, etc)
            if save_mid_file:
                results, mid_file = self.inference(
                    data_list, points, fpn_masks_sort,
                    out_cls_logits, out_offsets, save_mid_file
                )
                return results, mid_file
            else:
                time_infer = time.time()
                results = self.inference(
                    data_list, points, fpn_masks_sort,
                    out_cls_logits, out_offsets, save_mid_file
                )
                # print("-[mr] time for infer, %.2f" % (time.time() - time_infer))
                return results

    def forward(self, video_list, task_list, cur_epoch=None, max_epoch=None, extra_txt_feats=None,
                save_mid_file=False):
        '''1. batch the video list into feats (B, C, T) and masks (B, 1, T)'''
        feats_list = [video[list(video.keys())[0]] for video in video_list]
        batched_inputs, batched_masks = self.preprocessing(feats_list)

        '''2. backbone'''
        feats, masks = self.backbone(batched_inputs, batched_masks)

        '''3. neck'''
        fpn_feats, fpn_masks = self.neck(feats, masks)

        '''4. anchor'''
        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)

        '''5. forward head for tad & mr task'''
        tad_tasks = []
        mr_tasks = []
        for batch_i, one_batch in enumerate(video_list):
            for task_i, one_task in enumerate(task_list[batch_i]):
                if one_task == "tad":
                    tad_tasks.append([one_batch["tad"], batch_i])
                elif one_task == "mr":
                    mr_tasks.append([one_batch["mr"], batch_i])
        if len(tad_tasks):
            tad_res = self._forward_tad(
                tad_tasks, fpn_feats, fpn_masks, points, cur_epoch=cur_epoch,
                max_epoch=max_epoch, extra_txt_feats=extra_txt_feats, save_mid_file=save_mid_file
            )
        if len(mr_tasks):
            mr_res = self._forward_mr(
                mr_tasks, fpn_feats, fpn_masks, points, cur_epoch=cur_epoch,
                max_epoch=max_epoch, extra_txt_feats=None, save_mid_file=save_mid_file
            )

        # training
        if self.training:
            # res: losses, fpn_out_cls_val, None, fpn_masks
            return {
                       "tad": tad_res[0] if len(tad_tasks) else None,
                       "mr": mr_res[0] if len(mr_tasks) else None,
                   }, None, None, None

        # inference
        else:
            if save_mid_file:
                results = {
                    "tad": tad_res[0] if len(tad_tasks) else None,
                    "tad_mid_file": tad_res[1] if len(tad_tasks) else None,
                    "mr": mr_res[0] if len(mr_tasks) else None,
                    "mr_mid_file": mr_res[1] if len(mr_tasks) else None,
                }
            else:
                results = {
                    "tad": tad_res if len(tad_tasks) else None,
                    "mr": mr_res if len(mr_tasks) else None,
                }

            return results

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
            # if self.input_noise > 0:
            #     # trick, adding noise slightly increases the variability between input features.
            #     noise = torch.randn_like(batched_inputs) * self.input_noise
            #     batched_inputs += noise
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels,
                     out_cls_logits, out_offsets, valid_mask, cur_epoch, max_epoch):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch

        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)  # concat fpn Points
        gt_cls, gt_offset, align_matrics = [], [], []

        # loop over each video sample
        for i, (gt_segment, gt_label) in enumerate(zip(gt_segments, gt_labels)):
            assert self.assign_type == "class-wise"  # (temp) only support class wise regression
            cls_targets, reg_targets = self.label_independ_multi_label_points_single_video(
                concat_points, gt_segment, gt_label, i)
            align_metric = torch.empty((cls_targets.size(0), 1))
            # append to list (len = # images, each of size FT x C)
            # shape like `points`
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)
            align_matrics.append(align_metric)

        return gt_cls, gt_offset, align_matrics

    @torch.no_grad()
    def label_multi_label_points_single_video(self, concat_points, gt_segment, gt_label):
        """cls: multi label; reg: not class-wise offset"""
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))

        # ---> different from actionformer, cls make multi label gt
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)

        min_len_mask = (lens < float('inf')).to(reg_targets.dtype)
        # cls_targets: F T x C; reg_targets F T x 2
        cls_targets = min_len_mask @ gt_label_one_hot
        cls_targets.clamp_(min=0.0, max=1.0)

        # check whether gt all get assigned, min_len_mask: FT x N
        min_len_mask_ = torch.sum(min_len_mask, dim=0)
        if min_len_mask_.gt(0).all():
            print("== all assign")
            print(min_len_mask_)
        else:
            print("== not all assign")
            print(min_len_mask_)

        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)
        # # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        # min_len_mask = torch.logical_and(
        #     (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        # ).to(reg_targets.dtype)

        # reg_targets: F T X 2
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def label_independ_multi_label_points_single_video(self, concat_points, gt_segment, gt_label, idx):
        """cls: multi label; reg: class-wise offset"""
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]
        task = self.task
        if task == "tad":
            num_classes = self.num_classes
        elif task == "mr":
            num_classes = self.n_queries[idx]  # num_query == num_class
        else:
            raise

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, num_classes, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))

        # ---> different from actionformer, cls make multi label gt, reg cls-wise
        gt_label_one_hot = F.one_hot(
            gt_label, num_classes
        ).to(reg_targets.dtype)

        # per sample
        min_len_mask = (lens < float('inf')).to(reg_targets.dtype)
        # cls_targets: F T x C
        cls_targets = min_len_mask @ gt_label_one_hot
        cls_targets.clamp_(min=0.0, max=1.0)

        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)
        # # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        # min_len_mask = torch.logical_and(
        #     (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        # ).to(reg_targets.dtype)

        # FT, #n_cls/#n_query, 2
        # -> class-wise offset gt
        reg_targets_new = lens.new_full((num_pts, num_classes, 2), float("inf"))
        memory = lens.new_full((num_pts, num_gts), float("inf"))

        r_point, c_gt = torch.where(min_len_mask)
        for r, c in zip(r_point, c_gt):
            gt_cls = gt_label[c]
            gt_len = lens[r, c]
            if gt_len < memory[r, c]:
                memory[r, c] = gt_len
                reg_targets_new[r, gt_cls, :] = reg_targets[r, c, :]
        reg_targets = reg_targets_new
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None, None]

        # check whether gt all get assigned, min_len_mask: FT x N
        # min_len_mask_ = torch.sum(min_len_mask, dim=0)
        # if min_len_mask_.gt(0).all():
        #     print("== all assign")
        #     print(min_len_mask_)
        # else:
        #     print("== not all assign")
        #     print(min_len_mask_)
        # check whether regression of postive sample is effect (not inf)
        target_mask = cls_targets.bool()
        pos_reg = reg_targets[target_mask]
        if pos_reg.lt(float("inf")).all():
            # print("== all effect")
            pass
        else:
            print("== not all effect")

        return cls_targets, reg_targets

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))

        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # if len(torch.where(cls_targets.sum(-1) > 1)[0]) > 0:
        #     print("hook2")
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def losses(
            self, fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets, task="tad"
    ):
        if len(out_offsets[0].size()) == 4:
            assert self.assign_type == "class-wise"
        else:
            assert self.assign_type != "class-wise"

        # gt_cls_label: [FT, C]
        # gt_offsets: class-wise: [FT, #ncls/#nquery, 2]
        # others: [FT, 2]

        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)
        bs = valid_mask.size(0)
        # 1. classification loss
        # stack the list -> (B, FT, #ncls) -> (# Valid, #ncls/#nquery)
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        pos_mask_mul = torch.logical_and((gt_cls > 0), valid_mask[..., None])  # （b, FT, #ncls）
        gt_cls_pos = gt_cls[pos_mask]  # (#pos, #n_cls)

        # cat the predicted offsets
        if self.assign_type == "class-wise":
            # # pos：positive anchor; pos_new: belongs to one class postive sample anchor
            #  -> (B, FT, ,#ncls, 2 (xC)) -> # (#Pos, #ncls , 2 (xC)) -> (#pos_new, 2 (xC))
            # pred_offsets_prob = torch.cat(out_offsets, dim=1)[pos_mask]  # (n_pos_points, #ncls, 2)
            pred_offsets_prob_mul = torch.cat(out_offsets, dim=1)[pos_mask_mul]  # (n_pos, 2 (xC))
            pred_offsets_prob = pred_offsets_prob_mul
            # pred_offsets_prob = pred_offsets_prob[gt_cls_pos > 0]  # (#pos_new, 2 (xC))
            gt_offsets = torch.stack(gt_offsets)[pos_mask_mul]  # (#pos_new, 2)
            assert gt_offsets.lt(float("inf")).all()
        else:
            #  -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
            gt_offsets = torch.stack(gt_offsets)[pos_mask]
            raise Exception("not support yet")
        pred_offsets_prob = pred_offsets_prob

        pred_offsets = pred_offsets_prob

        # update the loss normalizer
        if self.assign_type == "class-wise":
            num_pos = pos_mask_mul.sum().item()
        else:
            num_pos = pos_mask.sum().item()
        if task == "tad":
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                    1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
        elif task == "mr":
            self.num_pos += num_pos
        else:
            raise

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]  # (#valid, #ncls)

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        weights = torch.ones(gt_target.size(0), device=gt_target.device)
        if self.cls_weight_list:
            cls_weight_tensor = torch.Tensor(self.cls_weight_list).to(gt_target.device)

            gt_valid_max, gt_valid_max_idx = torch.max(gt_target, dim=1)
            gt_valid_mask = gt_valid_max >= 1
            weights[gt_valid_mask] = cls_weight_tensor[gt_valid_max_idx[gt_valid_mask]]

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum',
            weighted=weights,
        )
        if task == "tad":
            cls_loss /= self.loss_normalizer
        elif task == "mr":
            pass
        else:
            raise

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            if task == "tad":
                reg_loss /= self.loss_normalizer
            elif task == "mr":
                pass
            else:
                raise

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)
        final_loss = cls_loss + reg_loss * loss_weight
        # assert self.reg_loss_weight >= 0 and self.reg_loss_weight <= 10
        # assert self.cls_loss_weight >= 0 and self.cls_loss_weight <= 10
        #
        # # return a dict of losses
        # final_loss = cls_loss * self.cls_loss_weight + reg_loss * self.reg_loss_weight

        return {'cls_loss': cls_loss,
                'reg_loss': reg_loss,
                'final_loss': final_loss}

    @torch.no_grad()
    def inference(
            self,
            video_list,
            points, fpn_masks,
            out_cls_logits, out_offsets, save_mid_file=False
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        save_dict = dict()
        cls_logits_per_vid = [x[0] for x in out_cls_logits]
        offsets_per_vid = [x[0] for x in out_offsets]
        fpn_masks_per_vid = [x[0] for x in fpn_masks]

        save_dict[vid_idxs[0]] = {
            "fps": vid_fps[0],
            "vid_len": vid_lens[0],
            "vid_ft_stride": vid_ft_stride[0],
            "vid_ft_nframes": vid_ft_nframes[0],
            "out_cls_logits": cls_logits_per_vid,
            "offsets_per_vid": offsets_per_vid,
            "fpn_masks": fpn_masks_per_vid,
            "points": points,
        }

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
                zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)
        if save_mid_file:
            return results, save_dict
        else:
            return results

    @torch.no_grad()
    def inference_single_video(
            self,
            points,
            fpn_masks,
            out_cls_logits,
            out_offsets,

    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for i, (cls_i, offsets_i, pts_i, mask_i) in enumerate(zip(
                out_cls_logits, out_offsets, points, fpn_masks
        )):

            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()  # (T, n_cls)

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]  # (#conf>thres)
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            if self.task == "tad":
                num_classes = self.num_classes
            elif self.task == "mr":
                num_classes = self.n_queries[0]  # because batch_size==1，so get idx 0
            else:
                raise
            pt_idxs = torch.div(
                topk_idxs, num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, num_classes)

            # 3. gather predicted offsets
            # naive model: offsets: [#pos, 2]
            # dfl model:   offsets: [#pos, 2*(reg_max+1)]
            if self.use_dfl:  # (#n_pt, #reg_val)
                if self.reg_multi:  # (#n_pt, n_cls, #reg_val)]
                    offsets = offsets_i[pt_idxs, cls_idxs, :]  # (#pos, n_cls, #reg_val)
                    offsets = self.projection(offsets)
                else:
                    offsets = offsets_i[pt_idxs]
                    offsets = self.projection(offsets)
            else:
                if self.reg_multi:  # (#n_pt, n_cls, #reg_val)]
                    offsets = offsets_i[pt_idxs, cls_idxs, :]  # (#pos, n_cls, #reg_val)
                else:
                    offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments': segs_all,
                   'scores': scores_all,
                   'labels': cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms=(self.test_nms_method == 'soft'),
                    multiclass=self.test_multiclass_nms,
                    sigma=self.test_nms_sigma,
                    voting_thresh=self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs <= 0.0] *= 0.0
                segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen

            # 4: repack the results
            processed_results.append(
                {'video_id': vidx,
                 'segments': segs,
                 'scores': scores,
                 'labels': labels}
            )

        return processed_results

    def cal_losses(self, fpn_masks,
                   out_cls_logits, out_offsets,
                   gt_cls_labels, gt_offsets, points,
                   align_metrics=None,
                   cur_epoch=None, max_epoch=None):
        '''compute the loss and return'''
        if self.task == "tad":
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets
            )
        elif self.task == "mr":
            losses = None  # calculate average loss
            self.num_pos = 0
            for batch_i in range(len(out_cls_logits[0])):
                losses_i = self.losses(
                    [f_m[batch_i][None] for f_m in fpn_masks],
                    [o_c_l[batch_i][None] for o_c_l in out_cls_logits],
                    [o_f[batch_i][None] for o_f in out_offsets],
                    [gt_cls_labels[batch_i]],
                    [gt_offsets[batch_i]],
                    task="mr"
                )
                if losses is None:
                    losses = losses_i
                else:
                    for k, v in losses_i.items():
                        losses[k] += v
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                    1 - self.loss_normalizer_momentum
            ) * max(self.num_pos, 1)
            if self.num_pos == 0:
                print("warning: num_pos == 0")
            for k, v in losses.items():
                losses[k] = v / self.loss_normalizer
        else:
            raise
        return losses

    def visual_multi_reg(
            self, fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets,
            points, use_dfl=True
    ):
        device = out_cls_logits[0].device
        valid_mask = torch.cat(fpn_masks, dim=1)
        bs = valid_mask.size(0)
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        pred_offsets_prob = torch.cat(out_offsets, dim=1)[pos_mask]  # (#pos, #n_cls, 2*(reg_max+1))
        gt_cls_pos = gt_cls[pos_mask]
        # gt_cls_pos_mask = gt_cls_pos == 1

        new_points = []
        for p_tensor in points:
            fpn_max_len = p_tensor.size(0)
            new_points.append(
                torch.cat([
                    p_tensor,
                    torch.ones(fpn_max_len).to(p_tensor.device)[:, None] * fpn_max_len
                ], dim=-1))

        points = torch.cat(new_points, dim=0)[None]
        points = points.repeat(bs, 1, 1)[pos_mask]

        # pos的regress
        n_pos, n_cls, _ = pred_offsets_prob.size()
        if use_dfl:
            # (#Pos, # n_cls, 2*(reg_max+1)) -> (#Pos, #n_cls,2)
            pred_offsets = torch.zeros((n_pos, n_cls, 2), dtype=torch.float32, device=device)
            for cls_i in range(n_cls):
                # offsets = self.projection(pred_offsets_prob[:, cls_i, :])
                # pred_offsets[:, cls_i, :] = distance2bbox(points, offsets, max_shape=True)
                pred_offsets[:, cls_i, :] = self.projection(pred_offsets_prob[:, cls_i, :])
                # pred_offsets = self.projection(pred_offsets_prob)
        else:
            assert torch.min(pred_offsets_prob) >= 0
            pred_offsets = pred_offsets_prob

        gt_offsets = torch.stack(gt_offsets)[pos_mask]
        # pred_bbox = distance2bbox(points, pred_offsets, max_shape=True)
        # gt_bbox = distance2bbox(points, gt_offsets, max_shape=True)

        # get neg_reg_loss, pos_reg_loss
        whole_loss = torch.zeros((n_pos, n_cls), dtype=torch.float32, device=device)
        for cls_i in range(n_cls):
            whole_loss[:, cls_i] = ctr_diou_loss_1d(
                pred_offsets[:, cls_i, :],
                gt_offsets,
                reduction='none',
                weighted=None
            )

        # gt_cls_idx = torch.argmax(gt_cls_pos, dim=-1, keepdim=False)
        pos_loss = whole_loss[gt_cls_pos == 1]
        neg_loss = whole_loss[gt_cls_pos == 0]

        pos_min = torch.min(pos_loss)
        pos_max = torch.max(pos_loss)
        pos_avg = torch.mean(pos_loss)

        neg_min = torch.min(neg_loss)
        neg_max = torch.max(neg_loss)
        neg_avg = torch.mean(neg_loss)

        return {
            "pos_min": pos_min,
            "pos_max": pos_max,
            "pos_avg": pos_avg,
            "neg_min": neg_min,
            "neg_max": neg_max,
            "neg_avg": neg_avg,
        }
