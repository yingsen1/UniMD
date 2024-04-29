import copy
import os
import json
import random

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats, truncate_feats_tad_mr
from ..utils import remove_duplicate_annotations


@register_dataset("anet_tad_mr")
class ANetTadMrDataset(Dataset):
    def __init__(
            self,
            is_training,  # if in training mode
            split,  # train_split, val_split
            tad_json,  # tad gt
            mr_train_json,  # mr gt
            mr_val_json_1,
            mr_val_json_2,
            tad_weight,  # clip tad weights
            mr_weights,  # clip mr weights
            tad_weight_avg,
            clip_feats_dir,  # clip backbone features
            feat_folder,  # main backbone features
            feat_stride,
            num_frames,
            default_fps,  # default fps
            downsample_rate,  # downsample rate for feats
            max_seq_len,  # maximum sequence length during training
            trunc_thresh,  # threshold for truncate an action segment
            crop_ratio,  # a tuple (e.g., (0.9, 1.0)) for random cropping
            input_dim,  # input feat dim
            num_classes,  # number of action categories
            # file_prefix,  # feature file prefix if any
            file_ext,  # feature file extension if any
            force_upsampling,  # force to upsample to max_seq_len
    ):
        # file path
        if not isinstance(feat_folder, (list, tuple)):
            feat_folder = (feat_folder,)
        assert all([os.path.exists(folder) for folder in feat_folder])

        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2

        self.tad_weight_path = tad_weight
        self.mr_weights_dir = mr_weights
        self.feat_folder = feat_folder
        self.clip_feats_dir = clip_feats_dir
        self.file_ext = file_ext
        # load tad weight
        self.tad_weight_avg = tad_weight_avg
        if self.tad_weight_avg:
            assert num_classes == 1
        self.tad_classifier = self._load_classifier(self.tad_weight_path)

        # annotation
        self.tad_json = tad_json
        self.mr_train_json = mr_train_json
        self.mr_val_json_1 = mr_val_json_1
        self.mr_val_json_2 = mr_val_json_2

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio
        self.force_upsampling = force_upsampling
        print("foce upsampling to fix length, %s" % self.force_upsampling)

        # load database and select the subset
        if "training" in self.split:
            # tad
            tad_dict_db, tad_label_dict = self._load_json_db(self.tad_json)
            self.json_file = self.tad_json
            # mr
            mr_dict_db = self._load_json_db_mr(self.mr_train_json)

        elif "validation" in self.split:
            raise Exception("pleases select specific validation, validationV1 or validationV2")
        elif "validationV1" in self.split or "validationV2" in self.split:
            # tad
            tad_dict_db, tad_label_dict = self._load_json_db(self.tad_json)
            self.json_file = self.tad_json
            # mr
            if "validationV1" in self.split:
                mr_dict_db = self._load_json_db_mr(self.mr_val_json_1)
            elif "validationV2" in self.split:
                mr_dict_db = self._load_json_db_mr(self.mr_val_json_2)
            else:
                raise
        elif "test" in self.split:
            raise
        else:
            raise
        self.label_dict = tad_label_dict

        # proposal vs action categories
        assert (num_classes == 1) or (len(tad_label_dict) == num_classes)
        self.data_list = {
            "tad": tad_dict_db,
            "mr": mr_dict_db,
        }
        self.count = 0
        # default: get both
        self.get_type("all")

        self.db_attributes = {
            'dataset_name': 'ActivityNet 1.3',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            'empty_label_ids': []
        }
        self.tad_keys = set(tad_dict_db.keys())  # only tad
        self.mr_keys = set(mr_dict_db.keys())  # only mr
        self.all_keys = self.tad_keys.union(self.mr_keys)  # # whole
        self.both_having_keys = self.tad_keys.intersection(self.mr_keys)  # get overlap
        self.tad_diff = self.tad_keys.difference(self.both_having_keys)  # get non-overlap in tad
        self.mr_diff = self.mr_keys.difference(self.both_having_keys)  # get non-overlap in mr
        # to list
        self.tad_keys = list(self.tad_keys)
        self.mr_keys = list(self.mr_keys)
        self.all_keys = list(self.all_keys)
        self.both_having_keys = list(self.both_having_keys)
        self.tad_diff = list(self.tad_diff)
        self.mr_diff = list(self.mr_diff)
        # fetch mode
        self.fetch_mode = None
        print("tad num, %d, same:%d, diff:%d" % (len(self.tad_keys), len(self.both_having_keys), len(self.tad_diff)))
        print("mr num, %d, same:%d, diff:%d" % (len(self.mr_keys), len(self.both_having_keys), len(self.mr_diff)))
        print("total_num: %d" % len(self.all_keys))

    def get_attributes(self):
        return self.db_attributes

    def get_type(self, type="all"):
        self.type = type

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = dict()
        split = self.split
        if "validationV1" in self.split or "validationV2" in self.split:
            split += ["validation"]

        for key, value in json_db.items():
            key = "v_" + key
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue

            # get fps if available
            if self.default_fps:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            # if 'duration' in value:
            duration = value['duration']
            # else:
            #     print("==warning: %s have no duration" % key)
            #     duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                valid_acts = remove_duplicate_annotations(value['annotations'])
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    if self.num_classes == 1:  # only proposal
                        labels[idx] = 0
                    else:
                        labels[idx] = label_dict[act['label']]
            else:
                segments = None
                labels = None

            dict_db[key] = {
                'id': key,
                'fps': fps,
                'duration': duration,
                'segments': segments,
                'labels': labels,
                "num_classes": self.num_classes,
            }

        return dict_db, label_dict

    def _load_json_db_mr(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data

        # fill in the db (immutable afterwards)
        dict_db = dict()
        for key, value in json_db.items():
            # skip the video if not in the split
            # if value['subset'].lower() not in self.split:
            #     continue

            # get fps if available
            if self.default_fps:
                # if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            # if 'duration' in value:
            duration = value['duration']
            # else:
            #     print("==warning: %s have no duration" % key)
            #     duration = 1e8

            # get annotations if available
            assert len(value["timestamps"]) == len(value["sentences"])
            if len(value["timestamps"]) > 0:
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels, description = [], [], []

                for act_i, act in enumerate(value['sentences']):
                    st = value["timestamps"][act_i][0]
                    et = value["timestamps"][act_i][1]
                    if st > et:
                        print("left > right, switch", st, et)
                        segments.append([et, st])
                        print(segments[-1])
                    else:
                        segments.append([st, et])
                    labels.append([act_i])
                    description.append(act)
                    # anno_uids.append(act['annotation_uid'])
                    # anno_idxs.append(act['annotation_idx'])
                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = []
                description = []
                # anno_uids = []
                # anno_idxs = []

            dict_db[key] = {
                'id': key,
                'fps': fps,
                'duration': duration,
                'segments': segments,
                'labels': labels,
                'description': description,
                "num_classes": len(labels),
                # "anno_uids": anno_uids,
                # "anno_idxs": anno_idxs,
            }

        return dict_db

    def set_fetch_mode(self, mode):
        assert mode in ["both", "diff"]
        self.fetch_mode = mode
        return self.fetch_mode

    def __len__(self):
        # get overlap
        if self.fetch_mode == "both":
            return len(self.both_having_keys)
        # get non-overlap videos for tad and mr respectively
        elif self.fetch_mode == "diff":
            if self.type == "tad":
                return len(self.tad_diff)
            elif self.type == "mr":
                return len(self.mr_diff)
            else:
                raise Exception
        # default:normal
        else:
            if self.type == "mr":
                return len(self.mr_keys)
            elif self.type == "tad":
                return len(self.tad_keys)
            else:
                return len(self.all_keys)

    def _load_classifier(self, file):
        assert os.path.exists(file)
        feats = np.load(file)["features"].astype(np.float32)
        feats = torch.from_numpy(feats)  # (n_cls, 512)
        feats /= feats.norm(dim=-1, keepdim=True)
        if self.tad_weight_avg:
            feats = feats.mean(dim=0, keepdims=True)
            print("==TAD的分类器取平均", feats.shape)
        return feats

    def get_classes(self):
        # get ordered label dict
        # cls_id = sorted(self.label_dict.values(), key=lambda x: int(x))
        cls_names = sorted(self.label_dict.keys(), key=lambda x: int(self.label_dict[x]))
        cls_ids = [self.label_dict[i] for i in cls_names]
        print("== cls_name == ", cls_names)
        print("== cls id   ==", cls_ids)
        return cls_names, cls_ids

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        assert self.type in ["mr", "tad", "all"]
        if self.fetch_mode == "both":
            vid = self.both_having_keys[idx]
        elif self.fetch_mode == "diff":
            assert self.type != "all"
            if self.type == "tad":
                vid = self.tad_diff[idx]
            elif self.type == "mr":
                vid = self.mr_diff[idx]
            else:
                raise Exception
        else:  # default:normal
            if self.type == "mr":
                vid = self.mr_keys[idx]
            elif self.type == "tad":
                vid = self.tad_keys[idx]
            else:
                vid = self.all_keys[idx]

        feat_list = []
        for feat_dir in self.feat_folder:
            filename = os.path.join(feat_dir, vid + self.file_ext)

            if filename.endswith(".npz"):
                feats = np.load(filename)["features"].astype(np.float32)
            elif filename.endswith(".npy"):
                feats = np.load(filename).astype(np.float32)
            else:
                raise
            # norm->1
            feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True))
            # deal with downsampling (= increased feat stride)
            feats = feats[::self.downsample_rate, :]
            feat_list.append(feats)
        if len(feat_list) == 1:
            feats = feat_list[0]
        else:
            feats = self.concat_all_feats(feat_list)
        # check length, for fix feat len
        if self.feat_stride > 0:  # normal
            pass
        else:
            seq_len = feats.shape[0]
            assert seq_len <= self.max_seq_len
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        # resize the features if needed
        if (self.feat_stride <= 0) and (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
            resize_feats = F.interpolate(
                feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            feats = resize_feats.squeeze(0)

        # we support both fixed length features / variable length features
        if self.type in ["tad", "all"]:  # get tad
            tad_data_dict, is_tad_data = self._load_tad_gt_feats(vid, feats.shape[1])
            if is_tad_data:
                tad_data_dict["q_feats"] = self.tad_classifier
                tad_data_dict["feats"] = feats
        else:
            tad_data_dict = None
            is_tad_data = False

        if self.type in ["mr", "all"]:  # get mr
            mr_data_dict, is_mr_data = self._load_mr_gt_feats(vid, feats.shape[1])
            if is_mr_data:
                mr_data_dict["feats"] = feats
        else:
            mr_data_dict = None
            is_mr_data = False

        # truncate the features during training
        if self.is_training:
            if self.type == "tad" or (is_tad_data and (not is_mr_data)):  # only tad source
                if tad_data_dict["segments"] is not None:
                    tad_data_dict = truncate_feats(
                        tad_data_dict, self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
            elif self.type == "mr" or (is_mr_data and (not is_tad_data)):  # only mr source
                if mr_data_dict["segments"] is not None:
                    mr_data_dict = truncate_feats(
                        mr_data_dict, self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
            elif self.type == "all":  # get both
                if tad_data_dict["segments"] is None and mr_data_dict["segments"] is not None:
                    mr_data_dict = truncate_feats(
                        mr_data_dict, self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
                elif tad_data_dict["segments"] is not None and mr_data_dict["segments"] is None:
                    tad_data_dict = truncate_feats(
                        tad_data_dict, self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
                elif tad_data_dict["segments"] is not None and mr_data_dict["segments"] is not None:
                    tad_data_dict, mr_data_dict = truncate_feats_tad_mr(
                        tad_data_dict, mr_data_dict,
                        self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
                else:
                    pass
            else:
                raise

        self.count += 1

        return {"tad": tad_data_dict, "mr": mr_data_dict, }

    def _load_tad_gt_feats(self, vid, feat_len):
        if vid not in self.data_list["tad"]:
            return None, False

        video_item = self.data_list["tad"][vid]
        # we support both fixed length features / variable length features
        if self.feat_stride > 0:
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                # feats = feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        else:
            # deal with fixed length featu
            assert self.force_upsampling
            # reset to max_seq_len
            seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            # center the features
            num_frames = feat_stride * self.num_frames

        feat_offset = 0.5 * num_frames / feat_stride
        self.feat_offset = feat_offset

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= feat_len:
                        # skip an action outside of the feature map
                        continue
                    # truncate an action boundary
                    valid_seg_list.append(seg.clamp(max=feat_len))
                    # some weird bug here if not converting to size 1 tensor
                    valid_label_list.append(label.view(1))
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id': video_item['id'],
                     'segments': segments,  # N x 2
                     'labels': labels,  # N
                     'fps': video_item['fps'],
                     'duration': video_item['duration'],
                     'feat_stride': feat_stride,
                     'feat_num_frames': num_frames,
                     }

        return data_dict, True

    def _load_mr_gt_feats(self, vid, feat_len):
        if vid not in self.data_list["mr"]:
            return None, False

        video_item = self.data_list["mr"][vid]
        # we support both fixed length features / variable length features
        if self.feat_stride > 0:
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                # feats = feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        else:
            # deal with fixed length featu
            assert self.force_upsampling
            # reset to max_seq_len
            seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            # center the features
            num_frames = feat_stride * self.num_frames

        feat_offset = 0.5 * self.num_frames / feat_stride
        self.feat_offset = feat_offset

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            subset = self.split[0]
            mr_query_weight = os.path.join(self.mr_weights_dir, vid + "_" + subset + ".npz")
            assert os.path.exists(mr_query_weight)
            mr_query_weight = np.load(mr_query_weight)
            query_feats = torch.from_numpy(mr_query_weight["features"].astype(np.float32))
            queries = mr_query_weight["queries"]
            for lbl_i, lbl in enumerate(video_item["description"]):
                assert queries[lbl_i] == lbl
            labels = torch.from_numpy(video_item['labels'])
            description = video_item["description"]

            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                valid_seg_list, valid_label_list = [], []
                valid_queries_list = []
                valid_queries_query_feats = []
                valid_describe_list = []
                valid_count = 0

                for seg_idx, (seg, label) in enumerate(zip(segments, labels)):
                    if seg[0] >= feat_len:
                        # skip an action outside of the feature map
                        continue
                    # truncate an action boundary
                    valid_seg_list.append(seg.clamp(max=feat_len))
                    # some weird bug here if not converting to size 1 tensor
                    valid_label_list.append(valid_count)
                    valid_count += 1
                    valid_queries_list.append(queries[seg_idx])
                    valid_queries_query_feats.append(query_feats[seg_idx])
                    valid_describe_list.append(video_item["description"][seg_idx])
                query_feats = torch.stack(valid_queries_query_feats, dim=0)
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.tensor(valid_label_list, dtype=torch.int64)
                queries = valid_queries_list
                description = valid_describe_list
        else:
            segments = None
            queries = None
            query_feats = None
            labels = None
            description = None

        # return a data dict
        data_dict = {
            'video_id': video_item['id'],
            'segments': segments,  # N x 2
            'labels': labels,  # N
            'fps': video_item['fps'],
            'duration': video_item['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': num_frames,
            'q_feats': query_feats,
            'queries': queries,
            "description": description,
        }
        return data_dict, True
