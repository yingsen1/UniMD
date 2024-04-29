import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats, truncate_feats_tad_mr

Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
EPS = 1e-7


@register_dataset("charades")
class CharadesDataset(Dataset):
    def __init__(
            self,
            is_training,  # if in training mode
            split,  # train_split, val_split
            train_json_file,  # tad gt
            val_json_file,  # tad_gt
            train_sta_file,  # mr gt
            val_sta_file,  # mr gt
            tad_weights,  # clip tad weights
            mr_weights,  # clip mr weights
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
        # file path        assert os.path.exists(json_file)
        if isinstance(feat_folder, str):
            assert os.path.exists(feat_folder)
        elif isinstance(feat_folder, (tuple, list)):
            for p in feat_folder:
                assert os.path.exists(p)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        self.file_ext = file_ext
        self.train_json_file = train_json_file
        self.val_json_file = val_json_file
        self.train_sta_file = train_sta_file
        self.val_sta_file = val_sta_file
        self.tad_weights_path = tad_weights
        # load tad weights
        self.tad_classifier = self._load_classifier(self.tad_weights_path)
        self.mr_weights_dir = mr_weights
        self.clip_feats_dir = clip_feats_dir

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

        # load database and select the subset
        if "Train" in self.split:
            # tad
            tad_dict_db, tad_label_dict = self._load_json_db(self.train_json_file)
            # mr
            mr_dict_db = self._load_json_db_mr(self.train_sta_file)
            self.json_file = self.train_json_file

        elif "Test" in self.split:
            # tad
            tad_dict_db, tad_label_dict = self._load_json_db(self.val_json_file)
            # mr
            mr_dict_db = self._load_json_db_mr(self.val_sta_file)
            self.json_file = self.val_json_file

        else:
            raise

        assert len(tad_label_dict) == num_classes
        self.data_list = {
            "tad": tad_dict_db,
            "mr": mr_dict_db
        }

        self.label_dict = tad_label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'charades, charades-sta',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),  # 原来的方式
            # 'tiou_thresholds': np.linspace(0.1, 0.9, 9),  # Pointtad, det-mAP
            'empty_label_ids': [],
        }
        print("==tiou_thresholds", self.db_attributes["tiou_thresholds"])
        self.count = 0
        # default: get both
        self.get_type("all")
        self.tad_keys = set(tad_dict_db.keys())  # only tad
        self.mr_keys = set(mr_dict_db.keys())  # only mr
        self.all_keys = self.tad_keys.union(self.mr_keys)  # whole
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

    def set_fetch_mode(self, mode):
        assert mode in ["both", "diff"]
        self.fetch_mode = mode
        return self.fetch_mode

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
        for key, value in json_db.items():
            # get fps if available
            if self.default_fps:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                print("==warning: %s have no duration" % key)
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
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
        json_db = json_data['database']

        # fill in the db (immutable afterwards)
        dict_db = dict()
        for key, value in json_db.items():
            # get fps if available
            if self.default_fps:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                print("==warning: %s have no duration" % key)
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels, description = [], [], []
                for act_i, act in enumerate(value['annotations']):
                    if act["segment"][0] > act["segment"][1]:
                        print("left > right, switch", act["segment"][0], act["segment"][1])
                        segments.append([act["segment"][1], act["segment"][0]])
                        print(segments[-1])
                    else:
                        segments.append(act['segment'])
                    labels.append([act_i])
                    description.append(act["label"])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = []
                description = []

            dict_db[key] = {
                'id': key,
                'fps': fps,
                'duration': duration,
                'segments': segments,
                'labels': labels,
                'description': description,
                "num_classes": len(labels),
            }

        return dict_db

    def get_classes(self):
        # get ordered label dict
        # cls_id = sorted(self.label_dict.values(), key=lambda x: int(x))
        cls_names = sorted(self.label_dict.keys(), key=lambda x: int(self.label_dict[x]))
        cls_ids = [self.label_dict[i] for i in cls_names]
        print("== cls_name == ", cls_names)
        print("== cls id   ==", cls_ids)
        return cls_names, cls_ids

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
        else:
            if self.type == "mr":
                vid = self.mr_keys[idx]
            elif self.type == "tad":
                vid = self.tad_keys[idx]
            else:
                vid = self.all_keys[idx]

        # load features
        filename = os.path.join(self.feat_folder, vid + self.file_ext)
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
        feat_stride = self.feat_stride * self.downsample_rate
        self.feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))  # !

        if self.type in ["tad", "all"]:  # get tad
            tad_data_dict = self._load_tad_gt_feats(vid)
            tad_data_dict["feats"] = feats
            tad_data_dict["q_feats"] = self.tad_classifier
        else:
            tad_data_dict = None

        if self.type in ["mr", "all"]:  # get mr
            mr_data_dict, in_tad_data = self._load_mr_gt_feats(vid)
            if in_tad_data:
                mr_data_dict["feats"] = feats
        else:
            mr_data_dict, in_tad_data = None, False

        # truncate the features during training
        if self.is_training:
            if (not in_tad_data) or self.type == "tad":  # only tad source
                if tad_data_dict["segments"] is not None:
                    tad_data_dict = truncate_feats(
                        tad_data_dict, self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
            elif self.type == "mr":
                if mr_data_dict["segments"] is not None:  # only mr source
                    mr_data_dict = truncate_feats(
                        mr_data_dict, self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
            elif self.type == "all":  # get both
                if tad_data_dict["segments"] is None and mr_data_dict["segments"] is not None:
                    mr_data_dict = truncate_feats(
                        mr_data_dict, self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
                    tad_data_dict["feats"] = mr_data_dict["feats"]
                elif tad_data_dict["segments"] is not None and mr_data_dict["segments"] is None:
                    tad_data_dict = truncate_feats(
                        tad_data_dict, self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
                    mr_data_dict["feats"] = tad_data_dict["feats"]
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

    def _load_classifier(self, file):
        assert os.path.exists(file)
        feats = np.load(file)["features"].astype(np.float32)
        feats = torch.from_numpy(feats).transpose(1, 0).contiguous()  # (n_cls, 512)
        feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    def _load_tad_gt_feats(self, vid):
        vid_item = self.data_list["tad"][vid]

        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = self.feat_offset
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if vid_item['segments'] is not None:
            segments = torch.from_numpy(
                vid_item['segments'] * vid_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(vid_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id': vid_item['id'],
                     'segments': segments,  # N x 2
                     'labels': labels,  # N
                     'fps': vid_item['fps'],
                     'duration': vid_item['duration'],
                     'feat_stride': feat_stride,
                     'feat_num_frames': self.num_frames,
                     }
        return data_dict

    def _load_mr_gt_feats(self, vid):
        if vid not in self.data_list["mr"]:
            return None, False

        vid_item = self.data_list["mr"][vid]

        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = self.feat_offset
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here

        if vid_item['segments'] is not None:
            segments = torch.from_numpy(
                vid_item['segments'] * vid_item['fps'] / feat_stride - feat_offset
            )
            mr_query_weight = os.path.join(self.mr_weights_dir, vid + ".npz")
            assert os.path.exists(mr_query_weight)
            mr_query_weight = np.load(mr_query_weight)
            query_feats = torch.from_numpy(mr_query_weight["features"].astype(np.float32))
            queries = mr_query_weight["queries"]

            for lbl_i, lbl in enumerate(vid_item["description"]):
                assert queries[lbl_i] == lbl
            labels = torch.from_numpy(vid_item['labels'])
        else:
            segments = None
            queries = None
            query_feats = None
            labels = None

        # return a data dict
        data_dict = {
            'video_id': vid_item['id'],
            'segments': segments,  # N x 2
            'labels': labels,  # N
            'fps': vid_item['fps'],
            'duration': vid_item['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': self.num_frames,
            'q_feats': query_feats,
            'queries': queries,
        }
        return data_dict, True


@register_dataset("charades_clip")
class CharadesCLIPDataset(CharadesDataset):
    def __init__(
            self,
            is_training,  # if in training mode
            split,  # train_split, val_split
            train_json_file,  # tad gt
            val_json_file,  # tad_gt
            train_sta_file,  # mr gt
            val_sta_file,  # mr gt
            tad_weights,  # clip tad weights
            mr_weights,  # clip mr weights
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
        super(CharadesCLIPDataset, self).__init__(
            is_training=is_training,  # if in training mode
            split=split,  # train_split, val_split
            train_json_file=train_json_file,  # tad gt
            val_json_file=val_json_file,  # tad_gt
            train_sta_file=train_sta_file,  # mr gt
            val_sta_file=val_sta_file,  # mr gt
            tad_weights=tad_weights,  # clip tad weights
            mr_weights=mr_weights,  # clip mr weights
            clip_feats_dir=clip_feats_dir,  # clip backbone features
            feat_folder=feat_folder,  # main backbone features
            feat_stride=feat_stride,
            num_frames=num_frames,
            default_fps=default_fps,  # default fps
            downsample_rate=downsample_rate,  # downsample rate for feats
            max_seq_len=max_seq_len,  # maximum sequence length during training
            trunc_thresh=trunc_thresh,  # threshold for truncate an action segment
            crop_ratio=crop_ratio,  # a tuple (e.g., (0.9, 1.0)) for random cropping
            input_dim=input_dim,  # input feat dim
            num_classes=num_classes,  # number of action categories
            # file_prefix,  # feature file prefix if any
            file_ext=file_ext,  # feature file extension if any
            force_upsampling=force_upsampling,  # force to upsample to max_seq_len
        )

        # clip video_feats
        self.clip_feats_dir = clip_feats_dir
        assert os.path.exists(self.clip_feats_dir)
        self.count = 0

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
        for key, value in json_db.items():
            # get fps if available
            if self.default_fps:
                # if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                print("==warning: %s have no duration" % key)
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
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
        json_db = json_data['database']

        # fill in the db (immutable afterwards)
        dict_db = dict()
        for key, value in json_db.items():
            # get fps if available
            if self.default_fps:
                # if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                print("==warning: %s have no duration" % key)
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels, description = [], [], []
                for act_i, act in enumerate(value['annotations']):
                    if act["segment"][0] > act["segment"][1]:
                        print("left > right, switch", act["segment"][0], act["segment"][1])
                        segments.append([act["segment"][1], act["segment"][0]])
                        print(segments[-1])
                    else:
                        segments.append(act['segment'])
                    labels.append([act_i])
                    description.append(act["label"])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = []
                description = []

            dict_db[key] = {
                'id': key,
                'fps': fps,
                'duration': duration,
                'segments': segments,
                'labels': labels,
                'description': description,
                "num_classes": len(labels),
            }

        return dict_db

    def concat_all_feats(self, feat_list):
        if self.count == 0 and len(feat_list) > 1:
            # print(feat_list[0].shape, feat_list[1].shape)
            if not (feat_list[0].shape[0] == feat_list[1].shape[0]):
                print("=== warning ===")
                print("feat not match strictly, %d-%d" % (
                    feat_list[0].shape[0], feat_list[1].shape[0]
                ))

        min_len = min(len(e) for e in feat_list)
        feat_list = [e[:min_len] for e in feat_list]
        feats_cat = np.concatenate(feat_list, axis=1)
        return feats_cat

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        assert self.type in ["mr", "tad", "all"]
        vid = None
        if self.fetch_mode == "both":
            vid = self.both_having_keys[idx]
        elif self.fetch_mode == "diff":
            if self.type == "tad":
                vid = self.tad_diff[idx]
            elif self.type == "mr":
                vid = self.mr_diff[idx]
            else:
                raise Exception
        else:
            if self.type == "mr":
                vid = self.mr_keys[idx]
            else:
                vid = self.tad_keys[idx]

        ### load features
        filename = os.path.join(self.feat_folder, vid + self.file_ext)
        if filename.endswith(".npz"):
            feats = np.load(filename)["features"].astype(np.float32)
        elif filename.endswith(".npy"):
            feats = np.load(filename).astype(np.float32)
        else:
            raise
        # norm->1
        feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True))

        # load clip feats
        clip_filename = os.path.join(self.clip_feats_dir, vid + '.npz')
        clip_feats = np.load(clip_filename)["features"].astype(np.float32)
        # norm->1
        clip_feats = clip_feats / (np.linalg.norm(clip_feats, axis=-1, keepdims=True))
        feats = self.concat_all_feats([feats, clip_feats])

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        self.feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))  # !

        if self.type in ["tad", "all"]:  # get tad
            tad_data_dict = self._load_tad_gt_feats(vid)
            tad_data_dict["feats"] = feats
            tad_data_dict["q_feats"] = self.tad_classifier
        else:
            tad_data_dict = None

        if self.type in ["mr", "all"]:  # get mr
            mr_data_dict, in_tad_data = self._load_mr_gt_feats(vid)
            if in_tad_data:
                mr_data_dict["feats"] = feats
        else:
            mr_data_dict, in_tad_data = None, False

        # truncate the features during training
        if self.is_training:
            if (not in_tad_data) or self.type == "tad":  # only tad source
                if tad_data_dict["segments"] is not None:
                    tad_data_dict = truncate_feats(
                        tad_data_dict, self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
            elif self.type == "mr":
                if mr_data_dict["segments"] is not None:  # only mr source
                    mr_data_dict = truncate_feats(
                        mr_data_dict, self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
            elif self.type == "all":  # get both
                if tad_data_dict["segments"] is None and mr_data_dict["segments"] is not None:
                    mr_data_dict = truncate_feats(
                        mr_data_dict, self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
                    tad_data_dict["feats"] = mr_data_dict["feats"]
                elif tad_data_dict["segments"] is not None and mr_data_dict["segments"] is None:
                    tad_data_dict = truncate_feats(
                        tad_data_dict, self.max_seq_len, self.trunc_thresh, self.feat_offset, self.crop_ratio
                    )
                    mr_data_dict["feats"] = tad_data_dict["feats"]
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

    def _load_classifier(self, file):
        assert os.path.exists(file)
        feats = np.load(file)["features"].astype(np.float32)
        feats = torch.from_numpy(feats).transpose(1, 0).contiguous()  # (n_cls, 512)
        feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    def _load_tad_gt_feats(self, vid):
        vid_item = self.data_list["tad"][vid]

        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = self.feat_offset
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if vid_item['segments'] is not None:
            segments = torch.from_numpy(
                vid_item['segments'] * vid_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(vid_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id': vid_item['id'],
                     'segments': segments,  # N x 2
                     'labels': labels,  # N
                     'fps': vid_item['fps'],
                     'duration': vid_item['duration'],
                     'feat_stride': feat_stride,
                     'feat_num_frames': self.num_frames,
                     }
        return data_dict

    def _load_mr_gt_feats(self, vid):
        if vid not in self.data_list["mr"]:
            return None, False

        vid_item = self.data_list["mr"][vid]

        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = self.feat_offset
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here

        if vid_item['segments'] is not None:
            segments = torch.from_numpy(
                vid_item['segments'] * vid_item['fps'] / feat_stride - feat_offset
            )
            mr_query_weight = os.path.join(self.mr_weights_dir, vid + ".npz")
            assert os.path.exists(mr_query_weight)
            mr_query_weight = np.load(mr_query_weight)
            query_feats = torch.from_numpy(mr_query_weight["features"].astype(np.float32))
            queries = mr_query_weight["queries"]

            for lbl_i, lbl in enumerate(vid_item["description"]):
                assert queries[lbl_i] == lbl
            labels = torch.from_numpy(vid_item['labels'])
        else:
            segments = None
            queries = None
            query_feats = None
            labels = None

        # return a data dict
        data_dict = {
            'video_id': vid_item['id'],
            'segments': segments,  # N x 2
            'labels': labels,  # N
            'fps': vid_item['fps'],
            'duration': vid_item['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': self.num_frames,
            'q_feats': query_feats,
            'queries': queries,
        }
        return data_dict, True
