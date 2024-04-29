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


@register_dataset("ego4d_tad_mr")
class EGO4DTadMrDataset(Dataset):
    def __init__(
            self,
            is_training,  # if in training mode
            split,  # train_split, val_split
            train_mq_file,  # tad gt
            test_mq_file,  # tad_gt
            train_nlq_file,  # mr gt
            test_nlq_file,  # mr gt
            mq_weights,  # clip tad weights
            nlq_weights,  # clip mr weights
            clip_feats_dir,  # clip backbone features
            mq_feat_folder,  # main backbone features
            nlq_feat_folder,
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
        if not isinstance(mq_feat_folder, (list, tuple)):
            mq_feat_folder = (mq_feat_folder,)
        assert all([os.path.exists(folder) for folder in mq_feat_folder])
        if not isinstance(nlq_feat_folder, (list, tuple)):
            nlq_feat_folder = (nlq_feat_folder,)
        assert all([os.path.exists(folder) for folder in nlq_feat_folder])

        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2

        self.mq_feat_folder = mq_feat_folder
        self.nlq_feat_folder = nlq_feat_folder

        self.file_ext = file_ext
        self.train_mq_file = train_mq_file
        self.test_mq_file = test_mq_file
        self.train_nlq_file = train_nlq_file
        self.test_nlq_file = test_nlq_file

        self.mq_weights_path = mq_weights
        # load tad weights
        self.mq_classifier = self._load_classifier(self.mq_weights_path)
        self.mr_weights_dir = nlq_weights
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
        if "training" in self.split:
            # mq
            mq_dict_db, mq_label_dict = self._load_json_db(self.train_mq_file)
            # nlq
            nlq_dict_db = self._load_json_db_mr(self.train_nlq_file)
            self.json_file = self.train_mq_file

        elif "validation" in self.split:
            # mq
            mq_dict_db, mq_label_dict = self._load_json_db(self.train_mq_file)
            # nlq
            nlq_dict_db = self._load_json_db_mr(self.train_nlq_file)
            self.json_file = self.train_mq_file
        elif "test" in self.split:
            # mq
            mq_dict_db, mq_label_dict = self._load_json_db(self.test_mq_file)
            # nlq
            nlq_dict_db = self._load_json_db_mr(self.test_nlq_file)
            self.json_file = self.test_mq_file
        else:
            raise
        if "test" not in self.split:  assert len(mq_label_dict) == num_classes

        self.data_list = {
            "tad": mq_dict_db,
            "mr": nlq_dict_db,
        }

        self.label_dict = mq_dict_db

        self.count = 0
        # default: get both tad and mr tasks
        self.get_type("all")
        # dataset specific attributes // ego4d setting
        self.db_attributes = {
            'dataset_name': 'ego4d',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            'empty_label_ids': []
        }
        self.tad_keys = set(mq_dict_db.keys())  # videos of tad
        self.mr_keys = set(nlq_dict_db.keys())  # videos of mr
        self.all_keys = self.tad_keys.union(self.mr_keys)  # whole videos
        self.both_having_keys = self.tad_keys.intersection(self.mr_keys)  # overlap videos
        self.tad_diff = self.tad_keys.difference(self.both_having_keys)
        self.mr_diff = self.mr_keys.difference(self.both_having_keys)
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

    def get_type(self, type="all"):
        self.type = type

    def get_attributes(self):
        return self.db_attributes

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
            if 'duration' in value:
                duration = value['duration']
            else:
                print("==warning: %s have no duration" % key)
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
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
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue

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
                segments, labels, description = [], [], []
                anno_uids = []
                anno_idxs = []
                for act_i, act in enumerate(value['annotations']):
                    if "test" in self.split:
                        pass
                    else:
                        if act["segment"][0] > act["segment"][1]:
                            print("left > right, switch", act["segment"][0], act["segment"][1])
                            segments.append([act["segment"][1], act["segment"][0]])
                            print(segments[-1])
                        else:
                            segments.append(act['segment'])
                    labels.append([act_i])
                    description.append(act["query"])
                    anno_uids.append(act['annotation_uid'])
                    anno_idxs.append(act['annotation_idx'])
                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = []
                description = []
                anno_uids = []
                anno_idxs = []

            dict_db[key] = {
                'id': key,
                'fps': fps,
                'duration': duration,
                'segments': segments,
                'labels': labels,
                'description': description,
                "num_classes": len(labels),
                "anno_uids": anno_uids,
                "anno_idxs": anno_idxs,
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

    def get_classes(self):
        # get ordered label dict
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
        # get tad
        if self.type in ["tad", "all"]:
            tad_data_dict, is_tad_data = self._load_tad_gt_feats(vid)
            if is_tad_data:
                tad_data_dict["q_feats"] = self.mq_classifier
        else:
            tad_data_dict = None
            is_tad_data = False
        # get mr
        if self.type in ["mr", "all"]:
            mr_data_dict, is_mr_data = self._load_mr_gt_feats(vid)
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

    def _load_classifier(self, file):
        assert os.path.exists(file)
        feats = np.load(file)["features"].astype(np.float32)
        feats = torch.from_numpy(feats)  # (n_cls, 512)
        feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    def concat_all_feats(self, feat_list):
        if self.count == 0 and len(feat_list) > 1:
            if not (feat_list[0].shape[0] == feat_list[1].shape[0]):
                print("=== warning ===")
                print("feat not match strictly, %d-%d" % (
                    feat_list[0].shape[0], feat_list[1].shape[0]
                ))
        min_len = min(len(e) for e in feat_list)
        feat_list = [e[:min_len] for e in feat_list]
        feats_cat = np.concatenate(feat_list, axis=1)
        return feats_cat

    def _load_tad_gt_feats(self, vid):
        if vid not in self.data_list["tad"]:
            return None, False

        feat_list = []
        for feat_dir in self.mq_feat_folder:
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
            # feats = np.concatenate(feat_list, axis=1)

        feat_stride = self.feat_stride * self.downsample_rate
        self.feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))  # !

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
                     'feats': feats,  # C x T
                     'segments': segments,  # N x 2
                     'labels': labels,  # N
                     'fps': vid_item['fps'],
                     'duration': vid_item['duration'],
                     'feat_stride': feat_stride,
                     'feat_num_frames': self.num_frames,
                     }

        return data_dict, True

    def _load_mr_gt_feats(self, vid):
        if vid not in self.data_list["mr"]:
            return None, False

        feat_list = []
        for feat_dir in self.nlq_feat_folder:
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

        feat_stride = self.feat_stride * self.downsample_rate
        self.feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))  # !

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
                assert queries[lbl_i] == (lbl.strip("?").strip(".") + "?"), \
                    print(queries[lbl_i], (lbl.strip("?").strip(".") + "?"))
            labels = torch.from_numpy(vid_item['labels'])
        else:
            segments = None
            queries = None
            query_feats = None
            labels = None

        # return a data dict
        data_dict = {
            'video_id': vid_item['id'],
            'feats': feats,  # C x T
            'segments': segments,  # N x 2
            'labels': labels,  # N
            'fps': vid_item['fps'],
            'duration': vid_item['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': self.num_frames,
            'q_feats': query_feats,
            'queries': queries,
            "anno_uids": vid_item["anno_uids"],
            "anno_idxs": vid_item["anno_idxs"],
            # "q_feats": [],
        }
        # print("fps: ", data_dict["fps"])
        return data_dict, True
