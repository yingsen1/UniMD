import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pandas as pd
import numpy as np
from collections import defaultdict
import time


def load_charades_localized(gtpath, frames_per_video):
    # Loads the ground truth annotations from the csv file
    df = pd.read_csv(gtpath)

    ntest = df.shape[0]
    framechar = [f"{i:03d}" for i in range(1, 51)]  # for speed
    gtids = [None] * (frames_per_video * ntest)
    gtclasses = [None] * (frames_per_video * ntest)
    c = 0
    for i in range(ntest):
        id = df.loc[i, 'id']
        classes = df.loc[i, 'actions']
        time = float(df.loc[i, 'length'])
        if pd.isnull(classes):
            missing = True
        else:
            missing = False
            classes = classes.split(';')
            classes = [list(map(float, x.replace("c", "").split(' '))) for x in classes]  # for speed
            # classes is C by 3 matrix where each row is [class start end] for an action
        for j in range(frames_per_video):
            frameclasses = np.zeros(50)  # for speed
            fc = 0
            timepoint = (j / frames_per_video) * time
            if not missing:
                for k in range(len(classes)):
                    # if missing: continue
                    if (classes[k][1] <= timepoint) and (timepoint <= classes[k][2]):
                        frameclasses[fc] = classes[k][0]
                        fc += 1
            frameid = id + '-' + framechar[j]  # for speed
            gtids[c] = frameid
            gtclasses[c] = frameclasses[:fc].astype(np.int64)
            c += 1
    gtids = gtids[:c]
    gtclasses = gtclasses[:c]
    return gtids, gtclasses


def THUMOSeventclspr(conf, labels):
    sortind = np.argsort(-conf)
    tp = (labels[sortind] == 1).astype(int)
    fp = (labels[sortind] != 1).astype(int)
    npos = len(np.where(labels == 1)[0])

    # compute precision/recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / npos
    prec = tp / (fp + tp)

    # compute average precision
    ap = 0
    tmp = (labels[sortind] == 1).astype(int)
    for i in range(len(conf)):
        if tmp[i] == 1:
            ap += prec[i]
    ap /= npos

    return rec, prec, ap


def convert_result_to_charades(result, ori_info, frames_per_video=25):
    start = time.time()
    num_cls = 157
    # for i, one_res in enumerate(results):
    vid = result['video_id']
    duration = ori_info['duration']
    output = torch.zeros(frames_per_video, num_cls)
    for i in range(frames_per_video):
        cur_time = duration / frames_per_video * i
        # print(cur_time)
        for j, one_seg in enumerate(result["segments"]):
            st, et = one_seg.tolist()
            cls = result["labels"][j].item()
            conf = result["scores"][j].item()
            if cur_time <= et and cur_time >= st:
                output[i][cls] = max(conf, output[i][cls])
    output_format = []
    for i, out in enumerate(output):
        one_out = [vid, i] + output[i].tolist()
        output_format.append(one_out)

    return output_format


def Charades_v1_localize(clsfilename,
                         gtpath="./data/TAD/charades/Charades_v1_test.csv"):
    start = time.time()
    print('Loading Charades Annotations:')
    frames_per_video = 25
    gtids, gtclasses = load_charades_localized(gtpath, frames_per_video)
    nclasses = 157
    ntest = len(gtids)
    print("Time taken: ", time.time() - start)

    start = time.time()
    print('Reading Submission File:')
    test_data = pd.read_csv(clsfilename, header=None, sep="\s+")
    testids = test_data[0].values
    framenr = test_data[1].values
    testscores = test_data.values[:, 2:]
    # if min(framenr) == 0:
    #     print('Warning: Frames should be 1 indexed')
    #     print('Warning: Adding 1 to all frames numbers')
    #     framenr = framenr + 1
    print("Time taken: ", time.time() - start)

    start = time.time()
    print('Parsing Submission Scores:')
    if len(testscores) < ntest:
        print('Warning: %d Total frames missing' % (ntest - len(testscores)))
    testscoresparsed = [list(i) for i in testscores]
    frameids = [f"{i}-{str(j + 1).zfill(3)}" for i, j in zip(testids, framenr)]  # gt: 从1 ；test: 从0
    predictions = dict(zip(frameids, testscoresparsed))
    print("Time taken: ", time.time() - start)

    start = time.time()
    print('Constructing Ground Truth Matrix:')
    gtlabel = np.zeros((ntest, nclasses))
    test = -np.inf * np.ones((ntest, nclasses))
    for i in range(ntest):
        id = gtids[i]
        # gtlabel[i, np.array(gtclasses[i]) + 1] = 1
        # one_gtclass = gtclasses[]
        gtlabel[i, list(gtclasses[i])] = 1
        if id in predictions:
            test[i, :] = predictions[id]
        else:
            print("id not in predictions", id)
    print("Time taken: ", time.time() - start)

    rec_all = np.zeros((ntest, nclasses))
    prec_all = np.zeros((ntest, nclasses))
    ap_all = np.zeros(nclasses)
    for i in range(nclasses):
        rec_all[:, i], prec_all[:, i], ap_all[i] = THUMOSeventclspr(test[:, i], gtlabel[:, i])
    map = np.mean(ap_all)
    wap = np.sum(ap_all * np.sum(gtlabel, 0)) / np.sum(gtlabel)
    print('\n\n')
    print('Per-Frame MAP: %f' % map)
    print('Per-Frame WAP: %f (weighted by size of each class)' % wap)
    print('\n\n')

    return rec_all, prec_all, ap_all, map


if __name__ == '__main__':
    gt_csv = "./data/TAD/Charades/Charades_v1_test.csv"
    test_txt = "./actionFormer_new/ckpt/charades_i3d_exp1/eval_results_tad.txt"
    # gt_ids, gt_classes = load_charades_localized(gt_csv, 25)
    rec_all, prec_all, ap_all, map = Charades_v1_localize(test_txt, gt_csv)
    # print("recall all", rec_all)
    # print("precision all", )
    print("[tad] mAP", map)
    print("finish!")
