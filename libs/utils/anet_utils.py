import sys
import json
import numpy as np
import itertools


def get_iou(pred, gt):
    """ Get tIoU of two segments
    """
    start_pred, end_pred = pred
    start, end = gt
    intersection = max(0, min(end, end_pred) - max(start, start_pred))
    union = min(max(end, end_pred) - min(start, start_pred), end - start + end_pred - start_pred)
    iou = float(intersection) / (union + 1e-8)

    return iou


def get_miou(predictions, groundtruths):
    """ Get mean IoU
    """
    ious = []
    missing_num = 0
    all_num = len(groundtruths)
    for idx in groundtruths.keys():
        if idx not in predictions:
            missing_num += 1
            continue
        pred = predictions[idx]["timestamp"][0]  # top1
        ious.append(get_iou(pred, groundtruths[idx]['timestamp']))

    miou = sum(ious) / all_num
    print('Calculating mIOU: total videos: {}, missing videos: {}'.format(all_num, missing_num))
    return miou


def get_recall_at_k(predictions, groundtruths, iou_threshold=0.5, max_proposal_num=5, detail=False):
    """ Get R@k for all predictions
    R@k: Given k proposals, if there is at least one proposal has higher tIoU than iou_threshold, R@k=1; otherwise R@k=0
    The predictions should have been sorted by confidence
    """
    vid_hits = dict()
    hit = np.zeros(shape=(len(groundtruths.keys()),), dtype=np.float32)
    all_num = len(groundtruths)
    missing_num = 0
    for idd, idx in enumerate(groundtruths.keys()):
        # output details
        if detail:
            try:
                vid, anno_id = idx.split("-", 1)
            except:
                print(idx, "err")
            if vid not in vid_hits:
                vid_hits[vid] = {"label_num": 0, "hit_num": 0}
            vid_hits[vid]["label_num"] += 1
        if idx not in predictions.keys():
            missing_num += 1
        if idx in predictions.keys():
            preds = predictions[idx]["timestamp"][:max_proposal_num]
            for pred in preds:
                if get_iou(pred, groundtruths[idx]['timestamp']) >= iou_threshold:
                    hit[idd] = 1.
        if detail:
            if int(hit[idd]) == 1:
                vid_hits[vid]["hit_num"] += 1

    avg_recall = np.sum(hit) / len(hit)
    print('Calculating Recall@{}: total videos: {}, missing videos: {}'.format(max_proposal_num, all_num, missing_num))
    if detail:
        return avg_recall, vid_hits
    return avg_recall


def eval_result(result_file, gt_file, detail=False):
    """
    Calculate mIoU, recalls for a given result file
    :param result_file: input .json result file
    :param gt_file: ground-truth file
    :return: None
    """
    results = json.load(open(result_file, 'r'))['results']
    groundtruth_data = json.load(open(gt_file, 'r'))
    video_ids = list(groundtruth_data.keys())
    out_grounding_data = {}
    # key:  video_id + '-' + str(timestamp_idx)
    for video_id in video_ids:
        gd = groundtruth_data[video_id]
        for anno_id in range(len(gd['timestamps'])):
            unique_anno_id = video_id + '-' + str(anno_id)
            out_grounding_data[unique_anno_id] = {
                'video_id': video_id,
                'anno_id': anno_id,
                'timestamp': gd['timestamps'][anno_id]
            }
    groundtruth_data = out_grounding_data

    miou = get_miou(results, groundtruth_data)
    print('mIoU: {}'.format(miou))
    scores = {}
    scores['mIOU'] = miou

    # for iou, max_proposal_num in list(itertools.product([0.7, 0.5, 0.3, 0.1], [1, 5])):
    new_vid_details = dict()
    for iou, max_proposal_num in list(itertools.product([0.7, 0.5], [1, 5])):
        if detail:
            recall, vid_details = get_recall_at_k(results, groundtruth_data, iou_threshold=iou,
                                                  max_proposal_num=max_proposal_num, detail=detail)
            for vid in vid_details.keys():
                if vid not in new_vid_details:
                    new_vid_details[vid] = dict()
                new_vid_details[vid][
                    "%d-%.2f" % (max_proposal_num, iou)
                    ] = vid_details[vid]["hit_num"] / (vid_details[vid]["label_num"] + 1e-4)
        else:
            recall = get_recall_at_k(results, groundtruth_data, iou_threshold=iou, max_proposal_num=max_proposal_num)
        print('R@{}, IoU={}: {}'.format(max_proposal_num, iou, recall))
        scores['R@{}IOU{}'.format(max_proposal_num, iou)] = recall
    if detail:
        return scores, new_vid_details
    else:
        return scores


def convert_result_to_anet_cap(results, mr_vid_qid_res):
    # unique_anno_id = video_id + '-' + str(anno_id)
    # out_grounding_data[unique_anno_id] = {
    #     'video_id': video_id,
    #     'anno_id': anno_id,
    #     'timestamp': gd['timestamps'][anno_id]
    # }
    # num_res = len(results["video-id"])
    # segs = results["segments"][0].cpu().numpy()
    # scores = results["scores"][0].cpu().numpy()
    # labels = results["label"][0].cpu().numpy()

    # results: from model
    sorted_res = dict()
    for one_vid in results:
        vid = one_vid["video_id"]
        sorted_res[vid] = dict()
        for i in range(len(one_vid["segments"])):
            label = one_vid["labels"][i].cpu().item()
            seg = one_vid["segments"][i].cpu().tolist()
            score = one_vid["scores"][i].cpu().item()
            sorted_res[vid].setdefault(label, []).append([seg[0], seg[1], score])
        for lbl in sorted_res[vid].keys():
            sorted_res[vid][lbl].sort(key=lambda x: x[-1], reverse=True)
            sorted_res[vid][lbl] = sorted_res[vid][lbl][:5]  # 最多取top5
            pred_period = [x[:2] for x in sorted_res[vid][lbl]]
            mr_vid_qid_res["%s-%s" % (vid, lbl)] = {
                "video_id": vid,
                "anno_id": lbl,
                "timestamp": pred_period,
            }
    return mr_vid_qid_res


if __name__ == '__main__':
    eval_result(sys.argv[1], sys.argv[2], sys.argv[3])
