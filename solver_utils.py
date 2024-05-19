import math

from sklearn.metrics import precision_recall_curve, f1_score

from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata

def knapsack_dp(values,weights,n_items,capacity,return_all=False):
    min_value = np.min(values)
    if min_value < 0:
      values = np.array(values) - min_value
    # print(values
    # print(weights[0:5])
    # print(values[0:5])
    # print(n_items)
    # print(capacity)
    # check inputs
    # check_inputs(values,weights,n_items,capacity)
    table = np.zeros((n_items+1,capacity+1),dtype=np.float32)
    keep = np.zeros((n_items+1,capacity+1),dtype=np.float32)

    for i in range(1,n_items+1):
        for w in range(0,capacity+1):
            wi = weights[i-1] # weight of current item
            vi = values[i-1] # value of current item
            if (wi <= w) and (vi + table[i-1,w-wi] > table[i-1,w]):
                table[i,w] = vi + table[i-1,w-wi]
                keep[i,w] = 1
            else:
                table[i,w] = table[i-1,w]

    picks = []
    K = capacity

    for i in range(n_items,0,-1):
        if keep[i,K] == 1:
            picks.append(i)
            K -= weights[i-1]

    picks.sort()
    picks = [x-1 for x in picks] # change to 0-index

    if return_all:
        max_val = table[n_items,capacity]
        return picks,max_val
    return picks


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    

def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):        
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr   



import torch
def represent_features(mask, video_embeddings):
    for i in range(len(mask)):
        if mask[i] == 0:
            video_embeddings[i] = video_embeddings[i] * 0.
    return video_embeddings

# def represent_features(mask, video_embeddings, change_points: list, positions: list, device: str):
#     video_average_feature = torch.zeros(video_embeddings[0].shape)
#     video_average_feature = video_average_feature.to(device)
#     filled_map = torch.zeros(len(change_points))
#     for segment_idx, segment in enumerate(change_points):
#         start, end = segment
#         average_feature = torch.zeros(video_embeddings[0].shape)
#         average_feature = average_feature.to(device)
#         selected_count = 0
#         for i in range(start, end+1):
#             if i in positions and mask[positions.index(i)] == 1:
#                 video_average_feature += video_embeddings[positions.index(i)]
#                 average_feature += video_embeddings[positions.index(i)]
#                 selected_count += 1
#         if selected_count > 0:
#             average_feature /= selected_count
#             for j in range(start, end+1):
#                 if j in positions and mask[positions.index(j)] == 0:
#                     video_embeddings[positions.index(j)] = average_feature
#             filled_map[segment_idx] = 1
#     video_average_feature /= np.sum(mask)
#     # print(np.sum(mask))
#     for segment_idx, segment in enumerate(change_points):
#         if filled_map[segment_idx] == 0:
#             start, end = segment
#             for i in range(start, end+1):
#                 if i in positions and mask[positions.index(i)] == 0:
#                     video_embeddings[positions.index(i)] = video_average_feature
#     return video_embeddings

import numpy as np
from ortools.algorithms.python import knapsack_solver



def knapsack_ortools(values, weights, items, capacity ):
    solver = knapsack_solver.KnapsackSolver(
            knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
            "KnapsackExample")
    scale = 1000
    values = np.array(values)
    weights = np.array(weights)
    values = (values * scale).astype(np.int64)
    weights = (weights).astype(np.int64)
    capacity = capacity

    solver.init(values.tolist(), [weights.tolist()], [capacity])
    computed_value = solver.solve()
    packed_items = [x for x in range(0, len(weights))
                    if solver.best_solution_contains( x)]

    return packed_items



import math


def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    # print(ypred.shape, len(positions))
    """Generate keyshot-based video summary i.e. a binary vector.
    Args:
    ---------------------------------------------
    - ypred: predicted importance scores.
    - cps: change points, 2D matrix, each row contains a segment.
    - n_frames: original number of frames.
    - nfps: number of frames per segment.
    - positions: positions of subsampled frames in the original video.
    - proportion: length of video summary (compared to original video length).
    - method: defines how shots are selected, ['knapsack', 'rank'].
    """
    #print('Computing scores of all frames based on scores of sub-sampled frames')
    n_segs = cps.shape[0]
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    #print('Computing scores of segments based on scores of all frames and change points (shot boundary))')
    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx,0]), int(cps[seg_idx,1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))

    if method == 'knapsack':
        picks = knapsack_dp(seg_score, nfps, n_segs, limits)
        # picks = knapsack_ortools(seg_score, nfps, n_segs, limits)
    elif method == 'rank':
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        raise KeyError("Unknown method {}".format(method))

    summary = np.zeros((1), dtype=np.float32) # this element should be deleted
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0) # delete the first element

    #return summary

    # thêm trả về frame_scores để tính kendal tau
    return summary, frame_scores


def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """Compare machine summary with user summary (keyshot-based).
    Args:
    --------------------------------
    machine_summary and user_summary should be binary vectors of ndarray type.
    eval_metric = {'avg', 'max'}
    'avg' averages results of comparing multiple human summaries.
    'max' takes the maximum (best) out of multiple comparisons.
    """
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users,n_frames = user_summary.shape

    # binarization
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []
    prec_arr = []
    rec_arr = []

    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx,:]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)

    if eval_metric == 'avg':
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == 'max':
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]

    return final_f_score, final_prec, final_rec

from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata
from collections import Counter

def get_rc_func(metric = "kendalltau"):
    if metric == "kendalltau":
        f = lambda x, y: kendalltau(rankdata(-x), rankdata(-y))
    elif metric == "spearman":
        f = lambda x, y: spearmanr(x, y)
    else:
        raise RuntimeError
    return f

def calculate_rank_order_statistics(
    frame_scores: list,
    user_anno: list,
    metric
) -> float:
  """
  Calculate rank_order_statistics by
  compare each user annotate with frame_scores
  """
  list_coeff = []
  frame_scores = np.array(frame_scores)

  corr_func = get_rc_func(metric)
  for idx in range(len(user_anno)):
    true_user_score = user_anno[idx]
    coeff, _ = corr_func(frame_scores,true_user_score)
    list_coeff.append(coeff)

  return np.array(list_coeff).mean()

