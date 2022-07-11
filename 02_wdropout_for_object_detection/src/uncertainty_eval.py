
import argparse
import os
import re
import numpy as np
from config import kitti_squeezeDet_config 
import pickle
import scipy.optimize as spo
from scipy.stats import moment, kurtosis, skew, norm, kstest, wasserstein_distance
from multiprocessing.pool import Pool as ProcessPool
from multiprocessing import cpu_count
import sys
import gzip
from sklearn.cluster import KMeans
import hdbscan
from easydict import EasyDict
import yaml

IOU_THRESH = 0.1

ANCHOR_POS = [0, 1, 9, 10]

SAMPLES_POS = 2
BBOX_BEGIN_POS = 3
BBOX_END_POS = 7
BBOX_VAR_BEGIN_POS = 11
BBOX_VAR_END_POS = 15
CLS_POS = 7
SCORE_POS = 8
PROBS_BEGIN_POS = 11

EPS = 1e-10

""" From squeezedet """
def iou(box1, box2):
  """Compute the Intersection-Over-Union of two given boxes.

  Args:
    box1: array of 4 elements [cx, cy, width, height].
    box2: same as above
  Returns:
    iou: a float number in range [0, 1]. iou of the two boxes.
  """

  lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
      max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
  if lr > 0:
    tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
        max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    if tb > 0:
      intersection = tb*lr
      union = box1[2]*box1[3]+box2[2]*box2[3]-intersection

      return intersection/union

  return 0

def batch_iou(boxes, box):
  """Compute the Intersection-Over-Union of a batch of boxes with another
  box.

  Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
  Returns:
    ious: array of a float number in range [0, 1].
  """
  lr = np.maximum(
      np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
      np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
      0
  )
  tb = np.maximum(
      np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
      np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
      0
  )
  inter = lr*tb
  union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
  return inter/union

def bbox_transform(bbox):
  """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
  for numpy array or list of tensors.
  """
  cx, cy, w, h = bbox
  out_box = [[]]*4
  out_box[0] = cx-w/2.
  out_box[1] = cy-h/2.
  out_box[2] = cx+w/2.
  out_box[3] = cy+h/2.

  return out_box

def bbox_transform_inv(bbox):
  """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
  for numpy array or list of tensors.
  """
  xmin, ymin, xmax, ymax = bbox
  out_box = [[]]*4

  width       = xmax - xmin + 1.0
  height      = ymax - ymin + 1.0
  out_box[0]  = xmin + 0.5*width
  out_box[1]  = ymin + 0.5*height
  out_box[2]  = width
  out_box[3]  = height

  return out_box

def parse_paths(eval_dir, global_step=None, train_eval=False, uncertainty_method='mc'):

    print("\n", eval_dir)

    if train_eval:
        val_dir = os.path.join(eval_dir, 'train')
    else:
        val_dir = os.path.join(eval_dir, 'val')
    if not os.path.isdir(val_dir):
        raise Exception("Could not find the directory {:s}".format(val_dir))

    def _get_detdir(gs):
        detdir = os.path.join(val_dir, 'detection_files_{:d}'\
                            .format(gs))
        if not os.path.isdir(detdir):
            return None
        return detdir

    def _find_latest_detdir():
        latest_detdir, latest_gs = None, -1
        val_subdirs = os.listdir(val_dir)
        matcher = re.compile(r'detection_files_(\d+)')
        for val_subdir in val_subdirs:
            match = matcher.match(val_subdir)
            if match is not None:
                latest_gs = max(latest_gs, int(match.group(1)))
        return _get_detdir(latest_gs), latest_gs

    def _get_all_detdirs():
        detdirs, cur_detdir = [], None
        val_subdirs = os.listdir(val_dir)
        matcher = re.compile(r'detection_files_(\d+)')
        for val_subdir in val_subdirs:
            match = matcher.match(val_subdir)
            if match is not None:
                cur_detdir = _get_detdir(int(match.group(1)))
                if cur_detdir is not None:
                  detdirs.append(cur_detdir)
        return detdirs

    det_dir = None
    if global_step is not None:
        if global_step == 'all':
            det_dir = _get_all_detdirs()
            if len(det_dir) < 1:
                det_dir = None
        else:
          det_dir = _get_detdir(global_step)

    if det_dir is None:
        det_dir, global_step = _find_latest_detdir()

    if det_dir is None:
        raise Exception("No files found")

    def _get_subdirs_of_detdir(detdir):

        no_mc_dir = os.path.join(detdir, 'nomc')
        if not os.path.isdir(no_mc_dir):
            raise Exception("Could not locate directory containing no mc files")

        mc_dir = None
        if uncertainty_method in ['mc']:
          mc_dir = os.path.join(detdir, 'mcsamples')
          if not os.path.isdir(mc_dir):
              raise Exception("Could not locate directory containing mc files")

        gt_file = os.path.join(detdir, 'gt.pkl')
        if not os.path.isfile(gt_file):
            raise Exception("Could not locate GT dict")

        for suffix in ['pkl', 'yaml']:
          meta_file = os.path.join(detdir, 'meta.%s' % suffix)
          if os.path.isfile(meta_file):
            break

        if meta_file is None:
          print("WARNING: No meta pickle found. Using KITTI model config by default")

        return detdir, mc_dir, no_mc_dir, gt_file, meta_file

    if type(det_dir) == list:
        res = []
        for detdir in det_dir:
            try:
                res.append(_get_subdirs_of_detdir(detdir))
            except Exception as e:
                print("Skipping detdir: {} Exception caught: {}".format(detdir, e.message))

        if len(res) < 1:
            raise Exception("Could not locate proper files in any of the subdirectories")

        return res
    else:
      return _get_subdirs_of_detdir(det_dir)

# Expects shape: (GT, dim) for y/mean, (GT, dim, dim) for cov
# Return shape: (GT,)
def nll(y, mean, cov, treat_invalid='raise'):
    dim = mean.shape[-1]
    const = (dim/2.) * np.log(2*np.pi)
    norm = 0.5 * np.log(np.maximum(np.linalg.det(cov), EPS))
    cov_eps = cov.copy()
    cov_eps[:, np.arange(dim), np.arange(dim)] = np.maximum(cov_eps[:, np.arange(dim), np.arange(dim)], EPS)
    res = norm + 0.5 * np.matmul((y - mean)[:, None, :], np.matmul(np.linalg.inv(cov_eps), (y-mean)[:, :, None]))

    if treat_invalid == 'raise' and np.any(np.isnan(res)):
        raise ArithmeticError("Result contains NaN values")

    return res[:, 0, 0]

# Expects shape: (GT, dim) for y/mean, (GT, dim, dim) for cov
# Return shape (GT,)
def mahalanobis(y, mean, cov, treat_invalid='raise'):
    dim = mean.shape[-1]
    cov_eps = cov + np.diag(EPS * np.ones(dim))[None, :, :]
    res = np.sqrt(np.matmul((y - mean)[:, None, :], np.matmul(np.linalg.inv(cov_eps), (y-mean)[:, :, None])))
    if treat_invalid == 'raise' and np.any(np.isnan(res)):
        raise ArithmeticError("Result contains NaN values")
    return res[:, 0, 0]

def ece(error_quantiles):
    bins = np.linspace(0, 1+EPS, 20, endpoint=False)
    rel_freqs = np.zeros(20)
    digitized = np.digitize(error_quantiles, bins)-1
    unique, counts = np.unique(digitized, return_counts=True)
    rel_freqs[unique] = counts / float(len(error_quantiles))
    ece = np.abs(rel_freqs - 0.05).sum()
    return ece

# Expects shape: (n_data, 4)
# Return shape: len(q_list) x 4
def etl(pred_residual_normed, q_list):

  abs_res_normed = np.abs(pred_residual_normed)
  q = np.quantile(abs_res_normed, q_list, axis=0) # len(q_list) x 4
  filtered = np.stack([abs_res_normed for _ in range(len(q_list))]) # len(q_list) x n_data x 4
  filtered[abs_res_normed[None, :, :] <= q[:, None, :]] = np.nan
  return np.nanmean(filtered, axis=1) # len(q_list) x 4

def _bip_match_iou(gt_bboxes, dets, bbox_begin_pos=BBOX_BEGIN_POS, bbox_end_pos=BBOX_END_POS):

  # batch_iou: Returns (n_gt,) array of ious between gts and iths entry in lines_sample
  # np.array(...) results in (n_lines_samples, n_gt) matrix (rows are samples columns are gt)
  cost_matrix = -np.array([batch_iou(gt_bboxes, dets[i, bbox_begin_pos:bbox_end_pos])
                           for i in range(len(dets))])

  # bipartite matching
  row_ind, col_ind = spo.linear_sum_assignment(cost_matrix)
  assert np.all(np.unique(col_ind, return_counts=True)[1] == 1)

  return row_ind, col_ind, cost_matrix


def _bip_match_gt_to_det(gt_idx_to_det, gt_bboxes, dets,
                         bbox_begin_pos=BBOX_BEGIN_POS,
                         bbox_end_pos=BBOX_END_POS):

    row_ind, col_ind, cost_matrix = _bip_match_iou(gt_bboxes, dets, bbox_begin_pos, bbox_end_pos)

    for i in range(len(row_ind)):
        if col_ind[i] not in gt_idx_to_det:
            gt_idx_to_det[col_ind[i]] = []
        gt_idx_to_det[col_ind[i]].append(dets[row_ind[i], :])

    det_no_match_mask = np.ones(len(dets), dtype=np.bool)
    det_no_match_mask[row_ind] = False
    det_no_match = dets[det_no_match_mask, :]

    order = np.argsort(col_ind)
    return cost_matrix[row_ind[order], col_ind[order]], det_no_match # ious sorted by gt_idx, detections without match

def _filter_by_iou(gt_idx_to_det, gt_bboxes, bbox_begin_pos=BBOX_BEGIN_POS,
                   bbox_end_pos=BBOX_END_POS):
    bboxes_filtered = dict()
    for gt_idx in gt_idx_to_det:
        dets = []
        bboxes_filtered[gt_idx] = []
        for det in gt_idx_to_det[gt_idx]:
            if iou(gt_bboxes[gt_idx, :4], det[bbox_begin_pos:bbox_end_pos]) >= IOU_THRESH:
                dets.append(det)
            else:
                bboxes_filtered[gt_idx].append(det)
        gt_idx_to_det[gt_idx] = dets

    return bboxes_filtered

def _read_pred_file(dir, img_file, use_scaling=None):

    with open(os.path.join(dir, img_file)) as f:
        lines = f.readlines()

    lines_val = np.array([[float(val) for val in line.strip().split(' ')] for line in lines])

    # Transform bbox representation to cx, cy, w, h
    if np.shape(lines_val)[0] != 0:
      lines_bbox = lines_val[:, BBOX_BEGIN_POS:BBOX_END_POS]
      lines_bbox_inv = np.array(bbox_transform_inv([lines_bbox[:, 0],
                                                    lines_bbox[:, 1],
                                                    lines_bbox[:, 2],
                                                    lines_bbox[:, 3]])).T

      if use_scaling is not None:
        lines_bbox_inv[:, [0,2]] *= use_scaling[0]
        lines_bbox_inv[:, [1,3]] *= use_scaling[1]
      lines_val[:, BBOX_BEGIN_POS:BBOX_END_POS] = lines_bbox_inv

    return lines_val


def _ap_matching(cls_to_gt_boxes, detections, iou_thresh=IOU_THRESH,
                 score_pos=SCORE_POS, cls_pos=CLS_POS, bbox_begin_pos=BBOX_BEGIN_POS, bbox_end_pos=BBOX_END_POS):

  # AP: Uses greedy matching per class
  dets = {k: [] for k in cls_to_gt_boxes}

  # iterate all detections
  detected = dict()
  for no_mc_line_val in sorted(detections, key=lambda x: x[score_pos], reverse=True):
    bbox = no_mc_line_val[bbox_begin_pos:bbox_end_pos]
    cls = int(no_mc_line_val[cls_pos])
    score = no_mc_line_val[score_pos]

    if cls not in detected:
      detected[cls] = set()

    if cls in cls_to_gt_boxes:
      cls_gt_bboxes = np.array(cls_to_gt_boxes[cls])
      ious = batch_iou(cls_gt_bboxes, bbox) # compute iou to all gt's of same class

      # greedy matching
      max_iou = np.max(ious)
      best_idx = np.argmax(ious)

      # TP if over iou thresh and not already detected
      decision = (max_iou > iou_thresh) and (best_idx not in detected[cls])
      dets[cls].append((score, decision))

      if decision is True:
        detected[cls].add(best_idx)

    else: # detection without gt is a fp
      if cls not in dets:
        dets[cls] = []
      dets[cls].append((score, False))

  return dets


def process_image(inp):

    img_idx, gt_data, no_mc_dir, mc_dir, mc, uncertainty_method, use_scaling, aggregation_method, beta_inv = inp

    results = {'img_idx': img_idx,
               'error': [False, 0, 0],  # detection filtered, singular, nan
               'n_gt': 0,  # total number of gt bboxes
               'n_gt_cls': dict(), # number of gt bboxes per class
               'n_net_proposals': 0,  # number of all network proposals
               'n_net_proposals_cls': dict(), # number of all network proposals per class
               'n_net_matched': 0, # number of object propsals that have been matched and have iou >= IOU_THRESH
               'n_net_matched_cls_correct': 0,  # number of correct detections (matched & IOU >= 0.1 & correct class)
               'n_net_matched_cls_error': 0,  # number of detections with wrong class (matched & IOU >= 0.1 & wrong class)
               'n_bg_error': 0, # number of detections with IOU < IOU_THRESH plus number of detections that are not associated to a gt (not matched or (matched and IOU < 0.1))
               'dets': dict(), # For AP calculation: Maps ap threshold to dict that maps class to list of tuples; each tuple represents a detected bbox; (score, bool(tp))
               'filtered_nomc': None,
               'gt_idxs_no_det': None,
               'gt_idxs_nomc': None,
               'ious_nomc': None,
               'pred_mean': None, 'pred_cov': None,
               'pred_residual': None, 'pred_residual_normed': None,
               'pred_error_quantile': None,
               'pred_mahalanobis': None,
               'std': None,
               'spread': None,
               'nll': None,
               'nll_x': None, 'nll_y': None, 'nll_w': None, 'nll_h': None,
               'kitti_annotation': None
               }

    if uncertainty_method in ['mc']:
      results.update({
        'filtered_mc': None,
        'gt_idxs_mc': None,
        'ious_mc': None,
        'ious_mean_mc': None,
        'pred_nomc': None
      })

      gt_idx_to_mc_det = {}

    # Find no mc predictions corresponding to image
    image_nomcdet_file = '{:d}.nomc'.format(img_idx)
    if image_nomcdet_file not in os.listdir(no_mc_dir):
        raise Exception("Could not locate nomc detection corresponding to {:d}".format(img_idx))
    no_mc_lines_val = _read_pred_file(no_mc_dir, image_nomcdet_file, use_scaling=use_scaling)

    if np.shape(no_mc_lines_val)[0] == 0:  # when no_mc_lines_val is empty, return the default results
      return results

    # Find mc predictions corresponding to image
    if uncertainty_method in ['mc']:
      image_mcdet_file = '{:d}.mc'.format(img_idx)
      if image_mcdet_file not in os.listdir(mc_dir):
        raise Exception("Could not locate mc detection corresponding to {:d}".format(img_idx))
      lines_val = _read_pred_file(mc_dir, image_mcdet_file, use_scaling=use_scaling)

    gt_bboxes = np.array([a[:5] for a in gt_data])
    if use_scaling is not None:
      gt_bboxes[:, [0,2]] *= use_scaling[0] # scale cx-coord and w
      gt_bboxes[:, [1,3]] *= use_scaling[1] # scale cy-coord and h
    results['n_gt'] = len(gt_bboxes)
    gt_idx_to_nomc_det = {}

    """ Nomc matching: Done for MC for computing 'spread'"""

    # Match no-mc detections using bipartite maching; this returns a bijective map between gt and detections
    # if there are more detections than gt, some detections will not have associated gt and vice versa
    ious_nomc, dets_no_match_nomc  = _bip_match_gt_to_det(gt_idx_to_nomc_det, gt_bboxes[:, :4], no_mc_lines_val)
    ious_nomc = -ious_nomc

    bboxes_filtered = _filter_by_iou(gt_idx_to_nomc_det, gt_bboxes)  # Filter low IOU predictions out
    results['filtered_nomc'] = {k: len(v) for k, v in bboxes_filtered.items()}
    results['ious_nomc'] = ious_nomc[ious_nomc >= IOU_THRESH]

    # gt_idxs of nomc predictions that are not filtered by IOU_THRESH or bipartite matching
    gt_idxs_nomc = np.array([gt_idx for gt_idx in sorted(gt_idx_to_nomc_det)
                             if len(gt_idx_to_nomc_det[gt_idx]) >= 1], dtype=np.int)

    """ Matching MC samples """

    if uncertainty_method == 'mc':

        if aggregation_method == 'individual': # Individually match samples to detections
            ious = []
            for k in range(mc.MC_DROP_SAMPLES): # Iterate samples

                # Get all lines of a specific sample
                lines_sample = lines_val[lines_val[:, SAMPLES_POS] == k]

                # handling the mc file edge case: when predictions from certain forward pass are empty.
                if np.shape(lines_sample)[0] == 0:
                  continue

                iou_, _ = _bip_match_gt_to_det(gt_idx_to_mc_det, gt_bboxes[:, :4], lines_sample)
                iou_ = -iou_
                if not np.any(np.isnan(iou_)):
                    ious += list(iou_)

            raise Exception("Individual matching is not supported anymore. Use 'anchor' or 'clustering' instead")

        elif aggregation_method == 'anchor': # Aggregate/Cluster via anchor

            anchor_aggregates = dict()
            for line in lines_val: # Iterate all detections
                anchor_coord = tuple(line[ANCHOR_POS])
                if anchor_coord not in anchor_aggregates:
                    anchor_aggregates[anchor_coord] = []
                anchor_aggregates[anchor_coord].append(line) # Map anchor to all detections with same anchor

            clusters, cluster_means, cluster_mean_probs, anchor_coords = [], [], [], []
            for c, anchor_coord in enumerate(sorted(anchor_aggregates)):
                cluster = anchor_aggregates[anchor_coord]
                if len(cluster) < 2: # ignore anchors that have less than 2 corresponding detections
                    continue
                clusters.append(cluster)
                cluster_means.append(np.mean(cluster[:, BBOX_BEGIN_POS:BBOX_END_POS], axis=0))
                cluster_mean_probs.append(np.mean(cluster[:, PROBS_BEGIN_POS:PROBS_BEGIN_POS + mc.CLASSES], axis=0))
                anchor_coords.append(anchor_coord)

        elif aggregation_method == 'kmeans': # Aggregate/Cluster by kmeans

            # Find average number of proposals; Used as k for kmeans
            n_centers = max(1, int(np.mean([len(lines_val[lines_val[:, SAMPLES_POS] == k]) for k in range(mc.MC_DROP_SAMPLES)])))

            # Cluster all detections via kmeans
            kmeans = KMeans(n_centers, random_state=42)
            kmeans.fit(lines_val[:, BBOX_BEGIN_POS:BBOX_END_POS])

            # Find cluster means of each cluster
            clusters, cluster_means, cluster_mean_probs = [], [], []
            for c in range(n_centers):
                cluster = lines_val[kmeans.labels_ == c]
                if len(cluster) > 4: # ignore clusters that are too small
                    clusters.append(cluster)
                    cluster_means.append(np.mean(cluster[:, BBOX_BEGIN_POS:BBOX_END_POS], axis=0))
                    cluster_mean_probs.append(np.mean(cluster[:, PROBS_BEGIN_POS:PROBS_BEGIN_POS+mc.CLASSES], axis=0))

        elif aggregation_method == 'hdbscan': # Aggregate/Cluster by hdbscan

            clusterer = hdbscan.HDBSCAN(min_cluster_size=20, gen_min_span_tree=True)
            clusterer.fit(lines_val[:, BBOX_BEGIN_POS:BBOX_END_POS])

            # Find cluster means of each cluster
            clusters, cluster_means, cluster_mean_probs = [], [], []
            for c in range(clusterer.labels_.max()):
                cluster = lines_val[clusterer.labels_ == c]
                clusters.append(cluster)
                cluster_means.append(np.mean(cluster[:, BBOX_BEGIN_POS:BBOX_END_POS], axis=0))
                cluster_mean_probs.append(np.mean(cluster[:, PROBS_BEGIN_POS:PROBS_BEGIN_POS+mc.CLASSES], axis=0))

        else:
          raise Exception("Given aggregation_method is not recognized")

        # hdbscan sometimes yields only outlier clusters
        if np.shape(cluster_means)[0] == 0:
            results['error'][0] = True
            return results

        # Match cluster means to gt
        det_ind, gt_ind, cost_matrix = _bip_match_iou(gt_bboxes[:, :4], np.array(cluster_means),
                                                      bbox_begin_pos=0, bbox_end_pos=4)

        # map gt_idx to all samples corresponding to the matched mean box
        order = np.argsort(gt_ind)
        ious = cost_matrix[det_ind[order], gt_ind[order]]
        for k in range(len(det_ind)):
            gt_idx_to_mc_det[gt_ind[k]] = clusters[det_ind[k]]

        results['ious_mc'] = ious


    """ Per class metrics """

    classes = mc['CLASS_NAMES']
    num_classes = len(classes)

    # map class to gt boxes
    cls_to_gt_boxes = {}
    for gt_vals in gt_bboxes:
      gt_cls = int(gt_vals[4])
      if gt_cls not in cls_to_gt_boxes:
        cls_to_gt_boxes[gt_cls] = []
      cls_to_gt_boxes[gt_cls].append(gt_vals[:4])

    # Compute AP based on NOMC predictions
    if uncertainty_method == 'mc':
        cluster_mean_dets = [cluster_means[c].tolist() + [np.max(cluster_mean_probs[c]), np.argmax(cluster_mean_probs[c])]
                             for c in range(len(cluster_means))]

        for iou_thresh in range(50, 100, 5):
            results['dets'][iou_thresh / 100.] = _ap_matching(cls_to_gt_boxes, cluster_mean_dets,
                                                              iou_thresh=iou_thresh / 100.,
                                                              bbox_begin_pos=0, bbox_end_pos=4,
                                                              score_pos=4, cls_pos=5)

    results['n_gt_cls'] = {cls: len(cls_to_gt_boxes[cls]) for cls in cls_to_gt_boxes}

    """ Matching statistics """

    if uncertainty_method == 'mc':

        results['n_net_proposals'] = len(cluster_means)

        # iou filtering
        idxs_matched_mean_dets, cluster_sizes = [], []
        for k in range(len(det_ind)):
            if iou(cluster_means[det_ind[k]], gt_bboxes[gt_ind[k]]) >= IOU_THRESH:
                idxs_matched_mean_dets.append(k)
                cluster_sizes.append(len(clusters[det_ind[k]]))
            else:
                gt_idx_to_mc_det[gt_ind[k]] = []  # Reject clusters with mean box with iou not over IOU_THRESH
        results['cluster_sizes'] = cluster_sizes if len(cluster_sizes) > 0 else None

        results['n_net_matched'] = len(idxs_matched_mean_dets)
        results['n_net_matched_cls_correct'] = np.count_nonzero([gt_bboxes[gt_ind[k], 4] \
                                                                 == np.argmax(cluster_mean_probs[det_ind[k]])
                                                                 for k in idxs_matched_mean_dets])

    """ Global uncertainty metrics """

    # Compute mean, cov, residual
    if uncertainty_method in ['mc']:
        det_samples_mean, det_samples_cov = [], []
        det_samples_corrcoef = []
        gt_idxs_no_det, gt_idxs_mc = [], []
        for gt_idx in sorted(gt_idx_to_mc_det):
            det_sample_lines = np.array(gt_idx_to_mc_det[gt_idx])
            if len(det_sample_lines) < 2:  # It can happen that all detections are filtered
                gt_idxs_no_det.append(gt_idx)
                continue

            det_sample_bboxs = det_sample_lines[:, BBOX_BEGIN_POS:BBOX_END_POS] # n_samples x 4
            det_samples_mean.append(np.mean(det_sample_bboxs, axis=0))

            if uncertainty_method == 'mc' and beta_inv is not None: # add inverse beta factor (aleatoric part) to covariance (see gal et. al.)
                det_samples_cov.append(np.cov(det_sample_bboxs.T) + np.diag(np.ones(det_sample_bboxs.shape[1])*float(beta_inv)))
            else:
                det_samples_cov.append(np.cov(det_sample_bboxs.T))
            det_samples_corrcoef.append(np.corrcoef(det_sample_bboxs.T))
            gt_idxs_mc.append(gt_idx)

        if len(gt_idxs_mc) == 0:
            results['error'][0] = True
            print("All Detections got filtered out")
            return results

        det_samples_mean, det_samples_cov = np.array(det_samples_mean), np.array(det_samples_cov)
        gt_idxs_no_det, gt_idxs_mc = np.array(gt_idxs_no_det), np.array(gt_idxs_mc)
        det_samples_corrcoef = np.array(det_samples_corrcoef)
        det_samples_std = np.array([np.sqrt(np.diag(cov)) for cov in det_samples_cov])

        results['pred_mean'] = det_samples_mean
        results['pred_cov'] = det_samples_cov
        results['gt_idxs_no_det'] = gt_idxs_no_det
        results['gt_idxs_mc'] = gt_idxs_mc
        results['ious_mean_mc'] = np.array([iou(det_samples_mean[i], gt_bboxes[gt_idx, :4])
                                          for i, gt_idx in enumerate(gt_idxs_mc)])

        if uncertainty_method == 'mc':

            total_std = det_samples_std

            # To find spread, we need to find the overlap between mc samples and nomc samples
            gt_idxs_nomc = np.array([gt_idx for gt_idx in gt_idxs_mc
                                     if gt_idx in gt_idx_to_nomc_det and len(gt_idx_to_nomc_det[gt_idx]) >= 1])
            spread = None
            if len(gt_idxs_nomc) > 0:
                gt_idxs_nomc_index = np.nonzero(gt_idxs_mc == gt_idxs_nomc[:, None])[1]
                no_mc_bboxs = np.array(
                  [gt_idx_to_nomc_det[gt_idx][0][BBOX_BEGIN_POS:BBOX_END_POS] for gt_idx in gt_idxs_nomc])
                spread = np.abs(no_mc_bboxs - det_samples_mean[gt_idxs_nomc_index])
            residual_mean = det_samples_mean
            results['std'] = total_std

        results['spread'] = spread
        results['pred_residual'] = residual_mean - gt_bboxes[gt_idxs_mc, :4]

    results['pred_residual_normed'] = results['pred_residual'] / np.maximum(total_std, EPS)
    results['pred_error_quantile'] = np.round(norm.cdf(results['pred_residual_normed']), 2)

    try:
        if uncertainty_method == 'mc':
            results['pred_mahalanobis'] = mahalanobis(gt_bboxes[gt_idxs_mc, :4], residual_mean,
                                                               det_samples_cov)
    except np.linalg.LinAlgError:
        results['error'][1] += 1
        print("Skipping mahalanobis distance")
    except ArithmeticError:
        results['error'][2] += 1
        print("NaN encountered")

    # Compute NLL
    try:
        if uncertainty_method == 'mc':
            results['nll'] = nll(gt_bboxes[gt_idxs_mc, :4], residual_mean, det_samples_cov)
        else:
            results['nll'] = nll(gt_bboxes[gt_idxs_nomc, :4], residual_mean, results['pred_cov'])
    except np.linalg.LinAlgError:
        results['error'][1] += 1
        print("Skipping 4d NLL (singular matrix)")
    except ArithmeticError:
        results['error'][2] += 1
        print("NaN encountered")

    for i, ident in enumerate(['nll_x', 'nll_y', 'nll_w', 'nll_h']):
        try:
            if uncertainty_method == 'mc':
                results[ident] = nll(gt_bboxes[gt_idxs_mc, i, None],
                                          residual_mean[:, i, None],
                                          det_samples_cov[:, i, i, None, None])
        except ArithmeticError:
            results['error'][2] += 1
            print("NaN encountered")

    if uncertainty_method in ['mc']:
        results['kitti_annotation'] = [a[4:] for i, a in enumerate(gt_data) if i in gt_idxs_mc]
    return results

def _get_meta_info(paths):
  det_dir, mc_dir, no_mc_dir, gt_file, meta_file = paths

  mc, image_idx = None, None
  if meta_file is not None:
    if meta_file.endswith('.pkl'):
      with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
      mc = meta['mc']
      if 'image_idx' in meta:
        image_idx = set(meta['image_idx'])
      elif 'use_subsample' in meta and meta['use_subsample']:
        print('Warning: use_subsample is True, but image_idx is not found in meta.pkl; '
              'This might cause a problem if the image indices in gt.pkl do not match the image indices in the nomc/mc dir')

    elif meta_file.endswith('.yaml'):
      with open(meta_file, 'r') as f:
        mc = EasyDict(yaml.safe_load(f))
        mc.MC_DROP_SAMPLES = mc.mc_dropout_samples
        mc.CLASS_NAMES = mc.class_names
        mc.CLASSES = len(mc.CLASS_NAMES)
  else:
    mc = kitti_squeezeDet_config()

  return mc, image_idx

# Calls dispatch_func for all global_steps specified if they are found in eval_dir
def _call_global_steps(eval_dir, dispatch_func, uncertainty_method='mc', global_step=None, train_eval=False):
  if global_step == 'all':
    paths = parse_paths(eval_dir, 'all', train_eval, uncertainty_method=uncertainty_method)
    for paths_ in paths:
      dispatch_func(paths_)
  else:
    paths = parse_paths(eval_dir, global_step, train_eval, uncertainty_method=uncertainty_method)
    dispatch_func(paths)

# Only show previously computed results
def show_results(eval_dir, uncertainty_method='mc', global_step=None, train_eval=False):
  def show_global_step_result(paths):
    det_dir, mc_dir, no_mc_dir, gt_file, meta_file = paths
    mc, _ = _get_meta_info(paths)

    agg_path = os.path.join(det_dir, 'aggregated_results.pkl')
    if not os.path.isfile(agg_path):
      raise Exception('No aggregated_results.pkl has been found to print. You need to run the evaluation first.')

    with gzip.open(os.path.join(det_dir, 'aggregated_results.pkl'), 'rb') as f:
      aggregated_results = pickle.load(f)
    _show_results(aggregated_results, mc, list(range(len(mc.CLASS_NAMES))))
  _call_global_steps(eval_dir, show_global_step_result,
                     uncertainty_method=uncertainty_method, global_step=global_step, train_eval=train_eval)

def evaluate_uncertainty(eval_dir, global_step=None, uncertainty_method='mc', train_eval=False,
                         use_scaling=None, aggregation_method='individual', n_processes=cpu_count(), beta_inv=None):
    def evaluate_global_step(paths):
      _evaluate_uncertainty(paths, uncertainty_method=uncertainty_method, use_scaling=use_scaling,
                            aggregation_method=aggregation_method, n_processes=n_processes, beta_inv=beta_inv)
    _call_global_steps(eval_dir, evaluate_global_step,
                       uncertainty_method=uncertainty_method, global_step=global_step, train_eval=train_eval)

def _evaluate_uncertainty(paths, uncertainty_method='mc', use_scaling=None, aggregation_method='individual', n_processes=cpu_count(), beta_inv=None):

    """ Parse paths, load gt & additional information """
    det_dir, mc_dir, no_mc_dir, gt_file, meta_file = paths
    mc, image_idx = _get_meta_info(paths)
    print(det_dir, mc_dir, no_mc_dir, gt_file, meta_file)

    # Obtain GT
    with open(gt_file, 'rb') as f:
        gt = pickle.load(f)
        gt = {int(k): v for k, v in gt.items() if len(v) > 0}

    """ Main loop over images """

    if image_idx is None:
      image_idx = list(gt.keys())
    else:
      image_idx = [int(img_idx) for img_idx in image_idx]
      image_idx = [img_idx for img_idx in gt if img_idx in image_idx]

    if use_scaling is not None and type(use_scaling) != list:
      use_scaling = [use_scaling, use_scaling]

    arg_list =  [(img_idx, gt[img_idx], no_mc_dir, mc_dir, mc, uncertainty_method, use_scaling, aggregation_method, beta_inv)
                 for img_idx in image_idx]

    # Standard loop for debugging
    #for arg in arg_list:
    #  process_image(arg)

    pool = ProcessPool(processes=n_processes)
    img_res = pool.map(process_image, arg_list)
    pool.close()
    pool.join()
    results = {res['img_idx']: res for res in img_res}

    n_det_filtered = np.count_nonzero([results[img_idx]['error'][0] for img_idx in results])
    n_singular = np.sum([results[img_idx]['error'][1] for img_idx in results])
    n_nan = np.sum([results[img_idx]['error'][2] for img_idx in results])

    """ Aggregate uncertainty / performance metrics """

    def _aggregate_metric(ident, use_slice=None, reduce_func=np.mean):
        if use_slice is None:
            all_values = [results[img_idx][ident] for img_idx in results if results[img_idx][ident] is not None]
        else:
            all_values = [results[img_idx][ident][use_slice]
                          for img_idx in results if results[img_idx][ident] is not None]

        if len(all_values) > 0:
            if np.all([np.isscalar(val) for val in all_values]):
                return reduce_func(all_values)
            return reduce_func(np.concatenate(all_values))
        return None

    # Aggregate results
    norm_rand = np.random.randn(100000)
    n_dets = _aggregate_metric('n_net_proposals', reduce_func=np.sum)
    n_objs = _aggregate_metric('n_gt', reduce_func=np.sum)
    n_net_matched = _aggregate_metric('n_net_matched', reduce_func=np.sum)
    n_correct = _aggregate_metric('n_net_matched_cls_correct', reduce_func=np.sum)
    etls = _aggregate_metric('pred_residual_normed', reduce_func=lambda a: etl(a, [0.95, 0.99]))
    aggregated_results = {'rmse': _aggregate_metric('pred_residual', reduce_func=lambda x: np.sqrt(np.mean(x**2))),
                          'mean_std_x': _aggregate_metric('std', use_slice=(slice(None), 0)),
                          'mean_std_y': _aggregate_metric('std', use_slice=(slice(None), 1)),
                          'mean_std_w': _aggregate_metric('std', use_slice=(slice(None), 2)),
                          'mean_std_h': _aggregate_metric('std', use_slice=(slice(None), 3)),
                          'mean_nll': _aggregate_metric('nll'),
                          'mean_nll_x': _aggregate_metric('nll_x'),
                          'mean_nll_y': _aggregate_metric('nll_y'),
                          'mean_nll_w': _aggregate_metric('nll_w'),
                          'mean_nll_h': _aggregate_metric('nll_h'),
                          'median_nll_x': _aggregate_metric('nll_x', reduce_func=np.median),
                          'median_nll_y': _aggregate_metric('nll_y', reduce_func=np.median),
                          'median_nll_w': _aggregate_metric('nll_w', reduce_func=np.median),
                          'median_nll_h': _aggregate_metric('nll_h', reduce_func=np.median),
                          'std_nll_x': _aggregate_metric('nll_x', reduce_func=np.std),
                          'std_nll_y': _aggregate_metric('nll_y', reduce_func=np.std),
                          'std_nll_w': _aggregate_metric('nll_w', reduce_func=np.std),
                          'std_nll_h': _aggregate_metric('nll_h', reduce_func=np.std),
                          'mean_mahalanobis': _aggregate_metric('pred_mahalanobis'),
                          'ws_dist': _aggregate_metric('pred_mahalanobis',
                                                       reduce_func=lambda x: wasserstein_distance(x, norm_rand)),
                          'ws_dist_x': _aggregate_metric('pred_residual_normed', use_slice=(slice(None), 0),
                                                         reduce_func=lambda x: wasserstein_distance(x, norm_rand)),
                          'ws_dist_y': _aggregate_metric('pred_residual_normed', use_slice=(slice(None), 1),
                                                         reduce_func=lambda x: wasserstein_distance(x, norm_rand)),
                          'ws_dist_w': _aggregate_metric('pred_residual_normed', use_slice=(slice(None), 2),
                                                         reduce_func=lambda x: wasserstein_distance(x, norm_rand)),
                          'ws_dist_h': _aggregate_metric('pred_residual_normed', use_slice=(slice(None), 3),
                                                         reduce_func=lambda x: wasserstein_distance(x, norm_rand)),
                          'ece_x': _aggregate_metric('pred_error_quantile', use_slice=(slice(None), 0),
                                                     reduce_func=ece),
                          'ece_y': _aggregate_metric('pred_error_quantile', use_slice=(slice(None), 1),
                                                     reduce_func=ece),
                          'ece_w': _aggregate_metric('pred_error_quantile', use_slice=(slice(None), 2),
                                                     reduce_func=ece),
                          'ece_h': _aggregate_metric('pred_error_quantile', use_slice=(slice(None), 3),
                                                     reduce_func=ece),
                          'mean_ious_nomc': _aggregate_metric('ious_nomc'),
                          'std_ious_nomc': _aggregate_metric('ious_nomc', reduce_func=np.std),
                          'n_net_proposals': n_dets,
                          'n_net_matched': n_net_matched,
                          'n_net_matched_cls_correct': n_correct,
                          'n_gt': n_objs,
                          'correct_class_perc': n_correct/float(n_net_matched),
                          'perc_cls_error': _aggregate_metric('n_net_matched_cls_error', reduce_func=np.sum) / float(n_net_matched),
                          'perc_bg_error': _aggregate_metric('n_bg_error', reduce_func=np.sum) / float(n_dets),
                          'recall': n_correct / float(n_objs),
                          'etl_99_x': etls[1, 0],
                          'etl_99_y': etls[1, 1],
                          'etl_99_w': etls[1, 2],
                          'etl_99_h': etls[1, 3],
                          'etl_95_x': etls[0, 0],
                          'etl_95_y': etls[0, 1],
                          'etl_95_w': etls[0, 2],
                          'etl_95_h': etls[0, 3],
                         }
    if uncertainty_method in ['mc']:
        aggregated_results.update({
          'mean_ious_mc': _aggregate_metric('ious_mc'),
          'std_ious_mc': _aggregate_metric('ious_mc', reduce_func=np.std),
          'mean_ious_mean_mc': _aggregate_metric('ious_mean_mc'),
          'std_ious_mean_mc': _aggregate_metric('ious_mean_mc', reduce_func=np.std)
        })

    """ Average precision """

    # Merge dictionaries
    ap_dets = dict()
    n_gt_cls = dict()
    cls_idxs = set()
    for img_idx in results:
      for cls in results[img_idx]['n_gt_cls']:
        if cls not in n_gt_cls:
          n_gt_cls[cls] = 0
        n_gt_cls[cls] += results[img_idx]['n_gt_cls'][cls]

      for iou_thresh in results[img_idx]['dets']:
        if iou_thresh not in ap_dets:
          ap_dets[iou_thresh] = dict()

        for cls in results[img_idx]['dets'][iou_thresh]:
          cls_idxs.add(cls)
          if cls not in ap_dets[iou_thresh]:
             ap_dets[iou_thresh][cls] = []
          ap_dets[iou_thresh][cls] += results[img_idx]['dets'][iou_thresh][cls]

    # Iterate over classes, accumulate true positives / false positives
    aps = {k: dict() for k in ap_dets}
    for iou_thresh in ap_dets:
      for cls in ap_dets[iou_thresh]:

        # Sort boxes by confidence score, get threshold decision
        #det_decisions = np.array([over_iou_thresh for _, over_iou_thresh in sorted(ap_dets[iou_thresh][cls],
        #                                                                           key=lambda x: x[0],
        #                                                                           reverse=True)],
        #                         dtype=np.bool)
        confs = np.array([item[0] for item in ap_dets[iou_thresh][cls]])
        confs_order = np.argsort(confs)[::-1]
        det_decisions = np.array([item[1] for item in ap_dets[iou_thresh][cls]], dtype=np.bool)[confs_order]

        cum_tp = np.cumsum(det_decisions, dtype=np.float)
        cum_fp = np.cumsum(~det_decisions, dtype=np.float)

        cum_precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
        cum_recall = cum_tp / n_gt_cls[cls]

        cum_recall_ = np.concatenate(([0.], cum_recall, [1.]))
        cum_precision_ = np.concatenate(([0.], cum_precision, [0.]))

        # compute the precision envelope
        cum_precision_ = np.maximum.accumulate(cum_precision_[::-1])[::-1]

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        interp_idx = np.where(cum_recall_[1:] != cum_recall_[:-1])[0]
        area_under_interp_pr_curve = np.sum((cum_recall_[interp_idx + 1] - cum_recall_[interp_idx]) * cum_precision_[interp_idx + 1])
        aps[iou_thresh][cls] = area_under_interp_pr_curve

    aggregated_results['mAP (@iou=0.5)'] = np.mean(list(aps[0.5].values()))
    aggregated_results['mAP (@iou=0.75)'] = np.mean(list(aps[0.75].values()))
    aggregated_results['mAP (@iou=0.5:0.95)'] = np.mean([np.mean([aps[it/100.][cls] for it in range(50,100,5)])
                                                    for cls in aps[0.5]])
    aggregated_results.update({'AP_{:s} (@iou=0.5)'.format(mc.CLASS_NAMES[cls]): aps[0.5][cls] for cls in aps[0.5]})
    aggregated_results.update({'AP_{:s} (@iou=0.75)'.format(mc.CLASS_NAMES[cls]): aps[0.75][cls] for cls in aps[0.75]})
    aggregated_results.update({'AP_{:s} (@iou=0.5:0.95)'.format(mc.CLASS_NAMES[cls]): np.mean([aps[it/100.][cls]
                                                                                          for it in range(50,100,5)])
                               for cls in aps[0.5]})

    # output
    _show_results(aggregated_results, mc, cls_idxs)

    with gzip.open(os.path.join(det_dir, 'full_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    with gzip.open(os.path.join(det_dir, 'aggregated_results.pkl'), 'wb') as f:
        pickle.dump(aggregated_results, f)

def _pprint_dict(d, sections, suffix_dict=None):

  def _key_str(key):
    return str(key) + ( (' ' + str(suffix_dict[key])) if suffix_dict is not None and key in suffix_dict else '')

  max_key_len = np.max([len(_key_str(key)) for key in d])
  first = True
  for section_header, section_keys in sections:
    if not first:
      print("")

    print(section_header)
    for key in section_keys:
      print(("%s:" % _key_str(key)).ljust(max_key_len) + "\t" + str(d[key]))

    if first:
      first = False

def _show_results(aggregated_results, mc, cls_idxs):
  suffix_dict = {'recall': '(d)', 'rmse': '(d)', 'mean_ious_mean_mc': '(u)',
                 'mAP (@iou=0.5:0.95)': '(u)', 'mAP (@iou=0.5)': '(u)', 'mAP (@iou=0.75)': '(u)',
                 'mean_nll_x': '(d)', 'mean_nll_y': '(d)', 'mean_nll_w': '(d)', 'mean_nll_h': '(d)',
                 'ece_x': '(d)', 'ece_y': '(d)', 'ece_w': '(d)', 'ece_h': '(d)',
                 'ws_dist_x': '(d)', 'ws_dist_y': '(d)', 'ws_dist_w': '(d)', 'ws_dist_h': '(d)',
                 'mean_ious_nomc': '(u)', 'correct_class_perc': '(u)'}
  suffix_dict.update({'AP_{:s} (@iou=0.5:0.95)'.format(mc.CLASS_NAMES[cls]): '(d)' for cls in cls_idxs})
  suffix_dict.update({'AP_{:s} (@iou=0.5)'.format(mc.CLASS_NAMES[cls]): '(d)' for cls in cls_idxs})
  suffix_dict.update({'AP_{:s} (@iou=0.75)'.format(mc.CLASS_NAMES[cls]): '(d)' for cls in cls_idxs})
  _pprint_dict(aggregated_results, [
    ('matching statistics', ['n_gt', 'n_net_proposals', 'n_net_matched', 'n_net_matched_cls_correct', 'recall']),
    ('performance (without APs)', ['rmse', 'mean_ious_mean_mc']),
    ('performance (APs)', ['mAP (@iou=0.5:0.95)', 'mAP (@iou=0.5)', 'mAP (@iou=0.75)']),
    ('uncertainty', ['mean_std_x', 'mean_std_y', 'mean_std_w', 'mean_std_h',
                     'mean_nll_x', 'mean_nll_y', 'mean_nll_w', 'mean_nll_h',
                     'median_nll_x', 'median_nll_y', 'median_nll_w', 'median_nll_h',
                     'ece_x', 'ece_y', 'ece_w', 'ece_h', 'ws_dist_x', 'ws_dist_y', 'ws_dist_w', 'ws_dist_h',
                     'etl_95_x', 'etl_95_y', 'etl_95_w', 'etl_95_h',
                     'etl_99_x', 'etl_99_y', 'etl_99_w', 'etl_99_h']),
    ('verbose/other', ['AP_{:s} (@iou=0.5:0.95)'.format(mc.CLASS_NAMES[cls]) for cls in cls_idxs] +
     ['AP_{:s} (@iou=0.5)'.format(mc.CLASS_NAMES[cls]) for cls in cls_idxs] +
     ['AP_{:s} (@iou=0.75)'.format(mc.CLASS_NAMES[cls]) for cls in cls_idxs] +
     ['std_nll_x', 'std_nll_y', 'std_nll_w', 'std_nll_h', 'std_ious_mean_mc',
      'mean_ious_nomc', 'std_ious_nomc', 'correct_class_perc'])
  ], suffix_dict)

if __name__ == '__main__':

    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', help='Absolute path to directory that contains the /train or /val directory with *.mc, *.nomc and gt.pkl files')
    parser.add_argument('--global_step', default=None, help='"None" (Default): Use latest global step available; "All": Evaluate on all available global steps; Can also be an integer')
    parser.add_argument('--uncertainty_method', default="mc", help='mc')
    parser.add_argument('--train_eval', default="False", help='True/False if evaluation should be done on training data')
    parser.add_argument('--show_results_only', default='False', help='True/False if only want to show previously evaluated results')
    parser.add_argument('--use_scaling', default=None, help='Either None, scale (float) or [x_scale,y_scale] (list of float); Scales all boxes by these factors')
    parser.add_argument('--aggregation_method', default='kmeans', help='Method used for aggregating MC-samples. Either "individual" (individually maps sample box to gt via bipartite matching) '
                                                                           'or "anchor" (anchor based aggregation; map mean box to gt via bipartite matching)'
                                                                           'or "kmeans" (group boxes together based on a k-means clustering; mean box is matched to ground truth via bipartite matching))')
    parser.add_argument('--beta_inv', default=None, help='MC Dropouts constant aleatoric term')
    args = parser.parse_args()

    # Parse params
    eval_dir = args.eval_dir.strip()

    global_step = None
    if args.global_step is not None:
      if args.global_step.strip().lower() == "all":
        global_step = 'all'
      elif args.global_step.strip().lower() != "none":
        global_step = int(args.global_step.strip())

    uncertainty_method = args.uncertainty_method.strip().lower()
    train_eval = args.train_eval.strip().lower() == 'true'
    show_results_only = args.show_results_only.strip().lower() == 'true'

    if uncertainty_method not in ['mc']:
      raise Exception("Given uncertainty_method is not supported. Has to be 'mc'")

    if show_results_only:
      show_results(eval_dir, uncertainty_method=uncertainty_method, global_step=global_step, train_eval=train_eval)
    else:
      use_scaling = None
      if args.use_scaling is not None:
        use_scaling_str = args.use_scaling.strip()
        try:
          use_scaling = float(use_scaling_str)
        except ValueError:
          pat = re.compile(r'\[([0-9.]+),([0-9.]+)\]')
          mat = pat.match(use_scaling_str)
          if mat is not None:
            grps = mat.groups()
            if grps[0] is not None and grps[1] is not None:
              use_scaling = [float(grps[0]), float(grps[1])]

          if use_scaling is None:
            raise Exception("Given scaling argument is not of correct form")

      if uncertainty_method in ['mc']:
        aggregation_method = args.aggregation_method.strip()
        if aggregation_method not in ['individual', 'anchor', 'kmeans', 'hdbscan']:
          raise Exception("Argument 'matching_type' has to be either 'bip' or 'anchor' or 'kmeans' or 'hdbscan'")

      evaluate_uncertainty(eval_dir, global_step=global_step, uncertainty_method=uncertainty_method,
                           train_eval=train_eval, use_scaling=use_scaling, aggregation_method=aggregation_method,
                           beta_inv=args.beta_inv)

