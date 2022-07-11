# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Utility functions."""

import numpy as np
import time
import tensorflow as tf
import re

def pdm(tensor):
  eps = tf.constant(1e-7)

  with tf.device('/cpu:0'):
    eigvals, eigvecs = tf.linalg.eigh(tensor)

  shape = eigvecs.get_shape().as_list()
  tp_perm = list(range(len(shape)))
  tp_perm[-1] = len(shape) -2
  tp_perm[-2] = len(shape) -1

  eigvals = tf.maximum(eps, eigvals)
  reconstr_pdm = tf.matmul(eigvecs, tf.expand_dims(eigvals, -1) * tf.transpose(eigvecs, tp_perm))
  return reconstr_pdm

def sqrtm(tensor):
  eps = tf.constant(1e-7)

  with tf.device('/cpu:0'):
    eigvals, eigvecs = tf.linalg.eigh(tensor) # input shape ... x N x N; output shapes: [..., N], [..., N, N]

  shape = eigvecs.get_shape().as_list()
  tp_perm = list(range(len(shape)))
  tp_perm[-1] = len(shape) - 2
  tp_perm[-2] = len(shape) - 1

  eigvals_sqrt = tf.sqrt(tf.maximum(eps, eigvals))

  matrix_sqrt = tf.matmul(eigvecs, tf.expand_dims(eigvals_sqrt, -1) * tf.transpose(eigvecs, tp_perm))
  return matrix_sqrt

""" Parses list of elements of type list_cast or a single integer from string """
def parse_params(inp, list_cast=float):
  if inp is not None:
    inp = str(inp).strip()
    if inp.startswith('[') and inp.endswith(']'):
      return [list_cast(num.strip()) for num in inp[1:-1].split(',')]
    try:
      return int(inp)
    except Exception as e:
      pass
  return None

def parse_list_of_type(inp, cast=float):
  if inp is not None:
    inp = str(inp).strip()
    if inp.startswith('[') and inp.endswith(']'):
        return [cast(num.strip()) for num in inp[1:-1].split(',')]
  return None

def parse_bool(inp):
  if inp is not None:
    inp = str(inp)
    if inp.strip().lower() == 'true':
      return True
  return False


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

def nms(boxes, probs, threshold):
  """Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format)
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """

  order = probs.argsort()[::-1]
  keep = [True]*len(order)

  for i in range(len(order)-1):
    ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
    for j, ov in enumerate(ovps):
      if ov > threshold:
        keep[order[j+i+1]] = False
  return keep

def standard_nms_score_update(scores, remaining_boxes_mask, ious, threshold):

  scores[remaining_boxes_mask.nonzero()[0][ious > threshold]] = 0.
  return scores

def generalized_nms(boxes, probs, score_update, score_thresh=0.):
  """
  Generalized nms that uses a score_update function
  :param boxes: array of [cx, cy, w, h] (center format)
  :param probs: array of probabilities
  :param score_update: function that takes probs, remaining_boxes_mask and ious as input
  it sets the score of boxes with high iou to 0, so that they are not considered anymore
  read: https://openaccess.thecvf.com/content_ICCV_2017/papers/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.pdf

  NOTE: using score_update = standard_nms_score_update is almost the same as the squeezedet nms implementation
  Only ALMOST, because of the following scenario:
  Box A filters box B. This function does not consider box B anymore.
  vs the squeezeDet implementation would additionally filter all boxes that B filters
  :return:
  """

  # Maintains bounding box scores that are updated throughout the procedure
  scores = np.array(probs)

  # Masks remaining boxes to consider
  remaining_boxes_mask = np.ones(len(boxes), dtype=np.bool)

  n_boxes = 0
  while n_boxes < len(boxes):

    # Keep box with highest score
    remaining_idxs = remaining_boxes_mask.nonzero()[0]
    best_idx = remaining_idxs[scores[remaining_boxes_mask].argmax()]

    # Do not consider that box after
    remaining_boxes_mask[best_idx] = False

    # compute iou between best box and all other boxes
    ious = batch_iou(boxes[remaining_boxes_mask], boxes[best_idx])

    # update score of boxes according to given score_update func
    scores = score_update(scores, remaining_boxes_mask, ious)

    n_boxes += 1

  # Only keep boxes with sufficient scores
  return scores > score_thresh


# TODO(bichen): this is not equivalent with full NMS. Need to improve it.
def recursive_nms(boxes, probs, threshold, form='center'):
  """Recursive Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format) or [xmin, ymin, xmax, ymax]
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """

  assert form == 'center' or form == 'diagonal', \
      'bounding box format not accepted: {}.'.format(form)

  if form == 'center':
    # convert to diagonal format
    boxes = np.array([bbox_transform(b) for b in boxes])

  areas = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
  hidx = boxes[:, 0].argsort()
  keep = [True]*len(hidx)

  def _nms(hidx):
    order = probs[hidx].argsort()[::-1]

    for idx in range(len(order)):
      if not keep[hidx[order[idx]]]:
        continue
      xx2 = boxes[hidx[order[idx]], 2]
      for jdx in range(idx+1, len(order)):
        if not keep[hidx[order[jdx]]]:
          continue
        xx1 = boxes[hidx[order[jdx]], 0]
        if xx2 < xx1:
          break
        w = xx2 - xx1
        yy1 = max(boxes[hidx[order[idx]], 1], boxes[hidx[order[jdx]], 1])
        yy2 = min(boxes[hidx[order[idx]], 3], boxes[hidx[order[jdx]], 3])
        if yy2 <= yy1:
          continue
        h = yy2-yy1
        inter = w*h
        iou = inter/(areas[hidx[order[idx]]]+areas[hidx[order[jdx]]]-inter)
        if iou > threshold:
          keep[hidx[order[jdx]]] = False

  def _recur(hidx):
    if len(hidx) <= 20:
      _nms(hidx)
    else:
      mid = len(hidx)/2
      _recur(hidx[:mid])
      _recur(hidx[mid:])
      _nms([idx for idx in hidx if keep[idx]])

  _recur(hidx)

  return keep

def sparse_to_dense(sp_indices, output_shape, values, default_value=0):
  """Build a dense matrix from sparse representations.

  Args:
    sp_indices: A [0-2]-D array that contains the index to place values.
    shape: shape of the dense matrix.
    values: A {0,1}-D array where values corresponds to the index in each row of
    sp_indices.
    default_value: values to set for indices not specified in sp_indices.
  Return:
    A dense numpy N-D array with shape output_shape.
  """

  assert len(sp_indices) == len(values), \
      'Length of sp_indices is not equal to length of values'

  array = np.ones(output_shape) * default_value
  for idx, value in zip(sp_indices, values):
    array[tuple(idx)] = value
  return array

def bgr_to_rgb(ims):
  """Convert a list of images from BGR format to RGB format."""
  out = []
  for im in ims:
    out.append(im[:,:,::-1])
  return out

def bbox_transform(bbox):
  """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
  for numpy array or list of tensors.
  """
  with tf.variable_scope('bbox_transform') as scope:
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
  with tf.variable_scope('bbox_transform_inv') as scope:
    xmin, ymin, xmax, ymax = bbox
    out_box = [[]]*4

    width       = xmax - xmin + 1.0
    height      = ymax - ymin + 1.0
    out_box[0]  = xmin + 0.5*width 
    out_box[1]  = ymin + 0.5*height
    out_box[2]  = width
    out_box[3]  = height

  return out_box

class Timer(object):
  def __init__(self):
    self.total_time   = 0.0
    self.calls        = 0
    self.start_time   = 0.0
    self.duration     = 0.0
    self.average_time = 0.0

  def tic(self):
    self.start_time = time.time()

  def toc(self, average=True):
    self.duration = time.time() - self.start_time
    self.total_time += self.duration
    self.calls += 1
    self.average_time = self.total_time/self.calls
    if average:
      return self.average_time
    else:
      return self.duration

def safe_exp(w, thresh):
  """Safe exponential function for tensors."""

  slope = np.exp(thresh)
  with tf.variable_scope('safe_exponential'):
    lin_bool = w > thresh
    lin_region = tf.to_float(lin_bool)

    lin_out = slope*(w - thresh + 1.)
    exp_out = tf.exp(tf.where(lin_bool, tf.zeros_like(w), w))

    out = lin_region*lin_out + (1.-lin_region)*exp_out
  return out


