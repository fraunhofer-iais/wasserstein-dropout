# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016
# Modified by Sicking et al. for incorporation of the method "Wasserstein dropout"

"""Evaluation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import *
from dataset import kitti, bdd, nightowls, synscapes, a2d2, nuimages, kitti_3cls, bdd_3cls, nightowls_3cls, synscapes_3cls, a2d2_3cls, nuimages_3cls
from utils.util import bbox_transform, Timer, parse_list_of_type, parse_bool, parse_params
from nets import *

from multiprocessing import Manager, cpu_count
from multiprocessing.pool import Pool
from functools import wraps

import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently support kitti, bdd, nightowls, synscapes, a2d2 or nuimages  dataset (and their 3-class variants.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'test',
                           """Only used for VOC data."""
                           """Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for VOC data""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/bichen/logs/squeezeDet/eval',
                            """Directory where to write event logs """)
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/bichen/logs/squeezeDet/train',
                            """Path to the training checkpoint.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                             """How often to check if new cpt is saved.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                             """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                           """Neural net architecture.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

tf.app.flags.DEFINE_string('show_stats_only', 'False', """Whether or not to only show stats from an already computed eval""")
tf.app.flags.DEFINE_string('mc_keep_probs', '[1.0,0.5]', """First value: Other layer drop chance. 
                                                            Second value: Last layer drop chance.
                                                            Only has an effect if mc.MC_DROP_SAMPLES >= 1""")
tf.app.flags.DEFINE_string('uncertainty_method', 'mc', """The uncertainty method to use: mc""")
tf.app.flags.DEFINE_string('pred_filtering_method', 'nms', """The method to use for prediction filtering. """
                                                           """One out of [nms_orig, nms]""")
tf.app.flags.DEFINE_string('global_step', None, """The global step of the checkpoint to use (None uses latest) """ )
tf.app.flags.DEFINE_float('sample_frac', -1., """If this parameter is between 0 and 1, a sample of the image set of size sample_frac * len(images) will be used""")

def eval_once(
    saver, ckpt_path, summary_writer, eval_summary_ops, eval_summary_phs, imdb,
    model):

  config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Session(config=config) as sess:

    # Restores from checkpoint
    saver.restore(sess, ckpt_path)
    # Assuming model_checkpoint_path looks something like:
    #   /ckpt_dir/model.ckpt-0,
    # extract global_step from it.
    global_step = ckpt_path.split('/')[-1].split('-')[-1]

    num_images = len(imdb.image_idx) # Returns image_idx or subsample of image_idx if flag is set

    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    if (model.mc.UNCERTAINTY_METHOD == 'mc') and (model.mc.MC_DROP_SAMPLES > 0):
        all_boxes_mc_samples = [[[[] for _ in xrange(num_images)]
                                     for _ in xrange(imdb.num_classes)]
                                     for _ in xrange(model.mc.MC_DROP_SAMPLES)]

    _t = {'im_detect': Timer(), 'im_read': Timer(), 'misc': Timer(), 'mc_drop':
         Timer()}

    """
    Performs nms
    batch_anchor_idxs: If not None, does not perform nms from scratch. The predictions/scores of are chosen from the anchors specified by batch_anchor_idxs (shape: N_BATCH x FILTERED_ANCHORS) 
    additional_scores: Either None (No additional score is appended to the result) or of shape BATCH_SIZE x ANCHORS x n_scores
    """
    def _filter_boxes(all_boxes, det_boxes, det_probs, det_class, det_all_probs, scales,
                      batch_anchor_idxs=None,
                      additional_scores=None):
      num_detection = 0
      if batch_anchor_idxs is None:
        batch_anchor_idxs_ = []
      else:
        batch_anchor_idxs_ = batch_anchor_idxs

      for j in range(len(det_boxes)): # batch
        # rescale
        det_boxes[j, :, 0::2] /= scales[j][0]
        det_boxes[j, :, 1::2] /= scales[j][1]

        if batch_anchor_idxs is None:
          # nms, thresholding ...
          # Result shape: [FILTERED_ANCHORS, 4] / [FILTERED_ANCHORS]
          # Results are ordered by class

          if additional_scores is not None:
            det_bbox, score, det_class, anchor_idxs, all_scores, add_scores = model.filter_prediction(
              det_boxes[j], det_probs[j], det_class[j],
              return_anchor_idxs=True,
              det_all_probs=det_all_probs[j],
              additional_scores=additional_scores[j])
          else:
            det_bbox, score, det_class, anchor_idxs, all_scores = model.filter_prediction(
              det_boxes[j], det_probs[j], det_class[j],
              det_all_probs=det_all_probs[j],
              return_anchor_idxs=True)
          batch_anchor_idxs_.append(anchor_idxs)
        else:
          anchor_idxs = batch_anchor_idxs[j]
          det_bbox, score, det_class, all_scores = det_boxes[j][anchor_idxs, :], det_probs[j][anchor_idxs], \
                                                      det_class[j][anchor_idxs], det_all_probs[j][anchor_idxs]

          if additional_scores is not None:
            add_scores = additional_scores[j, anchor_idxs, :]

        # Add filtered predictions to all_boxes
        # "Shape" of all_boxes: [CLASSES, BATCHSIZE, FILTERED_ANCHORS, 5]
        # where 5 is: xmin, ymin, xmax, ymax + probability score and FILTERED_ANCHORS is variable
        # last dimension is 9 if with_anchor_idxs=True (last 4 are anchor_x,y,w,h)
        num_detection += len(det_bbox)

        if additional_scores is not None:
          for c, b, s, a, all_s, add_score in zip(det_class, det_bbox, score, anchor_idxs, all_scores, add_scores):
            ax, ay, aw, ah = model.mc.ANCHOR_BOX[a]
            all_boxes[c][i].append(bbox_transform(b) + [s, ax, ay, aw, ah] + all_s.tolist() + [score for score in add_score])
        else:
          for c, b, s, a, all_s in zip(det_class, det_bbox, score, anchor_idxs, all_scores):
            ax, ay, aw, ah = model.mc.ANCHOR_BOX[a]
            all_boxes[c][i].append(bbox_transform(b) + [s, ax, ay, aw, ah] + all_s.tolist())
        return num_detection, batch_anchor_idxs_
 
    num_detection = 0.0
    for i in xrange(num_images):
      _t['im_read'].tic()
      images, scales = imdb.read_image_batch(shuffle=False)
      _t['im_read'].toc()

      _t['im_detect'].tic()
      # Getting detections without dropout
      # det_boxes shape: [BATCHSIZE, ANCHORS, 4] (4 = x,y,w,h)
      # det_probs shape: [BATCHSIZE, ANCHORS] (chosen class probability)
      # det_class shape: [BATCHSIZE, ANCHORS] (chosen class idx)
      det_boxes, det_probs, det_class, det_all_probs = sess.run(
      [model.det_boxes, model.det_probs, model.det_class, model.det_all_probs],
          feed_dict={model.image_input:images})
      _t['im_detect'].toc()

      _t['misc'].tic()
      n_det, nomc_batch_anchors = _filter_boxes(all_boxes, det_boxes, det_probs,
                                     det_class, det_all_probs, scales)
      num_detection += n_det
      _t['misc'].toc()

      # Getting MC dropout detections
      _t['mc_drop'].tic()
      if (model.mc.UNCERTAINTY_METHOD == 'mc') and (model.mc.MC_DROP_SAMPLES > 0):
        det_boxes_mc_samples, det_probs_mc_samples, det_class_mc_samples, det_all_probs_mc_samples = sess.run(
          [model.det_boxes_mc_samples, model.det_probs_mc_samples,
           model.det_class_mc_samples, model.det_all_probs_mc_samples],
          feed_dict={model.image_input:images})
        assert len(det_boxes_mc_samples) == len(det_probs_mc_samples)
        assert len(det_boxes_mc_samples) == len(det_class_mc_samples)
        assert len(det_boxes_mc_samples) == len(det_all_probs_mc_samples)
        assert len(det_boxes_mc_samples) == model.mc.MC_DROP_SAMPLES

        for k in range(model.mc.MC_DROP_SAMPLES):
          det_boxes = det_boxes_mc_samples[k]
          det_probs = det_probs_mc_samples[k]
          det_class = det_class_mc_samples[k]
          det_all_probs = det_all_probs_mc_samples[k]

          _filter_boxes(all_boxes_mc_samples[k], det_boxes, det_probs,
                        det_class, det_all_probs, scales)
      _t['mc_drop'].toc()

      print ('im_detect: {:d}/{:d} im_read: {:.3f}s '
             'detect: {:.3f}s misc: {:.3f}s mc: {:.3f}s'.format(
                i+1, num_images, _t['im_read'].average_time,
                _t['im_detect'].average_time, _t['misc'].average_time,
             _t['mc_drop'].average_time))

    print("Storing No MC predictions")
    det_file_dir = os.path.join(
      FLAGS.eval_dir, 'detection_files_{:s}'.format(global_step))
    if not os.path.isdir(det_file_dir):
      os.makedirs(det_file_dir)

    # Saving GT
    imdb.save_gtrois_at(os.path.join(det_file_dir, 'gt.pkl'))

    # Writing nomc predictions
    nomc_file_dir = os.path.join(det_file_dir, 'nomc')
    if not os.path.isdir(nomc_file_dir):
      os.makedirs(nomc_file_dir)
    for i, idx in enumerate(imdb.image_idx):
      with open(os.path.join(nomc_file_dir, '{:d}.nomc'.format(int(idx))), 'w') as f:
        for c in xrange(imdb.num_classes):
          for vals in all_boxes[c][i]:
            cx, cy, w, h, s, ax, ay, aw, ah = vals[:9]
            probs = vals[9:]
            format_str = '{:.5f} {:.5f} 0 {:.5f} {:.5f} {:.5f} {:.5f} {:d} {:.5f} {:.5f} {:.5f}'
            format_str += ''.join([' {:.5f}' for _ in range(imdb.num_classes)])
            format_str += '\n'
            f.write(format_str.format(ax, ay, cx, cy, w, h, c, s, aw, ah, *probs))

    # Writing mc samples to textfile
    if (model.mc.UNCERTAINTY_METHOD == 'mc') and (model.mc.MC_DROP_SAMPLES > 0):
      print('Storing MC Dropout samples')
      mc_file_dir = os.path.join(det_file_dir, 'mcsamples')
      if not os.path.isdir(mc_file_dir):
        os.makedirs(mc_file_dir)

      for i, idx in enumerate(imdb.image_idx):
        with open(os.path.join(mc_file_dir, '{:d}.mc'.format(int(idx))), 'w') as f:
          for k in xrange(model.mc.MC_DROP_SAMPLES):
            for c in xrange(imdb.num_classes):
              for vals in all_boxes_mc_samples[k][c][i]:
                  cx, cy, w, h, s, ax, ay, aw, ah = vals[:9]
                  probs = vals[9:]
                  format_str = '{:.5f} {:.5f} {:d} {:.5f} {:.5f} {:.5f} {:.5f} {:d} {:.5f} {:.5f} {:.5f}'
                  format_str += ''.join([' {:.5f}' for _ in range(imdb.num_classes)])
                  format_str += '\n'
                  f.write(format_str.format(ax, ay, k, cx, cy, w, h, c, s, aw, ah, *probs))


    if FLAGS.dataset.strip().upper() == 'KITTI':
      # Computes APs
      print ('Evaluating detections...')
      aps, ap_names = imdb.evaluate_detections(
          FLAGS.eval_dir, global_step, all_boxes)

      with open(os.path.join(det_file_dir, 'summary_ap.pkl'), 'wb') as f:
        pickle.dump((aps, ap_names), f)

    print ('Evaluation summary:')
    print ('  Average number of detections per image: {}:'.format(
      num_detection/num_images))
    print ('  Timing:')
    print ('    im_read: {:.3f}s detect: {:.3f}s misc: {:.3f}s'.format(
      _t['im_read'].average_time, _t['im_detect'].average_time,
      _t['misc'].average_time))

    feed_dict = {}  # Make stats available in tensorboard, aggregate them
    if FLAGS.dataset.strip().upper() == 'KITTI':
      print('  Average precisions:')
      for cls, ap in zip(ap_names, aps):
        feed_dict[eval_summary_phs['APs/'+cls]] = ap
        print ('    {}: {:.3f}'.format(cls, ap))

      print ('    Mean average precision: {:.3f}'.format(np.mean(aps)))
      feed_dict[eval_summary_phs['APs/mAP']] = np.mean(aps)

    feed_dict[eval_summary_phs['timing/im_detect']] = \
        _t['im_detect'].average_time
    feed_dict[eval_summary_phs['timing/im_read']] = \
        _t['im_read'].average_time
    feed_dict[eval_summary_phs['timing/post_proc']] = \
        _t['misc'].average_time
    feed_dict[eval_summary_phs['num_det_per_image']] = \
        num_detection/num_images

    if FLAGS.dataset.strip().upper() == 'KITTI':
      # Compute percentage of correct, wrong, missing, ... detections
      # Predictions are associated to a ground truth by selecting the GT with
      # largest IOU to the prediction
      print ('Analyzing detections...')
      stats, ims = imdb.do_detection_analysis_in_eval(
          FLAGS.eval_dir, global_step)

      with open(os.path.join(det_file_dir, 'summary.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    eval_summary_str = sess.run(eval_summary_ops, feed_dict=feed_dict)
    for sum_str in eval_summary_str:
      summary_writer.add_summary(sum_str, global_step)

    meta = {'name': imdb.name,
            'classes': imdb.classes,
            'image_idx': imdb.image_idx,
            'image_set': imdb.image_set,
            'use_subsample': imdb.use_subsample,
            'mc': model.mc}
    with open(os.path.join(det_file_dir, 'meta.pkl'), 'wb') as f:
      pickle.dump(meta, f)

def evaluate():
  """Evaluate."""

  mc_keep_probs = parse_list_of_type(FLAGS.mc_keep_probs)
  uncertainty_method = FLAGS.uncertainty_method.strip().lower()
  if uncertainty_method not in ['mc']:
    raise Exception("Given uncertainty method {:s} is not supported. Has to be one of ['mc', 'pu']".format(uncertainty_method))
  pred_filtering_method = FLAGS.pred_filtering_method.strip().lower()
  if pred_filtering_method not in ['nms_orig', 'nms']:
    raise Exception("Given pred filtering method {:s} is not supported. Has to be one of ..".format(pred_filtering_method))

  dataset = FLAGS.dataset.strip().upper()
  assert dataset in ['KITTI', 'BDD', 'NIGHTOWLS', 'SYNSCAPES', 'A2D2', 'NUIMAGES',
                     'KITTI_3CLS', 'BDD_3CLS', 'NIGHTOWLS_3CLS', 'SYNSCAPES_3CLS', 'A2D2_3CLS', 'NUIMAGES_3CLS'], \
      'Currently only supports KITTI, BDD, NIGHTOWLS, SYNSCAPES, A2D2, NUIMAGES dataset and their _3CLS variants'

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  with tf.Graph().as_default() as g:

    assert FLAGS.net == 'squeezeDet', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)
    if FLAGS.net == 'squeezeDet':
      if dataset == 'KITTI':
        mc = kitti_squeezeDet_config() # note that by default mc.IS_TRAINING = False
      elif dataset == 'BDD':
        mc = bdd_squeezeDet_config()
      elif dataset == 'NIGHTOWLS':
        mc = nightowls_squeezeDet_config()
      elif dataset == 'SYNSCAPES':
        mc = synscapes_squeezeDet_config()
      elif dataset == 'A2D2':
        mc = a2d2_squeezeDet_config()
      elif dataset == 'NUIMAGES':
        mc = nuimages_squeezeDet_config()
      elif dataset == 'KITTI_3CLS':
        mc = kitti_3cls_squeezeDet_config()
      elif dataset == 'BDD_3CLS':
        mc = bdd_3cls_squeezeDet_config()
      elif dataset == 'NIGHTOWLS_3CLS':
        mc = nightowls_3cls_squeezeDet_config()
      elif dataset == 'SYNSCAPES_3CLS':
        mc = synscapes_3cls_squeezeDet_config()
      elif dataset == 'A2D2_3CLS':
        mc = a2d2_3cls_squeezeDet_config()
      elif dataset == 'NUIMAGES_3CLS':
        mc = nuimages_3cls_squeezeDet_config()
        
      mc.BATCH_SIZE = 1 # TODO(bichen): allow batch size > 1
      mc.LOAD_PRETRAINED_MODEL = False
      mc.KEEP_PROB_FULL_DROP = mc_keep_probs[0]
      mc.KEEP_PROB = mc_keep_probs[1]
      mc.UNCERTAINTY_METHOD = uncertainty_method
      mc.PREDICTION_FILTERING_METHOD = pred_filtering_method
      model = SqueezeDet(mc)

    print("Using the data path: ", FLAGS.data_path)
    print("Using the checkpoint path: ", FLAGS.checkpoint_path)
    if dataset == 'KITTI':
      imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)
    elif dataset == 'BDD':
      imdb = bdd.BDD(FLAGS.image_set, FLAGS.data_path, mc)
    elif dataset == 'NIGHTOWLS':
      imdb = nightowls.NIGHTOWLS(FLAGS.image_set, FLAGS.data_path, mc)
    elif dataset == 'SYNSCAPES':
      imdb = synscapes.SYNSCAPES(FLAGS.image_set, FLAGS.data_path, mc)
    elif dataset == 'A2D2':
      imdb = a2d2.A2D2(FLAGS.image_set, FLAGS.data_path, mc)
    elif dataset == 'NUIMAGES':
      imdb = nuimages.NUIMAGES(FLAGS.image_set, FLAGS.data_path, mc)
    elif dataset == 'KITTI_3CLS':
      imdb = kitti_3cls.KITTI_3CLS(FLAGS.image_set, FLAGS.data_path, mc)
    elif dataset == 'BDD_3CLS':
      imdb = bdd_3cls.BDD_3CLS(FLAGS.image_set, FLAGS.data_path, mc)
    elif dataset == 'NIGHTOWLS_3CLS':
      imdb = nightowls_3cls.NIGHTOWLS_3CLS(FLAGS.image_set, FLAGS.data_path, mc)
    elif dataset == 'SYNSCAPES_3CLS':
      imdb = synscapes_3cls.SYNSCAPES_3CLS(FLAGS.image_set, FLAGS.data_path, mc)
    elif dataset == 'A2D2_3CLS':
      imdb = a2d2_3cls.A2D2_3CLS(FLAGS.image_set, FLAGS.data_path, mc)
    elif dataset == 'NUIMAGES_3CLS':
      imdb = nuimages_3cls.NUIMAGES_3CLS(FLAGS.image_set, FLAGS.data_path, mc)

    # Use subsample of all data if flag is set properly
    if 0. < FLAGS.sample_frac <= 1.:
      imdb._subsample_image_set(FLAGS.sample_frac)
      imdb.use_subsample = True

    # add summary ops and placeholders
    eval_summary_ops = []
    eval_summary_phs = {}
    if dataset == 'KITTI':
      ap_names = []
      for cls in imdb.classes:
        ap_names.append(cls+'_easy')
        ap_names.append(cls+'_medium')
        ap_names.append(cls+'_hard')

      for ap_name in ap_names:
        ph = tf.placeholder(tf.float32)
        eval_summary_phs['APs/'+ap_name] = ph
        eval_summary_ops.append(tf.summary.scalar('APs/'+ap_name, ph))

      ph = tf.placeholder(tf.float32)
      eval_summary_phs['APs/mAP'] = ph
      eval_summary_ops.append(tf.summary.scalar('APs/mAP', ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['timing/im_detect'] = ph
    eval_summary_ops.append(tf.summary.scalar('timing/im_detect', ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['timing/im_read'] = ph
    eval_summary_ops.append(tf.summary.scalar('timing/im_read', ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['timing/post_proc'] = ph
    eval_summary_ops.append(tf.summary.scalar('timing/post_proc', ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['num_det_per_image'] = ph
    eval_summary_ops.append(tf.summary.scalar('num_det_per_image', ph))

    saver = tf.train.Saver(model.model_params)

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    
    ckpts = set()

    global_step = None
    if FLAGS.run_once: # global_step can only be specified if run_once == True
      global_step = parse_params(FLAGS.global_step, list_cast=int)

    if global_step is None:
      global_step = {-1}
    elif type(global_step) == int:
      global_step = {global_step}
    global_step = set(global_step) # set of all global steps that are about to be evaluated
    cur_global_step = -1

    while True:

      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
      model_checkpoint_path = None

      if ckpt:
        if -1 in global_step: # use latest checkpoint if global step == -1
          model_checkpoint_path = ckpt.model_checkpoint_path
        elif ckpt.all_model_checkpoint_paths: # only possible if run_once == True
          all_available_global_steps = {int(ckpt_path.split('/')[-1].split('-')[-1]): ckpt_path
                                        for ckpt_path in ckpt.all_model_checkpoint_paths}
          intersection_global_steps = global_step.intersection(set(all_available_global_steps.keys()))
          if len(intersection_global_steps) < 1:
            print('There does not exist a checkpoint corresponding to (any of) the global step(s) given')
          else:
            cur_global_step = next(iter(intersection_global_steps))
            model_checkpoint_path = all_available_global_steps[cur_global_step]


      if model_checkpoint_path: # None if no checkpoint exist or none with specified global steps

        # There is a problem if the directory the checkpoint resides in was renamed, therefore this check
        if os.path.dirname(model_checkpoint_path) != FLAGS.checkpoint_path:
          model_checkpoint_path = os.path.join(FLAGS.checkpoint_path,
                                                    os.path.basename(model_checkpoint_path))

        if FLAGS.run_once:

          global_step.remove(cur_global_step)
          print('Evaluating {}...'.format(model_checkpoint_path))
          eval_once(
              saver, model_checkpoint_path, summary_writer, eval_summary_ops,
              eval_summary_phs, imdb, model)

          if len(global_step) < 1:
            return

        else:
          if model_checkpoint_path in ckpts:
            # Do not evaluate on the same checkpoint
            print ('Wait {:d}s for new checkpoints to be saved ... '
                      .format(FLAGS.eval_interval_secs))
            time.sleep(FLAGS.eval_interval_secs)
          else:
            ckpts.add(model_checkpoint_path)
            print ('Evaluating {}...'.format(model_checkpoint_path))
            eval_once(
                saver, model_checkpoint_path, summary_writer,
                eval_summary_ops, eval_summary_phs, imdb, model)

      else:
        print('No checkpoint file(s) found')
        if FLAGS.run_once:
          return
        else:
          print ('Wait {:d}s for new checkpoints to be saved ... '
                    .format(FLAGS.eval_interval_secs))
          time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument

  if not tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()

if __name__ == '__main__':
  tf.app.run()
