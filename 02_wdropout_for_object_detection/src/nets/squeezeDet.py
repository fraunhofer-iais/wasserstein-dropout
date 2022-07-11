# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016
# Modified by Sicking et al. for incorporation of the method "Wasserstein dropout"

"""SqueezeDet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton

class SqueezeDet(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    conv1 = self._conv_layer(
        'conv1', self.image_input, filters=64, size=3, stride=2,
        padding='SAME', freeze=True)
    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride=2, padding='SAME')

    fire2 = self._fire_layer(
        'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False, drop_prob=self.keep_prob_full_mc)
    fire3 = self._fire_layer(
        'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False, drop_prob=self.keep_prob_full_mc)
    pool3 = self._pooling_layer(
        'pool3', fire3, size=3, stride=2, padding='SAME')

    fire4 = self._fire_layer(
        'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False, drop_prob=self.keep_prob_full_mc)
    fire5 = self._fire_layer(
        'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False, drop_prob=self.keep_prob_full_mc)
    pool5 = self._pooling_layer(
        'pool5', fire5, size=3, stride=2, padding='SAME')

    fire6 = self._fire_layer(
        'fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=False, drop_prob=self.keep_prob_full_mc)
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False, drop_prob=self.keep_prob_full_mc)
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False, drop_prob=self.keep_prob_full_mc)
    fire9 = self._fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=False, drop_prob=self.keep_prob_full_mc)

    # Two extra fire modules that are not trained before
    fire10 = self._fire_layer(
        'fire10', fire9, s1x1=96, e1x1=384, e3x3=384, freeze=False, drop_prob=self.keep_prob_full_mc)

    fire11 = self._fire_layer(
        'fire11', fire10, s1x1=96, e1x1=384, e3x3=384, freeze=False)

    dropout11 = tf.nn.dropout(fire11, self.keep_prob, name='drop11')

    if self.uncertainty_method == 'pu':
      num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 2*4) # Each bbox coordinate gets an additional variance score
    else:
      num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)

    conv12_shared_input = []
    if mc.IS_TRAINING:
      if (self.uncertainty_method == 'mc') and (mc.N_SAMPLES is not None):
        conv12_shared_input += [fire11]

      if self.uncertainty_method in ['exact_wdrop']: # Take mc.N_SAMPLES many dropout passes
        if mc.N_SAMPLES is None or mc.N_SAMPLES < 2:
          raise Exception('If using uncertainty_method = exact_wdrop, you must pass the number of samples as N_SAMPLES.')
        conv12_shared_input = [tf.nn.dropout(fire11, self.keep_prob, name='drop_11_%d' % i) for i in range(1, mc.N_SAMPLES)]

      # On training, use dropout layer outputs as predictions
      preds_drop, conv12_shared_out = self._conv_layers_shared(
          'conv12', dropout11, conv12_shared_input, filters=num_output, size=3, stride=1,
          padding='SAME', xavier=False, relu=False, stddev=0.0001)

      if (self.uncertainty_method == 'mc') and (mc.N_SAMPLES is not None):
        self.preds_no_drop = conv12_shared_out[0]

      self.mc_drop_samples = [preds_drop]
      if self.uncertainty_method in ['exact_wdrop']:
        self.mc_drop_samples += conv12_shared_out

    else:
      if ((self.uncertainty_method == 'mc') and (mc.N_SAMPLES is not None)) or mc.MC_DROP_SAMPLES > 0:
        conv12_shared_input += [dropout11]
      conv12_shared_input += [tf.nn.dropout(fire11, self.keep_prob, name='drop_11_%d' % i) for i in range(1, mc.MC_DROP_SAMPLES)]

      self.preds_no_drop, conv12_shared_out = self._conv_layers_shared(
         'conv12', fire11, conv12_shared_input, filters=num_output, size=3, stride=1, 
          padding='SAME', xavier=False, relu=False, stddev=0.0001)

      self.mc_drop_samples = conv12_shared_out

  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,
      freeze=False, drop_prob=1.0):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', stddev=stddev, freeze=freeze)

    out = tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')
    drop = tf.nn.dropout(out, drop_prob, name=layer_name+'_dropout')
    return drop
