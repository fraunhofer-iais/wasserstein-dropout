# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016
# Modified by Sicking et al. for incorporation of the method "Wasserstein dropout"

"""Neural network model base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf

from utils.util import batch_iou, sqrtm, pdm

def _add_loss_summaries(total_loss):
  """Add summaries for losses
  Generates loss summaries for visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  """
  losses = tf.get_collection('losses')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    tf.summary.scalar(l.op.name, l)

  for l in tf.get_collection('wdrop_losses'):
      tf.summary.scalar(l.op.name, l)

def _variable_on_device(name, shape, initializer, trainable=True):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # TODO(bichen): fix the hard-coded data type below
  dtype = tf.float32
  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

class ModelSkeleton:
  """Base class of NN detection models."""
  def __init__(self, mc):
    self.mc = mc
    # a scalar tensor in range (0, 1]. Usually set to 0.5 in training phase and
    # 1.0 in evaluation phase
    self.keep_prob = mc.KEEP_PROB if (mc.IS_TRAINING or mc.MC_DROP_SAMPLES >= 1) else 1.0
    self.keep_prob_full_mc = mc.KEEP_PROB_FULL_MC if (mc.IS_TRAINING or mc.MC_DROP_SAMPLES >= 1) else 1.0
    self.uncertainty_method = mc.UNCERTAINTY_METHOD

    # image batch input
    self.ph_image_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
        name='image_input'
    )
    # A tensor where an element is 1 if the corresponding box is "responsible"
    # for detection an object and 0 otherwise.
    self.ph_input_mask = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, 1], name='box_mask')
    # Tensor used to represent bounding box deltas.
    self.ph_box_delta_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, 4], name='box_delta_input')
    # Tensor used to represent bounding box coordinates.
    self.ph_box_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, 4], name='box_input')
    # Tensor used to represent labels
    self.ph_labels = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES], name='labels')

    # IOU between predicted anchors with ground-truth boxes
    self.ious = tf.Variable(
      initial_value=np.zeros((mc.BATCH_SIZE, mc.ANCHORS)), trainable=False,
      name='iou', dtype=tf.float32
    )

    self.FIFOQueue = tf.FIFOQueue(
        capacity=mc.QUEUE_CAPACITY,
        dtypes=[tf.float32, tf.float32, tf.float32, 
                tf.float32, tf.float32],
        shapes=[[mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
                [mc.ANCHORS, 1],
                [mc.ANCHORS, 4],
                [mc.ANCHORS, 4],
                [mc.ANCHORS, mc.CLASSES]],
    )

    self.enqueue_op = self.FIFOQueue.enqueue_many(
        [self.ph_image_input, self.ph_input_mask,
         self.ph_box_delta_input, self.ph_box_input, self.ph_labels]
    )

    self.image_input, self.input_mask, self.box_delta_input, \
        self.box_input, self.labels = tf.train.batch(
            self.FIFOQueue.dequeue(), batch_size=mc.BATCH_SIZE,
            capacity=mc.QUEUE_CAPACITY) 

    #debug
    self.image_input_nan_count = tf.count_nonzero(tf.is_nan(self.image_input))

    # model parameters
    self.model_params = []

    # model size counter
    self.model_size_counter = [] # array of tuple of layer name, parameter size
    # flop counter
    self.flop_counter = [] # array of tuple of layer name, flop number
    # activation counter
    self.activation_counter = [] # array of tuple of layer name, output activations
    self.activation_counter.append(('input', mc.IMAGE_WIDTH*mc.IMAGE_HEIGHT*3))


  def _add_forward_graph(self):
    """NN architecture specification."""
    raise NotImplementedError

  def _add_interpretation_graph(self):
    """Interpret NN output."""
    mc = self.mc

    with tf.variable_scope('interpret_output') as scope:

      # number of object. Used to normalize bbox and classification loss
      self.num_objects = tf.reduce_sum(self.input_mask, name='num_objects')

      if mc.IS_TRAINING:
        preds_list = self.mc_drop_samples # Contains one dropout output / mc.n_samples many in case of exact_wdrop
      else:
        # In inference mode: Do standard evaluation without dropout
        # + additionally last layer dropout samples
        preds_list = [self.preds_no_drop] + self.mc_drop_samples

      self.pred_box_delta_mc_samples, self.pred_class_probs_mc_samples, self.pred_conf_mc_samples = [], [], []
      self.det_boxes_mc_samples, self.det_probs_mc_samples, self.det_class_mc_samples = [], [], []
      self.det_all_probs_mc_samples = []

      # preds has shape: BATCH_SIZE x ANCHOR_W x ANCHOR_H x ANCHOR_PER_GRID*(N_CLASSES + 1 (conf) + 4 (bb coords, in case of pu 2*4))
      for preds_idx, preds in enumerate(preds_list):

        suffix = '_%d' % preds_idx if preds_idx > 0 else ''

        # probability
        num_class_probs = mc.ANCHOR_PER_GRID*mc.CLASSES
        pred_class_probs = tf.reshape(
            tf.nn.softmax(
                tf.reshape(
                    preds[:, :, :, :num_class_probs],
                    [-1, mc.CLASSES]
                )
            ),
            [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
            name='pred_class_probs' + suffix
        )
        if preds_idx == 0: # this idx stands for the output without dropout in case of inference
          self.pred_class_probs = pred_class_probs
        else:
          self.pred_class_probs_mc_samples.append(pred_class_probs)

        # confidence
        num_confidence_scores = mc.ANCHOR_PER_GRID+num_class_probs
        pred_conf = tf.sigmoid(
            tf.reshape(
                preds[:, :, :, num_class_probs:num_confidence_scores],
                [mc.BATCH_SIZE, mc.ANCHORS]
            ),
            name='pred_confidence_score' + suffix
        )
        if preds_idx == 0:
          self.pred_conf = pred_conf
        else:
          self.pred_conf_mc_samples.append(pred_conf)

        # bbox_delta
        num_bboxes = num_confidence_scores+mc.ANCHOR_PER_GRID*4
        pred_box_delta = tf.reshape(
            preds[:, :, :, num_confidence_scores:num_bboxes],
            [mc.BATCH_SIZE, mc.ANCHORS, 4],
            name='bbox_delta' + suffix
        )
        if preds_idx == 0:
          self.pred_box_delta = pred_box_delta
        else:
          self.pred_box_delta_mc_samples.append(pred_box_delta)

        with tf.variable_scope('bbox') as scope:
          with tf.variable_scope('stretching'):
            delta_x, delta_y, delta_w, delta_h = tf.unstack(
                pred_box_delta, axis=2) # All have shape [BATCH_SIZE, ANCHORS]

            # shape of mc.ANCHOR_BOX: [ANCHORS, 4] 
            anchor_x = mc.ANCHOR_BOX[:, 0]
            anchor_y = mc.ANCHOR_BOX[:, 1]
            anchor_w = mc.ANCHOR_BOX[:, 2]
            anchor_h = mc.ANCHOR_BOX[:, 3]

            # "add" delta to anchors; result shapes: [BATCH_SIZE, ANCHORS]
            box_center_x = tf.identity(
                anchor_x + delta_x * anchor_w, name='bbox_cx' + suffix)
            box_center_y = tf.identity(
                anchor_y + delta_y * anchor_h, name='bbox_cy' + suffix)
            box_width = tf.identity(
                anchor_w * util.safe_exp(delta_w, mc.EXP_THRESH),
                name='bbox_width' + suffix)
            box_height = tf.identity(
                anchor_h * util.safe_exp(delta_h, mc.EXP_THRESH),
                name='bbox_height' + suffix)

            if preds_idx == 0:
              self._activation_summary(delta_x, 'delta_x')
              self._activation_summary(delta_y, 'delta_y')
              self._activation_summary(delta_w, 'delta_w')
              self._activation_summary(delta_h, 'delta_h')

              self._activation_summary(box_center_x, 'bbox_cx')
              self._activation_summary(box_center_y, 'bbox_cy')
              self._activation_summary(box_width, 'bbox_width')
              self._activation_summary(box_height, 'bbox_height')

          with tf.variable_scope('trimming'):
            # Converts from x, y, w, h to xmin, ymin, xmax, ymax
            # result shape: [BATCH_SIZE, ANCHORS]
            xmins, ymins, xmaxs, ymaxs = util.bbox_transform(
                [box_center_x, box_center_y, box_width, box_height])

            # The max x position is mc.IMAGE_WIDTH - 1 since we use zero-based
            # pixels. Same for y.
            # Clipping xmin, xmax, ... to proper values; result shape: [BATCH_SIZE,
            # ANCHORS]
            xmins = tf.minimum(
                tf.maximum(0.0, xmins), mc.IMAGE_WIDTH-1.0, name='bbox_xmin' + suffix)

            if preds_idx == 0:
              self._activation_summary(xmins, 'box_xmin')

            ymins = tf.minimum(
                tf.maximum(0.0, ymins), mc.IMAGE_HEIGHT-1.0, name='bbox_ymin' + suffix)

            if preds_idx == 0:
              self._activation_summary(ymins, 'box_ymin')

            xmaxs = tf.maximum(
                tf.minimum(mc.IMAGE_WIDTH-1.0, xmaxs), 0.0, name='bbox_xmax' + suffix)

            if preds_idx == 0:
              self._activation_summary(xmaxs, 'box_xmax')

            ymaxs = tf.maximum(
                tf.minimum(mc.IMAGE_HEIGHT-1.0, ymaxs), 0.0, name='bbox_ymax' + suffix)

            if preds_idx == 0:
              self._activation_summary(ymaxs, 'box_ymax')

            # converting xmin, ymin, xmax, ymax back to x,y,w,h
            # result shape: [BATCH_SIZE, ANCHORS, 4]
            det_boxes = tf.transpose(
                tf.stack(util.bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
                (1, 2, 0), name='bbox' + suffix
            )
            if preds_idx == 0:
              self.det_boxes = det_boxes
            else:
              self.det_boxes_mc_samples.append(det_boxes)

        if preds_idx == 0:
          with tf.variable_scope('IOU'):
            # input of shape: [4, BATCH_SIZE, ANCHORS] in xmin,ymin,xmax,ymax encoding
            def _tensor_iou(box1, box2):
              with tf.variable_scope('intersection'):
                xmin = tf.maximum(box1[0], box2[0], name='xmin')
                ymin = tf.maximum(box1[1], box2[1], name='ymin')
                xmax = tf.minimum(box1[2], box2[2], name='xmax')
                ymax = tf.minimum(box1[3], box2[3], name='ymax')

                w = tf.maximum(0.0, xmax-xmin, name='inter_w')
                h = tf.maximum(0.0, ymax-ymin, name='inter_h')
                intersection = tf.multiply(w, h, name='intersection')

              with tf.variable_scope('union'):
                w1 = tf.subtract(box1[2], box1[0], name='w1')
                h1 = tf.subtract(box1[3], box1[1], name='h1')
                w2 = tf.subtract(box2[2], box2[0], name='w2')
                h2 = tf.subtract(box2[3], box2[1], name='h2')

                union = w1*h1 + w2*h2 - intersection

              return intersection/(union+mc.EPSILON) \
                  * tf.reshape(self.input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

            # Result shape: [BATCH_SIZE, ANCHORS]
            self.ious = self.ious.assign(
                _tensor_iou(
                    util.bbox_transform(tf.unstack(det_boxes, axis=2)),
                    util.bbox_transform(tf.unstack(self.box_input, axis=2))
                )
            )
            self._activation_summary(self.ious, 'conf_score')

        with tf.variable_scope('probability') as scope:
          if preds_idx == 0:
            self._activation_summary(pred_class_probs, 'class_probs')

          # Multiply softmax probs with confidence scores
          # Result shape: [BATCHSIZE, ANCHORS, CLASSES]
          probs = tf.multiply(
              pred_class_probs,
              tf.reshape(pred_conf, [mc.BATCH_SIZE, mc.ANCHORS, 1]),
              name='final_class_prob' + suffix
          )

          # Pick out class with maximum probability
          # Result shape: [BATCHSIZE, ANCHORS]
          det_probs = tf.reduce_max(probs, 2, name='score' + suffix)
          det_class = tf.argmax(probs, 2, name='class_idx' + suffix)

          if preds_idx == 0:
            self._activation_summary(probs, 'final_class_prob')
            self.det_probs = det_probs
            self.det_class = det_class
            self.det_all_probs = probs
          else:
            self.det_probs_mc_samples.append(det_probs)
            self.det_class_mc_samples.append(det_class)
            self.det_all_probs_mc_samples.append(probs)

  def _add_loss_graph(self):
    """Define the loss operation."""
    mc = self.mc

    with tf.variable_scope('class_regression') as scope:
      # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
      # add a small value into log to prevent blowing up
      self.class_loss = tf.truediv(
          tf.reduce_sum(
              (self.labels*(-tf.log(self.pred_class_probs+mc.EPSILON))
               + (1-self.labels)*(-tf.log(1-self.pred_class_probs+mc.EPSILON)))
              * self.input_mask * mc.LOSS_COEF_CLASS),
          self.num_objects,
          name='class_loss'
      )
      tf.add_to_collection('losses', self.class_loss)

    with tf.variable_scope('confidence_score_regression') as scope:
      input_mask = tf.reshape(self.input_mask, [mc.BATCH_SIZE, mc.ANCHORS])
      self.conf_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.square((self.ious - self.pred_conf)) 
              * (input_mask*mc.LOSS_COEF_CONF_POS/self.num_objects
                 +(1-input_mask)*mc.LOSS_COEF_CONF_NEG/(mc.ANCHORS-self.num_objects)),
              reduction_indices=[1]
          ),
          name='confidence_loss'
      )
      tf.add_to_collection('losses', self.conf_loss)
      tf.summary.scalar('mean iou', tf.reduce_sum(self.ious)/self.num_objects)

    with tf.variable_scope('bounding_box_regression') as scope:

     if self.uncertainty_method == 'exact_wdrop':

        eps = tf.constant(1e-8)
        pred_box_delta_mc = tf.stack([self.pred_box_delta] + self.pred_box_delta_mc_samples)
        self.pred_box_delta_mc = pred_box_delta_mc

        # compute moments
        pred_box_delta_mc_mean = tf.reduce_mean(pred_box_delta_mc, axis=0)
        pred_box_delta_mc_sm = tf.reduce_mean(tf.square(pred_box_delta_mc), axis=0)

        pred_box_delta_mc_var = pred_box_delta_mc_sm - tf.square(pred_box_delta_mc_mean)
        pred_box_delta_mc_sigma = tf.sqrt(tf.maximum(pred_box_delta_mc_var, eps))

        y_bootstrap_var = tf.square(pred_box_delta_mc_mean - self.box_delta_input) + pred_box_delta_mc_var
        y_bootstrap_sigma = tf.sqrt(tf.maximum(y_bootstrap_var, eps))

        self.pred_box_delta_mc_var = pred_box_delta_mc_var
        self.y_bootstrap_var = y_bootstrap_var

        self.bbox_loss = tf.truediv(
          mc.LOSS_COEF_BBOX * tf.reduce_sum(
              self.input_mask * (
                  tf.square(pred_box_delta_mc_mean - self.box_delta_input) + tf.square(pred_box_delta_mc_sigma - y_bootstrap_sigma)
              )
          ),
          self.num_objects,
          name='bbox_loss'
        )
     else:
        self.bbox_loss = tf.truediv(
              tf.reduce_sum(
                  mc.LOSS_COEF_BBOX * tf.square(
                      self.input_mask*(self.pred_box_delta-self.box_delta_input))),
              self.num_objects,
              name='bbox_loss'
        )

     tf.add_to_collection('losses', self.bbox_loss)

    # add above losses as well as weight decay losses to form the total loss
    self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

  def _add_train_graph(self):
    """Define the training operation."""
    mc = self.mc

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                    self.global_step,
                                    mc.DECAY_STEPS,
                                    mc.LR_DECAY_FACTOR,
                                    staircase=True)

    tf.summary.scalar('learning_rate', lr)

    _add_loss_summaries(self.loss)

    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mc.MOMENTUM)
    grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())

    var_names = []
    with tf.variable_scope('clip_gradient') as scope:
      for i, (grad, var) in enumerate(grads_vars):
        grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)
        var_names.append(var.name)

    self.grads_vars = grads_vars
    self.var_names = var_names

    apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads_vars:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    with tf.control_dependencies([apply_gradient_op]):
      self.train_op = tf.no_op(name='train')

  def _add_viz_graph(self):
    """Define the visualization operation."""
    mc = self.mc
    self.image_to_show = tf.placeholder(
        tf.float32, [None, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
        name='image_to_show'
    )
    self.viz_op = tf.summary.image('sample_detection_results',
        self.image_to_show, collections='image_summary',
        max_outputs=mc.BATCH_SIZE)

  def _conv_bn_layer(
      self, inputs, conv_param_name, bn_param_name, scale_param_name, filters,
      size, stride, padding='SAME', freeze=False, relu=True,
      conv_with_bias=False, stddev=0.001):
    """ Convolution + BatchNorm + [relu] layer. Batch mean and var are treated
    as constant. Weights have to be initialized from a pre-trained model or
    restored from a checkpoint.

    Args:
      inputs: input tensor
      conv_param_name: name of the convolution parameters
      bn_param_name: name of the batch normalization parameters
      scale_param_name: name of the scale parameters
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      conv_with_bias: whether or not add bias term to the convolution output.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """
    mc = self.mc

    with tf.variable_scope(conv_param_name) as scope:
      channels = inputs.get_shape()[3]

      if mc.LOAD_PRETRAINED_MODEL:
        cw = self.caffemodel_weight
        kernel_val = np.transpose(cw[conv_param_name][0], [2,3,1,0])
        if conv_with_bias:
          bias_val = cw[conv_param_name][1]
        mean_val   = cw[bn_param_name][0]
        var_val    = cw[bn_param_name][1]
        gamma_val  = cw[scale_param_name][0]
        beta_val   = cw[scale_param_name][1]
      else:
        kernel_val = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        if conv_with_bias:
          bias_val = tf.constant_initializer(0.0)
        mean_val   = tf.constant_initializer(0.0)
        var_val    = tf.constant_initializer(1.0)
        gamma_val  = tf.constant_initializer(1.0)
        beta_val   = tf.constant_initializer(0.0)

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))
      self.model_params += [kernel]
      if conv_with_bias:
        biases = _variable_on_device('biases', [filters], bias_val,
                                     trainable=(not freeze))
        self.model_params += [biases]
      gamma = _variable_on_device('gamma', [filters], gamma_val,
                                  trainable=(not freeze))
      beta  = _variable_on_device('beta', [filters], beta_val,
                                  trainable=(not freeze))
      mean  = _variable_on_device('mean', [filters], mean_val, trainable=False)
      var   = _variable_on_device('var', [filters], var_val, trainable=False)
      self.model_params += [gamma, beta, mean, var]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, stride, stride, 1], padding=padding,
          name='convolution')
      if conv_with_bias:
        conv = tf.nn.bias_add(conv, biases, name='bias_add')

      conv = tf.nn.batch_normalization(
          conv, mean=mean, variance=var, offset=beta, scale=gamma,
          variance_epsilon=mc.BATCH_NORM_EPSILON, name='batch_norm')

      self.model_size_counter.append(
          (conv_param_name, (1+size*size*int(channels))*filters)
      )
      out_shape = conv.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((conv_param_name, num_flops))

      self.activation_counter.append(
          (conv_param_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      if relu:
        return tf.nn.relu(conv)
      else:
        return conv


  def _conv_layer(
      self, layer_name, inputs, filters, size, stride, padding='SAME',
      freeze=False, xavier=False, relu=True, stddev=0.001):
    """Convolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    mc = self.mc
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
        bias_val = cw[layer_name][1]
        # check the shape
        if (kernel_val.shape == 
              (size, size, inputs.get_shape().as_list()[-1], filters)) \
           and (bias_val.shape == (filters, )):
          use_pretrained_param = True
        else:
          print ('Shape of the pretrained parameter of {} does not match, '
              'use randomly initialized parameter'.format(layer_name))
      else:
        print ('Cannot find {} in the pretrained model. Use randomly initialized '
               'parameters'.format(layer_name))

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape()[3]

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val , dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.0)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)

      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

      biases = _variable_on_device('biases', [filters], bias_init, 
                                trainable=(not freeze))
      self.model_params += [kernel, biases]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, stride, stride, 1], padding=padding,
          name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
  
      if relu:
        out = tf.nn.relu(conv_bias, 'relu')
      else:
        out = conv_bias

      self.model_size_counter.append(
          (layer_name, (1+size*size*int(channels))*filters)
      )
      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      return out
 
  def _conv_layers_shared(
      self, layer_name, inputs, shared_inputs, filters, size, stride, padding='SAME',
      freeze=False, xavier=False, relu=True, stddev=0.001):
    """Convolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      shared_inputs: input for shared layers (single tensor or list of tensors)
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    mc = self.mc
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
        bias_val = cw[layer_name][1]
        # check the shape
        if (kernel_val.shape == 
              (size, size, inputs.get_shape().as_list()[-1], filters)) \
           and (bias_val.shape == (filters, )):
          use_pretrained_param = True
        else:
          print ('Shape of the pretrained parameter of {} does not match, '
              'use randomly initialized parameter'.format(layer_name))
      else:
        print ('Cannot find {} in the pretrained model. Use randomly initialized '
               'parameters'.format(layer_name))

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))


    with tf.variable_scope(layer_name) as scope:
 
      channels = inputs.get_shape()[3]

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val , dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.0)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)

      kernel = _variable_with_weight_decay(
            'kernels', shape=[size, size, int(channels), filters],
            wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

      biases = _variable_on_device('biases', [filters], bias_init, 
                                trainable=(not freeze))
      self.model_params += [kernel, biases]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, stride, stride, 1], padding=padding,
          name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
  
      if relu:
        out = tf.nn.relu(conv_bias, 'relu')
      else:
        out = conv_bias

      # TODO: those stats are not correct anymore
      self.model_size_counter.append(
          (layer_name, (1+size*size*int(channels))*filters)
      )
      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

    if type(shared_inputs) == list and len(shared_inputs) == 0:
        return out, []

    with tf.variable_scope(layer_name, reuse=True) as scope:

      kernel_shared = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

      biases_shared = _variable_on_device('biases', [filters], bias_init,
                                trainable=(not freeze))

      if type(shared_inputs) == list:
        out_shared = []

        for shared_input in shared_inputs:
            conv_shared = tf.nn.conv2d(
                shared_input, kernel_shared, [1, stride, stride, 1], padding=padding,
                name='convolution')
            conv_bias_shared = tf.nn.bias_add(conv_shared, biases_shared, name='bias_add')

            if relu:
                out_shared.append(tf.nn.relu(conv_bias_shared, 'relu'))
            else:
                out_shared.append(conv_bias_shared)
      else:
          conv_shared = tf.nn.conv2d(
              shared_inputs, kernel_shared, [1, stride, stride, 1],padding=padding,
              name='convolution')
          conv_bias_shared = tf.nn.bias_add(conv_shared, biases_shared,
                                            name='bias_add')
          if relu:
              out_shared = tf.nn.relu(conv_bias_shared, 'relu')
          else:
              out_shared = conv_bias_shared

    return out, out_shared

  def _pooling_layer(
      self, layer_name, inputs, size, stride, padding='SAME'):
    """Pooling layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
    Returns:
      A pooling layer operation.
    """

    with tf.variable_scope(layer_name) as scope:
      out =  tf.nn.max_pool(inputs, 
                            ksize=[1, size, size, 1], 
                            strides=[1, stride, stride, 1],
                            padding=padding)
      activation_size = np.prod(out.get_shape().as_list()[1:])
      self.activation_counter.append((layer_name, activation_size))
      return out

  
  def _fc_layer(
      self, layer_name, inputs, hiddens, flatten=False, relu=True,
      xavier=False, stddev=0.001):
    """Fully connected layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      hiddens: number of (hidden) neurons in this layer.
      flatten: if true, reshape the input 4D tensor of shape 
          (batch, height, weight, channel) into a 2D tensor with shape 
          (batch, -1). This is used when the input to the fully connected layer
          is output of a convolutional layer.
      relu: whether to use relu or not.
      xavier: whether to use xavier weight initializer or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A fully connected layer operation.
    """
    mc = self.mc

    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        use_pretrained_param = True
        kernel_val = cw[layer_name][0]
        bias_val = cw[layer_name][1]

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      input_shape = inputs.get_shape().as_list()
      if flatten:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs = tf.reshape(inputs, [-1, dim])
        if use_pretrained_param:
          try:
            # check the size before layout transform
            assert kernel_val.shape == (hiddens, dim), \
                'kernel shape error at {}'.format(layer_name)
            kernel_val = np.reshape(
                np.transpose(
                    np.reshape(
                        kernel_val, # O x (C*H*W)
                        (hiddens, input_shape[3], input_shape[1], input_shape[2])
                    ), # O x C x H x W
                    (2, 3, 1, 0)
                ), # H x W x C x O
                (dim, -1)
            ) # (H*W*C) x O
            # check the size after layout transform
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            # Do not use pretrained parameter if shape doesn't match
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match, '
                   'use randomly initialized parameter'.format(layer_name))
      else:
        dim = input_shape[1]
        if use_pretrained_param:
          try:
            kernel_val = np.transpose(kernel_val, (1,0))
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match, '
                   'use randomly initialized parameter'.format(layer_name))

      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val, dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.0)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)

      weights = _variable_with_weight_decay(
          'weights', shape=[dim, hiddens], wd=mc.WEIGHT_DECAY,
          initializer=kernel_init)
      biases = _variable_on_device('biases', [hiddens], bias_init)
      self.model_params += [weights, biases]
  
      outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
      if relu:
        outputs = tf.nn.relu(outputs, 'relu')

      # count layer stats
      self.model_size_counter.append((layer_name, (dim+1)*hiddens))

      num_flops = 2 * dim * hiddens + hiddens
      if relu:
        num_flops += 2*hiddens
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append((layer_name, hiddens))

      return outputs

  def filter_prediction(self, boxes, probs, cls_idx,
                        return_anchor_idxs=False,
                        det_all_probs=None,
                        additional_scores=None):
    """Filter bounding box predictions with probability threshold and
    non-maximum supression.

    Args:
      boxes: array of [cx, cy, w, h]. Shape: [ANCHOR, 4]
      probs: array of probabilities. Shape: [ANCHOR]
      cls_idx: array of class indices. Shape: [ANCHOR]
      additional_scores: (optional) array of additional scores: Shape: [ANCHOR, N_SCORES]
    Returns:
        final_boxes: array of filtered bounding boxes. Shape:
            [FILTERED_ANCHORS, 4]
        final_probs: array of filtered probabilities. Shape:
                [FILTERED_ANCHORS]
        final_cls_idx: array of filtered class indices. Shape:
                    [FILTERED_ANCHORS]
        if additional_scores is not None:
        final_additional_scores. Shape: [FILTERED_ANCHORS, N_SCORES]
        The outputs are ordered by class
    """
    n_diff, n_total = 0, 0

    mc = self.mc
    order = None

    #debug
    _n_probs_before = len(probs)

    # Get the TOP_N_DETECTIONS (top n probability)
    if mc.TOP_N_DETECTION < len(probs) and mc.TOP_N_DETECTION > 0:
      order = probs.argsort()[:-mc.TOP_N_DETECTION-1:-1]
      probs = probs[order]
      boxes = boxes[order]
      cls_idx = cls_idx[order]
      if det_all_probs is not None:
        all_probs = det_all_probs[order]
      if additional_scores is not None:
        additional_scores = additional_scores[order]
    else: # if there are less than TOP_N_DETECTIONS
          # threshold probability
      filtered_idx = np.nonzero(probs>mc.PROB_THRESH)[0]
      probs = probs[filtered_idx]
      boxes = boxes[filtered_idx]
      cls_idx = cls_idx[filtered_idx]
      if det_all_scores is not None:
        all_probs = det_all_scores[filtered_idx]
      if additional_scores is not None:
        additional_scores = additional_scores[filtered_idx]

    final_boxes = []
    final_probs = []
    final_cls_idx = []
    anchor_idxs = []
    final_all_probs = []
    final_additional_scores = []

    for c in range(mc.CLASSES):  # Iterate all classes
      # Apply nms on all boxes classified with class c
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]

      if mc.PREDICTION_FILTERING_METHOD == 'nms_orig':
        keep = util.nms(boxes[idx_per_class], probs[idx_per_class], mc.NMS_THRESH)
      elif mc.PREDICTION_FILTERING_METHOD == 'nms':
        keep = util.generalized_nms(boxes[idx_per_class], probs[idx_per_class],
                                 lambda s, bi, i: util.standard_nms_score_update(s, bi, i, mc.NMS_THRESH))

      n_total += len(idx_per_class)

      # put all boxes that "survive" to a list
      for i in range(len(keep)):
        if keep[i]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)
          anchor_idxs.append(idx_per_class[i])
          if det_all_probs is not None:
            final_all_probs.append(all_probs[idx_per_class[i]])
          if additional_scores is not None:
            final_additional_scores.append(additional_scores[idx_per_class[i]])

    out = [final_boxes]
    out += [final_probs, final_cls_idx]

    if return_anchor_idxs:
      if order is not None:
        anchor_idxs = order[anchor_idxs]
      else:
        anchor_idxs = filtered_idx[anchor_idxs]
      out.append(anchor_idxs)

    if det_all_probs is not None:
      out.append(final_all_probs)

    if additional_scores is not None:
      out.append(final_additional_scores)

    return out

  def _activation_summary(self, x, layer_name):
    """Helper to create summaries for activations.

    Args:
      x: layer output tensor
      layer_name: name of the layer
    Returns:
      nothing
    """
    with tf.variable_scope('activation_summary') as scope:
      tf.summary.histogram(
          'activation_summary/'+layer_name, x)
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/sparsity', tf.nn.zero_fraction(x))
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/average', tf.reduce_mean(x))
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/max', tf.reduce_max(x))
      tf.summary.scalar(
          'activation_summary/'+layer_name+'/min', tf.reduce_min(x))
