import pickle
import cv2
import yaml
import sys
import time
import os
import numpy as np
import tensorflow as tf
from absl import logging
from modules.dataset import load_tfrecord_dataset
from widerface_evaluate.evaluation import (dataset_pr_info, image_eval, img_pr_info, voc_ap)


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                logging.info(
                    "Detect {} Physical GPUs, {} Logical GPUs.".format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)


def load_dataset(cfg, priors, split):
  """load dataset"""
  logging.info("load dataset from {}".format(cfg['dataset_root']))

  if split is 'train':
    batch_size = cfg['batch_size']
    shuffle = True
    using_flip = cfg['using_flip']
    using_distort = cfg['using_distort']
    using_encoding = True
    buffer_size = 4000
    number_cycles = 2
    threads = tf.data.experimental.AUTOTUNE
  else:
    batch_size = 1
    shuffle = False
    using_flip = False
    using_distort = False
    using_encoding = False
    buffer_size = 4000
    number_cycles = 1
    threads = tf.data.experimental.AUTOTUNE

  dataset = load_tfrecord_dataset(dataset_root=cfg['dataset_root'],
                              split=split,
                              threads=threads,
                              number_cycles=number_cycles,
                              batch_size=batch_size,
                              img_dim=cfg['input_size'],
                              using_bin=cfg['using_bin'],
                              using_flip=using_flip,
                              using_distort=using_distort,
                              using_encoding=using_encoding,
                              priors=priors,
                              match_thresh=cfg['match_thresh'],
                              ignore_thresh=cfg['ignore_thresh'],
                              variances=cfg['variances'],
                              shuffle=shuffle,
                              buffer_size=buffer_size)
  return dataset



class ProgressBar(object):
    """A progress bar which can print the progress modified from
       https://github.com/hellock/cvbase/blob/master/cvbase/progress.py"""
    def __init__(self, task_num=0, completed=0, bar_width=25):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width
                          if bar_width <= max_bar_width else max_bar_width)
        self.completed = completed
        self.first_step = completed
        self.warm_up = False

    def _get_max_bar_width(self):
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        else:
            from backports.shutil_get_terminal_size import get_terminal_size
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            logging.info('terminal width is too small ({}), please consider '
                         'widen the terminal for better progressbar '
                         'visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def reset(self):
        """reset"""
        self.completed = 0
        self.fps = 0

    def update(self, inf_str=''):
        """update"""
        self.completed += 1

        if not self.warm_up:
            self.start_time = time.time() - 1e-1
            self.warm_up = True

        if self.completed > self.task_num:
            self.completed = self.completed % self.task_num
            self.start_time = time.time() - 1 / self.fps
            self.first_step = self.completed - 1
            sys.stdout.write('\n')

        elapsed = time.time() - self.start_time
        self.fps = (self.completed - self.first_step) / elapsed
        percentage = self.completed / float(self.task_num)
        mark_width = int(self.bar_width * percentage)
        bar_chars = '>' * mark_width + ' ' * (self.bar_width - mark_width)
        stdout_str = '\rTraining [{}] {}/{}, {}  {:.1f} step/sec'
        sys.stdout.write(stdout_str.format(
            bar_chars, self.completed, self.task_num, inf_str, self.fps))

        sys.stdout.flush()


###############################################################################
#   Testing                                                                   #
###############################################################################
def pad_input_image(img, max_steps):
    """pad image to suitable shape"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT, value=padd_val.tolist())
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return img, pad_params


def recover_pad_output(outputs, pad_params):
    """recover the padded output effect"""
    img_h, img_w, img_pad_h, img_pad_w = pad_params
    recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
        [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
    outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

    return outputs


def labels_to_boxes(labels):
  gt_boxes = labels[0, :, 0:4].numpy().astype('float')
  gt_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]  # width
  gt_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]  # height
  return gt_boxes


class WiderFaceEval(object):

  def __init__(self, split, thresh_num=1000, iou_thresh=0.5):
    with open('./widerface_evaluate/ground_truth/widerface_gt.pickle', 'rb') as handle:
      widerface_gt = pickle.load(handle)

    self.split = split
    self.widerface = widerface_gt
    self.count_face = 0
    self.pr_curve = np.zeros((thresh_num, 2)).astype('float')

    self.iou_thresh = iou_thresh
    self.thresh_num = thresh_num

  def update(self, outputs, gt, img_name):

    pred_info = outputs

    gt_boxes = gt
    keep_index = self.widerface[self.split][img_name]
    self.count_face += len(keep_index)

    if len(gt_boxes) == 0 or len(pred_info) == 0:
      return
    ignore = np.zeros(gt_boxes.shape[0])
    if len(keep_index) != 0:
      ignore[keep_index - 1] = 1
    pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, self.iou_thresh)

    _img_pr_info = img_pr_info(self.thresh_num, pred_info, proposal_list, pred_recall)

    self.pr_curve += _img_pr_info

  def calculate_ap(self):

    pr_curve = dataset_pr_info(self.thresh_num, self.pr_curve, self.count_face)

    propose = pr_curve[:, 0]
    recall = pr_curve[:, 1]

    ap = voc_ap(recall, propose)

    return ap

  def reset(self):
    self.count_face = 0
    self.pr_curve = np.zeros((self.thresh_num, 2)).astype('float')


###############################################################################
#   Visulization                                                              #
###############################################################################
def draw_bbox_landm(img, ann, img_height, img_width):
    """draw bboxes and landmarks"""
    # bbox
    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                     int(ann[2] * img_width), int(ann[3] * img_height)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # confidence
    if len(ann) > 15:
        text = "{:.4f}".format(ann[15])
        cv2.putText(img, text, (int(ann[0] * img_width), int(ann[1] * img_height)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # landmark
    if ann[14] > 0:
        cv2.circle(img, (int(ann[4] * img_width),
                         int(ann[5] * img_height)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(ann[6] * img_width),
                         int(ann[7] * img_height)), 1, (0, 255, 255), 2)
        cv2.circle(img, (int(ann[8] * img_width),
                         int(ann[9] * img_height)), 1, (255, 0, 0), 2)
        cv2.circle(img, (int(ann[10] * img_width),
                         int(ann[11] * img_height)), 1, (0, 100, 255), 2)
        cv2.circle(img, (int(ann[12] * img_width),
                         int(ann[13] * img_height)), 1, (255, 0, 100), 2)


def draw_anchor(img, prior, img_height, img_width):
    """draw anchors"""
    x1 = int(prior[0] * img_width - prior[2] * img_width / 2)
    y1 = int(prior[1] * img_height - prior[3] * img_height / 2)
    x2 = int(prior[0] * img_width + prior[2] * img_width / 2)
    y2 = int(prior[1] * img_height + prior[3] * img_height / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
