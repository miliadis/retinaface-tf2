import os
import cv2
import numpy as np
import tensorflow as tf

from absl import app, flags, logging
from absl.flags import FLAGS
from modules.models import RetinaFaceModel
from modules.utils import (labels_to_boxes, load_dataset, load_yaml,
                           pad_input_image, recover_pad_output, set_memory_growth)
from evaluate.evaluation import WiderFaceEval


flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml', 'config file path')
flags.DEFINE_string('dataset_root', '/data', 'dataset root')
flags.DEFINE_string('checkpoint_dir', '/checkpoints', 'checkpoint dir')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_boolean('origin_size', True, 'whether use origin image size to evaluate')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.02, 'score threshold for nms')


def main(_):

  # init
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  logger = tf.get_logger()
  logger.disabled = True
  logger.setLevel(logging.FATAL)
  set_memory_growth([])

  cfg = load_yaml(FLAGS.cfg_path)
  cfg['dataset_root'] = FLAGS.dataset_root

  # define network
  model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th, score_th=FLAGS.score_th)

  # load dataset
  val_dataset = load_dataset(cfg, None, 'val', [])

  # load checkpoint
  checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, cfg['sub_name'])
  checkpoint = tf.train.Checkpoint(model=model)
  if tf.train.latest_checkpoint(checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("[*] load ckpt from {}.".format(tf.train.latest_checkpoint(checkpoint_dir)))
  else:
    print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
    exit()

  def test_step(inputs, img_name):
    _, img_height_raw, img_width_raw, _ = inputs.shape
    # pad input image to avoid unmatched shape problem
    img = inputs[0].numpy()
    if img_name == '6_Funeral_Funeral_6_618':
      resize = 0.5  # this image is too big to avoid OOM problem
      img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))
    input_img = img[np.newaxis, ...]
    outputs = model(input_img, training=False).numpy()
    #outputs = pred_to_outputs(cfg, predictions, input_img.shape).numpy()
    # recover padding effect
    outputs = recover_pad_output(outputs, pad_params)

    bboxs = outputs[:, :4]
    confs = outputs[:, -1]
    pred_boxes = []
    for box, conf in zip(bboxs, confs):
      x = int(box[0] * img_width_raw)
      y = int(box[1] * img_height_raw)
      w = int(box[2] * img_width_raw) - int(box[0] * img_width_raw)
      h = int(box[3] * img_height_raw) - int(box[1] * img_height_raw)
      pred_boxes.append([x, y, w, h, conf])

    pred_boxes = np.array(pred_boxes).astype('float')

    return pred_boxes

  widerface_eval_easy = WiderFaceEval(split='easy')
  widerface_eval_medium = WiderFaceEval(split='medium')
  widerface_eval_hard = WiderFaceEval(split='hard')

  dataset_iter = val_dataset
  num_of_samples = 2474
  #num_of_samples = 50

  for ii, (x_batch_val, y_batch_val, img_name) in enumerate(dataset_iter):

    if '/' in img_name.numpy()[0].decode():
      img_name = img_name.numpy()[0].decode().split('/')[1].split('.')[0]
    else:
      img_name = []

    print(" [{} / {}] det {}".format(ii + 1, num_of_samples, img_name))

    pred_boxes = test_step(x_batch_val, img_name)
    gt_boxes = labels_to_boxes(y_batch_val)
    widerface_eval_easy.update(pred_boxes, gt_boxes, img_name)
    widerface_eval_medium.update(pred_boxes, gt_boxes, img_name)
    widerface_eval_hard.update(pred_boxes, gt_boxes, img_name)

  ap_easy = widerface_eval_easy.calculate_ap()
  ap_medium = widerface_eval_medium.calculate_ap()
  ap_hard = widerface_eval_hard.calculate_ap()

  print("==================== Results ====================")
  print("Easy   Val AP: {}".format(ap_easy))
  print("Medium Val AP: {}".format(ap_medium))
  print("Hard   Val AP: {}".format(ap_hard))
  print("=================================================")


if __name__ == '__main__':
  app.run(main)
