import time

import cv2
import numpy as np

from absl import app, flags
from absl.flags import FLAGS
from modules.anchor import decode_tf, prior_box
from modules.dataset import load_tfrecord_dataset
from modules.utils import draw_anchor, draw_bbox_landm

flags.DEFINE_string('output_path', '/output', 'output path')
flags.DEFINE_string('dataset_path', '/data/wider_face', 'dataset path')
flags.DEFINE_string('split', 'train', 'split to check data')
flags.DEFINE_boolean('using_encoding', True, 'whether visualization or not')
flags.DEFINE_boolean('visualization', True, 'whether visualize dataset or not')


def main(_):

  min_sizes = [[16, 32], [64, 128], [256, 512]]
  steps = [8, 16, 32]
  clip = False

  img_dim = 640
  priors = prior_box((img_dim, img_dim), min_sizes, steps, clip)

  variances = [0.1, 0.2]
  match_thresh = 0.45
  ignore_thresh = 0.3
  batch_size = 1
  shuffle = True
  using_flip = True
  using_distort = True
  using_bin = True
  buffer_size = 4000
  number_cycles = 2
  threads = 2

  check_dataset = load_tfrecord_dataset(dataset_root=FLAGS.dataset_path,
                              split=FLAGS.split,
                              threads=threads,
                              number_cycles=number_cycles,
                              batch_size=batch_size,
                              hvd=[],
                              img_dim=img_dim,
                              using_bin=using_bin,
                              using_flip=using_flip,
                              using_distort=using_distort,
                              using_encoding=FLAGS.using_encoding,
                              priors=priors,
                              match_thresh=match_thresh,
                              ignore_thresh=ignore_thresh,
                              variances=variances,
                              shuffle=shuffle,
                              buffer_size=buffer_size)

  time.time()
  for idx, (inputs, labels, _) in enumerate(check_dataset):
    print("{} inputs:".format(idx), inputs.shape, "labels:", labels.shape)

    if not FLAGS.visualization:
      continue

    img = np.clip(inputs.numpy()[0], 0, 255).astype(np.uint8)
    if not FLAGS.using_encoding:
      # labels includes loc, landm, landm_valid.
      targets = labels.numpy()[0]
      for target in targets:
        draw_bbox_landm(img, target, img_dim, img_dim)
    else:
      # labels includes loc, landm, landm_valid, conf.
      targets = decode_tf(labels[0], priors, variances=variances).numpy()
      for prior_index in range(len(targets)):
        if targets[prior_index][-1] != 1:
          continue

        draw_bbox_landm(img, targets[prior_index], img_dim, img_dim)
        draw_anchor(img, priors[prior_index], img_dim, img_dim)

    cv2.imwrite('{}/{}.png'.format(FLAGS.output_path, str(idx)), img[:, :, ::-1])


if __name__ == '__main__':
  app.run(main)
