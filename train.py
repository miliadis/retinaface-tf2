from absl import app, flags, logging
import os
import random
import time
import argparse
import pickle
import numpy as np
import tensorflow as tf

from modules.models import RetinaFaceModel, pred_to_outputs
from modules.lr_scheduler import MultiStepWarmUpLR
from modules.losses import MultiBoxLoss
from modules.anchor import prior_box
from modules.utils import (ProgressBar, labels_to_boxes, load_dataset, load_yaml,
                           pad_input_image, recover_pad_output, set_memory_growth)
from evaluate.evaluation import WiderFaceEval


def reset_random_seeds():
  os.environ['PYTHONHASHSEED'] = str(2)
  tf.random.set_seed(2)
  np.random.seed(2)
  random.seed(2)


def train_retinaface(cfg):

    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if cfg['distributed']:
        import horovod.tensorflow as hvd
        # Initialize Horovod
        hvd.init()
    else:
        hvd = []
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    reset_random_seeds()

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth(hvd)

    # define network
    model = RetinaFaceModel(cfg, training=True)
    model.summary(line_length=80)

    # define prior box
    priors = prior_box((cfg['input_size'], cfg['input_size']),
                       cfg['min_sizes'],  cfg['steps'], cfg['clip'])

    # load dataset
    train_dataset = load_dataset(cfg, priors, 'train', hvd)
    if cfg['evaluation_during_training']:
        val_dataset = load_dataset(cfg, priors, 'val', [])

    # define optimizer
    if cfg['distributed']:
        init_lr = cfg['init_lr'] * hvd.size()
        min_lr = cfg['min_lr'] * hvd.size()
        steps_per_epoch = cfg['dataset_len'] // (cfg['batch_size'] * hvd.size())
    else:
        init_lr = cfg['init_lr']
        min_lr = cfg['min_lr']
        steps_per_epoch = cfg['dataset_len'] // cfg['batch_size']

    learning_rate = MultiStepWarmUpLR(
        initial_learning_rate=init_lr,
        lr_steps=[e * steps_per_epoch for e in cfg['lr_decay_epoch']],
        lr_rate=cfg['lr_rate'],
        warmup_steps=cfg['warmup_epoch'] * steps_per_epoch,
        min_lr=min_lr)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)

    # define losses function
    multi_box_loss = MultiBoxLoss()

    # load checkpoint
    checkpoint_dir = os.path.join(cfg['output_path'], 'checkpoints', cfg['sub_name'])
    checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0, name='epoch'),
                                     optimizer=optimizer,
                                     model=model)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'cfg.pickle'), 'wb') as handle:
        pickle.dump(cfg, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {}'.format(manager.latest_checkpoint))
    else:
        print("[*] training from scratch.")

    # define training step function
    @tf.function
    def train_step(inputs, labels, first_batch, epoch):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)

            losses = {}
            losses['reg'] = tf.reduce_sum(model.losses)
            losses['loc'], losses['landm'], losses['class'] = \
                multi_box_loss(labels, predictions)
            total_loss = tf.add_n([l for l in losses.values()])

        if cfg['distributed']:
            # Horovod: add Horovod Distributed GradientTape.
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if cfg['distributed'] and first_batch and epoch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        return total_loss, losses

    def test_step(inputs, img_name):
        _, img_height_raw, img_width_raw, _ = inputs.shape
        # pad input image to avoid unmatched shape problem
        img = inputs[0].numpy()
        # if img_name == '6_Funeral_Funeral_6_618':
        #     resize = 0.5 # this image is too big to avoid OOM problem
        #     img = cv2.resize(img, None, None, fx=resize, fy=resize,
        #                      interpolation=cv2.INTER_LINEAR)
        img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))
        input_img = img[np.newaxis, ...]
        predictions = model(input_img, training=False)
        outputs = pred_to_outputs(cfg, predictions, input_img.shape).numpy()
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

    #training loop
    summary_writer = tf.summary.create_file_writer(os.path.join(cfg['output_path'], 'logs', cfg['sub_name']))
    prog_bar = ProgressBar(steps_per_epoch, 0)

    if cfg['evaluation_during_training']:
        widerface_eval_hard = WiderFaceEval(split='hard')

    for epoch in range(cfg['epoch']):
        try:
            actual_epoch = epoch + 1

            if cfg['distributed']:
                if hvd.rank() == 0:
                    print("\nStart of epoch %d" % (actual_epoch,))
            else:
                print("\nStart of epoch %d" % (actual_epoch,))

            checkpoint.epoch.assign_add(1)
            start_time = time.time()

            #Iterate over the batches of the dataset.
            for batch, (x_batch_train, y_batch_train, img_name) in enumerate(train_dataset):
                total_loss, losses = train_step(x_batch_train, y_batch_train, batch == 0, epoch == 0)

                if cfg['distributed']:
                    if hvd.rank() == 0:
                        # prog_bar.update("epoch={}/{}, loss={:.4f}, lr={:.1e}".format(
                        #     checkpoint.epoch.numpy(), cfg['epoch'], total_loss.numpy(), optimizer._decayed_lr(tf.float32)))
                        if batch % 100 == 0:
                            print("batch={}/{},  epoch={}/{}, loss={:.4f}, lr={:.1e}".format(
                                batch, steps_per_epoch, checkpoint.epoch.numpy(), cfg['epoch'], total_loss.numpy(), optimizer._decayed_lr(tf.float32)))
                else:
                    prog_bar.update("epoch={}/{}, loss={:.4f}, lr={:.1e}".format(
                        checkpoint.epoch.numpy(), cfg['epoch'], total_loss.numpy(), optimizer._decayed_lr(tf.float32)))

            # Display metrics at the end of each epoch.
            # train_acc = train_acc_metric.result()
            # print("\nTraining loss over epoch: %.4f" % (float(total_loss.numpy()),))

            if cfg['distributed']:
                if hvd.rank() == 0:
                    print("Time taken: %.2fs" % (time.time() - start_time))
                    manager.save()
                    print("\n[*] save ckpt file at {}".format(manager.latest_checkpoint))
            else:
                print("Time taken: %.2fs" % (time.time() - start_time))
                manager.save()
                print("\n[*] save ckpt file at {}".format(manager.latest_checkpoint))

            if cfg['evaluation_during_training']:
                # Run a validation loop at the end of each epoch.
                for batch, (x_batch_val, y_batch_val, img_name) in enumerate(val_dataset.take(500)):
                    if '/' in img_name.numpy()[0].decode():
                        img_name = img_name.numpy()[0].decode().split('/')[1].split('.')[0]
                    else:
                        img_name = []
                    pred_boxes = test_step(x_batch_val, img_name)
                    gt_boxes = labels_to_boxes(y_batch_val)
                    widerface_eval_hard.update(pred_boxes, gt_boxes, img_name)

                ap_hard = widerface_eval_hard.calculate_ap()
                widerface_eval_hard.reset()

                if cfg['distributed']:
                    if hvd.rank() == 0:
                        print("Validation acc: %.4f" % (float(ap_hard),))
                else:
                    print("Validation acc: %.4f" % (float(ap_hard),))

            def tensorboard_writer():
                with summary_writer.as_default():
                    tf.summary.scalar('loss/total_loss', total_loss, step=actual_epoch)
                    for k, l in losses.items():
                        tf.summary.scalar('loss/{}'.format(k), l, step=actual_epoch)
                    tf.summary.scalar('learning_rate', optimizer._decayed_lr(tf.float32), step=actual_epoch)
                    if cfg['evaluation_during_training']:
                        tf.summary.scalar('Val AP', ap_hard, step=actual_epoch)

            if cfg['distributed']:
                if hvd.rank() == 0:
                    tensorboard_writer()
            else:
                tensorboard_writer()

        except Exception as E:
            print(E)
            continue

    if cfg['distributed']:
        if hvd.rank() == 0:
            manager.save()
            print("\n[*] training done! save ckpt file at {}".format(
                manager.latest_checkpoint))
    else:
        manager.save()
        print("\n[*] training done! save ckpt file at {}".format(
            manager.latest_checkpoint))


def get_args():
  parser = argparse.ArgumentParser(
      description='RetinaFace train')
  parser.add_argument('--dataset_root', required=True, help='Dataset path', type=str)
  parser.add_argument('--output_path', required=True, help='Output path', type=str)
  parser.add_argument('--cfg_path', required=True, help='Config file path', type=str)
  args = parser.parse_args()

  return args


if __name__ == '__main__':
  args = get_args()
  cfg = load_yaml(args.cfg_path)
  cfg['dataset_root'] = args.dataset_root
  cfg['output_path'] = args.output_path

  train_retinaface(cfg)


