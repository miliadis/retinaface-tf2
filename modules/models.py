import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2, ResNet50, ResNet152
from tensorflow.keras.layers import Input, Conv2D, ReLU, LeakyReLU
from modules.anchor import decode_tf, prior_box_tf


def _regularizer(weights_decay):
    """l2 regularizer"""
    return tf.keras.regularizers.l2(weights_decay)


def _kernel_init(scale=1.0, seed=None):
    """He normal initializer"""
    return tf.keras.initializers.he_normal()


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, center=True,
                 scale=True, name=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center,
            scale=scale, name=name, **kwargs)

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)

        return super().call(x, training)


def Backbone(backbone_type='ResNet50', use_pretrain=True, levels='3'):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'

    def backbone(x):
        if backbone_type == 'ResNet50':
            extractor = ResNet50(
                input_shape=x.shape[1:], include_top=False, weights=weights)
            pick_layer1 = 80  # [80, 80, 512]
            pick_layer2 = 142  # [40, 40, 1024]
            pick_layer3 = 174  # [20, 20, 2048]
            pick_layer4 = []
            preprocess = tf.keras.applications.resnet.preprocess_input
        elif backbone_type == 'ResNet152':
            extractor = ResNet152(
                input_shape=x.shape[1:], include_top=False, weights=weights)
            pick_layer1 = 38  # [160, 160, 256]
            pick_layer2 = 120  # [80, 80, 512]
            pick_layer3 = 482  # [40, 40, 1024]
            pick_layer4 = 514  # [20, 20, 2048]
            preprocess = tf.keras.applications.resnet.preprocess_input
        elif backbone_type == 'MobileNetV2':
            extractor = MobileNetV2(
                input_shape=x.shape[1:], include_top=False, weights=weights)
            pick_layer1 = 54  # [80, 80, 32]
            pick_layer2 = 116  # [40, 40, 96]
            pick_layer3 = 143  # [20, 20, 160]
            pick_layer4 = []
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        else:
            raise NotImplementedError(
                'Backbone type {} is not recognized.'.format(backbone_type))

        if levels == '5':
            model = Model(extractor.input,
                  (extractor.layers[pick_layer1].output,
                   extractor.layers[pick_layer2].output,
                   extractor.layers[pick_layer3].output,
                   extractor.layers[pick_layer4].output
                   ),
                  name=backbone_type + '_extrator')(preprocess(x))
        else:
            model = Model(extractor.input,
                  (extractor.layers[pick_layer1].output,
                   extractor.layers[pick_layer2].output,
                   extractor.layers[pick_layer3].output
                   ),
                  name=backbone_type + '_extrator')(preprocess(x))

        return model

    return backbone


class ConvUnit(tf.keras.layers.Layer):
    """Conv + BN + Act"""
    def __init__(self, f, k, s, wd, act=None, name='ConvBN', **kwargs):
        super(ConvUnit, self).__init__(name=name, **kwargs)
        self.conv = Conv2D(filters=f, kernel_size=k, strides=s, padding='same',
                           kernel_initializer=_kernel_init(),
                           kernel_regularizer=_regularizer(wd),
                           use_bias=False, name='conv')
        self.bn = BatchNormalization(name='bn')

        if act is None:
            self.act_fn = tf.identity
        elif act == 'relu':
            self.act_fn = ReLU()
        elif act == 'lrelu':
            self.act_fn = LeakyReLU(0.1)
        else:
            raise NotImplementedError(
                'Activation function type {} is not recognized.'.format(act))

    def call(self, x):
        return self.act_fn(self.bn(self.conv(x)))


class FPN(tf.keras.layers.Layer):
    """Feature Pyramid Network"""
    def __init__(self, out_ch, wd, name='FPN', levels='3', **kwargs):
        super(FPN, self).__init__(name=name, **kwargs)
        act = 'relu'
        if (out_ch <= 64):
            act = 'lrelu'

        self.levels = levels

        self.output1 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act)
        self.output2 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act)
        self.output3 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act)
        self.output4 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act)
        self.output5 = ConvUnit(f=out_ch, k=3, s=2, wd=wd, act=act)
        self.merge1 = ConvUnit(f=out_ch, k=3, s=1, wd=wd, act=act)
        self.merge2 = ConvUnit(f=out_ch, k=3, s=1, wd=wd, act=act)
        self.merge3 = ConvUnit(f=out_ch, k=3, s=1, wd=wd, act=act)
        self.merge4 = ConvUnit(f=out_ch, k=3, s=1, wd=wd, act=act)

    def call(self, x):
        output1 = self.output1(x[0])  # [160, 160, out_ch]
        output2 = self.output2(x[1])  # [80, 80, out_ch]
        output3 = self.output3(x[2])  # [40, 40, out_ch]

        if self.levels == '5':
            output4 = self.output4(x[3])  # [20, 20, out_ch]
            output5 = self.output5(x[3])  # [10, 10, out_ch]

            up_h, up_w = tf.shape(output4)[1], tf.shape(output4)[2]
            up5 = tf.image.resize(output5, [up_h, up_w], method='nearest')
            output4 = output4 + up5
            output4 = self.merge4(output4)

            up_h, up_w = tf.shape(output3)[1], tf.shape(output3)[2]
            up4 = tf.image.resize(output4, [up_h, up_w], method='nearest')
            output3 = output3 + up4
            output3 = self.merge3(output3)

        up_h, up_w = tf.shape(output2)[1], tf.shape(output2)[2]
        up3 = tf.image.resize(output3, [up_h, up_w], method='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up_h, up_w = tf.shape(output1)[1], tf.shape(output1)[2]
        up2 = tf.image.resize(output2, [up_h, up_w], method='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)

        if self.levels == '5':
            outputs = output1, output2, output3, output4, output5
        else:
            outputs = output1, output2, output3

        return outputs


class SSH(tf.keras.layers.Layer):
    """Single Stage Headless Layer"""
    def __init__(self, out_ch, wd, name='SSH', **kwargs):
        super(SSH, self).__init__(name=name, **kwargs)
        assert out_ch % 4 == 0
        act = 'relu'
        if (out_ch <= 64):
            act = 'lrelu'

        self.conv_3x3 = ConvUnit(f=out_ch // 2, k=3, s=1, wd=wd, act=None)

        self.conv_5x5_1 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=act)
        self.conv_5x5_2 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=None)

        self.conv_7x7_2 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=act)
        self.conv_7x7_3 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=None)

        self.relu = ReLU()

    def call(self, x):
        conv_3x3 = self.conv_3x3(x)

        conv_5x5_1 = self.conv_5x5_1(x)
        conv_5x5 = self.conv_5x5_2(conv_5x5_1)

        conv_7x7_2 = self.conv_7x7_2(conv_5x5_1)
        conv_7x7 = self.conv_7x7_3(conv_7x7_2)

        output = tf.concat([conv_3x3, conv_5x5, conv_7x7], axis=3)
        output = self.relu(output)

        return output


class BboxHead(tf.keras.layers.Layer):
    """Bbox Head Layer"""
    def __init__(self, num_anchor, wd, name='BboxHead', **kwargs):
        super(BboxHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = Conv2D(filters=num_anchor * 4, kernel_size=1, strides=1)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, 4])


class LandmarkHead(tf.keras.layers.Layer):
    """Landmark Head Layer"""
    def __init__(self, num_anchor, wd, name='LandmarkHead', **kwargs):
        super(LandmarkHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = Conv2D(filters=num_anchor * 10, kernel_size=1, strides=1)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, 10])


class ClassHead(tf.keras.layers.Layer):
    """Class Head Layer"""
    def __init__(self, num_anchor, wd, name='ClassHead', **kwargs):
        super(ClassHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = Conv2D(filters=num_anchor * 2, kernel_size=1, strides=1)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, 2])


class PadInputImage(tf.keras.layers.Layer):
    def __init__(self, max_steps):
        super(PadInputImage, self).__init__(name='PadInputImage')
        self.max_steps = max_steps

    def call(self, x):
        img_h, img_w = tf.shape(x)[1], tf.shape(x)[2]

        tmp_h = img_h % self.max_steps
        img_pad_h = tf.cond(tmp_h > 0, lambda: self.max_steps - img_h % self.max_steps, lambda: 0)

        tmp_w = img_w % self.max_steps
        img_pad_w = tf.cond(tmp_w > 0, lambda: self.max_steps - img_w % self.max_steps, lambda: 0)

        paddings = [[0, 0], [0, img_pad_h], [0, img_pad_w], [0, 0]]
        x = tf.pad(x, paddings, mode='REFLECT', name=None)

        pad_params = [img_h, img_w, img_pad_h, img_pad_w]

        return x, pad_params


class RecoverPadOutputs(tf.keras.layers.Layer):
    def __init__(self):
        super(RecoverPadOutputs, self).__init__(name='RecoverPadOutputs')

    def call(self, x, pad_params):

        img_h, img_w, img_pad_h, img_pad_w = pad_params

        first_input = x[:, :14]
        second_input = x[:, 14:]

        recover_xy = tf.reshape(first_input, [-1, 7, 2]) * \
                     [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
        first_input = tf.reshape(recover_xy, [-1, 14])

        return tf.concat([first_input, second_input], axis=1)


class ProcessOutput(tf.keras.layers.Layer):

    def __init__(self):
        super(ProcessOutput, self).__init__(name='ProcessOutput')

    def call(self, outputs):

        lmks = outputs[:, 4:14]

        lmks = tf.reshape(lmks, (-1, 5, 2))
        bboxes = outputs[:, :4]
        confs = outputs[:, -1]

        return bboxes, confs, lmks


def pred_to_outputs(cfg, output, inp_shape, iou_th=0.4, score_th=0.02):

  bbox_regressions, landm_regressions, classifications = output

  # only for batch size 1
  preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
      [
          bbox_regressions[0], landm_regressions[0],
          tf.ones_like(classifications[0, :, 0][..., tf.newaxis]),
          classifications[0, :, 1][..., tf.newaxis]
      ], 1)
  priors = prior_box_tf((inp_shape[1], inp_shape[2]), cfg['min_sizes'], cfg['steps'], cfg['clip'])
  decode_preds = decode_tf(preds, priors, cfg['variances'])

  selected_indices = tf.image.non_max_suppression(boxes=decode_preds[:, :4],
                                                  scores=decode_preds[:, -1],
                                                  max_output_size=tf.shape(decode_preds)[0],
                                                  iou_threshold=iou_th,
                                                  score_threshold=score_th)

  out = tf.gather(decode_preds, selected_indices)

  return out


def RetinaFaceModel(cfg, training=False, iou_th=0.4, score_th=0.02,
                    name='RetinaFaceModel'):
    """Retina Face Model"""
    input_size = cfg['input_size'] if training else None
    wd = cfg['weights_decay']
    out_ch = cfg['out_channel']
    num_anchor = len(cfg['min_sizes'][0])
    backbone_type = cfg['backbone_type']
    levels = '3'

    # define model
    x = input_img = Input([input_size, input_size, 3], name='input_image')
    inputs = input_img

    if not training:
        input_img, pad_params = PadInputImage(max(cfg['steps']))(x)
        x = input_img

    x = Backbone(backbone_type=backbone_type, levels=levels)(x)

    fpn = FPN(out_ch=out_ch, wd=wd, levels=levels)(x)

    features = [SSH(out_ch=out_ch, wd=wd, name=f'SSH_{i}')(f)
                for i, f in enumerate(fpn)]

    bbox_regressions = tf.concat(
        [BboxHead(num_anchor, wd=wd, name=f'BboxHead_{i}')(f)
         for i, f in enumerate(features)], axis=1)
    landm_regressions = tf.concat(
        [LandmarkHead(num_anchor, wd=wd, name=f'LandmarkHead_{i}')(f)
         for i, f in enumerate(features)], axis=1)
    classifications = tf.concat(
        [ClassHead(num_anchor, wd=wd, name=f'ClassHead_{i}')(f)
         for i, f in enumerate(features)], axis=1)

    classifications = tf.keras.layers.Softmax(axis=-1)(classifications)

    if training:
        out = (bbox_regressions, landm_regressions, classifications)
    else:
        # only for batch size 1
        preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
            [bbox_regressions[0], landm_regressions[0],
             tf.ones_like(classifications[0, :, 0][..., tf.newaxis]),
             classifications[0, :, 1][..., tf.newaxis]], 1)
        priors = prior_box_tf((tf.shape(input_img)[1], tf.shape(input_img)[2]),
                              cfg['min_sizes'],  cfg['steps'], cfg['clip'])
        decode_preds = decode_tf(preds, priors, cfg['variances'])

        selected_indices = tf.image.non_max_suppression(
            boxes=decode_preds[:, :4],
            scores=decode_preds[:, -1],
            max_output_size=tf.shape(decode_preds)[0],
            iou_threshold=iou_th,
            score_threshold=score_th)

        out = tf.gather(decode_preds, selected_indices)

        out = RecoverPadOutputs()(out, pad_params)

        bboxes, confs, lmks = ProcessOutput()(out)

        out = [bboxes, confs, lmks]

    return Model(inputs, out, name=name)
