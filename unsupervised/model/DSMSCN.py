import tensorflow as tf
from chrx_util.net_util import conv_2d, max_pool_2d, avg_pool_2d, fully_connected


class DSMSCN(object):
    def get_model(self, Input_X, Input_Y, data_format='NHWC', is_training=True):
        net_X, feature_1_X, feature_2_X, feature_3_X = self._feature_extract_layer(inputs=Input_X, name='Fea_Ext_',
                                                                      data_format=data_format,
                                                                      is_training=is_training)
        net_Y, feature_1_Y, feature_2_Y, feature_3_Y = self._feature_extract_layer(inputs=Input_Y, name='Fea_Ext_',
                                                                      data_format=data_format,
                                                                      is_training=is_training,
                                                                      is_reuse=True)
        diff_fea_1 = tf.abs(feature_1_X - feature_1_Y)  # (B, H, W, 16)
        diff_fea_2 = tf.abs(feature_2_X - feature_2_Y)  # (B, H, W, 32)
        diff_fea_3 = tf.abs(feature_3_X - feature_3_Y)  # (B, H, W, 64)
        diff_net = tf.abs(net_X - net_Y)  # (B, H, W, 128)
        # (B, H, W, 128) --> (B, H, W, 256)
        if data_format == 'NHWC':
            diff_feature = tf.concat([diff_net, diff_fea_1, diff_fea_2], axis=-1)
        else:
            diff_feature = tf.concat([diff_net, diff_fea_1, diff_fea_2], axis=1)
        # (B, 1)
        diff_feature = conv_2d(inputs=diff_feature, kernel_size=[1, 1], output_channel=256, stride=[1, 1],
                               padding='SAME', name='diff_conv',
                               data_format=data_format, is_training=is_training, is_bn=True)
        diff_feature = tf.contrib.layers.dropout(inputs=diff_feature, is_training=is_training, keep_prob=0.5)
        net, pred = self._change_judge_layer(inputs=diff_feature, name='Cha_Jud_', data_format=data_format,
                                             is_training=is_training)
        return net, pred

    def _feature_extract_layer(self, inputs, name='Fea_Ext_', data_format='NHWC', is_training=True, is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse:
                scope.reuse_variables()
            # (B, H, W, C) --> (B, H, W, 32)
            layer_1 = conv_2d(inputs=inputs, kernel_size=[3, 3], output_channel=16, stride=[1, 1], name='layer_1_conv',
                              padding='SAME', data_format=data_format, is_training=is_training, is_bn=False,
                              activation=tf.nn.relu)
            layer_2 = conv_2d(inputs=layer_1, kernel_size=[3, 3], output_channel=16, stride=[1, 1], name='layer_2_conv',
                              padding='SAME', data_format=data_format, is_training=is_training, is_bn=False,
                              activation=tf.nn.relu)

            drop_layer_2 = tf.contrib.layers.dropout(inputs=layer_2, is_training=is_training, keep_prob=0.8)
            feature_1 = drop_layer_2

            # (B, H/2, W/2, 32) --> (B, H/2, W/2, 64)
            layer_3 = conv_2d(inputs=drop_layer_2, kernel_size=[3, 3], output_channel=32, stride=[1, 1], padding='SAME',
                              name='layer_3_conv', data_format=data_format, is_training=is_training, is_bn=False,
                              activation=tf.nn.relu)
            layer_4 = conv_2d(inputs=layer_3, kernel_size=[3, 3], output_channel=32, stride=[1, 1], padding='SAME',
                              name='layer_4_conv', data_format=data_format, is_training=is_training, is_bn=False,
                              activation=tf.nn.relu)

            drop_layer_4 = tf.contrib.layers.dropout(inputs=layer_4, is_training=is_training, keep_prob=0.6)
            feature_2 = drop_layer_4
            # (B, H/2, W/2, 64) --> (B, H/2, W/2, 64)
            layer_5 = self._Inception_model_2(inputs=drop_layer_4, name='Inception_1_', stride=[1, 1],
                                              data_format='NHWC',
                                              is_training=is_training, activation=tf.nn.relu)
            layer_6 = self._Inception_model_1(inputs=layer_5, name='Inception_2_', stride=[1, 1], data_format='NHWC',
                                              is_training=is_training, activation=tf.nn.relu)
            drop_layer_6 = tf.contrib.layers.dropout(inputs=layer_6, is_training=is_training, keep_prob=0.5)
            feature_3 = drop_layer_6
            # (B, H, W, 64) --> (B, H, W, 128)
            layer_7 = self._Inception_model_2(inputs=layer_6, name='Inception_3_', data_format='NHWC', stride=[1, 1],
                                          is_training=is_training, activation=tf.nn.relu)
            net = self._Inception_model_1(inputs=layer_7, name='Inception_4_', data_format='NHWC', stride=[1, 1],
                                          is_training=is_training, activation=tf.nn.relu)
           
            return net, feature_1, feature_2,feature_3

    def _change_judge_layer(self, inputs, name='Cha_Jud_', data_format='NHWC', is_training=True):
        with tf.variable_scope(name) as scope:
            # (B, H, W, 256) --> (B, H, W, 512)
            layer_1 = self._Inception_model_2(inputs=inputs, name='Inception_1_', stride=[1, 1],
                                              data_format=data_format,
                                              is_training=is_training,
                                              activation=tf.nn.relu)
          
            layer_2 = tf.contrib.layers.dropout(inputs=layer_1, is_training=is_training, keep_prob=0.5)
            layer_2 = avg_pool_2d(layer_2, kernel_size=[9, 9], stride=[1, 1], padding='VALID')
            if data_format == 'NHWC':
                layer_2 = tf.squeeze(layer_2, axis=[1, 2])
            else:
                layer_2 = tf.squeeze(layer_2, axis=[2, 3])

            logits = fully_connected(layer_2, num_outputs=2, is_training=is_training, is_bn=False)

            pred = tf.nn.softmax(logits)
            return logits, pred

    def _Inception_model_1(self, inputs, name, stride, data_format='NHWC', is_training=True, activation=None):
        """
        Inception model v1, which keep the channel of outputs is same with inputs
        :param inputs: (B, H, W, C)
        :param name: str
        :param data_format: str
        :param is_training: bool
        :return: net, (B, H, W, C)
        """
        with tf.variable_scope(name) as scope:
            if data_format == 'NHWC':
                inputs_channel = inputs.get_shape().as_list()[-1]
                concat_dim = 3
            else:
                inputs_channel = inputs.get_shape().as_list()[1]
                concat_dim = 1
            # 1x1 Conv
            branch_11conv = conv_2d(inputs=inputs, kernel_size=[1, 1], output_channel=inputs_channel // 4,
                                    stride=stride, name='11_conv', padding='SAME', data_format='NHWC',
                                    is_training=is_training, is_bn=False, activation=activation)
            # 3x3 Conv
            branch_33conv = conv_2d(inputs=inputs, kernel_size=[1, 1], output_channel=inputs_channel // 4,
                                    stride=stride, name='33_conv_1', padding='SAME', data_format='NHWC',
                                    is_training=is_training, is_bn=False, activation=activation)
            branch_33conv = conv_2d(inputs=branch_33conv, kernel_size=[3, 3], output_channel=inputs_channel // 2,
                                    stride=stride, name='33_conv_2', padding='SAME', data_format='NHWC',
                                    is_training=is_training, is_bn=False, activation=activation)

            # use two 3x3 conv layer to replace 5x5 conv layer, which can reduce parameter size and improve nonlinear
            branch_55conv = conv_2d(inputs=inputs, kernel_size=[1, 1], output_channel=inputs_channel // 4,
                                    stride=stride, name='55_conv_1', padding='SAME', data_format='NHWC',
                                    is_training=is_training, is_bn=False, activation=activation)
            branch_55conv = conv_2d(inputs=branch_55conv, kernel_size=[3, 3], output_channel=inputs_channel // 8,
                                    stride=stride, name='55_conv_2', padding='SAME', data_format='NHWC',
                                    is_training=is_training, is_bn=False, activation=activation)
            branch_55conv = conv_2d(inputs=branch_55conv, kernel_size=[3, 3], output_channel=inputs_channel // 8,
                                    name='55_conv_3', stride=stride, padding='SAME', data_format='NHWC',
                                    is_training=is_training, is_bn=False, activation=activation)
            # Max Pool
            branch_pool = max_pool_2d(inputs, kernel_size=[3, 3], stride=stride, padding='SAME')
            branch_pool = conv_2d(inputs=branch_pool, kernel_size=[1, 1], output_channel=inputs_channel // 8,
                                  stride=stride, name='max_pool_conv', padding='SAME', data_format='NHWC',
                                  is_training=is_training, is_bn=False, activation=activation)

            net = tf.concat(axis=concat_dim, values=[branch_11conv, branch_33conv, branch_55conv, branch_pool])

            return net

    def _Inception_model_2(self, inputs, name, stride, data_format='NHWC', is_training=True, activation=None):
        """
        Inception model v2, which keep the channel of outputs is twice than inputs
        :param inputs: (B, H, W, C)
        :param name: str
        :param data_format: str
        :param is_training: bool
        :return: net, (B, H, W, 2 * C)
        """
        with tf.variable_scope(name) as scope:
            if data_format == 'NHWC':
                inputs_channel = inputs.get_shape().as_list()[-1]
                concat_dim = 3
            else:
                inputs_channel = inputs.get_shape().as_list()[1]
                concat_dim = 1
            # 1x1 Conv
            branch_11conv = conv_2d(inputs=inputs, kernel_size=[1, 1], output_channel=inputs_channel // 2,
                                    name='11_conv',
                                    stride=stride, padding='SAME', data_format='NHWC', is_training=is_training,
                                    is_bn=False, activation=activation)
            # 3x3 Conv
            branch_33conv = conv_2d(inputs=inputs, kernel_size=[1, 1], output_channel=inputs_channel // 4,
                                    name='33_conv_1',
                                    stride=stride, padding='SAME', data_format='NHWC', is_training=is_training,
                                    is_bn=False, activation=activation)
            branch_33conv = conv_2d(inputs=branch_33conv, kernel_size=[3, 3], output_channel=inputs_channel,
                                    stride=stride, name='33_conv_2',
                                    padding='SAME', data_format='NHWC', is_training=is_training, is_bn=False,
                                    activation=activation)

            # use two 3x3 conv layer to replace 5x5 conv layer, which can reduce parameter size and improve nonlinear
            branch_55conv = conv_2d(inputs=inputs, kernel_size=[1, 1], output_channel=inputs_channel // 4,
                                    name='55_conv_1',
                                    stride=stride, padding='SAME', data_format='NHWC', is_training=is_training,
                                    is_bn=False, activation=activation)
            branch_55conv = conv_2d(inputs=branch_55conv, kernel_size=[3, 3], output_channel=inputs_channel // 4,
                                    name='55_conv_2',
                                    stride=stride, padding='SAME', data_format='NHWC', is_training=is_training,
                                    is_bn=False, activation=activation)
            branch_55conv = conv_2d(inputs=branch_55conv, kernel_size=stride, output_channel=inputs_channel // 4,
                                    name='55_conv_3',
                                    stride=[1, 1], padding='SAME', data_format='NHWC', is_training=is_training,
                                    is_bn=False, activation=activation)
            # Max Pool
            branch_pool = max_pool_2d(inputs, kernel_size=[3, 3], stride=stride, padding='SAME')
            branch_pool = conv_2d(inputs=branch_pool, kernel_size=[1, 1], output_channel=inputs_channel // 4,
                                  name='max_pool_conv',
                                  stride=[1, 1], padding='SAME', data_format='NHWC', is_training=is_training,
                                  is_bn=False, activation=activation)

            net = tf.concat(axis=concat_dim, values=[branch_11conv, branch_33conv, branch_55conv, branch_pool])

            return net
