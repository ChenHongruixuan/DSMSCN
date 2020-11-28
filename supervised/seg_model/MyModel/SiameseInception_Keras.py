import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Lambda, Subtract, Conv2DTranspose, \
    Multiply, GlobalAveragePooling2D
from keras.models import Input, Model


class SiameseInception(object):

    def get_model(self, input_size):
        inputs_tensor = Input(shape=input_size)
        Feature_Extract_Model = Model(inputs=[inputs_tensor], outputs=self._feature_extract_layer(inputs_tensor),
                                      name='FEM')
        Inputs_1 = Input(shape=input_size)
        Inputs_2 = Input(shape=input_size)
        net_X, feature_1_X, feature_2_X, feature_3_X, feature_4_X = Feature_Extract_Model(inputs=Inputs_1)
        net_Y, feature_1_Y, feature_2_Y, feature_3_Y, feature_4_Y = Feature_Extract_Model(inputs=Inputs_2)

        # both_net = Concatenate()([net_X, net_Y])
        diff_fea_1 = self.Abs_layer(Subtract()([feature_1_X, feature_1_Y]))  # (B, H, W, 16)
        diff_fea_2 = self.Abs_layer(Subtract()([feature_2_X, feature_2_Y]))  # (B, H/2, W/2, 32)
        diff_fea_3 = self.Abs_layer(Subtract()([feature_3_X, feature_3_Y]))  # (B, H/4, W/4, 64)
        diff_fea_4 = self.Abs_layer(Subtract()([feature_4_X, feature_4_Y]))  # (B, H/8, W/8. 128)

        pred = self._change_judge_layer(inputs=net_Y, diff_fea_1=diff_fea_1, diff_fea_2=diff_fea_2,
                                        diff_fea_3=diff_fea_3, diff_fea_4=diff_fea_4)
        FCI_model = Model(inputs=[Inputs_1, Inputs_2], outputs=pred)
        return FCI_model

    def _feature_extract_layer(self, inputs):
        """
        feature extraction layer
        :param inputs: (B, H, W, C)
        :return:
            net: (B, H/16, W/16, 256)
            feature_1: (B, H, W, 16)
            feature_2: (B, H/2, W/16, 32)
            feature_3: (B, H/4, W/16, 64)
            feature_4: (B, H/8, W/16, 128)
        """
        # (B, H, W, C) --> (B, H/2, W/2, 16)
        layer_1 = Conv2D(16, kernel_size=3, strides=[1, 1], activation='relu', padding='same',
                         kernel_initializer='he_normal', name='Conv_1')(inputs)
        layer_1 = Conv2D(16, kernel_size=3, strides=[1, 1], activation='relu', padding='same',
                         kernel_initializer='he_normal', name='Conv_2')(layer_1)
        # layer_1 = BatchNormalization()(layer_1)
        feature_1 = layer_1
        layer_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='Max_Pool_1')(layer_1)
        # drop_layer_1 = Dropout(0.2)(layer_1)

        # (B, H/2, W/2, 16) --> (B, H/4, W/4, 32)
        layer_2 = Conv2D(32, kernel_size=3, strides=[1, 1], activation='relu', padding='same',
                         kernel_initializer='he_normal', name='Conv_3')(layer_1)
        layer_2 = Conv2D(32, kernel_size=3, strides=[1, 1], activation='relu', padding='same',
                         kernel_initializer='he_normal', name='Conv_4')(layer_2)
        #  layer_2 = BatchNormalization()(layer_2)
        feature_2 = layer_2
        layer_2 = Dropout(0.2)(layer_2)
        layer_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='Max_Pool_2')(layer_2)

        # (B, H/4, W/4, 32) --> (B, H/8, W/8, 64)
        layer_3 = self._Inception_model_2(inputs=layer_2, strides=[1, 1], data_format='NHWC')
        layer_3 = self._Inception_model_1(inputs=layer_3, strides=[1, 1], data_format='NHWC')
        # layer_3 = self._Inception_model_1(inputs=layer_3, strides=[1, 1], data_format='NHWC')
        # layer_3 = Conv2D(64, kernel_size=1, strides=[1, 1], padding='same',
                         # kernel_initializer='he_normal', name='Conv_111')(layer_3)
        # layer_3 = BatchNormalization()(layer_3)
        feature_3 = layer_3
        layer_3 = Dropout(0.4)(layer_3)
        layer_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='Max_Pool_3')(layer_3)

        # (B, H/8, W/8, 64) --> (B, H/16, W/16, 128)
        layer_4 = self._Inception_model_2(inputs=layer_3, strides=[1, 1], data_format='NHWC')
        layer_4 = self._Inception_model_1(inputs=layer_4, strides=[1, 1], data_format='NHWC')
        # layer_4 = self._Inception_model_1(inputs=layer_4, strides=[1, 1], data_format='NHWC')
        # layer_4 = Conv2D(128, kernel_size=1, strides=[1, 1], padding='same',
                        #  kernel_initializer='he_normal', name='Conv_112')(layer_4)
        # layer_4 = BatchNormalization()(layer_4)
        feature_4 = layer_4
        layer_4 = Dropout(0.5)(layer_4)
        net = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='Max_Pool_4')(layer_4)

        return net, feature_1, feature_2, feature_3, feature_4

    def _change_judge_layer(self, inputs, diff_fea_1, diff_fea_2, diff_fea_3, diff_fea_4):
        # (B, H/16, W/16, 128) --> (B, H/8, W/8, 64)
        layer_1 = Conv2DTranspose(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(inputs))

        # attention_1 = self.Attention_layer(layer_1)
        #  diff_fea_4 = Multiply()([attention_1, diff_fea_4])
        concat_layer_1 = Concatenate()([layer_1, diff_fea_4])

        # layer_1 = Conv2D(128, 3, strides=[1, 1], activation='relu', padding='same', kernel_initializer='he_normal')(
        #     concat_layer_1)

        layer_1 = Conv2D(128, 3, strides=[1, 1], activation='relu', padding='same', kernel_initializer='he_normal')(
            concat_layer_1)
        # layer_1 = BatchNormalization()(layer_1)
        layer_1 = Dropout(0.5)(layer_1)
        layer_1 = Conv2D(64, 3, strides=[1, 1], activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_1)

        # (B, H/8, W/8, 64) --> (B, H/4, W/4, 32)
        layer_2 = Conv2DTranspose(64, 2, strides=[1, 1], activation='relu', padding='same',
                                  kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(layer_1))

        # attention_2 = self.Attention_layer(layer_2)
        # diff_fea_3 = Multiply()([attention_2, diff_fea_3])
        concat_layer_2 = Concatenate()([layer_2, diff_fea_3])

        # layer_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     concat_layer_2)
        layer_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat_layer_2)
        # layer_2 = BatchNormalization()(layer_2)
        layer_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_2)
        drop_layer_2 = Dropout(0.4)(layer_2)
        # (B, H/4, W/4, 32) --> (B, H/2, W/2, 16)
        layer_3 = Conv2DTranspose(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop_layer_2))

        # attention_3 = self.Attention_layer(layer_3)
        # diff_fea_2 = Multiply()([attention_3, diff_fea_2])
        concat_layer_3 = Concatenate()([layer_3, diff_fea_2])

        layer_3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat_layer_3)
        # layer_3 = BatchNormalization()(layer_3)
        layer_3 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_3)
        drop_layer_3 = Dropout(0.3)(layer_3)
        # (B, H/2, W/2, 16) --> (B, H, W, 1)
        layer_4 = Conv2DTranspose(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop_layer_3))

        # attention_4 = self.Attention_layer(layer_4)
        # diff_fea_1 = Multiply()([attention_4, diff_fea_1])
        concat_layer_4 = Concatenate()([layer_4, diff_fea_1])
        # drop_layer_4 = Dropout(0.2)(concat_layer_4)
        # layer_4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     concat_layer_4)
        # layer_3 = BatchNormalization()(layer_3)
        layer_4 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat_layer_4)
        logits = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(layer_4)
        logits = Lambda(self.squeeze)(logits)
        return logits

    def squeeze(self, tensor):
        return K.squeeze(tensor, axis=-1)

    def sum_func(self, tensor):
        return K.sum(tensor, axis=-1, keepdims=True)

    def Abs_layer(self, tensor):
        return Lambda(K.abs)(tensor)


    def Negative_layer(self, tensor):
        return Lambda(self.negative)(tensor)

    def negative(self, tensor):
        return -tensor

    def _Inception_model_1(self, inputs, strides, data_format='NHWC'):
        """
        Inception model v1, which keep the channel of outputs is same with inputs
        :param inputs: (B, H, W, C)
        :param data_format: str
        :return: net, (B, H, W, C)
        """
        # attention = tf.Variable(initial_value=[1, 1, 1, 1], dtype=tf.float32)
        if data_format == 'NHWC':
            inputs_channel = inputs.get_shape().as_list()[-1]

        else:
            inputs_channel = inputs.get_shape().as_list()[1]

        # 1x1 Conv
        branch_11conv = Conv2D(inputs_channel // 4, kernel_size=1, strides=strides, activation='relu',
                               padding='same',
                               kernel_initializer='he_normal')(inputs)
        # 3x3 Conv
        # branch_33conv = Conv2D(inputs_channel // 4, kernel_size=1, strides=[1, 1], activation='relu', padding='same',
        #                        kernel_initializer='he_normal')(inputs)
        branch_33conv = Conv2D(inputs_channel // 2, kernel_size=3, strides=strides, activation='relu',
                               padding='same',
                               kernel_initializer='he_normal')(inputs)
        # use two 3x3 conv layer to replace 5x5 conv layer, which can reduce parameter size and improve nonlinear
        branch_55conv = Conv2D(inputs_channel // 4, kernel_size=1, strides=strides, activation='relu',
                               padding='same',
                               kernel_initializer='he_normal')(inputs)
        branch_55conv = Conv2D(inputs_channel // 8, kernel_size=3, strides=strides, activation='relu',
                               padding='same',
                               kernel_initializer='he_normal')(branch_55conv)
        branch_55conv = Conv2D(inputs_channel // 8, kernel_size=3, strides=strides, activation='relu',
                               padding='same',
                               kernel_initializer='he_normal')(branch_55conv)
        # branch_55conv = Multiply()([attention[2], branch_55conv])
        # Max Pool
        branch_pool = MaxPooling2D(pool_size=[3, 3], strides=strides, padding='same')(inputs)
        branch_pool = Conv2D(inputs_channel // 8, kernel_size=[1, 1], strides=strides, activation='relu',
                             padding='same', kernel_initializer='he_normal')(branch_pool)
        # branch_pool = Multiply()([attention[3], branch_pool])

        net = Concatenate()([branch_11conv, branch_33conv, branch_55conv, branch_pool])

        return net

    def _Inception_model_2(self, inputs, strides, data_format='NHWC'):
        """
        Inception model v2, which keep the channel of outputs is twice than inputs
        :param inputs: (B, H, W, C)
        :param data_format: str
        :return: net, (B, H, W, 2 * C)
        """
        # attention = tf.Variable(initial_value=[1, 1, 1, 1], dtype=tf.float32)
        if data_format == 'NHWC':
            inputs_channel = inputs.get_shape().as_list()[-1]
            concat_dim = 3
        else:
            inputs_channel = inputs.get_shape().as_list()[1]
            concat_dim = 1
        # 1x1 Conv
        branch_11conv = Conv2D(inputs_channel // 2, 1, strides=strides, activation='relu', padding='same',
                               kernel_initializer='he_normal')(inputs)
        # branch_11conv = Multiply()([attention[0], branch_11conv])
        # 3x3 Conv
        # branch_33conv = Conv2D(inputs_channel // 2, 1, strides=strides, activation='relu', padding='same',
        #                        kernel_initializer='he_normal')(inputs)
        branch_33conv = Conv2D(inputs_channel, 3, strides=strides, activation='relu', padding='same',
                               kernel_initializer='he_normal')(inputs)
        # use two 3x3 conv layer to replace 5x5 conv layer, which can reduce parameter size and improve nonlinear

        branch_55conv = Conv2D(inputs_channel // 2, 1, strides=strides, activation='relu', padding='same',
                               kernel_initializer='he_normal')(inputs)

        branch_55conv = Conv2D(inputs_channel // 4, 3, strides=strides, activation='relu', padding='same',
                               kernel_initializer='he_normal')(branch_55conv)
        branch_55conv = Conv2D(inputs_channel // 4, 3, strides=strides, activation='relu', padding='same',
                               kernel_initializer='he_normal')(branch_55conv)
        # Max Pool
        branch_pool = MaxPooling2D(pool_size=[3, 3], strides=strides, padding='same')(inputs)
        branch_pool = Conv2D(inputs_channel // 4, 1, strides=strides, activation='relu', padding='same',
                             kernel_initializer='he_normal')(branch_pool)
        # branch_pool = Multiply()([attention[3], branch_pool])
        net = Concatenate(axis=concat_dim)([branch_11conv, branch_33conv, branch_55conv, branch_pool])

        return net


    def Expand_Dim_Layer(self, tensor):
        def expand_dim(tensor):
            return K.expand_dims(tensor, axis=1)

        return Lambda(expand_dim)(tensor)

    def get_loss(self, label, logits, pos_weight):
        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets=label, logits=logits, pos_weight=pos_weight,
                                                     name='weight_loss'))
        return loss
