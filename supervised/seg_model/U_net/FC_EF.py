import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Lambda, Subtract, Conv2DTranspose, \
    Multiply, GlobalAveragePooling2D
from keras.models import Input, Model


def get_FCEF_model(input_size, pre_weights=None):
    # get a Siamese Encoder
    inputs_tensor = Input(shape=input_size)
    Contract_Path_Model = Model(inputs=[inputs_tensor], outputs=contract_path(inputs_tensor))
    Inputs = Input(shape=input_size)
    net, feature_1, feature_2, feature_3, feature_4 = Contract_Path_Model(Inputs)

    # get a Decoder
    FSEF_model = Model(inputs=Inputs,
                       outputs=expansive_path(net, feature_1, feature_2, feature_3, feature_4))
    return FSEF_model


def Abs_layer(tensor):
    return Lambda(K.abs)(tensor)


def contract_path(Inputs):
    Conv_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Inputs)
    # Conv_1 = BatchNormalization()(Conv_1)
    Conv_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_1)
    feature_1 = Conv_1
    # Conv_1 = BatchNormalization()(Conv_1)
    Pool_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(Conv_1)

    Conv_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Pool_1)
    # Conv_2 = BatchNormalization()(Conv_2)
    Conv_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_2)
    feature_2 = Conv_2
    Merge_2 = Dropout(0.2)(Conv_2)
    # Conv_2 = BatchNormalization()(Conv_2)
    Pool_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(Merge_2)

    # Conv_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Pool_2)
    # Conv_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_3)
    # Conv_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_3)
    Conv_3 = _Inception_model_2(inputs=Pool_2, strides=[1, 1], data_format='NHWC')
    Conv_3 = _Inception_model_1(inputs=Conv_3, strides=[1, 1], data_format='NHWC')
    Conv_3 = _Inception_model_1(inputs=Conv_3, strides=[1, 1], data_format='NHWC')
    feature_3 = Conv_3
    Conv_3 = Dropout(0.3)(Conv_3)
    # Conv_3 = BatchNormalization()(Conv_3)
    Pool_3 = MaxPooling2D(pool_size=(2, 2), padding='same')(Conv_3)

    Conv_4 = _Inception_model_2(inputs=Pool_3, strides=[1, 1], data_format='NHWC')
    Conv_4 = _Inception_model_1(inputs=Conv_4, strides=[1, 1], data_format='NHWC')
    Conv_4 = _Inception_model_1(inputs=Conv_4, strides=[1, 1], data_format='NHWC')

    # Conv_4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Pool_3)
    # # Conv_4 = BatchNormalization()(Conv_4)
    # Conv_4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_4)
    # # Conv_4 = BatchNormalization()(Conv_4)
    # Conv_4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_4)
    feature_4 = Conv_4
    Drop_4 = Dropout(0.5)(Conv_4)
    #Pool_4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=2)(Drop_4)
    Pool_4 = MaxPooling2D(pool_size=(2, 2), padding='same')(Drop_4)

    return Pool_4, feature_1, feature_2, feature_3, feature_4


def expansive_path(feature, fea_1, fea_2, fea_3, fea_4):
    # layer_1 = Conv2DTranspose(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(feature))
    layer_1 = Conv2DTranspose(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(feature)
    # attention_1 = Attention_layer(layer_1)
    # diff_fea_4 = Multiply()([attention_1, fea_4])
    concat_layer_1 = Concatenate()([layer_1, fea_4])

    layer_1 = Conv2D(128, 3, strides=[1, 1], activation='relu', padding='same', kernel_initializer='he_normal')(
        concat_layer_1)
    layer_1 = Conv2D(128, 3, strides=[1, 1], activation='relu', padding='same', kernel_initializer='he_normal')(
        layer_1)
    layer_1 = Dropout(0.5)(layer_1)
    # layer_1 = BatchNormalization()(layer_1)
    layer_1 = Conv2D(64, 3, strides=[1, 1], activation='relu', padding='same', kernel_initializer='he_normal')(
        layer_1)

    # (B, H/8, W/8, 64) --> (B, H/4, W/4, 32)
    layer_2 = Conv2DTranspose(64, 2, strides=[1, 1], activation='relu', padding='same',
                              kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer_1))

    # attention_2 = Attention_layer(layer_2)
    # diff_fea_3 = Multiply()([attention_2, fea_3])
    concat_layer_2 = Concatenate()([layer_2, fea_3])

    layer_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        concat_layer_2)
    layer_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_2)
    # layer_2 = BatchNormalization()(layer_2)
    layer_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_2)
    drop_layer_2 = Dropout(0.4)(layer_2)
    # (B, H/4, W/4, 32) --> (B, H/2, W/2, 16)
    layer_3 = Conv2DTranspose(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop_layer_2))

    # attention_3 = Attention_layer(layer_3)
    # diff_fea_2 = Multiply()([attention_3, fea_2])
    concat_layer_3 = Concatenate()([layer_3, fea_2])

    layer_3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat_layer_3)
    # layer_3 = BatchNormalization()(layer_3)
    layer_3 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_3)
    drop_layer_3 = Dropout(0.3)(layer_3)
    # (B, H/2, W/2, 16) --> (B, H, W, 1)
    layer_4 = Conv2DTranspose(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop_layer_3))

    # attention_4 = Attention_layer(layer_4)
    # diff_fea_1 = Multiply()([attention_4, fea_1])
    concat_layer_4 = Concatenate()([layer_4, fea_1])
    # drop_layer_4 = Dropout(0.2)(concat_layer_4)
    # layer_4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     concat_layer_4)
    # layer_3 = BatchNormalization()(layer_3)
    layer_4 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat_layer_4)
    logits = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(layer_4)
    logits = Lambda(squeeze)(logits)
    # Up_1 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(feature))
    # Merge_1 = Concatenate()([fea_4, Up_1])
    # Deconv_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Merge_1)
    # #  Deconv_1 = BatchNormalization()(Deconv_1)
    # Deconv_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_1)
    # # Deconv_1 = BatchNormalization()(Deconv_1)
    # Deconv_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_1)
    # #  Deconv_1 = BatchNormalization()(Deconv_1)
    # Up_2 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(Deconv_1))
    # Merge_2 = Concatenate(axis=-1)([fea_3, Up_2])
    # Merge_2 = Dropout(0.5)(Merge_2)
    # Deconv_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Merge_2)
    # #  Deconv_2 = BatchNormalization()(Deconv_2)
    # Deconv_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_2)
    # # Deconv_2 = BatchNormalization()(Deconv_2)
    # Deconv_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_2)
    # # Deconv_2 = BatchNormalization()(Deconv_2)
    # Up_3 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(Deconv_2))
    # Merge_3 = Concatenate(axis=-1)([fea_2, Up_3])
    # Merge_3 = Dropout(0.3)(Merge_3)
    # Deconv_3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Merge_3)
    # # Deconv_3 = BatchNormalization()(Deconv_3)
    # Deconv_3 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_3)
    # #  Deconv_3 = BatchNormalization()(Deconv_3)
    # Up_4 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(Deconv_3))
    # Merge_4 = Concatenate(axis=-1)([fea_1, Up_4])
    # Merge_4 = Dropout(0.2)(Merge_4)
    # Deconv_4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Merge_4)
    # # Deconv_4 = BatchNormalization()(Deconv_4)
    # Deconv_4 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_4)
    # # Deconv_4 = BatchNormalization()(Deconv_4)
    # logits = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(Deconv_4)
    # logits = Lambda(squeeze)(logits)
    return logits


def Global_Attention(high_feature, low_feature, low_fea_dim, high_fea_dim):
    weight = GlobalAveragePooling2D()(high_feature)
    weight = Expand_Dim_Layer(tensor=weight)
    weight = Expand_Dim_Layer(tensor=weight)
    weight = Conv2D(high_fea_dim, kernel_size=1, strides=[1, 1], activation='sigmoid', padding='same',
                    kernel_initializer='glorot_uniform')(weight)
    low_feature = Conv2D(low_fea_dim, kernel_size=3, strides=[1, 1], activation='relu',
                         padding='same',
                         kernel_initializer='he_normal')(low_feature)
    weight_low_feature = Multiply()([weight, low_feature])
    return weight_low_feature


def Expand_Dim_Layer(tensor):
    def expand_dim(tensor):
        return K.expand_dims(tensor, axis=1)

    return Lambda(expand_dim)(tensor)


def _Inception_model_1(inputs, strides, data_format='NHWC'):
    """
    Inception model v1, which keep the channel of outputs is same with inputs
    :param inputs: (B, H, W, C)
    :param data_format: str
    :return: net, (B, H, W, C)
    """

    if data_format == 'NHWC':
        inputs_channel = inputs.get_shape().as_list()[-1]

    else:
        inputs_channel = inputs.get_shape().as_list()[1]

    # 1x1 Conv
    branch_11conv = Conv2D(inputs_channel // 4, kernel_size=1, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(inputs)
    # 3x3 Conv
    # branch_33conv = Conv2D(inputs_channel // 4, kernel_size=1, strides=[1, 1], activation='relu', padding='same',
    #                        kernel_initializer='he_normal')(inputs)
    branch_33conv = Conv2D(inputs_channel // 2, kernel_size=3, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(inputs)

    # use two 3x3 conv layer to replace 5x5 conv layer, which can reduce parameter size and improve nonlinear
    branch_55conv = Conv2D(inputs_channel // 4, kernel_size=1, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(inputs)

    branch_55conv = Conv2D(inputs_channel // 8, kernel_size=3, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(branch_55conv)
    branch_55conv = Conv2D(inputs_channel // 8, kernel_size=3, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(branch_55conv)
    # Max Pool
    branch_pool = MaxPooling2D(pool_size=[3, 3], strides=strides, padding='same')(inputs)
    branch_pool = Conv2D(inputs_channel // 8, kernel_size=[1, 1], strides=strides, activation='relu',
                         padding='same', kernel_initializer='he_normal')(branch_pool)

    net = Concatenate()([branch_11conv, branch_33conv, branch_55conv, branch_pool])

    return net


def _Inception_model_2(inputs, strides, data_format='NHWC'):
    """
    Inception model v2, which keep the channel of outputs is twice than inputs
    :param inputs: (B, H, W, C)
    :param data_format: str
    :return: net, (B, H, W, 2 * C)
    """
    if data_format == 'NHWC':
        inputs_channel = inputs.get_shape().as_list()[-1]
        concat_dim = 3
    else:
        inputs_channel = inputs.get_shape().as_list()[1]
        concat_dim = 1
    # 1x1 Conv
    branch_11conv = Conv2D(inputs_channel // 2, 1, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(inputs)
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

    net = Concatenate(axis=concat_dim)([branch_11conv, branch_33conv, branch_55conv, branch_pool])

    return net


def Attention_layer(tensor):
    # fea = Lambda(self.sum_func)(Lambda(K.square)(tensor))
    attention = Negative_layer(Conv2D(1, kernel_size=1, strides=[1, 1], activation='sigmoid', padding='same',
                                      kernel_initializer='glorot_uniform')(tensor))
    return attention


def Negative_layer(tensor):
    return Lambda(negative)(tensor)


def negative(tensor):
    return -tensor


def squeeze(tensor):
    return K.squeeze(tensor, axis=-1)
