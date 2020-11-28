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
    Conv_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_1)
    feature_1 = Conv_1
    Pool_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(Conv_1)

    Conv_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Pool_1)
    Conv_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_2)
    feature_2 = Conv_2
    Merge_2 = Dropout(0.2)(Conv_2)
    Pool_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(Merge_2)

    Conv_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Pool_2)
    Conv_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_3)
    Conv_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_3)

    feature_3 = Conv_3
    Conv_3 = Dropout(0.3)(Conv_3)
    Pool_3 = MaxPooling2D(pool_size=(2, 2), padding='same')(Conv_3)

    Conv_4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Pool_3)
    Conv_4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_4)
    Conv_4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_4)
    feature_4 = Conv_4
    Drop_4 = Dropout(0.5)(Conv_4)
    Pool_4 = MaxPooling2D(pool_size=(2, 2), padding='same')(Drop_4)

    return Pool_4, feature_1, feature_2, feature_3, feature_4


def expansive_path(feature, fea_1, fea_2, fea_3, fea_4):
    layer_1 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(feature))
    concat_layer_1 = Concatenate()([layer_1, fea_4])
    layer_1 = Conv2D(128, 3, strides=[1, 1], activation='relu', padding='same', kernel_initializer='he_normal')(
        concat_layer_1)
    layer_1 = Conv2D(128, 3, strides=[1, 1], activation='relu', padding='same', kernel_initializer='he_normal')(
        layer_1)
    layer_1 = Dropout(0.5)(layer_1)
    layer_1 = Conv2D(64, 3, strides=[1, 1], activation='relu', padding='same', kernel_initializer='he_normal')(
        layer_1)

    # (B, H/8, W/8, 64) --> (B, H/4, W/4, 32)
    layer_2 = Conv2DTranspose(64, 2, strides=[1, 1], activation='relu', padding='same',
                              kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer_1))

    concat_layer_2 = Concatenate()([layer_2, fea_3])
    layer_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        concat_layer_2)
    layer_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_2)
    layer_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_2)
    drop_layer_2 = Dropout(0.4)(layer_2)
    # (B, H/4, W/4, 32) --> (B, H/2, W/2, 16)
    layer_3 = Conv2DTranspose(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop_layer_2))

    concat_layer_3 = Concatenate()([layer_3, fea_2])

    layer_3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat_layer_3)
    layer_3 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer_3)
    drop_layer_3 = Dropout(0.3)(layer_3)
    # (B, H/2, W/2, 16) --> (B, H, W, 1)
    layer_4 = Conv2DTranspose(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop_layer_3))

    concat_layer_4 = Concatenate()([layer_4, fea_1])
    layer_4 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat_layer_4)
    logits = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(layer_4)
    logits = Lambda(squeeze)(logits)
    return logits


def Expand_Dim_Layer(tensor):
    def expand_dim(tensor):
        return K.expand_dims(tensor, axis=1)

    return Lambda(expand_dim)(tensor)


def Negative_layer(tensor):
    return Lambda(negative)(tensor)


def negative(tensor):
    return -tensor


def squeeze(tensor):
    return K.squeeze(tensor, axis=-1)
