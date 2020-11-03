import imageio
import gdal
import numpy as np
import tensorflow as tf

from DSMSCN.model.DSMSCN import DSMSCN


def norm_img(img):
    """
    normalization image
    :param img: (C, H, W)
    :return:
        norm_img: (C, H, W)
    """
    channel, img_height, img_width = img.shape
    img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
    mean = np.mean(img, axis=1, keepdims=True)  # (channel, 1)
    center = img - mean  # (channel, height * width)
    var = np.sum(np.power(center, 2), axis=1, keepdims=True) / (img_height * img_width)  # (channel, 1)
    std = np.sqrt(var)  # (channel, 1)
    nm_img = center / std  # (channel, height * width)
    nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    return nm_img


def load_data():
    data_set_X = gdal.Open('data/GF_2_2/T1')  # data set X
    data_set_Y = gdal.Open('data/GF_2_2/T2')  # data set Y

    img_width = data_set_X.RasterXSize  # image width
    img_height = data_set_X.RasterYSize  # image height

    img_X = data_set_X.ReadAsArray(0, 0, img_width, img_height)
    img_Y = data_set_Y.ReadAsArray(0, 0, img_width, img_height)

    img_X = norm_img(img_X)  # (C, H, W)
    img_Y = norm_img(img_Y)
    img_X = np.transpose(img_X, [1, 2, 0])  # (H, W, C)
    img_Y = np.transpose(img_Y, [1, 2, 0])  # (H, W, C)
    return img_X, img_Y


def load_model():
    patch_sz = 13
    edge = patch_sz // 2
    img_X, img_Y = load_data()
    img_X = np.pad(img_X, ((edge, edge), (edge, edge), (0, 0)), 'constant')
    img_Y = np.pad(img_Y, ((edge, edge), (edge, edge), (0, 0)), 'constant')
    img_height, img_width, channel = img_X.shape  # image width

    sample_X = []
    sample_Y = []
    for i in range(edge, img_height - edge):
        for j in range(edge, img_width - edge):
            sample_X.append(img_X[i - edge:i + edge + 1, j - edge:j + edge + 1, :])
            sample_Y.append(img_Y[i - edge:i + edge + 1, j - edge:j + edge + 1, :])
    sample_X = np.array(sample_X, dtype=np.float32)
    sample_Y = np.array(sample_Y, dtype=np.float32)

    epoch = sample_X.shape[0]

    Input_X = tf.placeholder(dtype=tf.float32, shape=[None, patch_sz, patch_sz, 4], name='Input_X')
    Input_Y = tf.placeholder(dtype=tf.float32, shape=[None, patch_sz, patch_sz, 4], name='Input_Y')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    model_path = 'model_param/dis'
    model = DSMSCN()
    net, pred = model.get_model(Input_X=Input_X, Input_Y=Input_Y, is_training=is_training)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    pred_results = []
    path = None
    print('start inferring')
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            path = ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)
        for _epoch in range(1000):
            pred_value = sess.run([pred], feed_dict={
                Input_X: sample_X[1000 * _epoch:1000 * (_epoch + 1)],
                Input_Y: sample_Y[1000 * _epoch:1000 * (_epoch + 1)],
                is_training: False
            })
            pred_value = np.argmax(pred_value, axis=-1)
            pred_results.append(pred_value)
    pred_results = np.array(pred_results)
    pred_results = np.reshape(pred_results, (1000, 1000)) 
    pred_results[pred_results == 1] = 255
    imageio.imsave('CIM.bmp', pred_results)

if __name__ == '__main__':
    load_model()
