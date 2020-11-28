import argparse
import cv2 as cv
import os
import pickle

import numpy as np
from keras.optimizers import Adam

from .acc_util import Recall, Precision, F1_score
from .acc_ass import accuracy_assessment
from .net_util import weight_binary_cross_entropy
from .seg_model.MyModel.SiameseInception_Keras import SiameseInception

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=250, help='epoch to run[default: 100]')
parser.add_argument('--batch_size', type=int, default=1, help='batch size during training[default: 512]')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='initial learning rate[default: 1e-4]')
parser.add_argument('--result_save_path', default='result/ACD', help='model param path')
parser.add_argument('--model_save_path', default='model_param/ACD', help='model param path')
parser.add_argument('--data_path', default='data/ACD', help='data path')
parser.add_argument('--data_set_name', default='Szada', help='dataset name')
parser.add_argument('--gpu_num', type=int, default=1, help='number of GPU to train')

# basic params
FLAGS = parser.parse_args()

BATCH_SZ = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
MAX_EPOCH = FLAGS.max_epoch
RESULT_SAVE_PATH = FLAGS.result_save_path
MODEL_SAVE_PATH = FLAGS.model_save_path
DATA_PATH = FLAGS.data_path
DATA_SET_NAME = FLAGS.data_set_name
GPU_NUM = FLAGS.gpu_num
BATCH_PER_GPU = BATCH_SZ // GPU_NUM


def train_model():
    result_path = os.path.join(RESULT_SAVE_PATH, DATA_SET_NAME)
    model_save_path = os.path.join(MODEL_SAVE_PATH, DATA_SET_NAME)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_X, train_Y, train_label, test_X, test_Y, test_label = get_data()
    si_model = SiameseInception()
    deep_model = si_model.get_model(input_size=[None, None, 3])
    opt = Adam(lr=LEARNING_RATE)
    deep_model.compile(optimizer=opt,
                       loss=weight_binary_cross_entropy, metrics=['accuracy', Recall, Precision, F1_score])
    deep_model.summary()

    best_loss = 1000
    best_kappa = 0
    best_F1 = 0
    best_acc = 0
    save_best = True
    iter_on_epoch = len(train_X)

    for _epoch in range(MAX_EPOCH):
        train_loss, train_acc, train_sen, train_spe, train_F1 = 0, 0, 0, 0, 0
        for i in range(iter_on_epoch):
            train_1 = np.reshape(train_X[i], (1, train_X[i].shape[0], train_X[i].shape[1], train_X[i].shape[2]))
            train_2 = np.reshape(train_Y[i], (1, train_Y[i].shape[0], train_Y[i].shape[1], train_Y[i].shape[2]))

            label = np.reshape(train_label[i],
                               (1, train_label[i].shape[0], train_label[i].shape[1]))

            temp_loss, temp_train_acc, temp_train_sen, temp_train_spe, temp_train_F1 = \
                deep_model.train_on_batch(x=[train_1, train_2], y=label)
            train_loss += temp_loss
            train_acc += temp_train_acc
            train_sen += temp_train_sen
            train_spe += temp_train_spe
            train_F1 += temp_train_F1

        train_loss /= iter_on_epoch
        train_acc /= iter_on_epoch
        train_sen /= iter_on_epoch
        train_spe /= iter_on_epoch
        train_F1 /= iter_on_epoch

        train_info = 'epoch %d, train loss is %.4f,  train sen is %.4f, train spe is %.4f, train F1 is %.4f, ' \
                     'train acc is %.4f' % (_epoch, train_loss, train_sen, train_spe, train_F1, train_acc)
        print(train_info)

        loss, acc, sen, spe, F1 = deep_model.evaluate(x=[test_X, test_Y], y=test_label, batch_size=1)

        binary_change_map = deep_model.predict([test_X, test_Y])
        binary_change_map = np.reshape(binary_change_map, (binary_change_map.shape[1], binary_change_map.shape[2]))
        idx_1 = binary_change_map > 0.5
        idx_2 = binary_change_map <= 0.5
        binary_change_map[idx_1] = 255
        binary_change_map[idx_2] = 0

        conf_mat, overall_acc, kappa = accuracy_assessment(
            gt_changed=np.reshape(255 * test_label, (test_label.shape[1], test_label.shape[2])),
            gt_unchanged=np.reshape(255. - 255 * test_label, (test_label.shape[1], test_label.shape[2])),
            changed_map=binary_change_map)

        test_info = 'epoch %d, test loss is %.4f,  test sen is %.4f, test spe is %.4f, test F1 is %.4f, ' \
                    'test acc is %.4f, test kappa is %.4f, ' % (
                        _epoch, loss, sen, spe, F1, overall_acc, kappa)
        print(test_info)
        print('confusion matrix is ', conf_mat)

        with open(os.path.join(result_path, 'log.txt'), 'a+') as f:
            f.write(train_info + '\n')
            f.write(test_info + '\n')
        if best_acc < overall_acc:
            best_acc = overall_acc
            save_best = True
        if loss < best_loss:
            best_loss = loss
            save_best = True
        if kappa > best_kappa:
            best_kappa = kappa
            save_best = True
        if F1 > best_F1:
            best_F1 = F1
            save_best = True
        if save_best:
            cv.imwrite(os.path.join(result_path, str(_epoch) + '_bcm.bmp'), binary_change_map)
            save_best = False
            deep_model.save(os.path.join(model_save_path, str(_epoch) + '_model.h5'))

    best_info = 'best loss is %.4f, best F1 is %.4f, best acc is %.4f, best kappa is %.4f, ' % (
        best_loss, best_F1, best_acc, best_kappa)
    print('train is done, ' + best_info)
    param_info = 'learning rate ' + LEARNING_RATE + ', Max Epoch is ' + MAX_EPOCH
    print('parameter is, ' + param_info)
    with open(os.path.join(result_path, 'log.txt'), 'a+') as f:
        f.write(best_info + '\n')
        f.write(param_info + '\n')


def get_data():
    path = os.path.join(DATA_PATH, DATA_SET_NAME)
    train_X, train_Y, train_label = load_train_data(path=path)
    test_X, test_Y, test_label = load_test_data(path=path)

    test_X = np.array(test_X) / 255.
    test_Y = np.array(test_Y) / 255.
    test_label = np.array(test_label) / 255.
    test_label = np.reshape(test_label, (test_label.shape[0], test_label.shape[1], test_label.shape[2]))
    for i in range(len(train_X)):
        train_X[i] = train_X[i] / 255.
        train_Y[i] = train_Y[i] / 255.
        train_label[i] = train_label[i] / 255.

    return train_X, train_Y, train_label, test_X, test_Y, test_label


def load_train_data(path):
    with open(os.path.join(path, 'train_sample_1.pickle'), 'rb') as file:
        train_X = pickle.load(file)
    with open(os.path.join(path, 'train_sample_2.pickle'), 'rb') as file:
        train_Y = pickle.load(file)
    with open(os.path.join(path, 'train_label.pickle'), 'rb') as file:
        train_label = pickle.load(file)

    return train_X, train_Y, train_label


def load_test_data(path):
    with open(os.path.join(path, 'test_sample_1.pickle'), 'rb') as file:
        test_X = pickle.load(file)
    with open(os.path.join(path, 'test_sample_2.pickle'), 'rb') as file:
        test_Y = pickle.load(file)
    with open(os.path.join(path, 'test_label.pickle'), 'rb') as file:
        test_label = pickle.load(file)
    return test_X, test_Y, test_label


def norm_img(img):
    """
    normalization image
    :param img: (C, H, W)
    :return:
        norm_img: (C, H, W)
    """
    height, width, channel = img.shape
    img = np.reshape(img, (height * width, channel))  # (height * width, channel)
    mean = np.mean(img, axis=0, keepdims=True)  # (1, channel)
    center = img - mean  # (height * width, channel)
    var = np.sum(np.power(center, 2), axis=0, keepdims=True) / (height * width)  # (1, channel)
    std = np.sqrt(var)  # (1, channel)
    _norm_img = center / std  # (height * width, channel)
    _norm_img = np.reshape(_norm_img, (height, width, channel))
    return _norm_img


if __name__ == '__main__':
    train_model()
