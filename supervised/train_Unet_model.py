import argparse
import cv2 as cv
import os
import pickle

import numpy as np
from keras.optimizers import Adam

from DSMSFCN.acc_util import Recall, Precision, F1_score
from DSMSFCN.seg_model.U_net.FC_Siam_Diff import get_FCSD_model
from DSMSFCN.seg_model.U_net.FC_Siam_Conc import get_FCSC_model
from DSMSFCN.seg_model.U_net.FC_EF import get_FCEF_model
from acc_ass import assess_accuracy
from net_util import weight_binary_cross_entropy
from scipy import misc

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=200, help='epoch to run[default: 100]')
parser.add_argument('--batch_size', type=int, default=1, help='batch size during training[default: 1]')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='initial learning rate[default: 3e-4]')
parser.add_argument('--result_save_path', default='result', help='model param path')
parser.add_argument('--model_save_path', default='model_param', help='model param path')
parser.add_argument('--data_path', default='data', help='data path')
parser.add_argument('--data_set_name', default='ACD/Szada', help='dataset name')
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
    train_X, train_Y, train_label, test_X, test_Y, test_label = get_data()
    test = np.concatenate([test_X, test_Y], axis=-1)
    FCEF_model = get_FCEF_model(input_size=[None, None, 6])
    opt = Adam(lr=LEARNING_RATE)
    FCEF_model.compile(optimizer=opt,
                       loss=weight_binary_cross_entropy, metrics=['accuracy', Recall, Precision, F1_score])
    FCEF_model.summary()

    best_acc = 0
    best_loss = 1000
    best_kappa = 0
    best_F1 = 0
    save_best = True
    for _epoch in range(MAX_EPOCH):
        for i in range(len(train_X)):
            train_1 = np.reshape(train_X[i], (1, train_X[i].shape[0], train_X[i].shape[1], train_X[i].shape[2]))
            train_2 = np.reshape(train_Y[i], (1, train_Y[i].shape[0], train_Y[i].shape[1], train_Y[i].shape[2]))
            train = np.concatenate([train_1, train_2], axis=-1)
            label = np.reshape(train_label[i],
                               (1, train_label[i].shape[0], train_label[i].shape[1]))

            FCEF_model.train_on_batch(x=train, y=label)

        loss, acc, sen, spe, F1 = FCEF_model.evaluate(x=test, y=test_label, batch_size=1)

        binary_change_map = FCEF_model.predict(test)
        binary_change_map = np.reshape(binary_change_map, (binary_change_map.shape[1], binary_change_map.shape[2]))
        idx_1 = binary_change_map > 0.5
        idx_2 = binary_change_map <= 0.5
        binary_change_map[idx_1] = 255
        binary_change_map[idx_2] = 0

        conf_mat, overall_acc, kappa = accuracy_assessment(
            gt_changed=np.reshape(255 * test_label, (test_label.shape[1], test_label.shape[2])),
            gt_unchanged=np.reshape(255. - 255 * test_label, (test_label.shape[1], test_label.shape[2])),
            changed_map=binary_change_map)

        info = 'epoch %d, loss is %.4f,  sen is %.4f, spe is %.4f, F1 is %.4f, acc is %.4f, kappa is %.4f, ' % (
            _epoch, loss, sen, spe, F1, overall_acc, kappa)
        print(info)
        print('confusion matrix is ', conf_mat)

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
            with open(os.path.join(RESULT_SAVE_PATH, os.path.join(DATA_SET_NAME, 'log.txt')), 'a+') as f:
                f.write(info + '\n')
            cv.imwrite(os.path.join(RESULT_SAVE_PATH, os.path.join(DATA_SET_NAME, str(_epoch) + '_bcm.bmp')),
                       binary_change_map)
            FCEF_model.save(os.path.join(MODEL_SAVE_PATH, os.path.join(DATA_SET_NAME, str(_epoch) + '_model.h5')))
            save_best = False

    best_info = 'best loss is %.4f, best F1 is %.4f, best acc is %.4f, best kappa is %.4f, ' % (
        best_loss, best_F1, best_acc, best_kappa)
    print('train is done, ' + best_info)
    param_info = 'learning rate ' + str(LEARNING_RATE) + ', Max Epoch is ' + str(MAX_EPOCH)
    print('parameter is: ' + param_info)
    with open(os.path.join(RESULT_SAVE_PATH, os.path.join(DATA_SET_NAME, 'log.txt')), 'a+') as f:
        f.write(best_info + '\n')
        f.write(param_info + '\n')


def get_data():
    path = os.path.join(DATA_PATH, DATA_SET_NAME)
    train_X, train_Y, train_label = load_train_data(path=path)
    test_X, test_Y, test_label = load_test_data(path=path)

    test_X = np.array(test_X) 
    test_Y = np.array(test_Y)
    test_label = np.array(test_label)
    test_label = np.reshape(test_label, (test_label.shape[0], test_label.shape[1], test_label.shape[2]))
    for i in range(len(train_X)):
        train_X[i] = train_X[i] 
        train_Y[i] = train_Y[i]
        train_label[i] = train_label[i]

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


if __name__ == '__main__':
    train_model()
