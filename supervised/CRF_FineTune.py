import argparse
import cv2 as cv
import os
import pickle

import time
import numpy as np
import pydensecrf.densecrf as dcrf
from keras.models import load_model
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian
import .acc_util as au
from .acc_ass import accuracy_assessment
from .net_util import weight_binary_cross_entropy
from keras.layers import Lambda
import keras.backend as K
from .seg_model.MyModel.SiameseInception_Keras import SiameseInception

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/ACD', help='data path')
parser.add_argument('--data_set_name', default='Szada', help='dataset name')
# basic params
FLAGS = parser.parse_args()

DATA_PATH = FLAGS.data_path
DATA_SET_NAME = FLAGS.data_set_name


def fine_tune_result():
    path = os.path.join(DATA_PATH, DATA_SET_NAME)
    test_X, test_Y, test_label = load_test_data(path=path)
    test_X = np.array(test_X) / 255.
    test_Y = np.array(test_Y) / 255.
    test_label = np.array(test_label) / 255.
    test_label = np.reshape(test_label, (test_label.shape[0], test_label.shape[1], test_label.shape[2]))
    # test = np.concatenate([test_X, test_Y], axis=-1)

    # MS_model = load_model("393_model.h5",
    #                       custom_objects={
    #                           'weight_binary_cross_entropy': weight_binary_cross_entropy,
    #                           'Recall': au.Recall,
    #                           'Precision': au.Precision,
    #                           'F1_score': au.F1_score
    #                       })

    #
    Network = SiameseInception()
    MS_model = Network.get_model(input_size=[None, None, 3])
    MS_model.load_weights("./model_param/ACD/Szada/DSMSFCN/27_model.h5", by_name=True)
    MS_model.compile(optimizer='Adam', loss=weight_binary_cross_entropy,
                     metrics=['accuracy', au.Recall, au.Precision, au.F1_score])
    loss, acc, sen, spe, F1 = MS_model.evaluate(x=[test_X, test_Y], y=test_label, batch_size=1)

    tic = time.time()
    for i in range(0, 1):
        change_prob = MS_model.predict([test_X, test_Y])
    toc = time.time()
    change_prob_2 = np.array(255 * change_prob, dtype=np.uint8)
    cv.imwrite('CIM.bmp', change_prob_2[0])

    print('network time: ', (toc - tic))

    diff = np.abs(test_X - test_Y)
    diff = 255 * (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    image = np.squeeze(diff, axis=0)
    test_label = np.squeeze(test_label, axis=0)

    binary_change_map = np.copy(np.reshape(change_prob, (change_prob.shape[1], change_prob.shape[2])))
    idx_1 = binary_change_map > 0.5
    idx_2 = binary_change_map <= 0.5
    binary_change_map[idx_1] = 255
    binary_change_map[idx_2] = 0

    conf_mat, overall_acc, kappa = accuracy_assessment(
        gt_changed=np.reshape(255 * test_label, (test_label.shape[0], test_label.shape[1])),
        gt_unchanged=np.reshape(255. - 255 * test_label, (test_label.shape[0], test_label.shape[1])),
        changed_map=binary_change_map)
    print(conf_mat)

    info = 'loss is %.4f,  sen is %.4f, spe is %.4f, F1 is %.4f, acc is %.4f, kappa is %.4f, ' % (
        loss, sen, spe, F1, overall_acc, kappa)
    print(info)

    # change_prob = np.expand_dims(change_prob, axis=-1)
    unchange_prob = 1. - change_prob
    softmax_result = np.concatenate([change_prob, unchange_prob], axis=0)
    # softmax_result = np.transpose(softmax_result, axes=[2, 0, 1])

    # unary potential
    # unary = softmax_to_unary(softmax_result)
    # unary = np.ascontiguousarray(unary)  # (2, n)
    unary = -np.log(softmax_result)
    unary = unary.reshape((2, -1))
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(binary_change_map.shape[1] * binary_change_map.shape[0], 2)

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=binary_change_map.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(7, 7), schan=(10, 10, 5),
                                      img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=4,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    tic = time.time()
    for i in range(0, 1):
        Q = d.inference(20)
    toc = time.time()
    print('FC-CRF time: ', (toc - tic))
    res = np.argmax(Q, axis=0).reshape((binary_change_map.shape[0], binary_change_map.shape[1]))

    recall = Recall(y_true=test_label, y_pred=1. - res)
    pre = Precision(y_true=test_label, y_pred=1. - res)
    f1 = F1_score(y_true=test_label, y_pred=1. - res)
    print(recall, pre, f1)

    idx_1 = res == 0
    idx_2 = res == 1
    res[idx_1] = 255
    res[idx_2] = 0
    cv.imwrite('another_result.bmp', res)
    conf_mat, overall_acc, kappa = accuracy_assessment(
        gt_changed=np.reshape(255 * test_label, (test_label.shape[0], test_label.shape[1])),
        gt_unchanged=np.reshape(255. - 255 * test_label, (test_label.shape[0], test_label.shape[1])),
        changed_map=res)
    print(conf_mat)
    print(overall_acc, kappa)


def load_test_data(path):
    with open(os.path.join(path, 'test_sample_1.pickle'), 'rb') as file:
        test_X = pickle.load(file)
    with open(os.path.join(path, 'test_sample_2.pickle'), 'rb') as file:
        test_Y = pickle.load(file)
    with open(os.path.join(path, 'test_label.pickle'), 'rb') as file:
        test_label = pickle.load(file)

    return test_X, test_Y, test_label


def Abs_layer(tensor):
    return Lambda(K.abs)(tensor)


def Recall(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + 1e-8)


def Precision(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_negatives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    return true_positives / (possible_negatives + 1e-8)


def F1_score(y_true, y_pred):
    R = Recall(y_true, y_pred)
    P = Precision(y_true, y_pred)
    return 2 * P * R / (R + P)


if __name__ == '__main__':
    fine_tune_result()
