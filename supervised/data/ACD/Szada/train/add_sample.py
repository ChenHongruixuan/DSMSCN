import cv2 as cv
import os
import pickle

import gdal
import numpy as np


def read_img_info(path):
    img_data_set = gdal.Open(path)
    img_geo_transform = img_data_set.GetGeoTransform()
    img_projection = img_data_set.GetProjection()
    img_width = img_data_set.RasterXSize
    img_height = img_data_set.RasterYSize
    img_data = img_data_set.ReadAsArray(0, 0, img_width, img_height)
    img_data = np.transpose(img_data, axes=[1, 2, 0])
    return img_data, img_geo_transform, img_projection


def rotate_imgs(img_list, k):
    rot_img_list = []
    for img in img_list:
        rot_img = np.rot90(img, k)
        rot_img_list.append(rot_img)
    return rot_img_list


def flip_imgs(img_list):
    flip_img_list = []
    for img in img_list:
        flip_img = cv.flip(img, flipCode=-1)
        flip_img_list.append(flip_img)
    return flip_img_list


def generate_train_sample():
    for i in range(1, 7):
        img_1 = cv.imread(str(i) + '_im1.bmp')
        img_2 = cv.imread(str(i) + '_im2.bmp')
        gt = cv.imread(str(i) + '_gt.bmp')

        flip_img_1 = cv.flip(img_1, flipCode=-1)
        flip_img_2 = cv.flip(img_2, flipCode=-1)
        flip_gt = cv.flip(gt, flipCode=-1)

        cv.imwrite(str(36 + i) + '_im1.bmp', flip_img_1)
        cv.imwrite(str(36 + i) + '_im2.bmp', flip_img_2)
        cv.imwrite(str(36 + i) + '_gt.bmp', flip_gt)


if __name__ == '__main__':
    generate_train_sample()
