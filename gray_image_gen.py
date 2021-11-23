import os
import cv2
import numpy as np


def read_path(file_pathname):
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv2.imread(file_pathname+'/'+filename, 1)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(r"C:\Users\dcs\Desktop\images" + '/' + filename, gray_img)


read_path(r"C:\Users\dcs\Desktop\hw\445pro2\starter_code111\data\images")
