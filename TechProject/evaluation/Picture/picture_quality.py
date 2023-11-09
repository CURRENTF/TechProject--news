import cv2
import math
import base64
import numpy as np

from io import BytesIO
from tkinter import Image
from PIL import Image

from tools.utils import fix_output_score


#1.计算图片像素密度
def getDensity(picture_list, l):
    if not isinstance(picture_list, list):
        picture_list = [picture_list]
    res = []
    for num, picture in enumerate(picture_list):
        # 读取影像
        byte_picture = base64.b64decode(picture)
        image = Image.open(BytesIO(byte_picture))
        image = np.asarray(image)
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        width = img.shape[0]
        height = img.shape[1]

        density = fix_output_score(math.sqrt(width * width + height * height) / l, _int=True)
        res.append(density)
    return res


#2.曝光检测
def checkExposure(picture_list, down, up):
    if not isinstance(picture_list, list):
        picture_list = [picture_list]
    res = []
    for picture in picture_list:
        # 读取影像
        byte_picture = base64.b64decode(picture)
        image = Image.open(BytesIO(byte_picture))
        image = np.asarray(image)
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        #转换为hsv空间
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(hsv)
        v = V.ravel()[np.flatnonzero(V)]
        #得到平均亮度
        average_v = sum(v) / len(v)

        if down <= average_v <= up:
            # 正常
            res.append(0.0)
        else:
            # 曝光
            res.append(1.0)
    return res


# 3.模糊影像检测函数，阈值默认为0.07
def blurImagesDetection(picture_list, thres=100):
    if not isinstance(picture_list, list):
        picture_list = [picture_list]
    res = []
    for picture in picture_list:
        # 读取影像
        byte_picture = base64.b64decode(picture)
        image = Image.open(BytesIO(byte_picture))
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
        blurness = cv2.Laplacian(img, cv2.CV_64F).var()
        # 如果影像模糊程度小于阈值就将其移动到存放模糊影像的文件夹中
        if blurness < thres:
            res.append(1.0)
        else:
            res.append(0.0)
    return res


#4.色彩检测
def checkColor(picture_list, thres=250):
    if not isinstance(picture_list, list):
        picture_list = [picture_list]
    res = []
    for picture in picture_list:
        # 读取影像
        byte_picture = base64.b64decode(picture)
        image = Image.open(BytesIO(byte_picture))
        np.asarray(image)
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        #获得三个通道
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        r, g, b = np.array(r), np.array(g), np.array(b)
        max_rgb = [b.max(), g.max(), r.max()]
        min_rgb = [b.min(), g.min(), r.min()]

        if max(max_rgb) - min(min_rgb) < thres:
            res.append(0.0)
        else:
            res.append(1.0)
    return res
