import base64
import os
from evaluation.baidu_api import *
from picture_quality import *

path = "D:\python\project\TechProject\evaluation\Picture\picture"

# imageNameList = os.listdir(path)
# picture_list = []
# for imageName in imageNameList:
#     imagePath = os.path.join(path, imageName)
#     #print(imagePath)
#         # 读取影像
#     # img = open(imagePath, 'rb').read()
#     img = Image.open(imagePath)
#     bytesIO = BytesIO()
#     img.save(bytesIO, format("PNG"))
#     picture_list.append(bytesIO.getvalue())
#     # picture_list.append(img)
#
# score1 = getDensity(picture_list, 27)
# #没有测试没有测试没有测试，测的全是合规图片
# print(len(picture_list))
# print(score1)


with open('D:/python/project/TechProject/evaluation/Picture/test.txt','r',encoding='utf-8') as f:
    content = f.read()
    # print(content)
    scores = getDensity(content, 27)
    scores2 = baidu_picture(content)
    print(scores)
    print(scores2)