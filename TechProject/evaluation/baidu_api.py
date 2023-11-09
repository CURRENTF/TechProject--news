import ssl
import simplejson
import numpy as np

from urllib.parse import urlencode

from logger import logger
from tools.baidu_utils import (
    fetch_token,
    request,
    merge,
    check_api_key
)

from config import config
from myException import MyException

# 防止https证书校验不正确
ssl._create_default_https_context = ssl._create_unverified_context

# 后续添加只需要在check_list和pictureType2name中添加相应指标名称即可
check_list = [
    "图片是否合规",
    "图片是否涉黄",
    "图片是否涉恐",
    "图片是否恶心",
    "图片是否涉政",
    "图片是否违禁",
    "图片是否包含广告"
]
# key值为百度api返回的type值，构造成subdict形式可以支持检索subtype字段
pictureType2name = {
    1: "图片是否涉黄",
    2: "图片是否涉恐",
    3: "图片是否恶心",
    4: "图片是否包含广告",
    5: "图片是否涉政",
    # "图文内容不合规"
    # 12: {
    #     0: "低质灌水",
    #     2: "文本色情",
    #     4: "恶意推广",
    #     5: "低俗辱骂",
    #     6: "恶意推广-联系方式",
    #     7: "恶意推广-软文推广",
    #     8: "广告法审核"
    # },
    16: "图片是否涉政",
    21: "图片是否违禁"
}


def getPictureIndex(pictureType, pictureSubType):
    try:
        name = pictureType2name.get(pictureType, None)
        if isinstance(name, dict):
            name = name.get(pictureSubType, None)
        if name is None:
            return -1
        else:
            if name not in check_list:
                return -1
            return check_list.index(name)
    except Exception as e:
        error_message = f"key:{pictureType}, subtype:{pictureSubType}\n请检查服务器getPictureIndex函数是否有错.\n" + str(e)
        logger.error_logging(error_message)
        raise MyException(error_code=config.SERVER_ERROR, message=error_message)


def baidu_picture(picture_list):
    if not isinstance(picture_list, list):
        picture_list = [picture_list]

    message = {}

    if len(picture_list) == 0:
        for k in check_list:
            message[k] = []
        return message

    token = fetch_token()
    image_url = config.IMAGE_CENSOR + "?access_token=" + token

    scores = []

    for picture in picture_list:
        score = [0.0] * len(check_list)

        result = request(image_url, urlencode({'image': picture}))
        result_dict = simplejson.loads(result)

        conclusionType = check_api_key(api_dict=result_dict, key="conclusionType")
        if conclusionType == 1:
            score[0] = 1.0
        else:
            msg_list = check_api_key(api_dict=result_dict, key="data")

            for msg in msg_list:
                msg_type = check_api_key(api_dict=msg, key="type")
                msg_subtype = check_api_key(api_dict=msg, key="subType")

                index = getPictureIndex(msg_type, msg_subtype)
                if index == -1:
                    warning_message = f"type:{msg_type}, 不在字典中, 请根据考虑是否添加. \n" + str(msg)
                    logger.warning_logging(warning_message)
                    continue

                score[index] = 1.0
        scores.append(score)

    scores = np.array(scores).transpose().tolist()

    for k, v in zip(check_list, scores):
        message[k] = v

    return message


def baidu_text(text):

    token = fetch_token()
    text_url = config.TEXT_CENSOR + "?access_token=" + token

    result = request(text_url, urlencode({'text': text}))
    result_dict = simplejson.loads(result)

    conclusionType = check_api_key(api_dict=result_dict, key="conclusionType")
    if conclusionType == 1:
        # 真实准确 导向正确
        score1, score2 = 1.0, 1.0
    else:
        res1 = []
        res2 = []

        msg_list = check_api_key(api_dict=result_dict, key="data")
        for msg in msg_list:
            msg_type = check_api_key(api_dict=msg, key="type")
            hits = check_api_key(api_dict=msg, key="hits")
            for hit in hits:
                wordHitPositions = check_api_key(api_dict=hit, key="wordHitPositions")
                if len(wordHitPositions) == 0:
                    positions = [[0, len(text) - 1]]
                else:
                    positions = check_api_key(api_dict=wordHitPositions[0], key="positions")

                if msg_type == 11:
                    res1.extend(positions)
                elif msg_type == 12:
                    res2.extend(positions)

        score1 = 1.0 - merge(res1) / len(text)
        score2 = 1.0 - merge(res2) / len(text)

    message = {
        '真实准确指数': score1,
        # '导向正确指数': score2
    }

    return message
