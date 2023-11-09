import os
import json
import uuid
import imghdr

from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode

from io import BytesIO
from PIL import Image

from config import config

from logger import logger
from myException import MyException


def fetch_token():
    """
        获取token
    """
    params = {'grant_type': 'client_credentials',
              'client_id': config.API_KEY,
              'client_secret': config.SECRET_KEY}

    post_data = urlencode(params)
    post_data = post_data.encode('utf-8')
    req = Request(config.TOKEN_URL, post_data)

    try:
        f = urlopen(req, timeout=5)
        result_str = f.read()
    except URLError as err:
        err_message = "获取token时网络连接错误."
        logger.error_logging(err_message + "\n" + str(err))
        raise MyException(error_code=config.WEB_ERROR, message=err_message)

    result_str = result_str.decode()
    result = json.loads(result_str)

    if 'access_token' in result.keys() and 'scope' in result.keys():
        if 'brain_all_scope' not in result['scope'].split(' '):
            logger.warning_logging("请确保百度功能都已开启")
        return result['access_token']
    else:
        err_message = "请确保百度API_KEY和SECRET_KEY填写正确."
        logger.error_logging(err_message)
        raise MyException(error_code=config.BAIDU_ERROR, message=err_message)


def check_api_key(api_dict, key):
    value = api_dict.get(key, None)
    if value is None:
        err_message = f"key:{key}不存在,请检查百度API返回关键词是否正确."
        logger.error_logging(err_message + "\n" + str(api_dict))
        raise MyException(error_code=config.BAIDU_ERROR, message=err_message)
    else:
        return value


def read_file(image_path):
    """
        读取文件
    """
    f = None
    try:
        f = open(image_path, 'rb')
        return f.read()
    except:
        print('read image file fail')
        return None
    finally:
        if f:
            f.close()


def request(url, data):
    """
        调用远程服务
    """
    req = Request(url, data.encode('utf-8'))

    try:
        f = urlopen(req)
        result_str = f.read()
        result_str = result_str.decode()
        return result_str
    except URLError as err:
        err_message = "请求百度API时发生错误."
        logger.error_logging(err_message + "\n" + str(err))
        raise MyException(error_code=config.REQUEST_ERROR, message=err_message)


def merge(intervals):
    """
    命中关键字区间去重
    :type intervals: List[List[int]]
    :rtype: int
    """
    if len(intervals) == 0:
        return 0

    res = []
    intervals = list(sorted(intervals))

    low = intervals[0][0]
    high = intervals[0][1]

    for i in range(1, len(intervals)):
        # 若当前区间和目前保存区间有交集，则进行判断后修改相应的区间参数；若当前区间和目前保存区间没有交集，则将目前保存区间放入到结果集合中，并将当前区间记录成目前保存区间
        if high >= intervals[i][0]:
            if high < intervals[i][1]:
                high = intervals[i][1]
        else:
            res.append([low, high])
            low = intervals[i][0]
            high = intervals[i][1]

    res.append([low, high])
    ans = 0
    #计算区间总长度
    for tem in res:
        ans += tem[1]-tem[0]+1

    return ans


def savePicture(picture_list):
    if not isinstance(picture_list, list):
        picture_list = [picture_list]

    for picture in picture_list:
        picture_name = uuid.uuid4().hex

        try:
            image = Image.open(BytesIO(picture))
            imgType = imghdr.what(None, picture)
            picture_name += ('.' + imgType)
            image.save(os.path.join(config.PIC_SACE_PATH, picture_name))
            image.close()
            logger.process_logging(f"保存图片 {picture_name} 成功。")
        except Exception as e:
            logger.error_logging(f"{picture_name} 存储出错。")

