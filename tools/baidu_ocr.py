"""
@Time       : 2023/11/7 21:51
@File       : baidu_ocr.py
@Description:
"""
import base64
import urllib
import json
import cv2
import requests

API_KEY = "Ck7aFc5NIGr5oPTCgrbw8I9D"
SECRET_KEY = "m7O2OwGF6e8lBmZVgtsLHj7oyZyV1Dmf"


def ocr_api(base64):
    """
    输入base64格式图片，返回文字识别内容
    :param base64: 图片
    :return:
    """
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token=" + get_access_token()

    payload = 'image='+base64+'&detect_direction=true&detect_language=true&paragraph=false&probability=true'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def encode_img(img_path):
    img = cv2.imread(img_path)
    img_str = cv2.imencode('.png', img)[1].tobytes()
    base64_data = base64.b64encode(img_str).decode("utf8")
    base64_data = urllib.parse.quote_plus(base64_data)
    return base64_data


def ocr(img_path):
    img = encode_img(img_path)
    d = ocr_api(img)
    d = json.loads(d)
    texts = ""
    if "words_result" not in d:
        return ""
    for dd in d['words_result']:
        texts += dd["words"] + "  ;  "
    return texts


if __name__ == '__main__':
    d = ocr("./pics/architecture.png")
    d = json.loads(d)
    print(d['words_result'])
    for dd in d['words_result']:
        print(dd["words"])
