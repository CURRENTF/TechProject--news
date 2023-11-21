# -*- coding: utf-8 -*-
import base64
import hashlib
import hmac
import json
import os
import time
import requests
import urllib
from moviepy.editor import AudioFileClip

lfasr_host = 'https://raasr.xfyun.cn/v2/api'
# 请求的接口名
api_upload = '/upload'
api_get_result = '/getResult'


class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path
        self.ts = str(int(time.time()))
        self.signa = self.get_signa()

    def get_signa(self):
        appid = self.appid
        secret_key = self.secret_key
        m2 = hashlib.md5()
        m2.update((appid + self.ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        return signa

    def upload(self):
        print("上传部分：")
        upload_file_path = self.upload_file_path
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)

        param_dict = {}
        param_dict['appId'] = self.appid
        param_dict['signa'] = self.signa
        param_dict['ts'] = self.ts
        param_dict["fileSize"] = file_len
        param_dict["fileName"] = file_name
        param_dict["duration"] = "200"
        print("upload参数：", param_dict)
        data = open(upload_file_path, 'rb').read(file_len)

        response = requests.post(url=lfasr_host + api_upload + "?" + urllib.parse.urlencode(param_dict),
                                 headers={"Content-type": "application/json"}, data=data)
        print("upload_url:", response.request.url)
        result = json.loads(response.text)
        print("upload resp:", result)
        return result

    def get_result(self):
        uploadresp = self.upload()
        orderId = uploadresp['content']['orderId']
        param_dict = {}
        param_dict['appId'] = self.appid
        param_dict['signa'] = self.signa
        param_dict['ts'] = self.ts
        param_dict['orderId'] = orderId
        param_dict['resultType'] = "transfer,predict"
        print("")
        print("查询部分：")
        print("get result参数：", param_dict)
        status = 3
        # 建议使用回调的方式查询结果，查询接口有请求频率限制
        while status == 3:
            response = requests.post(url=lfasr_host + api_get_result + "?" + urllib.parse.urlencode(param_dict),
                                     headers={"Content-type": "application/json"})
            # print("get_result_url:",response.request.url)
            result = json.loads(response.text)
            print(result)
            status = result['content']['orderInfo']['status']
            print("status=", status)
            if status == 4:
                break
            time.sleep(5)
        print("get_result resp:", result)
        return result

    def convert2sentence(self, data_voice2str):
        data_dict = json.loads(data_voice2str['content']['orderResult'])
        sentence_list = []
        for sentence_data in data_dict['lattice2']:
            sentence = ''
            for ws in sentence_data['json_1best']['st']['rt'][0]['ws']:
                sentence += ws['cw'][0]['w']
            sentence_list.append(sentence)
        return sentence_list


def convert2sentence(data_dict):
    for sentence_data in data_dict['lattice2']:
        sentence = ''
        for ws in sentence_data['json_1best']['st']['rt'][0]['ws']:
            sentence += ws['cw'][0]['w']
        print(sentence)


def upload_and_convert(upload_file_path, appid="ea4e59c5", secret_key="196c65249f2c8c85725860d6701cb01d"):
    api = RequestApi(appid=appid, secret_key=secret_key, upload_file_path=upload_file_path)
    data_voice2str = api.get_result()
    data_dict = json.loads(data_voice2str['content']['orderResult'])
    convert2sentence(data_dict)


if __name__ == '__main__':
    # 修改upload_file_path

    api = RequestApi(appid="ea4e59c5",
                     secret_key="196c65249f2c8c85725860d6701cb01d",
                     upload_file_path=r"test2.mp4")
    data_voice2str = api.get_result()
    # print(data_voice2str['content']['orderResult'])
    # print(type(data_voice2str['content']['orderResult']))
    data_dict = json.loads(data_voice2str['content']['orderResult'])


    # print(data_dict)
    def convert2sentence(data_dict):
        for sentence_data in data_dict['lattice2']:
            sentence = ''
            for ws in sentence_data['json_1best']['st']['rt'][0]['ws']:
                sentence += ws['cw'][0]['w']
            print(sentence)


    convert2sentence(data_dict)
    # data_str = api.analysis_json_lattice2(data_voice2str)
    # print(data_str)
