import os
import torch
import pandas as pd

from models.TextModel import SentenceModel, GPTModel


class BaseConfig(object):
    # 端口
    PORT = 8192

    # 目录
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    RES_DIR = os.path.join(ROOT_DIR, 'res')

    # 路径
    MODELS_STORE_PATH = os.path.join(ROOT_DIR, 'pretrained_models')
    DICT_FILE_PATH = os.path.join(RES_DIR, 'dict.txt')
    STOP_WORDS_PATH = os.path.join(RES_DIR, 'stop_words.txt')
    TEST_FILE_PATH = os.path.join(RES_DIR, 'spider.json')
    LOG_PATH = os.path.join(ROOT_DIR, 'log')
    LOG_FILE_PATH = os.path.join(LOG_PATH, 'log.txt')
    PIC_SACE_PATH = os.path.join(LOG_PATH, 'Imgs')

    # 模型
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BERT_MODEL_PATH = os.path.join(MODELS_STORE_PATH, 'bert-base-chinese')
    GPT_MODEL_PATH = os.path.join(MODELS_STORE_PATH, 'gpt2-base-chinese')

    # json key list
    JSON_KEY_LIST = ['title', 'paragraphs']

    # 百度 API
    API_KEY = 'rWKKOKTAcov73GpTRZkGu9zF'
    SECRET_KEY = 'Omas1jvYGTsQYOd45nETzpZmqlwNHEdx'
    IMAGE_CENSOR = "https://aip.baidubce.com/rest/2.0/solution/v1/img_censor/v2/user_defined"
    TEXT_CENSOR = "https://aip.baidubce.com/rest/2.0/solution/v1/text_censor/v2/user_defined"
    """  TOKEN start """
    TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'


class HyperparameterConfig(object):
    # 层次清晰
    HIERARCHY_ALPHA = 0.5
    # 简洁凝练
    CONCISE_ALPHA = 0.99999
    # 多样性
    DIVERSITY_NGRAM_N = 2
    DIVERSITY_BLEU_N = 4
    DIVERSITY_ALPHA = 0.5
    # 流畅度
    FLUENCY_TOPK = 1000
    # 像素密度
    DIAG_LEN = 12
    # 曝光检测
    LOWER_BOUND = 50
    UPPER_BOUND = 255
    # 饱和度
    COLOR_THRES = 150
    # 模糊影像检测
    DETECT_THRES = 100
    # 小数点位数
    REST_NUM = 4
    # 概括指数
    SUMMARIZE_ALPHA = 0.5


class ErrorCodeConfig(object):
    # susses code
    SUSSES_CODE = 0
    # input error
    INPUT_ERROR = 1
    # server error
    SERVER_ERROR = 2
    # token error
    TOKEN_ERROR = 3
    # web error
    WEB_ERROR = 4
    # baidu error
    BAIDU_ERROR = 5
    # request error
    REQUEST_ERROR = 6


class Config(BaseConfig, HyperparameterConfig, ErrorCodeConfig):
    def __init__(self):
        if not os.path.exists(self.LOG_PATH):
            os.makedirs(self.LOG_PATH)

        if not os.path.exists(self.PIC_SACE_PATH):
            os.makedirs(self.PIC_SACE_PATH)

        # 词表相关
        DICT_DF = pd.read_csv(self.DICT_FILE_PATH, sep=' ', header=None, names=['words', 'frequency', 'part'])
        self.WORDS = DICT_DF['words'].values

        with open(self.STOP_WORDS_PATH, 'r', encoding='utf-8') as f:
            self.STOP_WORDS_LIST = f.readlines()

        self.BERT_MODEL = SentenceModel(model_path=self.BERT_MODEL_PATH, truncation=False)
        self.BERT_MODEL = self.BERT_MODEL.to(self.DEVICE)

        self.GPT_MODEL = GPTModel(model_path=self.GPT_MODEL_PATH, truncation=False)
        self.GPT_MODEL = self.GPT_MODEL.to(self.DEVICE)

    def __getitem__(self, item):
        return getattr(self, item)


config = Config()
