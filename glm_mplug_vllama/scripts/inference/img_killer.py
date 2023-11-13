import pickle
import sys
import time
import re
import modelscope

sys.path.append(".")
import transformers
import visualcla
from peft import PeftModel
import logging
import torch
import os
from transformers import AutoTokenizer, AutoModel
import os
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
import argparse
from scripts.inference import img_captioner
import random

debugging = True


def init_img_killer_model(cuda_id, load_in_8bit, visualcla_model=None, text_model=None, vision_model=None,
                          lora_model=None):
    # Preparation on logging and storing HPs
    # logger.setLevel('INFO')
    # transformers.utils.logging.set_verbosity('INFO')
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # Load model
    load_type = torch.float16
    device = torch.device('cpu')
    device_map = {'': device}
    logger.info(f"Img Killer Running on {device}")
    load_in_8bit = load_in_8bit
    base_model, tokenizer, image_processor = visualcla.get_model_and_tokenizer_and_processor(
        visualcla_model=visualcla_model,
        text_model=text_model,
        vision_model=vision_model,
        lora_model=lora_model,
        torch_dtype=load_type,
        default_device=device,
        device_map=device_map,
        load_in_8bit=load_in_8bit and (visualcla_model is not None)
    )
    if isinstance(cuda_id, int):
        base_model = base_model.cuda(cuda_id)

    if lora_model is not None:
        logger.info(f"Model embedding size is {base_model.text_model.get_input_embeddings().weight.size(0)}. "
                    f"Resize embedding size to {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
        logger.info("Loading LoRA...")
        model = PeftModel.from_pretrained(
            base_model,
            lora_model,
        )
    else:
        model = base_model

    if device == torch.device('cpu'):
        model.float()
    model.eval()

    return model


def init_chatglm_model(device=None, load_in_kbit=None, model_name="THUDM/chatglm3-6b"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if load_in_kbit is not None:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                          device='cpu',
                                          torch_dtype=torch.float16).quantize(load_in_kbit)
        if device == 'cuda' or device == 'gpu' or isinstance(device, int):
            if isinstance(device, int):
                model = model.cuda(device)
            else:
                model = model.cuda()
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                          device=device if device is not None else 'cuda',
                                          torch_dtype=torch.float16)

    model = model.eval()
    return tokenizer, model


def kill(model, img_path, instruction, history_=None):
    # history example:
    #  [{'type': 'instruction', 'value': '介绍一下这张图片的内容', 'first_instruction': True},
    #  {'type': 'response', 'value': '这张图片展示了一张木制餐桌，上面摆放着各种食物和饮料。其中包括碗、杯子、面包、鸡蛋、牛奶和饼干等。'},
    #  {'type': 'instruction', 'value': '右下角的那个是什么？'},
    #  {'type': 'response', 'value': '在图像的右下角，有一个碗。'}]
    with torch.no_grad():
        history = []
        if history_ is not None:
            history = history_
        image_path = img_path
        if image_path is not None:
            logger.info(f'Image: {image_path}')
        text = instruction
        if type(text) != str:
            raise TypeError(f'Input type error! Expect \'str\' but get \'{type(text)}\'.')
        try:
            response, history = visualcla.chat(model, image=image_path, text=text, history=history, print_history=False,
                                               print_response=False)
            logger.info(f"img killer response: {response}")
        except FileNotFoundError:
            print(f"Cannot find file {image_path}. Clear history")

    logger.info("img killed ... ")
    return response


def desc_of_pic_by_llama(model,
                         img_path,
                         instruction=
                         "你是识别图片内容的专家，你要确保你识别的内容准确。"
                         "请详细地描述这张图片中有什么？"
                         "首先，识别图片中出现的中文和英文。但是请不要只识别文字，如果文字是水印，请忽略。"
                         # "考虑识别出的文字作为图片的相关信息，"
                         "你要描述图片中包含的主体（人或物），最好遍历图片中出现的物体。"
                         "请准确详细的识别，并描述物体之间的关系"
                         # "需要用 主题：...\n；内容：...\n；可识别的文字（中文及英文）：...\n；补充：...\n；这种格式回答"
                         ,
                         save_img_path="pics/",
                         history=None):
    if 'http' in img_path:
        img_path = download_image(img_path, save_img_path)
    return kill(model, img_path, instruction, history)


def direct_test_related(model, img_path, news_content, instruction="请判断新闻内容和图片是不是相关的？", history=None):
    return kill(model, img_path, f"{instruction} 新闻内容：{news_content}", history)
    # 效果不行


# 图片内容与题相符
def ___check_image_content_matches_topic(text_model, text_tokenizer, news_topic,
                                         img_path=None, img_content=None, img_model=None, img_model_fn=None,
                                         instruction="判断新闻的标题和新闻图片内容是否相关？并选择相关程度选项（）\n"
                                                     "A.非常相关 B.比较相关 C.勉强相关 D.不相关"
                                                     "由于图片内容往往比较简要，所以你可以把图片中概指的内容尽量向新闻标题靠拢，当实在不相关时，输出选项 D. 不相关"
                                                     "其他情况下根据相关程度选择选项"
                                         # "为了辅助你的判断，请你先根据新闻标题内的一些实体，生成一些相关的定义。"
                                         # "然后再给出你的选项。"
                                         # "下面是一个例子：\n\n\n"
                                         # "根据新闻标题中的实体，可以生成以下定义：\n"
                                         # "1. 上海迪士尼乐园：一个位于上海市的的主题公园，提供各种迪士尼相关的娱乐设施和表演。\n"
                                         # "2. 假期高峰日门票：一种特殊的门票，只在上海迪士尼乐园的假期高峰日发行。\n"
                                         # "3. 三级定价体系：一种价格策略，将门票价格分为不同的级别，以满足不同需求的市场。\n"
                                         # "根据新闻图片内容，可以得出以下描述：\n"
                                         # "图片展示了一个巨大的城堡，位于一片水域之上。"
                                         # "这座城堡的顶部有一个大圆顶，侧面有几物，包括一些人行道上的商店。"
                                         # "在图像的右上角可以看到一个时钟。可以看到几个人在城堡前行走或站立。"
                                         # "根据以上定义和描述，可以判断新闻标题和新闻图片内容的相关程度为："
                                         # "B.比较相关。虽然新闻标题和图片内容没有直接关联，但图片中的城堡和迪士尼乐园有一定的关联，可以认为它们是相关联的主题。\n\n\n"
                                         ,
                                         history=None):
    if (img_path and img_content) or (img_path is None and img_content is None):
        raise ValueError("输入path或content")
    if img_path is not None and img_content is None:
        if img_model is None:
            raise ValueError("need img_model for img_path")
        if 'http' in img_path:
            img_path = download_image(img_path, "./pics/")
        img_content = img_model_fn(img_model, img_path)
        if isinstance(img_model, modelscope.Pipeline):
            img_content = en2zh(text_model, text_tokenizer, img_content)
    history_ = []
    if history is not None:
        history_ = history
    if debugging:
        print("img_path", img_path)
        print(f"新闻标题：{news_topic}；新闻图片内容：{img_content}")
    res, history_ = text_model.chat(
        text_tokenizer, instruction + f"新闻标题：{news_topic}；新闻图片内容：{img_content}", history=history_)
    # print("debugging .. ", history_)
    return res


def summary(text_model, text_tokenizer, text):
    res, history_ = text_model.chat(
        text_tokenizer, f"简要总结内容: {text}", history=[])
    if debugging:
        print("原文本:", text)
        print("总结后文本", res)
    return res


def check_image_content_matches_topic_by_mplug(text_model, text_tokenizer, news_topic,
                                               img_path=None, img_content=None, img_model=None, vote_times=1, ocr=False,
                                               instruction="判断新闻的标题和新闻配图内容是否相关？并选择相关程度选项（）\n"
                                                           "A.非常相关 B.比较相关 C.勉强相关 D.不相关"
                                                           "由于图片内容往往比较简要，所以你可以把图片中概指的内容尽量向新闻标题靠拢。"
                                                           "当实在不相关时，输出选项 D. 不相关 。"
                                                           "其他情况下根据相关程度选择选项。"
                                                           "请一步一步仔细分析，最后再给出答案。"
                                                           "比如你可以分析新闻标题都涉及到了哪些人和哪些事，"
                                                           "图片中出现的人是不是新闻里提到的人，"
                                                           "图片是否和事件发生地有关系等等"
                                               # "为了辅助你的判断，请你先根据新闻标题内的一些实体，生成一些相关的定义。"
                                               # "然后再给出你的选项。"
                                               # "下面是一个例子：\n\n\n"
                                               # "根据新闻标题中的实体，可以生成以下定义：\n"
                                               # "1. 上海迪士尼乐园：一个位于上海市的的主题公园，提供各种迪士尼相关的娱乐设施和表演。\n"
                                               # "2. 假期高峰日门票：一种特殊的门票，只在上海迪士尼乐园的假期高峰日发行。\n"
                                               # "3. 三级定价体系：一种价格策略，将门票价格分为不同的级别，以满足不同需求的市场。\n"
                                               # "根据新闻图片内容，可以得出以下描述：\n"
                                               # "图片展示了一个巨大的城堡，位于一片水域之上。"
                                               # "这座城堡的顶部有一个大圆顶，侧面有几物，包括一些人行道上的商店。"
                                               # "在图像的右上角可以看到一个时钟。可以看到几个人在城堡前行走或站立。"
                                               # "根据以上定义和描述，可以判断新闻标题和新闻图片内容的相关程度为："
                                               # "B.比较相关。虽然新闻标题和图片内容没有直接关联，但图片中的城堡和迪士尼乐园有一定的关联，可以认为它们是相关联的主题。\n\n\n"
                                               ,
                                               history=None):
    if (img_path and img_content) or (img_path is None and img_content is None):
        raise ValueError("输入path或content")
    ocr_res = ""
    if img_path is not None and img_content is None:
        if img_model is None:
            raise ValueError("need img_model for img_path")
        if 'http' in img_path:
            img_path = download_image(img_path, "./pics/")
        img_content, ocr_res = img_captioner.mplug_img_des(img_model, img_path, ocr=ocr)
        img_content = en2zh(text_model, text_tokenizer, img_content)
        if ocr:
            ocr_res = summary(text_model, text_tokenizer, ocr_res)
    history_ = []
    if history is not None:
        history_ = history
    if debugging:
        print("img_path", img_path)
        print(instruction + f"\n\n新闻标题：{news_topic}；新闻图片内容：{img_content}")
        # print(f"新闻标题：{news_topic}；新闻图片内容：{img_content}")
    options = ["A?.{0,3}非常相关", "B?.{0,3}比较相关", "C?.{0,3}勉强相关", "D?.{0,3}不相关"]
    options.reverse()
    vote = 0
    for _ in range(vote_times):
        if ocr and _ == 0:
            res, history_ = text_model.chat(
                text_tokenizer, instruction + f"新闻标题：{news_topic}；新闻图片内容：{img_content}"
                                              f"新闻图片中识别的文字: {ocr_res}", history=history_)
        else:
            res, history_ = text_model.chat(
                text_tokenizer, instruction + f"新闻标题：{news_topic}；新闻图片内容：{img_content}", history=history_)
        if debugging:
            print("response :", res)
        for s, opt in enumerate(options):
            if re.search(opt, res):
                vote += s
    return options[min((vote + vote_times - 1) // vote_times, 3)]


def en2zh(text_model, text_tokenizer, text):
    res, history_ = text_model.chat(
        text_tokenizer, f"翻译到中文: {text}", history=[])
    return res


def download_image(url, path):
    if not os.path.isdir(path):
        os.makedirs(path)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        url_path = urlparse(url)
        filename = os.path.basename(url_path.path)
        file_path = os.path.join(path, filename)

        with open(file_path, 'wb') as f:
            f.write(response.content)

        return file_path
    else:
        print("Failed to get image at {}".format(url))
        return None


if __name__ == '__main__':
    glm_tokenizer, glm = init_chatglm_model(load_in_kbit=8, device=1)
    mm_model = img_captioner.init_img_caption_model()
    article_list = pickle.load(open("./article_list.pkl", "rb"))
    random.shuffle(article_list)
    pos_acc = 0
    for article_lines in article_list:
        img_path = None
        for line in article_lines:
            if "png" in line or "jpg" in line:
                img_path = line
                break
        if img_path is None:
            continue
        r = check_image_content_matches_topic_by_mplug(glm, glm_tokenizer, article_lines[0], img_path=img_path,
                                                       img_model=mm_model, vote_times=2)
        print(r)
        if r[0] != 'D':
            pos_acc += 1

    pos_acc = pos_acc / len(article_list)
    print("pos acc", pos_acc)

    neg_acc = 0
    for idx, article_lines in enumerate(article_list):
        img_path = None
        for line in article_lines:
            if "png" in line or "jpg" in line:
                img_path = line
                break
        if img_path is None:
            continue
        r = check_image_content_matches_topic_by_mplug(glm, glm_tokenizer, article_list[idx - 1][0], img_path=img_path,
                                                       img_model=mm_model, vote_times=2)
        print(r)
        if r[0] == 'D':
            neg_acc += 1

    neg_acc = neg_acc / len(article_list)
    print("neg acc", neg_acc)
