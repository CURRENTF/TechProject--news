import re
import sys
import requests
import json
sys.path.append(".")
from tools.tools import print_separator
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GenerationConfig
import os

debugging = True


class TextModel:
    instruction_for_img_match_topic = [
        "判断新闻的标题和新闻配图内容是否相关？并选择相关程度选项（）\n",
        "A.非常相关 B.比较相关 C.勉强相关 D.不相关",
        "由于图片内容往往比较简要，所以你可以把图片中概指的内容尽量向新闻标题靠拢。",
        "当实在不相关时，输出选项 D. 不相关 。",
        "其他情况下根据相关程度选择选项。",
        "请一步一步仔细分析，最后再给出答案。",
        "比如你可以分析新闻标题都涉及到了哪些人和哪些事，",
        "图片中出现的人是不是新闻里提到的人，",
        "图片是否和事件发生地有关系等等",
        # "为了辅助你的判断，请你先根据新闻标题内的一些实体，生成一些相关的定义。",
        # "然后再给出你的选项。",
        # "下面是一个例子：\n\n\n",
        # "根据新闻标题中的实体，可以生成以下定义：\n",
        # "1. 上海迪士尼乐园：一个位于上海市的的主题公园，提供各种迪士尼相关的娱乐设施和表演。\n",
        # "2. 假期高峰日门票：一种特殊的门票，只在上海迪士尼乐园的假期高峰日发行。\n",
        # "3. 三级定价体系：一种价格策略，将门票价格分为不同的级别，以满足不同需求的市场。\n",
        # "根据新闻图片内容，可以得出以下描述：\n",
        # "图片展示了一个巨大的城堡，位于一片水域之上。",
        # "这座城堡的顶部有一个大圆顶，侧面有几物，包括一些人行道上的商店。",
        # "在图像的右上角可以看到一个时钟。可以看到几个人在城堡前行走或站立。",
        # "根据以上定义和描述，可以判断新闻标题和新闻图片内容的相关程度为：",
        # "B.比较相关。虽然新闻标题和图片内容没有直接关联，"
        # "但图片中的城堡和迪士尼乐园有一定的关联，可以认为它们是相关联的主题。\n\n\n",
    ]
    options = ["A?.{0,3}非常相关", "B?.{0,3}比较相关", "C?.{0,3}勉强相关", "D?.{0,3}不相关"]
    options_re_list = f"({'|'.join(options)})"
    opt2score = {0: 95, 1: 80, 2: 60, 3: 40}
    api_key = "sk-JCgKDsdrCL6SqQyewix85nYQJgEuVKCp3bbsCPuaWEH4rbBY"

    def __init__(self, cuda_id=0, load_in_8bit=False, model_name="THUDM/chatglm3-6b"):
        self._chat = None
        if "glm" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name,
                                                   trust_remote_code=True,
                                                   device_map=cuda_id,
                                                   torch_dtype=torch.float16,
                                                   load_in_8bit=load_in_8bit).cuda(cuda_id).eval()
            self._chat = self.glm_chat
        elif "chuan" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                           use_fast=False, trust_remote_code=True)

            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=cuda_id,
                                                              torch_dtype=torch.float16, trust_remote_code=True,
                                                              load_in_8bit=load_in_8bit)

            self.model.generation_config = GenerationConfig.from_pretrained(model_name)
            self._chat = self.baichuan_chat
        elif "gpt" in model_name:
            self._chat = self.gpt_chat
        else:
            raise ValueError("model_name should be either glm or baichuan.")

    def baichuan_chat(self, tokenizer, message, history):
        messages = [] if history is None else history
        messages.append({"role": "user", "content": message})
        response = self.model.chat(self.tokenizer, messages)
        return response

    def glm_chat(self, tokenizer, message, history):
        res, history_ = self.model.chat(
            self.tokenizer,
            message,
            history=[] if history is None else history
        )
        return res

    def _gpt_chat(self, tokenizer, message, history):
        instruction = "You are a helpful assistant. " \
                      "You can help me by answering my questions. " \
                      "You can not ask me questions. "
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {
                    'role': 'system',
                    'content': instruction
                },
                {
                    'role': 'user',
                    'content': message
                }
            ]
        }
        response = requests.post('https://api.chatanywhere.com.cn', headers=headers,
                                 data=json.dumps(data))
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise ValueError(f"Error code {response.status_code} when calling API.")

    def gpt_chat(self, tokenizer, message, history):
        for i in range(10):
            try:
                return self._gpt_chat(tokenizer, message, history)
            except ValueError:
                pass
        raise ValueError("API call failed.")

    def chat(self, *args, **kwargs):
        return self._chat(*args, **kwargs), None

    def get_final_option_score(self, response):
        match_obj = re.findall(self.options_re_list, response)
        if match_obj is None or len(match_obj) == 0:
            other_form_options = ["A", "B", "C", "D"]
            for idx, opt in enumerate(other_form_options):
                if opt in response:
                    return self.opt2score[idx]
            return self.opt2score[3]
        opt = match_obj[-1]
        for i, r in enumerate(self.options):
            if re.match(r, opt):
                return self.opt2score[i]
        return self.opt2score[3]

    @print_separator(debugging)
    def img_match_topic_1(self, image_content, topic, instruction=None):
        if instruction is not None:
            instruction = ''.join(self.instruction_for_img_match_topic)
        res, history_ = self.chat(
            self.tokenizer,
            instruction + f"新闻标题：{topic}；新闻图片内容：{image_content}",
            history=[]
        )
        if debugging:
            print(res)
        return self.get_final_option_score(res)

    @print_separator(debugging)
    def img_match_topic_k(self, image_content, topic, vote_times=3):
        score = 0
        instructions = [
            ''.join(self.instruction_for_img_match_topic),
            ''.join(self.instruction_for_img_match_topic[:-3]),
            ''.join(self.instruction_for_img_match_topic[:-6]),
        ]
        for _ in range(vote_times):
            score += self.img_match_topic_1(image_content, topic, instruction=instructions[_ % len(instructions)])
        return score / vote_times

    def summary(self, text):
        res, history_ = self.chat(
            self.tokenizer,
            f"摘要，少于五个字：{text}",
            history=[]
        )
        return res

    def sentence_match(self, s1, s2):
        res, history_ = self.chat(
            self.tokenizer,
            f"判断两个句子是否相关：{s1}，{s2} \n"
            f"选择相关程度选项（）\n"
            f"A.非常相关 B.比较相关 C.勉强相关 D.不相关",
            history=[]
        )
        return res

    @print_separator(debugging)
    def img_match_topic_by_summary(self, image_content, topic):
        summary1 = image_content
        if len(summary1) > 20:
            summary1 = self.summary(image_content)
        summary2 = topic
        res = self.sentence_match(summary1, summary2)
        if debugging:
            print(f"{image_content} ===> {summary1}")
            print(f"{topic} ===> {summary2}")
            print(f"res: {res}")
        return self.get_final_option_score(res)

    def img_match_content(self, image_content, content):
        summary1 = image_content
        if len(summary1) > 20:
            summary1 = self.summary(summary1)
        summary2 = content
        if len(summary2) > 20:
            summary2 = self.summary(summary2)
        res = self.sentence_match(summary1, summary2)
        if debugging:
            print(f"{image_content} ===> {summary1}")
            print(f"{content} ===> {summary2}")
            print(f"res: {res}")
        return self.get_final_option_score(res)

    def en2zh(self, text_tokenizer, text):
        res, history_ = self.chat(
            text_tokenizer,
            f"翻译到中文: {text}",
            history=[]
        )
        return res


if __name__ == "__main__":
    glm = TextModel(load_in_8bit=False)
