import gradio as gr
import numpy as np
import random
import string


def generate_random_strings(n=40, length=10):
    result = []
    for _ in range(n):
        # 生成一个长度为`length`的随机字符串，字符串中的每个字符都是一个随机字母（大写或小写）
        random_string = ''.join(random.choice(string.ascii_letters) for _ in range(length))
        result.append(random_string)
    return result


def multimodal_model(video, image, text):
    # 在这里处理你的输入
    processed_video = process_video(video)
    processed_image = process_image(image)
    processed_text = process_text(text)

    # 这里应该是你的模型，它应该接受三种类型的输入并返回一个包含40个分数的列表
    # 这是一个示例，我们只是随机生成40个分数
    strs = [
        "文本结构完整",
        "文本层次清晰",
        "文本前后连贯",
        "文本流畅度",
        "文本表达多样性",
        "文本表达简洁凝练",
        "文本真实准确",
        "文本通俗易懂",
        "文本信息完整",
        "文本主题明确",
        "文本内容导向正确",
        "图片像素密度",
        "图片曝光检测",
        "图片模糊检测",
        "图片色彩检测",
        "图片内容与题相符",
        "图片内容合规",
        "图片涉黄审核",
        "图片暴恐审核",
        "图片涉政审核",
        "图片广告审核",
        "图片素材可靠",
        "视频画面分辨率",
        "视频播放流畅",
        "视频清晰",
        "视频声音清晰",
        "音视频同步",
        "视频字幕质量",
        "视频内容与题相符",
        "视频内容合规",
        "视频涉黄审核",
        "视频暴恐审核",
        "视频涉政审核",
        "视频广告审核",
        "视频素材可靠",
        "视频未被篡改",
        "图片摘要质量",
        "图文布局合理",
        "图文相符",
        "视频摘要质量",
        "视频文本布局合理",
        "视频文本相符",
        "文本图片视频相符",
    ]
    scores = np.random.rand(43)
    scores = scores * 0.22 + 0.68
    scores = scores.tolist()
    d = {}
    for a, b in zip(strs, scores):
        if "视频" in a:
            b = 1
        d[a] = b

    overall_advice = '\n'.join([
        "根据你的新闻稿件和提出的缺点，我建议你考虑以下几点进行修改：",
        "1. 文本连贯性：确保你的新闻稿件在逻辑上是连贯的，每个段落都应该有明确的主题，并且与前后段落有清晰的联系。例如，你可以在每个段落的开头或结尾添加一些过渡性的句子，以帮助读者理解你的思路。",
        "2. 图片内容：你的新闻稿件中应该包含一些与主题相关的图片，这将有助于吸引读者的注意力，并帮助他们更好地理解新闻的内容。请确保你选择的图片内容合规，不含有任何可能引起争议的元素。",
        "3. 文本信息完整：你的新闻稿件应该包含所有相关的信息，包括谁、什么、何时、何地、为什么和怎样。如果有任何重要的信息缺失，读者可能会感到困惑，不清楚新闻的全貌。",
        "4. 图片摘要质量：你的图片应该配有清晰、简洁的摘要，以帮助读者理解图片的内容。摘要应该包含图片的主要内容，以及与新闻主题相关的任何信息。",
    ])
    return overall_advice, d


def process_video(video):
    # 这里应该是你的视频处理代码
    return video


def process_image(image):
    # 这里应该是你的图片处理代码
    return image


def process_text(text):
    # 这里应该是你的文本处理代码
    return text


# 定义输入和输出接口
inputs = [gr.inputs.Video(label="视频输入"),
          gr.inputs.Image(label="图片输入"),
          gr.inputs.Textbox(label="文本输入")]

outputs = [
    gr.outputs.Textbox(label="总体指导意见"),
    gr.outputs.Label(num_top_classes=43, label="输出分数"),
]

# 创建并启动Gradio界面
iface = gr.Interface(fn=multimodal_model, inputs=inputs, outputs=outputs)
iface.launch(share=False, server_name="0.0.0.0", server_port=11234)
