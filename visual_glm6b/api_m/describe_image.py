import torch
from transformers import AutoTokenizer, AutoModel
import os
import sys
import pickle
import random
sys.path.append("./visual_glm6b/")
sys.path.append(".")
from tools.tools import download_image, print_separator
from tools.baidu_ocr import ocr
from glm_mplug_vllama.api.glm_text_only import TextModel
from matplotlib import pyplot as plt
import cv2
from visual_glm6b.api_m import video_reader

debugging = True


def is_jupyter():
    if 'ipykernel' in sys.modules:
        return True
    else:
        return False


run_on_jupyter = is_jupyter()


class VisualGLM:
    def __init__(self, cuda_id=0, load_in_8bit=False, load_in_4bit=False):
        if load_in_8bit or load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are not supported.")
        tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/visualglm-6b",
                                          trust_remote_code=True,
                                          device_map=cuda_id,
                                          load_in_8bit=load_in_8bit,
                                          load_in_4bit=load_in_4bit,
                                          torch_dtype=torch.float16).half().cuda(cuda_id).eval()
        self.tokenizer = tokenizer
        self.model = model

    def read_image(self, image_path):
        if "http" in image_path:
            image_path = download_image(image_path, "./pics")
        # check image_path
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found.")
        return image_path

    # local image only
    @print_separator(debugging)
    def describe_1_image(self, image_path, glm_model=None, no_ocr=True):
        image_path = self.read_image(image_path)
        ocr_content = ""
        if not no_ocr:
            ocr_content = ocr(image_path)
        if len(ocr_content) < 15:
            response, history = self.model.chat(self.tokenizer, image_path,
                                                "5个字描述这张图片，只描述图片里有的，不要延伸：",
                                                history=[])
        else:
            if glm_model is not None:
                ocr_content = glm_model.summary(ocr_content)
            response = ocr_content

        if debugging:
            if run_on_jupyter:
                img = cv2.imread(image_path)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
            print("image_path", image_path)
            print("image_content", response)

        return response

    @print_separator(debugging)
    def describe_image(self, image_path, glm_model=None, no_ocr=True):
        if debugging:
            print("image_path", image_path)
        if isinstance(image_path, list):
            return '\n\n'.join([self.describe_1_image(img_path, glm_model, no_ocr) for img_path in image_path])
        else:
            return self.describe_1_image(image_path, glm_model, no_ocr)

    @print_separator(debugging)
    def image_match_topic_check(self, image_path, topic):
        image_path = self.read_image(image_path)
        examples = [
            '示例1：'
            "这是一张新闻配图，图中有一只袋鼠正在草地上跳跃，请你根据图片内容判断这张配图和新闻标题'袋鼠在澳大利亚的生活'的关联程度，并在以下选项中选择一个最恰当的回答："
            "A.非常相关 B.比较相关 C.勉强相关 D.不相关。"
            '正确的回答应该是"A.非常相关"，因为图片和新闻标题都是关于袋鼠的。'
            '示例2：'
            "这是一张新闻配图，图中有一只袋鼠正在草地上跳跃，请你根据图片内容判断这张配图和新闻标题'澳大利亚的自然景色'的关联程度，并在以下选项中选择一个最恰当的回答："
            "A.非常相关 B.比较相关 C.勉强相关 D.不相关。"
            '正确的回答应该是"B.比较相关"，因为虽然图片中的袋鼠是澳大利亚的一部分，但新闻标题更广泛地涵盖了澳大利亚的自然景色，并不仅仅是关于袋鼠。'
            "示例3："
            "这是一张新闻配图，图中有一只袋鼠正在草地上跳跃，请你根据图片内容判断这张配图和新闻标题'非洲大象的生活'的关联程度，并在以下选项中选择一个最恰当的回答："
            "A.非常相关 B.比较相关 C.勉强相关 D.不相关。"
            '正确的回答应该是"D.不相关"，因为图片是关于袋鼠的，而新闻标题是关于非洲大象的，两者之间没有直接的关联。'
        ]
        response, history = self.model.chat(self.tokenizer, image_path,
                                            f"这是一张新闻配图，请你判断这张配图和新闻标题{topic}是否相关，请在相关选项()里选择："
                                            f"A.非常相关 B.比较相关 C.勉强相关 D.不相关\n"
                                            f"只输出选项就好，是个单选题\n"
                                            # f"比如\n"
                                            # f"Answer: B.比较相关"
                                            ,
                                            history=[])
        if debugging:
            if run_on_jupyter:
                img = cv2.imread(image_path)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
            print("image_path", image_path)
            print("topic", topic)
            print("response", response)

        options = "ABCD"
        for idx, opt in enumerate(options):
            if opt in response[:8]:
                return idx

        return 3
        # return response

    @print_separator(debugging)
    def describe_video(self, video_path, k_frames=5):
        # 在随机地址下保存视频的帧
        output_path = f"./videos/{random.randint(0, int(1e9))}"
        video_contents = []
        for image_path, timestamp in video_reader.extract_k_frames_with_timestamp(video_path, output_path, k_frames):
            content_ = self.describe_image(image_path)
            video_contents.append((content_, timestamp))
            if debugging:
                print("image_path", image_path)
                print("image_content", content_)

        return video_contents


def get_image_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                yield os.path.join(root, file)


if __name__ == "__main__":
    # print(f"running on jupyter = {run_on_jupyter}")
    run_on_jupyter = True
    debugging = True
    vglm = VisualGLM(
        cuda_id=0,
        # load_in_8bit=True,
        # load_in_4bit=True,
    )
    # cnt = 0
    # for image_path in get_image_files("./glm_mplug_vllama/pics"):
    #     print(image_path)
    #     print(vglm.describe_image(image_path))
    #     print("-" * 80)
    #     cnt += 1
    #     if cnt == 20:
    #         break

    article_list = pickle.load(open("article_list.pkl", "rb"))
    article_list = article_list[:20]
    random.shuffle(article_list)
    std = 2
    cnt = 0
    for article in article_list:
        title = article[0]
        img_path = None
        for line in article:
            if "png" in line or "jpg" in line:
                img_path = line
                break
        if img_path is None:
            continue
        # img_content = vglm.describe_image(img_path)
        score = vglm.image_match_topic_check(img_path, title)
        if score < std:
            cnt += 1
        print(score)

    print("pos acc", cnt / len(article_list))

    acc = 0
    for idx, article in enumerate(article_list):
        title = article_list[idx - 1][0]
        img_path = None
        for line in article:
            if "png" in line or "jpg" in line:
                img_path = line
                break
        if img_path is None:
            continue
        score = vglm.image_match_topic_check(img_path, title)
        if score >= std:
            acc += 1
    acc /= len(article_list)
    print("neg acc:", acc)
