from glm_mplug_vllama.api.glm_text_only import TextModel
from visual_glm6b.api_m.describe_image import VisualGLM
import pickle
import random


def test1():
    glm = TextModel(cuda_id=0, load_in_8bit=True)
    vglm = VisualGLM(cuda_id=1)
    article_list = pickle.load(open("article_list.pkl", "rb"))
    random.shuffle(article_list)
    article_list = article_list[:20]
    acc = 0
    for article in article_list:
        title = article[0]
        img_path = None
        for line in article:
            if "png" in line or "jpg" in line:
                img_path = line
                break
        if img_path is None:
            continue
        img_content = vglm.describe_image(img_path)
        score = glm.img_match_topic_k(img_content, title)
        if score >= 60:
            acc += 1
    acc /= len(article_list)
    print("pos acc:", acc)

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
        img_content = vglm.describe_image(img_path)
        score = glm.img_match_topic_k(img_content, title)
        if score < 60:
            acc += 1
    acc /= len(article_list)
    print("neg acc:", acc)


def test_image_topic_match_by_summary(std_score=60):
    article_list = pickle.load(open("article_list.pkl", "rb"))
    random.shuffle(article_list)
    # article_list = article_list[:20]
    acc = 0
    for info in article_list:
        title = info["topic"]
        if len(info["imgs"]) == 0:
            continue
        img_path = info["imgs"][0]
        img_content = vglm.describe_image(img_path, glm)
        score = glm.img_match_topic_by_summary(img_content, title)
        if score >= std_score:
            acc += 1
    acc /= len(article_list)
    print(f"pos acc ({std_score}):", acc)

    acc = 0
    for idx, info in enumerate(article_list):
        title = article_list[idx - 1]["topic"]
        if len(info["imgs"]) == 0:
            continue
        img_path = info["imgs"][0]
        img_content = vglm.describe_image(img_path, glm)
        score = glm.img_match_topic_by_summary(img_content, title)
        if score < std_score:
            acc += 1
    acc /= len(article_list)
    print(f"neg acc ({std_score}):", acc)


def test_images_topic_match_by_summary(std_score):
    article_list = pickle.load(open("article_list.pkl", "rb"))
    random.shuffle(article_list)
    # article_list = article_list[:20]
    acc = 0
    for info in article_list:
        title = info["topic"]
        if len(info["imgs"]) == 0:
            continue
        img_path = info["imgs"]
        img_content = vglm.describe_image(img_path, glm)
        score = glm.img_match_topic_by_summary(img_content, title)
        if score >= std_score:
            acc += 1
    acc /= len(article_list)
    print(f"pos acc ({std_score}):", acc)

    acc = 0
    for idx, info in enumerate(article_list):
        title = article_list[idx - 1]["topic"]
        if len(info["imgs"]) == 0:
            continue
        img_path = info["imgs"]
        img_content = vglm.describe_image(img_path, glm)
        score = glm.img_match_topic_by_summary(img_content, title)
        if score < std_score:
            acc += 1
    acc /= len(article_list)
    print(f"neg acc ({std_score}):", acc)


if __name__ == "__main__":
    glm = TextModel(cuda_id=0, load_in_8bit=False)
    vglm = VisualGLM(cuda_id=1)
    test_images_topic_match_by_summary(60)
    test_images_topic_match_by_summary(80)
