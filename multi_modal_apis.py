from glm_mplug_vllama.api.glm_text_only import TextModel
from visual_glm6b.api_m.describe_image import VisualGLM
import pickle
import random
from copy import deepcopy
from visual_glm6b.api_m import video_voice_recognize


#  related_paragraphs >= 1
def image_text_layout(text_model, images_path, images_content, article_imgs, related_paragraphs=1):
    cnt = 0
    score = 0
    only_text = deepcopy(article_imgs)
    i_idx = 0
    for idx, x in only_text:
        if x in images_path:
            only_text[idx] = f"[image]{i_idx}"
            i_idx += 1
    for idx, paragraph in enumerate(article_imgs):
        if "[image]" in paragraph:
            image_content = images_content[int(paragraph.split("[image]")[-1])]
            text = '\n'.join(only_text[idx - related_paragraphs:idx + related_paragraphs])
            score += text_model.img_match_content(image_content, text)
            cnt += 1
    return score / cnt


def video_text_layout(text_model, video_contents, article_text):
    num_imgs = len(video_contents)
    num_text = len(article_text)
    score = 0
    for i in range(num_imgs):
        score += text_model.img_match_content(video_contents[i],
                                              article_text[i * num_text // num_imgs: (i + 1) * num_text // num_imgs])
    return score / num_imgs


def return_image_multi_modal_scores(text_model, image_model, images, article_imgs, article_text, topic):
    scores = {}
    image_contents = [image_model.describe_1_image(image) for image in images]

    scores["image_match_topic"] = text_model.img_match_topic_k(image_contents, topic, k=2)
    scores["image_match_content"] = text_model.img_match_content(image_contents, article_text)
    scores["image_text_layout"] = image_text_layout(text_model, images, image_contents, article_imgs)
    return scores


def return_video_multi_modal_scores(text_model, image_model, video_path, topic):
    scores = {}
    video_contents = image_model.describe_video(video_path)
    video_contents = [i[0] for i in video_contents]
    article_text = video_voice_recognize.upload_and_convert(video_path)
    scores["video_match_topic"] = text_model.img_match_topic_k('\n'.join(video_contents), topic, k=2)
    scores["video_match_content"] = text_model.img_match_content('\n'.join(video_contents), article_text)
    scores["video_text_layout"] = video_text_layout(text_model, video_contents, article_text)
    return scores