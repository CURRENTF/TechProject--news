from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import sys
sys.path.append("scripts/inference")
from . import baidu_ocr


def init_img_caption_model(device='gpu'):
    model_id = 'damo/mplug_image-captioning_coco_large_en'
    # model_id = 'damo/mplug_image-captioning_coco_base_zh'
    pipeline_caption = pipeline(Tasks.image_captioning, model=model_id, device=device)
    return pipeline_caption


def mplug_img_des(mplug, img_path, ocr=False):
    # return {"caption": mplug(img_path)["caption"], "ocr": baidu_ocr.ocr(img_path)}
    caption_res = mplug(img_path)["caption"]
    ocr_res = ""
    if ocr:
        ocr_res = baidu_ocr.ocr(img_path)
    # ocr_res = ""
    return caption_res, ocr_res


if __name__ == "__main__":
    input_caption = 'https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/image_captioning.png'
