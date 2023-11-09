import torch
import hashlib
from config import config


def check_json(js):
    for key in config.JSON_KEY_LIST:
        if key not in js:
            return False

    return True


def tensor_to_cuda(tensor_dict):
    for k, v in tensor_dict.items():
        tensor_dict[k] = v.to(config.DEVICE)
    return tensor_dict


def fix_output_score(score, _int=False):
    if torch.is_tensor(score):
        score = score.item()

    if _int:
        return int(score)
    return round(score, config.REST_NUM)


def add_rand_variance(title, paragraphs, control_para=1):
    # 将标题和段落内容拼接在一起
    text = title + " " + ' '.join(paragraphs)

    # 计算哈希值
    hashed_text = hashlib.sha256(text.encode('utf-8')).hexdigest()

    # 将哈希值转换为一个-1到1之间的小数
    decimal = int(hashed_text, 16) / (16 ** len(hashed_text)) * 2 - 1

    # 将小数映射到-0.1到0.1之间
    number = decimal * 0.1 * control_para

    return number
