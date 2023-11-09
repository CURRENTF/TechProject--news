import torch
from tools.utils import add_rand_variance
from jieba.analyse import textrank


def integrity_score(paragraphs, title, model):
    # 信息完整指数

    model.eval()
    with torch.no_grad():
        logits = model([title, ''.join(paragraphs)])
        score = torch.cosine_similarity(logits[0].unsqueeze(0), logits[-1].unsqueeze(0))[0]

    return score


def theme_score(title, paragraphs, model, topk=4):
    # 主题明确指数
    context = ''.join(paragraphs)
    words = title + ' ' + context
    keywords = textrank(words, topK=topk, withWeight=False)

    model.eval()
    with torch.no_grad():
        logits = model([' '.join(keywords), context])
        score = torch.cosine_similarity(logits[0].unsqueeze(0), logits[-1].unsqueeze(0))[0]

    return score


def ending_score(paragraphs, model):
    # 概括升华主题指数
    if len(paragraphs) < 2:
        return 1.0
    model.eval()
    with torch.no_grad():
        logits = model([paragraphs[-1], ''.join(paragraphs[:-1])])
        score = torch.cosine_similarity(logits[0].unsqueeze(0), logits[-1].unsqueeze(0))[0]
    return score


def theme_keep_score(score, tile, paragraphs):
    return min(score + add_rand_variance(tile, paragraphs), 1.0)

def theme_summary_score(score1, score2, alpha):
    return min(alpha * score1 + (1.0 - alpha) * score2, 1.0)


def ending_clear_score(score, tile, paragraphs):
    return min(score + add_rand_variance(tile, paragraphs), 1.0)