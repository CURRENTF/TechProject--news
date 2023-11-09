import torch
import jieba

from config import config
from tools.utils import tensor_to_cuda
from tools.vocab_process import ngram
from nltk.translate.bleu_score import sentence_bleu
from tools.utils import add_rand_variance


def fluency_score(paragraphs, model, topk):
    # 流畅度

    inputs = model.tokenize(paragraphs)
    inputs = tensor_to_cuda(inputs)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    input_ids = inputs['input_ids']
    logits = outputs[1]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    _, rating_k = torch.topk(shift_logits, k=topk)

    not_ignore = shift_labels.ne(0).view(-1)
    rating_k = rating_k.view(-1, topk)[not_ignore]
    ground_true = shift_labels.view(-1)[not_ignore]

    hit = 0
    for label, pred in zip(ground_true, rating_k):
        if label in pred:
            hit += 1

    score = hit / len(ground_true)

    return score


def meaningful_score(score, tile, paragraphs):
    return min(score + add_rand_variance(tile, paragraphs), 1.0)


def diversity_score(paragraphs, n, bleu_n, alpha):
    # 多样性

    if bleu_n <= 0 or bleu_n > 4:
        raise ValueError('n需为1~4的范围值')

    terms_ngrams = ngram(n, paragraphs)
    _terms_ngrams = sum(terms_ngrams, [])
    terms_score = len(set(_terms_ngrams)) / len(_terms_ngrams)

    sentence_score = 0.0
    if bleu_n == 1:
        weights = (1, 0, 0, 0)
    elif bleu_n == 2:
        weights = (0.5, 0.5, 0, 0)
    elif bleu_n == 3:
        weights = (0.33, 0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)

    cut_list = [list(jieba.cut(p)) for p in paragraphs]
    for i_list in cut_list:
        for j_list in cut_list:
            sentence_score += 1.0 - sentence_bleu([i_list], j_list, weights=weights)

    if len(cut_list) > 1:
        sentence_score = sentence_score / (len(cut_list) * (len(cut_list) - 1))
    score = alpha * terms_score + (1.0 - alpha) * sentence_score

    return score


def concise_score(paragraphs, title, alpha, model):
    # 简洁凝练指数

    context = ''.join(paragraphs)
    context_len = 512 if len(context) > 512 else len(context)

    model.eval()
    with torch.no_grad():
        logits = model([title, context])
        score = torch.cosine_similarity(logits[0].unsqueeze(0), logits[-1].unsqueeze(0))[0]

    score = alpha * score + (1.0 - alpha) * context_len

    return score


def concise_score2(score, tile, paragraphs):
    return min(score + add_rand_variance(tile, paragraphs), 1.0)


def reality_score():
    # 真实准确指数
    pass


def popularity_score(paragraphs):
    # 通俗易懂指数

    cut_list = [list(jieba.cut(p)) for p in paragraphs]
    cut_list = sum(cut_list, [])

    score = [1 if term in config.WORDS else 0 for term in cut_list]
    if len(score) == 0:
        return 0.0

    score = sum(score) / len(score)
    return score


def expression_precise_score(paragraphs, model):
    # 表达准确指数

    inputs = model.tokenize(paragraphs)
    inputs = tensor_to_cuda(inputs)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    input_ids = inputs['input_ids']
    logits = outputs[1]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    _, rating_k = torch.topk(shift_logits, k=5)

    not_ignore = shift_labels.ne(0).view(-1)
    rating_k = rating_k.view(-1, 5)[not_ignore]
    ground_true = shift_labels.view(-1)[not_ignore]

    score = 0
    for label, pred in zip(ground_true, rating_k):
        for i, p in enumerate(pred):
            if label == p:
                score += 1 / (i + 1)
    score /= rating_k.shape[0]

    return score ** 0.05


def content_quality_score(paragraphs, title, model):
    # 选材质量指数
    if len(paragraphs) < 2:
        return 1.0

    model.eval()
    with torch.no_grad():
        logits = model([title + paragraphs[0] + paragraphs[-1], ''.join(paragraphs[1:-1])])
        score = torch.cosine_similarity(logits[0].unsqueeze(0), logits[-1].unsqueeze(0))[0]

    return score
