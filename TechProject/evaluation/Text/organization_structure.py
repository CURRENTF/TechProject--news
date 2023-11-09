import torch
from tools.utils import add_rand_variance

def structure_score(paragraphs, model):
    # 结构完整指数
    if len(paragraphs) < 3:
        return 1.0

    model.eval()
    with torch.no_grad():
        logits = model(paragraphs)

        head = logits[0]
        tail = logits[-1]
        contexts = logits[1:-1]

        score = torch.cosine_similarity(torch.vstack([head, tail]).unsqueeze(1),
                                        contexts.unsqueeze(0),
                                        dim=2)
        score = score.reshape(-1).mean()

    return score


def hierarchy_score(paragraphs, title, alpha, model):
    # 层次清晰指数
    if len(paragraphs) < 3:
        return 1.0

    model.eval()
    with torch.no_grad():
        score_1 = structure_score(paragraphs, model)

        logits = model([title, ''.join(paragraphs)])
        score_2 = torch.cosine_similarity(logits[0].unsqueeze(0), logits[-1].unsqueeze(0))[0]

        score = alpha * score_1 + (1.0 - alpha) * score_2

    return score


def linear_structure_score(score, tile, paragraphs):
    return min(score + add_rand_variance(tile, paragraphs), 1.0)


def coherence_score(paragraphs, model):
    # 前后连贯指数
    if len(paragraphs) < 2:
        return 1.0
    
    model.eval()
    with torch.no_grad():
        logits = model(paragraphs)

        score = torch.cosine_similarity(logits[:-1], logits[1:])
        score = score.reshape(-1).mean()

    return score
