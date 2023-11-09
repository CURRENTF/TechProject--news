#
#                     _ooOoo_
#                    o8888888o
#                    88" . "88
#                    (| -_- |)
#                     O\ = /O
#                 ____/`---'\____
#               .   ' \\| |// `.
#                / \\||| : |||// \
#              / _||||| -:- |||||- \
#                | | \\\ - /// | |
#              | \_| ''\---/'' | |
#               \ .-\__ `-` ___/-. /
#            ___`. .' /--.--\ `. . __
#         ."" '< `.___\_<|>_/___.' >'"".
#        | | : `- \`.;`\ _ /`;.`/ - ` : | |
#          \ \ `-. \_ __\ /__ _/ .-` / /
#  ======`-.____`-.___\_____/___.-`____.-'======
#                     `=---='
#
#  .............................................
#           佛祖保佑             永无BUG
#

from tools.web_process import *
from tools.vocab_process import pre_process

from evaluation.Text.organization_structure import *
from evaluation.Text.subject_content import *
from evaluation.Text.expression import *

from config import BERT_MODEL, GPT_MODEL
from config import (
    HIERARCHY_ALPHA,
    CONCISE_ALPHA,
    DIVERSITY_NGRAM_N,
    DIVERSITY_BLEU_N,
    DIVERSITY_ALPHA,
    FLUENCY_TOPK
)


def evaluate_news(text_json):
    text_json['title'] = pre_process(text_json['title'])[0]
    text_json['paragraphs'] = pre_process(text_json['paragraphs'])

    paragraphs = text_json['paragraphs']
    title = text_json['title']

    score1 = structure_score(paragraphs, BERT_MODEL)
    score2 = hierarchy_score(paragraphs, title, HIERARCHY_ALPHA, BERT_MODEL)
    score3 = coherence_score(paragraphs, BERT_MODEL)
    score4 = concise_score(paragraphs, title, CONCISE_ALPHA, BERT_MODEL)
    score5 = integrity_score(paragraphs, title, BERT_MODEL)
    score6 = theme_score(title, paragraphs, BERT_MODEL)
    score7 = popularity_score(paragraphs)

    score8 = fluency_score(paragraphs, GPT_MODEL, FLUENCY_TOPK)
    score9 = diversity_score(paragraphs, DIVERSITY_NGRAM_N, DIVERSITY_BLEU_N, DIVERSITY_ALPHA)

    print('结构完整指数：', score1)
    print('层次清晰指数: ', score2)
    print('前后连贯指数: ', score3)
    print('简洁凝练指数: ', score4)
    print('信息完整指数：', score5)
    print('主题明确指数：', score6)
    print('通俗易懂指数: ', score7)

    print('流畅度：', score8)
    print('多样性: ', score9)
    print()


if __name__ == '__main__':
    import json

    with open('res/spider.json', 'r', encoding='utf-8') as f:
        text_json = json.load(f)

    with open('res/spider.txt', 'r', encoding='utf-8') as fd:
        text = fd.readlines()
    text = ''.join(text)[1:-1]

    text_json2 = web_content_to_json(text)

    evaluate_news(text_json)
    evaluate_news(text_json2)
