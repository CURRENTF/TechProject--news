import json
import traceback
from flask import Flask, request, Response

from evaluation.Text.organization_structure import *
from evaluation.Text.subject_content import *
from evaluation.Text.expression import *
from evaluation.Text.complexity import *

from evaluation.Picture.picture_quality import *
from evaluation.baidu_api import baidu_text, baidu_picture

from tools.utils import check_json
from tools.utils import fix_output_score
from tools.baidu_utils import savePicture
from tools.vocab_process import pre_process

from myException import MyException
from config import config
from logger import logger

app = Flask(__name__)


@app.route("/evaluate_news", methods=["POST", "GET"])
def evaluate_news():
    request_json = request.json

    if request_json is None:
        return Response(json.dumps({'success': False,
                                    'message': '参数传输错误', 'code': config.INPUT_ERROR}),
                        mimetype='application/json')

    if not check_json(request_json):
        return Response(json.dumps({'success': False,
                                    'message': 'json格式输入错误', 'code': config.INPUT_ERROR}),
                        mimetype='application/json')

    paragraphs_list = []
    pictures_list = []
    videos_list = []
    audios_list = []

    for content in request_json['paragraphs']:
        if str(content['type']) == "0":
            # 段落
            if len(content['content']) == 0:
                continue
            paragraphs_list.append(content['content'])
        elif str(content['type']) == "1":
            # 图片
            pictures_list.append(content['content'])
        elif str(content['type']) == "2":
            # 视频
            videos_list.append(content['content'])
        elif str(content['type']) == "3":
            # 音频
            audios_list.append(content['content'])
        else:
            return Response(json.dumps({'success': False,
                                        'message': 'type类型不存在，请检查输入', 'code': config.INPUT_ERROR}),
                            mimetype='application/json')

    message = {}
    text_json = {"title": request_json['title'], "paragraphs": paragraphs_list}

    if len(text_json['title']) == 0:
        return Response(json.dumps({'success': False,
                                    'message': '缺少文本标题', 'code': config.INPUT_ERROR}),
                        mimetype='application/json')

    if len(text_json['paragraphs']) == 0:
        return Response(json.dumps({'success': False,
                                    'message': '缺少文本段落', 'code': config.INPUT_ERROR}),
                        mimetype='application/json')

    try:
        text_scores = get_text_scores(text_json)
        message.update(text_scores)

    except Exception as e:
        if isinstance(e, MyException):
            error_dict = e.get_error_dict()
            error_code = error_dict['code']
            error_message = error_dict['message']
        else:
            error_code = config.SERVER_ERROR
            error_message = "服务器文本处理错误"
        traceback.print_exc()

        return Response(json.dumps({'success': False,
                                    'message': error_message, 'code': error_code}),
                        mimetype='application/json')

    try:
        pictures_scores = get_picture_scores(pictures_list)
        message.update(pictures_scores)

    except Exception as e:
        if isinstance(e, MyException):
            error_dict = e.get_error_dict()
            error_code = error_dict['code']
            error_message = error_dict['message']
        else:
            error_code = config.SERVER_ERROR
            error_message = "服务器图片处理错误"

        return Response(json.dumps({'success': False,
                                    'message': error_message, 'code': error_code}),
                        mimetype='application/json')

    output = {
        'success': True,
        'message': message,
        'code': config.SUSSES_CODE
    }
    logger.process_logging("评分系统输出如下")
    logger.output_logging(message)

    return Response(json.dumps(output), mimetype='application/json')


def get_text_scores(text_json):
    paragraphs = text_json['paragraphs']
    title = text_json['title']

    logger.process_logging("开始文本指标。")
    logger.process_logging("文本标题为: %s " % title)
    logger.process_logging("文本内容为: %s " % '。\n'.join(paragraphs))

    text_json['title'] = pre_process(title)[0]
    text_json['paragraphs'] = pre_process(paragraphs)

    paragraphs = text_json['paragraphs']
    title = text_json['title']

    try:
        baidu_results = baidu_text(''.join(text_json['paragraphs']))
    except:
        baidu_results = {}

    # score1 = structure_score(paragraphs, config.BERT_MODEL)
    score2 = hierarchy_score(paragraphs, title, config.HIERARCHY_ALPHA, config.BERT_MODEL)
    logger.info("层次性得分为: %s" % score2)
    # score3 = coherence_score(paragraphs, config.BERT_MODEL)
    score4 = concise_score(paragraphs, title, config.CONCISE_ALPHA, config.BERT_MODEL)
    logger.info("简洁性得分为: %s" % score4)
    score5 = integrity_score(paragraphs, title, config.BERT_MODEL)
    logger.info("完整性得分为: %s" % score5)
    score6 = theme_score(title, paragraphs, config.BERT_MODEL)
    logger.info("主题性得分为: %s" % score6)
    score7 = popularity_score(paragraphs)
    logger.info("流行性得分为: %s" % score7)

    score8 = fluency_score(paragraphs, config.GPT_MODEL, config.FLUENCY_TOPK)
    logger.info("流畅性得分为: %s" % score8)
    score9 = diversity_score(paragraphs, config.DIVERSITY_NGRAM_N, config.DIVERSITY_BLEU_N, config.DIVERSITY_ALPHA)
    logger.info("多样性得分为: %s" % score9)
    score10 = ending_score(paragraphs, config.BERT_MODEL)
    logger.info("结尾性得分为: %s" % score10)
    score11 = theme_keep_score(score6, title, paragraphs)
    logger.info("主題一致性得分为: %s" % score10)
    score12 = theme_summary_score(score6, score10, config.SUMMARIZE_ALPHA)
    logger.info("主題概括性得分为: %s" % score10)
    score13 = ending_clear_score(score10, title, paragraphs)
    logger.info("结尾清晰性得分为: %s" % score10)
    score14 = concise_score2(score4, title, paragraphs)
    logger.info("简洁性得分为: %s" % score10)
    if baidu_results:
        score15 = meaningful_score(baidu_results['真实准确指数'], title, paragraphs)
        logger.info("百度得分为: %s" % score10)
    else:
        score15 = meaningful_score(score7, title, paragraphs)
        logger.info("百度得分为: %s" % score10)
    score16 = linear_structure_score(score2, title, paragraphs)
    logger.info("线性结构得分为: %s" % score10)
    score17 = expression_precise_score(paragraphs, config.GPT_MODEL)
    logger.info("表达准确性得分为: %s" % score10)
    score18 = content_quality_score(paragraphs, title, config.BERT_MODEL)
    logger.info("内容质量得分为: %s" % score10)
    score19 = measure_complexity_score(' '.join(paragraphs))
    logger.info("测量复杂性得分为: %s" % score10)

    logger.process_logging("结束文本指标检测。")

    message = {
        # '结构完整指数': score1,
        '层次清晰指数': score2,
        # '前后连贯指数': score3,
        '简洁凝练指数': score4,
        '信息完整指数': score5,
        '主题明确指数': score6,
        '通俗易懂指数': score7,
        '流畅度': score8,
        '多样性': score9,
        '概括升华主题指数': score10,
        '紧扣主题指数': score11,
        '概括主题指数': score12,
        '结束有力指数': score13,
        '短小精悍指数': score14,
        '言简意赅指数': score15,
        '线性结构指数': score16,
        '表达准确指数': score17,
        '选材质量指数': score18,
        '句式简洁指数': score19
    }
    message.update(baidu_results)
    for k, v in message.items():
        message[k] = fix_output_score(v)

    # 求均分，得到二元评分
    scores = [score2, score4, score5, score6, score7, score8, score9, score10, score11, score12,
              score13, score14, score15, score16, score17, score18, score19]
    total_score = sum(scores)
    average_score = total_score / len(scores)
    message['高质量稿件'] = True if average_score >= 0.8 else False

    # 根据不同的分数生成改稿意见标签
    tags = []
    for key, value in message.items():
        if value < 0.6:
            if key == '层次清晰指数':
                tags.append("层次不清晰")
            elif key == '简洁凝练指数':
                tags.append("表述啰嗦")
            elif key == '信息完整指数':
                tags.append("信息不完整")
            elif key == '主题明确指数':
                tags.append("主题不明确")
            elif key == '通俗易懂指数':
                tags.append("表达晦涩")
            elif key == '流畅度':
                tags.append("不够流畅")
            elif key == '多样性':
                tags.append("内容单一")
            elif key == '概括升华主题指数':
                tags.append("概括升华不足")
            elif key == '紧扣主题指数':
                tags.append("紧扣主题不足")
            elif key == '概括主题指数':
                tags.append("概括主题不足")
            elif key == '结束有力指数':
                tags.append("结束不够有力")
            elif key == '短小精悍指数':
                tags.append("篇幅冗余")
            elif key == '言简意赅指数':
                tags.append("表达不够简洁")
            elif key == '线性结构指数':
                tags.append("结构混乱")
            elif key == '表达准确指数':
                tags.append("表达不准确")
            elif key == '选材质量指数':
                tags.append("选材不合适")
            elif key == '句式简洁指数':
                tags.append("句式不够简洁")
            elif key == '真实准确指数':
                tags.append("真实准确度不足")
    message['改稿意见'] = tags

    return message


def get_picture_scores(pictures_list):
    if not isinstance(pictures_list, list):
        pictures_list = [pictures_list]

    byte_picture = [base64.b64decode(p) for p in pictures_list]
    savePicture(byte_picture)

    logger.process_logging("开始图片指标检测。")

    score1 = getDensity(pictures_list, config.DIAG_LEN)
    score2 = checkExposure(pictures_list, config.LOWER_BOUND, config.UPPER_BOUND)
    score3 = blurImagesDetection(pictures_list, config.DETECT_THRES)
    score4 = checkColor(pictures_list, config.COLOR_THRES)
    baidu_results = baidu_picture(pictures_list)

    logger.process_logging("图片指标计算完毕。")

    message = {
        '图片像素密度': score1,
        '图片是否曝光': score2,
        '图片是否模糊': score3,
        '图片是否饱和': score4
    }

    message.update(baidu_results)
    return message


@app.route("/test_text", methods=["POST", "GET"])
def test_text():
    if request.json is None:
        return Response(json.dumps({'success': False,
                                    'message': '参数传输错误', 'code': 1}),
                        mimetype='application/json')

    text_json = request.json
    if not check_json(text_json):
        return Response(json.dumps({'success': False,
                                    'message': 'json格式输入错误', 'code': 2}),
                        mimetype='application/json')

    try:
        message = get_text_scores(text_json)
    except:
        return Response(json.dumps({'success': False,
                                    'message': '服务器文本处理错误', 'code': 3}),
                        mimetype='application/json')

    output = {
        'success': True,
        'message': message,
        'code': 0
    }

    return Response(json.dumps(output), mimetype='application/json')


@app.route("/test_picture", methods=["POST", "GET"])
def test_picture():
    if request.files is None:
        return Response(json.dumps({'success': False,
                                    'message': '参数传输错误', 'code': 1}),
                        mimetype='application/json')

    obj_json = request.files
    pictures_list = []
    for _, content in obj_json.items():
        pictures_list.append(content.read())

    score1 = getDensity(pictures_list, 27)
    score2 = checkExposure(pictures_list, 80, 180)
    score3 = blurImagesDetection(pictures_list, 100)
    score4 = checkColor(pictures_list)
    baidu_results = baidu_picture(pictures_list)

    message = {
        '图片像素密度': score1,
        '图片是否曝光': score2,
        '图片是否模糊': score3,
        '图片是否饱和': score4
    }
    message.update(baidu_results)

    output = {
        'success': True,
        'message': message,
        'code': 0
    }

    return Response(json.dumps(output), mimetype='application/json')


def init():
    with open(config.TEST_FILE_PATH, 'r', encoding='utf-8') as f:
        text_json = json.load(f)
    get_text_scores(text_json)


if __name__ == '__main__':
    logger.process_logging("初始化系统。")
    init()
    logger.process_logging("初始化结束。")
    app.run(host='0.0.0.0', port=config.PORT)
