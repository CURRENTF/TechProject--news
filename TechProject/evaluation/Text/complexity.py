# -*- encoding: utf-8 -*-

import math
import re

from nltk.tree import Tree
from stanfordcorenlp import StanfordCoreNLP

CoreNLPPath = r'/data300GB/sdu/stanf3/stanford-corenlp-full-2018-10-05'


class SentenceComplexityAnalyzer(object):
    _nlp = None

    def __init__(self):
        if SentenceComplexityAnalyzer._nlp is None:
            SentenceComplexityAnalyzer._nlp = StanfordCoreNLP(CoreNLPPath, lang='zh')

    @staticmethod
    def normal_cut_sentence(text):
        text = re.sub('([。！？\?])([^’”])', r'\1\n\2', text)
        text = re.sub('(\.{6})([^’”])', r'\1\n\2', text)
        text = re.sub('(\…{2})([^’”])', r'\1\n\2', text)
        text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])', r'\1\n\2', text)
        return text.split("\n")

    # 分句
    @staticmethod
    def cut_sentence_with_quotation_marks(text):
        p = re.compile("“.*?”")
        list = []
        index = 0
        length = len(text)
        for i in p.finditer(text):
            temp = ''
            start = i.start()
            end = i.end()
            for j in range(index, start):
                temp += text[j]
            if temp != '':
                temp_list = SentenceComplexityAnalyzer.normal_cut_sentence(temp)
                list += temp_list
            temp = ''
            for k in range(start, end):
                temp += text[k]
            if temp != ' ':
                list.append(temp)
            index = end
        return list

    @staticmethod
    def get_tree_info(tree):
        if not isinstance(tree, Tree):
            return 0, 0
        width = len(tree)
        nodes = 1
        for i in range(len(tree)):
            res = SentenceComplexityAnalyzer.get_tree_info(tree[i])
            width = max(width, res[0])
            nodes += res[1]
        return width, nodes

    def get_text_score(self, text):
        sentences = SentenceComplexityAnalyzer.normal_cut_sentence(text)
        max_tree_height = 0
        max_tree_width = 0
        max_node_count = 0
        avg_tree_width = 0
        valid_cnt = 0
        for sentence in sentences:
            if len(sentence.strip()) <= 0:
                continue
            nlp_tree = Tree.fromstring(SentenceComplexityAnalyzer._nlp.parse(sentence.strip()))
            width, nodes = self.get_tree_info(nlp_tree)
            max_tree_width = max(width, max_tree_width)
            max_tree_height = max(nlp_tree.height(), max_tree_height)
            max_node_count = max(nodes, max_node_count)
            valid_cnt += 1
            avg_tree_width += width
        # 空句子的结构复杂评分当然是100
        if valid_cnt == 0:
            return 100
        avg_tree_width /= valid_cnt

        # Scoring via AST tree statistics
        #
        max_tree_height_score = max(0.0, 25 - max_tree_height / 2)
        max_tree_width_score = max(0.0, 25 - max_tree_width * 1.5)
        max_node_count_score = max(0.0, 25 - math.sqrt(max_node_count))
        avg_tree_width_score = max(0.0, 25 - avg_tree_width * 2.0)
        # print(max_tree_height_score, max_tree_width_score, max_node_count_score, avg_tree_width_score)
        return max_node_count_score + max_tree_width_score + max_tree_height_score + avg_tree_width_score


def measure_complexity_score(text):
    analyzer = SentenceComplexityAnalyzer()
    return (analyzer.get_text_score(text) / 100.0) ** 0.07


if __name__ == '__main__':
    print(measure_complexity_score("""中国外交官员涉嫌参与威胁加拿大国会议员的事件，已演变为中加两国互相驱逐外交官。中国外交部发言人周二（5月9日）宣布，将加拿大驻上海总领馆领事甄逸慧（Jennifer Lalonde）列为“不受欢迎的人”，并要求其在5月13日前离开中国。
在反对党的强烈要求下，加拿大周一决定驱逐中国驻多伦多领事馆的外交官员赵巍。“经过慎重考虑所有相关因素，加拿大已决定宣布赵巍先生为不受欢迎的人。”外交部长梅兰妮·乔利（Melanie Joly）重申，加拿大不会接受其内政遭到任何形式的外国干预。
加拿大国会2021年通过反对党保守党议员庄文浩（Michael Chong）提出的动议，宣布中国对待维吾尔穆斯林少数民族的方式是种族灭绝。此后，北京宣布对庄文浩实施制裁，中国情报机构据信还试图通过威胁他的亲戚来惩罚他，而赵巍被认为参与了这场阴谋。
加拿大外交部上周召见中国驻加拿大大使丛培武，表明了加拿大不会容忍这种干涉。中国外交部则否认有任何不当行为，坚称这起丑闻是“加拿大个别政客和媒体的炒作”。
中国外交部官网周二上午通过发言人公告：“5月9日，加拿大政府宣布将中国驻多伦多总领馆一名外交官列为不受欢迎的人。中方对此表示强烈谴责和坚决反对，已向加方提出严正交涉和强烈抗议。针对加方无理行径，中方决定采取对等反制措施，将加拿大驻上海总领馆领事甄逸慧列为不受欢迎的人，已要求其5月13日前离华。中方保留作出进一步反应的权利。”不过公告并未交代为何选择驱逐甄逸慧。
外交部发言人汪文斌同日在例行记者会上称，中方从不干涉别国内政，所谓“中国干涉加拿大内政”完全是无稽之谈，是对中方的污蔑抹黑和基于意识形态的政治操弄。加方以莫须有的罪名宣布中国外交人员为“不受欢迎的人”，违反国际关系基本准则，蓄意破坏中加关系，性质十分恶劣。
汪文斌称，加政府基于谎言，采取严重损害中国外交领事人员合法权益的错误举动，中方绝不接受，奉劝加方立即停止无理挑衅，如加方不听劝告，肆意妄为，中方必将坚决有力回击，由此产生的一切后果必须由加方承担。"""
                                   ))

    print(measure_complexity_score("""因为残酷镇压民主抗议并引发内战而被逐出阿拉伯国家联盟（Arab League）的叙利亚，时隔12年后于本周正式重返这一影响力巨大的地区联盟。
此举是大马士革与阿拉伯地区各国政府关系解冻的又一证据。
叙利亚的阿盟成员国资格恢复，意味着叙利亚总统巴沙尔·阿萨德（Bashar al-Assad）可以出席本月稍后于沙特阿拉伯举行的联盟峰会。
英美对这一举动表示批评。
美国国务院一名发言人称，叙利亚不值得被恢复成员国资格，但是美国支持阿拉伯联盟致力于解决叙利亚危机的长远目标。
英国外交、联邦及发展事务国务大臣阿哈默德勋爵（Lord Ahmad）表示，英国仍然“反对与阿萨德政权接触”，指阿萨德仍在继续“扣押、虐待及杀害无辜的叙利亚人”。
叙利亚外交部一份声明称，已经接收到阿盟的决定，并呼吁阿拉伯世界加强合作。
22个成员国中有13国的外交官较早前在开罗进行了会议，决定恢复叙利亚的阿盟成员国资格。
代表们强调有必要结束叙利亚内战，以及由此而来的难民偷渡及毒品走私问题。
BBC在去年曾报道，叙利亚越发严重的贫穷问题以及缺少工作机会的状况，令很多人转而进行毒品交易。
埃及、沙特阿拉伯、黎巴嫩、约旦和伊拉克等国将会组成委员会，帮助叙利亚实现相关目标。
阿拉伯联盟秘书长艾哈迈德·阿布乌尔·盖特（Ahmed Aboul Gheit）表示，此举是“逐步”解决叙利亚危机进程的开始。
他强调，有关决定并不意味着阿拉伯国家将恢复与叙利亚的外交关系，因为这将取决于每一个国家各自独立的决定。
根据联合国估计，叙利亚内战已经造成超过30万平民死亡，超过10万被扣押或者失踪。
内战前的总人口约有一半，即2100万人被迫无家可归，当中有些在叙利亚境内游离失所，有些则逃往国外成为难民。
在西北部由反叛武装控制的伊德利卜，失去家园的叙利亚人表示，他们对阿拉伯联盟的决定感到震惊。
“阿拉伯领袖们没有帮助我们，让我们离开那些令我们受苦的营地，而是为那个手上沾了我们的血的罪犯和杀人犯洗白，”法新社引述一名男子说。
另一名男子则说，阿盟将会付出“最惨重的代价”。
阿萨德在俄罗斯的帮助下于2015年开始夺回国家的控制权，迫使邻国开始考虑一个有阿萨德存在的未来。
今年2月土耳其与叙利亚发生毁灭性的地震灾害后，阿盟加快了与叙利亚重修关系的步伐。
本周较早前，伊朗总统易卜拉欣·莱希（Ebrahim Raisi）拜访阿萨德——一些分析人士指该 次到访给阿拉伯各国施加了额外压力，将叙利亚带回组织。"""))

    print(measure_complexity_score("""近平总书记在参加十四届全国人大一次会议江苏代表团审议时强调，推动高质量发展，必须完整、准确、全面贯彻新发展理念；必须更好统筹质的有效提升和量的合理增长；必须坚定不移深化改革开放、深入转变发展方式；必须以满足人民日益增长的美好生活需要为出发点和落脚点。“四个必须”展示了对高质量发展的最新规律性把握和创新性认识，为牢牢把握高质量发展这个首要任务提供了科学的方法论遵循，是习近平新时代中国特色社会主义思想世界观和方法论的生动体现。积极探索中国式现代化的开创性实践、谱写高质量发展新篇章，要深刻把握“四个必须”的方法论要求，坚持好、运用好贯穿其中的立场观点方法。
　　体现系统观念的创新运用
　　系统观念是具有基础性的思想和工作方法，是推进高质量发展必须遵循的科学准则和基本方法。推动高质量发展就是新发展理念的具体实践，而新发展理念本身是一个系统的理论体系，这就要求我们始终要坚持系统观念，以创新、协调、绿色、开放、共享的内在统一来把握发展、衡量发展、推动发展。
　　创新发展、协调发展、绿色发展、开放发展、共享发展五个方面之间是相互联系、有机统一的。它们所构成的系统理论体系科学回答了高质量发展的目的、动力、方式、路径等一系列理论和实践问题，阐明了高质量发展的政治立场、价值导向、发展模式、发展道路等重大问题。在推进高质量发展的实践中，必须坚持系统思维，协同发力，既不能畸轻畸重、以偏概全，又要结合实际、突出重点，防止生搬硬套。"""
                                   ))
