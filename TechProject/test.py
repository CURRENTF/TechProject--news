from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'/data300GB/sdu/stanford-corenlp-4.5.5', lang='zh')

sentence = '斯坦福大学自然语言处理包StanfordNLP'

print(nlp.word_tokenize(sentence))  # 分词

print(nlp.pos_tag(sentence))  # 词性标注

print(nlp.ner(sentence))  # 实体识别

print(nlp.parse(sentence))  # 语法树

print(nlp.dependency_parse(sentence))  # 依存句法

nlp.close()  # Do not forget to close! The backend server will consume a lot memery.