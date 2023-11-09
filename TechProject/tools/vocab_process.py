import re
import nltk

from config import config


def pre_process(paragraph_list, pattern=None):
    if not isinstance(paragraph_list, list):
        paragraph_list = [paragraph_list]

    if pattern is None:
        pattern = re.compile(r'[^\u4e00-\u9fa5]')

    paragraph_list = [re.sub(pattern, '', p) for p in paragraph_list]
    return paragraph_list


def ngram(n, paragraph_list, stop_word_list=None):
    stop_word_set = set(stop_word_list) if stop_word_list else config.STOP_WORDS_LIST

    ngram_list = []
    for sentence in paragraph_list:
        temp_list = []
        all_ngrams = nltk.ngrams(sentence, n)

        for ngram in all_ngrams:
            for tokens in ngram:
                if any(token not in stop_word_set for token in tokens):
                    temp_list.append(''.join(ngram))

        ngram_list.append(temp_list)

    return ngram_list
