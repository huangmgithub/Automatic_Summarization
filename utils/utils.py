import jieba
import re, os


def cut(string):
    """
    切词
    :param string:
    :return:
    """
    return list(jieba.cut(string))

def token(string):
    """
    正则匹配文字
    :param string:
    :return:
    """
    return re.findall(r'[\d\w]+', string)

def get_stopwords(filename):
    """
    停用词
    :param filename:
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f])


def split_sentence(text):
    """
    分割文本
    :param text:
    :return:
    """
    text = re.sub(r'\s+', '', text)
    # pattern = re.compile('[。，,.:：]')
    pattern = re.compile('[。?!！？.]')
    sentence_segments = pattern.sub(' ', text).split()
    return sentence_segments

# def split_sentence(text):
#     """分割文本"""
#     # pattern = re.compile('[。，,.]：')
#     # sentence_segments = pattern.sub(' ', sentence).split()
#     text = re.sub(r'\s+', '', text)
#     sentences = list(SentenceSplitter.split(text))
#     return sentences

def get_words_list(text, stopwords):
    """
    获得句子的全部词
    :param text:
    :param stopwords:
    :return: 返回词列表 [[],[]..]
    """
    word_list = []
    for sentence in split_sentence(text):
        tmp = cut(''.join(token(sentence)))
        word_list.append([word for word in tmp if word not in stopwords])
    return word_list