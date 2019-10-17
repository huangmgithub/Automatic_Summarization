import jieba
import re, os
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec

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
    pattern = re.compile('[。，,.:：]')
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

def get_content_file(source_filename, save_filename, stopwords):
    """
    将文本保存至文件
    :param source_filename:
    :param save_filename:
    :param stopwords:
    :return:
    """
    data = pd.read_csv(source_filename, encoding='gb18030')
    data = data.fillna('')
    with open(save_filename, 'w', encoding='utf-8') as f:
        for content in data['content'].tolist():
            words_list = get_words_list(content, stopwords)
            s = ' '.join([' '.join(words) for words in words_list])
            f.write(str(s) + '\n')

def get_weight_and_word_vec_file(model_filename, weight_filename, word_vec_filename):
    """
    将词向量和词频保存为文件
    :param text_file:
    :return:
    weight_file: each line is w word and its frequency, separate by space
    word_vec_file: each line is a word and its word vector
    both -> line[0] is word, line[1:] is vector, separate by space
    """
    model = Word2Vec.load(model_filename)
    vec_lookup = model.wv.vocab
    total_words = sum(vec_lookup[word].count for word in vec_lookup)
    with open(weight_filename, 'w', encoding = 'utf-8') as f_weight, \
        open(word_vec_filename, 'w', encoding = 'utf-8') as f_word_vec:
        for word in tqdm(vec_lookup):
            freq = vec_lookup[word].count / total_words
            f_weight.write(word + ' ' + str(freq) + '\n')
            f_word_vec.write(word + ' ' + ' '.join(map(str, model.wv[word])) + '\n')

if __name__ == "__main__":
    source_filename = '../data/sqlResult_1558435.csv'
    save_filename = '../data/news.txt'
    stopwords_filename = '../data/chinese_stopwords.txt'
    model_filename = '../model/word2vec/wiki.zh.model'
    weight_filename = '../data/weight_file.txt'
    word_vec_filename = '../data/word_vec_file.txt'

    stopwords = get_stopwords(stopwords_filename)
    get_content_file(source_filename, save_filename, stopwords)

    # 保存词频和词向量文件
    get_weight_and_word_vec_file(model_filename, weight_filename, word_vec_filename)



