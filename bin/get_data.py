import numpy as np
import os

# reference: https://github.com/PrincetonML/SIF
BASE_DATA_DIR = './data/'

def getWordmap(text_filename):
    """
    获得词embedding table and 词表
    :param textfilename:
    :return: words , We
    """
    words = {}
    filename = os.path.join(BASE_DATA_DIR, text_filename)
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    We = np.zeros((len(lines), 100))
    for i, line in enumerate(lines):
        word_with_vec = line.split()
        if len(word_with_vec) < 101: continue
        words[word_with_vec[0]] = i
        We[i, :] = np.array([float(num) for num in word_with_vec[1:]])
    return (words, We)

def prepare_data(list_of_seqs):
    """
    获得句子中每个词的索引，max_len（0表示没有）构成词索引矩阵
    :param list_of_seqs:
    :return:
    """
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask


def lookupIDX(words, w):
    """
    遍历词表
    :param words:
    :param w:
    :return:
    """
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#", "")
    if w in words:
        return words[w]
    elif 'UUUNKKK' in words:
        return words['UUUNKKK']
    else:
        return len(words) - 1

def getSeq(p1, words):
    """
    获得句子中每个词的索引
    :param p1: 句子
    :param words: 词表
    :return: 返回索引列表 [[],[]..]
    """
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words, i))
    return X1


def getSeqs(p1, p2, words):
    p1 = p1.split()
    p2 = p2.split()
    X1 = []
    X2 = []
    for i in p1:
        X1.append(lookupIDX(words, i))
    for i in p2:
        X2.append(lookupIDX(words, i))
    return X1, X2

def sentences2idx(sentences, words):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
    """
    seq1 = []
    for i in sentences:
        seq1.append(getSeq(i, words))
    x1, m1 = prepare_data(seq1)
    return x1, m1


def getWordWeight(weight_file, a=1e-3):
    """
    获得单词权重(SIF)
    :param weight_file: 词频文件
    :param a: 参数
    :return: 权重（字典{word：weight}）
    """
    word2weight = {}
    filename = os.path.join(BASE_DATA_DIR, weight_file)
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
    for i in lines:
        i = i.strip()
        if (len(i) > 0):
            i = i.split()
            if (len(i) == 2):
                word2weight[i[0]] = a / (a + float(i[1]))
            else:
                print(i)
    return word2weight

def getWeight(words, word2weight):
    """
    获得单词(索引)权重
    :param words:
    :param word2weight:
    :return: 权重（字典{word_idx: weight}）
    """
    weight4ind = {}
    for word, ind in words.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind


def seq2weight(seq, mask, weight4ind):
    """
    获得句子每个词的权重矩阵
    :param seq:
    :param mask:
    :param weight4ind:
    :return:
    """
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i, j] > 0 and seq[i, j] >= 0:
                weight[i, j] = weight4ind[seq[i, j]]
    weight = np.asarray(weight, dtype='float32')
    return weight

if __name__ == "__main__":
    word_file = 'word_vec_file.txt'
    words, vectors = getWordmap(word_file)

