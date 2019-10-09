from collections import Counter
from bin.get_data import cut, split_sentence
import numpy as np
import re
from scipy import linalg

def get_word_occurrence(fname):
    """获得词出现的频率"""
    with open(fname, 'r', encoding='utf-8') as f:
        contents = f.read()
        words = [word for content in contents for word in content.split()]
        words_counter = Counter(words)
        words_freq = {word: count / len(words) for word, count in words_counter.items()}
        occurrences_freq = sorted(list(words_freq.values()), reverse=True)
    return words_freq

def SVD(text_matrix):
    """获得第一奇异列向量"""
    U, S, V = linalg.svd(text_matrix)
    return U[:,0]

def get_sentences_matrix(text, model, words_freq, stopwords):
    """文本所有句子向量化"""
    sentences = split_sentence(text)
    text_matrix = np.zeros((model['江西'].shape[0], len(sentences)))
    for index, sentence in enumerate(split_sentence(text)):
        sentence_vec = get_sentence_embedding(sentence, model, words_freq, stopwords)
        text_matrix[:,index] = sentence_vec
    return text_matrix

def get_sentence_embedding(sentence, model, words_freq, stopwords, alpha = 1e-4):
    """句子向量化"""
    # Weight Average
    max_fre = max(list(words_freq.values()))
    words_old = cut(sentence)
    words_new = [word for word in words_old if word not in stopwords]
    sentence_vec = np.zeros_like(model['江西'])
    words_new = [w for w in words_new if w in model]
    for word in words_new:
        weight = alpha / (alpha + words_freq.get(word, max_fre))
        sentence_vec += weight * model[word]
    sentence_vec /= len(words_old)
    return sentence_vec

def sentence_embedding(sentence, model, text_matrix, words_freq, stopwords):
    """句子向量化via SVD"""
    sentence_vec = get_sentence_embedding(sentence, model, words_freq, stopwords)
    # SVD
    singular_vector = SVD(text_matrix).reshape(-1, 1)
    sentence_vec = sentence_vec - singular_vector @ singular_vector.T @ sentence_vec
    return sentence_vec