from gensim.models import LdaModel
from gensim.corpora import Dictionary
from bin.get_ldamodel import get_train_set
import os
from utils.utils import get_stopwords, token, cut, split_sentence
from sklearn.metrics.pairwise import cosine_similarity


BASE_DATA_DIR = './data/'
BASE_MODEL_DIR = './model/ldamodel'

def get_model(corpus_path, stopwords_path, model_path):
    """
    加载LDA 模型
    :param corpus_path:
    :param stopwords_path:
    :param model_path:
    :return:
    """
    train_set = get_train_set(corpus_path, stopwords_path)
    stopwords = get_stopwords(stopwords_path)

    lda = LdaModel.load(model_path)
    dictionary = Dictionary(train_set)
    return stopwords, dictionary, lda


def get_topic_inference(text_cut, stopwords, dictionary, lda):
    """
    获得文本主题推断
    :param text_cut:
    :param stopwords:
    :param dictionary:
    :param lda:
    :return: 主题推断 ndarray
    """
    bow = dictionary.doc2bow([w for w in text_cut if w not in stopwords ])
    inference_ndarray = lda.inference([bow])[0]
    print(inference_ndarray[0])
    return inference_ndarray[0]


def get_sim_with_content(sentence, content_topic_inference, stopwords, dictionary, lda):
    """
    比较文章整体和单个句子之间的主题相似度
    :param sentence:
    :param content_topic_inference:
    :param stopwords:
    :param dictionary:
    :param lda:
    :return: cosine相似度
    """
    sentence_topic_inference = get_topic_inference(cut(''.join(token(sentence))), stopwords, dictionary, lda).reshape(1, -1)

    return cosine_similarity(sentence_topic_inference, content_topic_inference)


def get_scores_by_lda(text, title, corpus_file, stopwords_file, model_file):
    """
    获得文章整体与所有句子的相似度
    :param text:
    :param title:
    :param corpus_file:
    :param stopwords_file:
    :param model_file:
    :return:
    -------------------
    scores -> {sentence_idx: cosine_similarity}
    """
    stopwords_path = os.path.join(BASE_DATA_DIR, stopwords_file)
    corpus_path = os.path.join(BASE_DATA_DIR, corpus_file)
    model_path = os.path.join(BASE_MODEL_DIR, model_file)

    stopwords, dictionary, lda = get_model(corpus_path, stopwords_path, model_path)
    sentences = split_sentence(text)
    if sentences == []:
        raise NameError
    scores = {}

    if title:
        text += title
    content_topic_inference = get_topic_inference(cut(''.join(token(text))), stopwords, dictionary, lda).reshape(1, -1)

    for i, sentence in enumerate(sentences):
        scores[i] = get_sim_with_content(sentence, content_topic_inference, stopwords, dictionary, lda )

    return scores, sentences