from gensim.corpora import Dictionary
from gensim.models import LdaModel
from utils.util import get_stopwords
import logging
from setting import news_file, stopwords_file, lda_model_path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_train_set():
    """
    获得LDA模型 Train Set
    :param corpus_path:
    :param stopwords_path:
    :return:
    """
    stopwords = get_stopwords()
    train_set = []

    with open(news_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            train_set.append([word for word in line if word not in stopwords])
    return train_set

def save_model():
    """
    保存LDA模型
    :param model_path:
    :return:
    -----------------
    corpus:[
            [('词ID', 词频),('词ID', 词频)...],
            [('词ID', 词频),('词ID', 词频)...],
            .......
            ] 稀疏向量集
    id2word: {'词1':0, '词2':1. ..}

    """
    train_set = get_train_set()
    word_dict = Dictionary(train_set)  # 生成文档的词典，每个词与一个整型索引值对应
    corpus_list = [word_dict.doc2bow(text) for text in train_set]  # 词频统计，转化成空间向量格式
    lda = LdaModel(corpus=corpus_list,
                   id2word=word_dict,
                   num_topics=100,
                   # passes=5, # epoch
                   alpha='auto')
    lda.print_topic(99)
    # 保存LDA 模型
    lda.save(lda_model_path)


if __name__ == "__main__":

    save_model()

