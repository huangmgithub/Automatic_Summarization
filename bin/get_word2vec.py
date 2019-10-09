from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(lineno)d -  %(message)s')
BASE_MODEL_DIR = '../model/word2vec/'
BASE_DATA_DIR = '../data/'

def word2vec_train(train_file, save_model_file, save_vector_file):
    """词向量训练"""
    f_wiki = open(train_file, "r", encoding="utf-8")
    sentences = LineSentence(f_wiki)
    model = Word2Vec(sentences, size = 100, window = 5, min_count = 7, workers = multiprocessing.cpu_count())
    model.save(save_model_file)
    model.wv.save_word2vec_format(save_vector_file, binary = False)

def load_model(fname):
    """加载模型"""
    return Word2Vec.load(fname)

if __name__ == "__main__":
    train_file = os.path.join(BASE_DATA_DIR, 'news.txt')
    save_model_file = os.path.join(BASE_MODEL_DIR, 'wiki.zh.model')
    save_vector_file = os.path.join(BASE_MODEL_DIR, 'wiki.zh.vectors')

    # 训练
    word2vec_train(train_file,save_model_file,save_vector_file)
    # 导入模型
    model = Word2Vec.load(save_model_file)
    # 词向量
    # print("江西", word2vec_model['江西']) # 获得词向量
    print(model.most_similar('江西'))