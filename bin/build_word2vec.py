from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import logging
from setting import news_file, word2vec_model_path, word2vec_vectors_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(lineno)d -  %(message)s')

def word2vec_train():
    """
    词向量训练
    :param train_file:
    :param save_model_file:
    :param save_vector_file:
    :return:
    """
    f_wiki = open(news_file, "r", encoding="utf-8")
    sentences = LineSentence(f_wiki)
    model = Word2Vec(sentences, size = 100, window = 5, min_count = 7, workers = multiprocessing.cpu_count())
    model.save(word2vec_model_path)
    model.wv.save_word2vec_format(word2vec_vectors_path, binary = False)

def load_model(fname):
    """
    加载模型
    :param fname:
    :return:
    """
    return Word2Vec.load(fname)

if __name__ == "__main__":
    # 训练
    word2vec_train()
    # 导入模型
    model = Word2Vec.load(word2vec_model_path)
    # 词向量
    # print("江西", word2vec_model['江西']) # 获得词向量
    print(model.most_similar('江西'))