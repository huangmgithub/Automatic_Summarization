import os

# file
# stopwords_file = os.path.join(os.path.abspath("./"), "data", "chinese_stopwords.txt")
# weight_file= os.path.join(os.path.abspath("./"), "data", "weight_file.txt")
# news_file = os.path.join(os.path.abspath("./"), "data", "news.txt")
# word2vec_file = os.path.join(os.path.abspath("./"), "data", "word_vec_file.txt")
#
# # model
# word2vec_model_path = os.path.join(os.path.abspath("./"), "model", "word2vec", "wiki.zh.model")
# lda_model_path = os.path.join(os.path.abspath("./"), "model", "ldamodel", "news.model")
# word2vec_vectors_path = os.path.join(os.path.abspath("./"), "model", "word2vec", "wiki.zh.vectors")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
# file
stopwords_file = BASE_DIR + r"\data\chinese_stopwords.txt"
weight_file= BASE_DIR + r"\data\weight_file.txt"
news_file = BASE_DIR + r"\data\news.txt"
word2vec_file = BASE_DIR + r"\data\word_vec_file.txt"

# model
word2vec_model_path = BASE_DIR + r"\model\word2vec\wiki.zh.model"
lda_model_path = BASE_DIR + r"\model\ldamodel\news.model"
word2vec_vectors_path = BASE_DIR + r"\model\word2vec\wiki.zh.vectors"

summary_ratio = 0.6
