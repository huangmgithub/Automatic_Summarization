from scipy.spatial.distance import cosine
from bin.get_sentence_embedding import sentence_embedding, get_sentences_matrix, get_word_occurrence
from bin.get_word2vec import load_model
from bin.get_data import get_stopwords, split_sentence, get_words_list
from collections import defaultdict
import numpy as np
from jieba import analyse  # TextRank
from gensim import corpora, models  # LDA
import networkx
import pandas as pd

class AutoSummaryBySim:
    """
    Optimal
    ① Knn_smooth
    ② Title
    ③ keyword
    ④ Topic
    """
    def __init__(self, text, title, model, stopwords, words_freq, constraint=180):
        self.w_title = 1
        self.w_keyword = 1
        self.constraint = constraint
        self.text = text
        self.title = title
        self.model = model
        self.stopwords = stopwords
        self.words_freq = words_freq
        self.keyword = self._keyword()

    def _get_correlation_from_sentence(self, isSorted=True):
        """获得子句和文本之间 and 子句和标题之间的相似度"""

        if isinstance(self.text, list): text = ' '.join(self.text)
        sub_sentences = split_sentence(self.text)
        print(sub_sentences)

        text_matrix = get_sentences_matrix(self.text, self.model, self.words_freq, self.stopwords)  # 文本矩阵
        sentence_vec = sentence_embedding(self.text, self.model, text_matrix, self.words_freq, self.stopwords)
        title_vec = sentence_embedding(self.title, self.model, text_matrix, self.words_freq, self.stopwords)

        correlations = {}

        for sub_sentence in sub_sentences:
            if self.keyword in sub_sentence:
                w_keyword = self.w_keyword * 1.5  # 关键字权重
            else:
                w_keyword = self.w_keyword
            sub_sen_vec = sentence_embedding(sub_sentence, self.model, text_matrix, self.words_freq, self.stopwords)
            correlation = cosine(sentence_vec, sub_sen_vec) + self.w_title * cosine(title_vec, sub_sen_vec)  # 增加与title相似度
            correlations[sub_sentence] = w_keyword * correlation
        if isSorted:
            return sorted(correlations.items(), key=lambda x: x[1], reverse=True)  # 列表中元组元素
        return correlations  # 字典

    def _knn_smooth(self):
        """knn平滑操作"""
        correlations = self._get_correlation_from_sentence()
        correlations_copy = correlations.copy()
        correlate_dict = defaultdict(int)
        for sen_i, cor_i in correlations:
            for sen_j, cor_j in correlations_copy:
                correlate_dict[sen_i] += np.sqrt(np.square(cor_i - cor_j))
        return sorted(correlate_dict.items(), key=lambda x: x[1])

    def _keyword(self):
        """关键字权重"""
        text_rank = analyse.textrank
        keywords = text_rank(self.text)
        return keywords[0]

    def _topic(self):
        word_list = get_words_list(self.text, self.stopwords)
        word_dict = corpora.Dictionary(word_list)  # 生成文档的词典，每个词与一个整型索引值对应
        corpus_list = [word_dict.doc2bow(text) for text in word_list]  # 词频统计，转化成空间向量格式
        lda = models.ldamodel.LdaModel(corpus=corpus_list,
                                       id2word=word_dict,
                                       num_topics=5,
                                       passes=20,
                                       alpha='auto')
        for pattern in lda.show_topics():
            print(pattern)

        lda.get_document_topics(corpus_list[0])
        lda.show_topic(1, topn=20)

    def get_summarizaton(self):
        """获得文摘"""
        sub_sentences = split_sentence(self.text)
        ranked_sentences = self._knn_smooth()
        selected_text = set()
        current_text = ''
        # 限制字数
        for sen, _ in ranked_sentences:
            if len(current_text) <= self.constraint:
                current_text += sen
                selected_text.add(sen)
            else:
                break

        # 获得摘要
        summarized = []
        for i, sen in enumerate(sub_sentences):  # 按顺序打印
            if sen in selected_text:
                summarized.append(sen)
        return ''.join(summarized)

class AutoSummaryByTextRank:
    def __init__(self, text, title, model, stopwords, words_freq, constraint=180):
        self.constraint = constraint
        self.text = text
        self.title = title
        self.model = model
        self.stopwords = stopwords
        self.words_freq = words_freq

    def _get_connect_graph_by_text_rank(self, sentences):
        """
        获得子句之间的相似度
        :param sentences:
        :return:
        """
        sentences_length = len(sentences)

        text_matrix = get_sentences_matrix(self.text, self.model, self.words_freq, self.stopwords)  # 文本矩阵

        graph = np.zeros((sentences_length, sentences_length))
        print(sentences_length)
        for x in range(sentences_length):
            sen_vec_x = sentence_embedding(sentences[x], self.model, text_matrix, self.words_freq, self.stopwords)
            for y in range(sentences_length):
                sen_vec_y = sentence_embedding(sentences[y], self.model, text_matrix, self.words_freq, self.stopwords)
                sentences_similarity = cosine(sen_vec_x, sen_vec_y)
                graph[x, y] = sentences_similarity
                graph[y, x] = sentences_similarity

        nx_graph = networkx.from_numpy_matrix(graph)
        return nx_graph

    def _sentence_ranking_by_text_ranking(self, sentences):
        """
        TextRank
        :param split_sentence:
        :return:
        """
        sentence_graph = self._get_connect_graph_by_text_rank(sentences)
        ranking_sentences_index = networkx.pagerank(sentence_graph, max_iter=500)
        ranking_sentences_index = sorted(ranking_sentences_index.items(), key=lambda x: x[1], reverse=True)
        ranking_sentences = [(sentences[index], weight) for index, weight in ranking_sentences_index]

        return ranking_sentences

    def _get_summarization(self):
        """获得文本摘要"""
        sub_sentence = split_sentence(text)
        ranking_sentences = self._sentence_ranking_by_text_ranking(sub_sentence)
        selected_text = set()
        current_text = ''

        for sen, _ in ranking_sentences:
            if len(current_text) <= self.constraint:
                current_text += str(sen)
                selected_text.add(sen)
            else:
                break

        summarized = []
        for sen in sub_sentence:  # print the selected sentence by sequent
            if sen in selected_text:
                summarized.append(sen)
        return summarized

    def get_summarization(self):
        """
        获得文本摘要 by TextRank
        :param text:
        :param constraint:
        :return:
        """
        summarized = self._get_summarization()
        return ''.join(summarized)


if __name__ == "__main__":
    # text = """虽然至今夏普智能手机在市场上无法排得上号，已经完全没落，并于 2013 年退出中国市场，
    # 但是今年 3 月份官方突然宣布回归中国，预示着很快就有夏普新机在中国登场了。那么，第一款夏普手机什么时候登陆中国呢？
    # 又会是怎么样的手机呢？近日，一款型号为 FS8016 的夏普神秘新机悄然出现在 GeekBench 的跑分库上。从其中相关信息了解到，
    # 这款机子并非旗舰定位，所搭载的是高通骁龙 660 处理器，配备有 4GB 的内存。骁龙 660 是高通今年最受瞩目的芯片之一，
    # 采用 14 纳米工艺，八个 Kryo 260 核心设计，集成 Adreno 512 GPU 和 X12 LTE 调制解调器。
    # 当前市面上只有一款机子采用了骁龙 660 处理器，那就是已经上市销售的 OPPO R11。骁龙 660 尽管并非旗舰芯片，
    # 但在多核新能上比去年骁龙 820 强，单核改进也很明显，所以放在今年仍可以让很多手机变成高端机。不过，由于 OPPO 与高通签署了排他性协议，
    # 可以独占两三个月时间。考虑到夏普既然开始测试新机了，说明只要等独占时期一过，夏普就能发布骁龙 660 新品了。
    # 按照之前被曝光的渲染图了解，夏普的新机核心竞争优势还是全面屏，因为从 2013 年推出全球首款全面屏手机 EDGEST 302SH 至今，
    # 夏普手机推出了多达 28 款的全面屏手机。在 5 月份的媒体沟通会上，惠普罗忠生表示：“我敢打赌，12 个月之后，在座的各位手机都会换掉。
    # 因为全面屏时代的到来，我们怀揣的手机都将成为传统手机。”
    # """
    # title = "配骁龙660 全面屏鼻祖夏普新机酝酿中"
    data = pd.read_csv("./data/sqlResult_1558435.csv", encoding='gb18030')
    data = data.fillna('')
    text = data.iloc[12]['content']
    title = data.iloc[12]['title']
    print(text)
    print(title)
    model_filename = './model/word2vec/wiki.zh.model'
    model = load_model(model_filename)  # 加载模型
    stopwords = get_stopwords('./data/chinese_stopwords.txt')
    words_freq = get_word_occurrence('./data/news.txt')
    summary_1 = AutoSummaryBySim(text, title,  model, stopwords, words_freq)
    result_1 = summary_1.get_summarizaton()
    print(result_1)
    print("TextRank Start")
    summary_2 = AutoSummaryByTextRank(text, title,  model, stopwords, words_freq)
    result_2 = summary_2.get_summarization()
    print(result_2)