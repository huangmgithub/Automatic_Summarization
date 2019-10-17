from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from bin.get_data import getWordmap, getWeight, getWordWeight, sentences2idx, seq2weight
from bin.get_file import split_sentence, cut, token, get_stopwords
from bin.SIF_embedding import SIF_embedding
import numpy as np
import pandas as pd
from jieba import analyse  # TextRank
from gensim import corpora, models  # LDA
import networkx
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoSummary:
    """
    Optimal
    ① Knn_smooth
    ② Title
    ③ keyword
    ④ Topic
    """
    def __init__(self, a=1e-3):
        self.w_title = 1
        self.w_keyword = 1


        self.word_file = 'word_vec_file.txt'
        self.weight_file = 'weight_file.txt'
        self.a = a # SIF param
        self.words, self.vectors = getWordmap(self.word_file)
        self.word2weight = getWordWeight(self.weight_file, self.a)
        self.weight4ind = getWeight(self.words, self.word2weight)

    def _pre_processing(self, text, use_sif=True):
        """
        split text to sentences, use SIF weighted or average word embedding to get sentence embedding
        :param text:
        :param use_sif:
        :return:
        sentences: List[str], cut from text
        embedding: np.array, sentence embedding
        """
        sentences = split_sentence(text)
        sentences_cut = [' '.join(word for word in cut(sentence)) for sentence in sentences]
        x, m = sentences2idx(sentences_cut, self.words)
        if use_sif:
            w = seq2weight(x, m, self.weight4ind)
            embedding = SIF_embedding(self.vectors, x, w, 1)
        else:
            embedding = np.zeros((len(sentences), 100))
            for i in range(embedding.shape[0]):
                tmp = np.zeros((1, 100))
                count = 0
                for j in range(x.shape[1]):
                    if m[i, j] > 0 and x[i, j] >= 0:
                        tmp += self.vectors[x[i, j]]
                        count += 1
                embedding[i, :] = tmp / count # 平均句子向量
        return sentences, embedding

    def get_summary(self, text, title, constraint=200, algorithm='TextRank', use_sif=True):
        """
        Use TextRank or Cosine Similarity to summary
        :param text: extract summary for the text
        :param constraint: select constraint sentences as the summary of text
        :param algorithm: default TextRank, or Cosine
        :param use_sif: whether to use sif embedding
        :return:
        """
        if algorithm == "TextRank":
            scores, sentences = self._text_rank(text, use_sif)
        elif algorithm == "Cosine":
            scores, sentences = self._summary_by_similarity(text, title, use_sif)
        else:
            raise ValueError('only support TextRank and Cosine Similarity!')

        selected_text = set()
        current_text = ''
        # 限制字数
        for idx, score in scores:
            if len(current_text) <= constraint:
                current_text += sentences[idx]
                selected_text.add(sentences[idx])
            else:
                break

        # 获得摘要
        summarized = []
        for i, sen in enumerate(sentences):  # 按顺序打印
            if sen in selected_text:
                summarized.append(sen)
        return ''.join(summarized)

    def _text_rank(self, text, use_sif):
        """
        Use TextRank to get scores of scores
        :param text:
        :param use_sif:
        :return:
        scores: dict, key is index of sentence, value is score used in rank
        sentences: list, a list of sentences
        """
        sentences, sentences_embedding = self._pre_processing(text, use_sif)
        sentences_length = sentences_embedding.shape[0]
        graph = np.zeros((sentences_length, sentences_length))

        for i in range(sentences_length):
            for j in range(i +1, sentences_length):
                graph[i, j] = cosine_similarity(sentences_embedding[i, None], sentences_embedding[j, None])
                graph[j, i] = graph[i, j]

        nx_graph = networkx.from_numpy_matrix(graph)

        scores = self._page_rank(nx_graph)
        return scores, sentences

    def _page_rank(self, graph):
        """
        Page_Rank
        :param split_sentence:
        :return:
        """
        ranking_sentences = networkx.pagerank(graph, max_iter=500)
        ranking_sentences = sorted(ranking_sentences.items(), key=lambda x: x[1], reverse=True)
        return ranking_sentences

    def _summary_by_similarity(self, text, title, use_sif):
        """
        Use cosine similarity to calculate the similarity between each sentence and the text
        :param text:
        :param use_sif:
        :return:
        scores: dict, key is index of sentence, value is score used in rank
        sentences: list, a list of sentences
        """
        sentences, sentences_embedding = self._pre_processing(text, use_sif)
        title, title_embedding = self._pre_processing(title, use_sif=False)

        text_embedding = sentences_embedding.mean(axis=0, keepdims=True) # 平均
        title_embedding = title_embedding.mean(axis=0, keepdims=True)

        scores = {}

        key_word = self._keyword(text) # 关键字
        for i, sub_sentence in enumerate(sentences):
            if key_word in sub_sentence:
                w_keyword = self.w_keyword * 1.5
            else:
                w_keyword = self.w_keyword
            sim_text = cosine_similarity(sentences_embedding[i].reshape(1, -1), text_embedding)
            sim_title = cosine_similarity(sentences_embedding[i].reshape(1, -1), title_embedding)
            total_sim = sim_text + self.w_title * sim_title
            scores[i] = total_sim * w_keyword

        scores = self._knn_smooth(scores) # KNN处理
        return scores, sentences

    def _knn_smooth(self, scores):
        """
        knn smooth
        :param scores:
        :return:
        """
        scores_copy = scores.copy()
        correlate_dict = defaultdict(int)
        for sen_i, cor_i in scores.items():
            for sen_j, cor_j in scores_copy.items():
                correlate_dict[sen_i] += np.sqrt(np.square(cor_i - cor_j))
        return sorted(correlate_dict.items(), key=lambda x: x[1])

    def _keyword(self, text):
        """
        keyword in text
        :param text:
        :return:
        """
        text_rank = analyse.textrank
        keywords = text_rank(text)
        return keywords[0]

    # def _topic(self):
    #     word_list = get_words_list(self.text, self.stopwords)
    #     word_dict = corpora.Dictionary(word_list)  # 生成文档的词典，每个词与一个整型索引值对应
    #     corpus_list = [word_dict.doc2bow(text) for text in word_list]  # 词频统计，转化成空间向量格式
    #     lda = models.ldamodel.LdaModel(corpus=corpus_list,
    #                                    id2word=word_dict,
    #                                    num_topics=5,
    #                                    passes=20,
    #                                    alpha='auto')
    #     for pattern in lda.show_topics():
    #         print(pattern)
    #
    #     lda.get_document_topics(corpus_list[0])
    #     lda.show_topic(1, topn=20)

if __name__ == "__main__":
    text_1 = """虽然至今夏普智能手机在市场上无法排得上号，已经完全没落，并于 2013 年退出中国市场，
    但是今年 3 月份官方突然宣布回归中国，预示着很快就有夏普新机在中国登场了。那么，第一款夏普手机什么时候登陆中国呢？
    又会是怎么样的手机呢？近日，一款型号为 FS8016 的夏普神秘新机悄然出现在 GeekBench 的跑分库上。从其中相关信息了解到，
    这款机子并非旗舰定位，所搭载的是高通骁龙 660 处理器，配备有 4GB 的内存。骁龙 660 是高通今年最受瞩目的芯片之一，
    采用 14 纳米工艺，八个 Kryo 260 核心设计，集成 Adreno 512 GPU 和 X12 LTE 调制解调器。
    当前市面上只有一款机子采用了骁龙 660 处理器，那就是已经上市销售的 OPPO R11。骁龙 660 尽管并非旗舰芯片，
    但在多核新能上比去年骁龙 820 强，单核改进也很明显，所以放在今年仍可以让很多手机变成高端机。不过，由于 OPPO 与高通签署了排他性协议，
    可以独占两三个月时间。考虑到夏普既然开始测试新机了，说明只要等独占时期一过，夏普就能发布骁龙 660 新品了。
    按照之前被曝光的渲染图了解，夏普的新机核心竞争优势还是全面屏，因为从 2013 年推出全球首款全面屏手机 EDGEST 302SH 至今，
    夏普手机推出了多达 28 款的全面屏手机。在 5 月份的媒体沟通会上，惠普罗忠生表示：“我敢打赌，12 个月之后，在座的各位手机都会换掉。
    因为全面屏时代的到来，我们怀揣的手机都将成为传统手机。”
    """
    title_1 = "配骁龙660 全面屏鼻祖夏普新机酝酿中"
    # data = pd.read_csv("./data/sqlResult_1558435.csv", encoding='gb18030')
    # data = data.fillna('')
    # text = data.iloc[12]['content']
    # title = data.iloc[12]['title']
    # print(text)
    # print(title)

    text_2 = """网易娱乐7月21日报道林肯公园主唱查斯特·贝宁顿 Chester Bennington于今天早上,
    在洛杉矶帕洛斯弗迪斯的一个私人庄园自缢身亡,年仅41岁。此消息已得到洛杉矶警方证实。
    洛杉矶警方透露, Chester的家人正在外地度假, Chester独自在家,上吊地点是家里的二楼。
    一说是一名音乐公司工作人员来家里找他时发现了尸体,也有人称是佣人最早发现其死亡。
    林肯公园另一位主唱麦克信田确认了 Chester Bennington自杀属实,并对此感到震惊和心痛,称稍后官方会发布声明。
    Chester昨天还在推特上转发了一条关于曼哈顿垃圾山的新闻。粉丝们纷纷在该推文下留言,不相信 Chester已经走了。
    外媒猜测,Chester选择在7月20日自杀的原因跟他极其要好的朋友Soundgarden(声音花园)乐队以及AudioslaveChris乐队主唱 Cornell有关,
    因为7月20日是 Chris CornellChris的诞辰。而 Cornell于今年5月17日上吊自杀,享年52岁。 Chris去世后, Chester还为他写下悼文。
    对于 Chester的自杀,亲友表示震惊但不意外,因为 Chester曾经透露过想自杀的念头,他曾表示自己童年时被虐待,导致他医生无法走出阴影,
    也导致他长期酗酒和嗑药来疗伤。目前,洛杉矶警方仍在调查Chester的死因。据悉, Chester与毒品和酒精斗争多年,年幼时期曾被成年男子性侵,
    导致常有轻生念头。 Chester生前有过2段婚姻,育有6个孩子。林肯公园在今年五月发行了新专辑《多一丝曙光OneMoreLight》,
    成为他们第五张登顶ilboard排行榜的专辑。而昨晚刚刚发布新单《 Talking To Myself》MV
    """

    title_2 = "肯公园主唱查斯特·贝宁顿今早自缢身亡"

    summary_1 = AutoSummary()
    result_1 = summary_1.get_summary(text_2, title_2, constraint=200, algorithm='Cosine', use_sif=True)
    print(result_1)
