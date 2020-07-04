from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from utils.preprocess_text import getWordmap, getWeight, getWordWeight, sentences2idx, seq2weight, get_stopwords
from utils.util import split_sentence, cut
from utils.SIF_embedding import SIF_embedding
import numpy as np
from textrank4zh import TextRank4Keyword
import logging
from setting import summary_ratio, stopwords_file


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

        self.a = a # SIF param
        self.words, self.vectors = getWordmap() # 词索引表{word:index}和词嵌入表(vocab,embed_size)
        self.word2weight = getWordWeight(self.a) # SIF权重（字典{word：weight}）
        self.weight4ind = getWeight(self.words, self.word2weight) # SIF权重（字典{index: weight}）
        self.stopwords = get_stopwords() # 停用词[stopword,..]

    def _pre_processing_text(self, text_cut, use_sif=True):
        """
        split text to sentences, use SIF weighted or average word embedding to get sentence embedding
        :param text: cut text
        :param use_sif:
        :return:
        sentences: List[str], cut from text
        embedding: np.array, sentence embedding
        """

        x, m = sentences2idx(text_cut, self.words) # 索引：x[词索引,...] m[1,1,1,0,0,..]
        if use_sif:
            w = seq2weight(x, m, self.weight4ind) # 权重： [[w1, w2,..], [w1, w2,..]]
            embedding = SIF_embedding(self.vectors, x, w, 1)  # SIF 加权句子向量
        else:
            embedding = np.zeros((len(text_cut), 100))
            for i in range(embedding.shape[0]):
                tmp = np.zeros((1, 100))
                count = 0
                for j in range(x.shape[1]):
                    if m[i, j] > 0 and x[i, j] >= 0:
                        tmp += self.vectors[x[i, j]]
                        count += 1
                embedding[i, :] = tmp / count # 平均句子向量
        return embedding

    def get_summary(self, text, title, use_sif):
        """
        Use cosine similarity to calculate the similarity between each sentence and the text
        :param text:
        :param use_sif:
        :return:
        scores: dict, key is index of sentence, value is score used in rank
        sentences: list, a list of sentences
        """

        # 文章
        sentences, flags = split_sentence(text)
        sentences_cut = [' '.join(word for word in cut(sentence) if word not in self.stopwords) for sentence in
                         sentences]
        # 使用切分好的文本，获得句子向量
        sentences_embedding = self._pre_processing_text(sentences_cut, use_sif)

        # 标题
        if title:
            title_cut = [' '.join(word for word in cut(title) if word not in self.stopwords)]
            # 使用切分好的文本，获得标题向量
            title_embedding = self._pre_processing_text(title_cut, use_sif)

        # 整体文本向量
        text_embedding = sentences_embedding.mean(axis=0, keepdims=True)  # 平均

        # 整体文本和句子的相似度
        sims_text = []
        # 标题和句子的相似度
        sims_title = []

        for i, sub_sentence in enumerate(sentences):

            sim_text = cosine_similarity(sentences_embedding[i].reshape(1, -1), text_embedding)[0, 0]
            sims_text.append(sim_text)
            if title:
                sim_title = cosine_similarity(sentences_embedding[i].reshape(1, -1), title_embedding)[0, 0]
                sims_title.append(sim_title)

        # 关键字
        keywords = self._get_keyword(text)

        # 关键字更新权重
        sims_text = self._add_keyword_weight(keywords, sims_text, sentences)

        # 若有标题，则更新权重
        if title:
            sims = self._add_title_weight(sims_text, sims_title)
        else:
            sims = sims_text

        # 开头位置更新权重
        sims = self._add_start_weight(sentences, sims)

        # KNN平滑处理
        sims = self._knn_soft(sims)

        # 排序
        idx2sim = {}
        for i, sim in enumerate(sims):
            idx2sim[i] = sim
        idx2sim = sorted(idx2sim.items(), key=lambda x: x[1], reverse=True)
        # 最大长度
        max_len = len(sentences + flags) // 6
        idx2sim = idx2sim[:max_len]
        summary = []
        if flags:
            for idx, sim in idx2sim:
                summary.append(sentences[idx])
                summary.append(flags[idx])
        else:
            summary = [sentences[idx] for idx, sim in idx2sim]

        return "".join(summary)

    def _get_keyword(self, text):
        """
        获得关键词
        :param text:
        :return:
        """
        tr4w = TextRank4Keyword()
        tr4w.analyze(text=text, window=4, lower=True)
        keyword_items = tr4w.get_keywords(10, word_min_len=2)
        # 将权重标准化
        keyword_items = sorted(keyword_items, key=lambda x: x.weight)
        over_length = keyword_items[-1].weight
        for wp in keyword_items:
            wp.weight /= over_length
        return keyword_items

    def _add_keyword_weight(self, keywords, sims, sentences):
        """
        关键词对摘要的权重
        :param keywords:
        :param sims:
        :param sentences:
        :return:
        """
        tokens = [[word for word in cut(sentence) if word not in self.stopwords] for sentence in sentences]
        for wp in keywords:
            for i, token in enumerate(tokens):
                if wp.weight in token:
                    sims[i] = sims[i] + 0.02 * wp.weight
        return sims

    def _add_title_weight(self, sims_text, sims_title):
        """
        标题对摘要的权重
        :param sims_text:
        :param sims_title:
        :return:
        """
        sims_text = np.array(sims_text)
        sims_title = np.array(sims_title)
        p = 0.7 # 1 - p为文本权重
        sims = p * sims_text + (1 - p) * sims_title
        return list(sims)

    def _add_start_weight(self, sentences, sims):
        """
        文本开头对摘要的权重
        :param sentences:
        :param sims:
        :return:
        """
        if len(sentences[0]) > 20:
            sims[0] = sims[0] + 0.1
        return sims

    def _knn_soft(self, sims):
        """
        KNN平滑处理
        :param sims:
        :return:
        """
        window = 2
        weight = np.array([0.1, 0.125, 0.5, 0.125, 0.1]) # 权重
        sims = [sims[0]] * window + sims + [sims[-1]] * window
        sims = np.array(sims)
        sims = [np.dot(sims[i - window:i + window + 1], weight)
               for i in range(window, len(sims) - window)]
        return sims

if __name__ == "__main__":

    text1 = """网易娱乐7月21日报道林肯公园主唱查斯特·贝宁顿 Chester Bennington于今天早上,
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
    成为他们第五张登顶ilboard排行榜的专辑。而昨晚刚刚发布新单《 Talking To Myself》MV。
    """

    title1 = "林肯公园主唱查斯特·贝宁顿 Chester Bennington自杀"

    summary = AutoSummary()


    result_3 = summary.get_summary(text1, title1,  use_sif=True)
    print("\n",result_3)

    # Cosine_similarity + Average
    result_4 = summary.get_summary(text1, title1,  use_sif=False)
    print("\n", result_4)

