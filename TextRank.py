import networkx as nx
import math
import re
from setting import stopwords_file, summary_ratio
from utils.util import get_stopwords, token, cut


def split_sentence(text):
    """
    分割文本, 两个句子为整体
    :param text:
    :return:
    """
    text = re.sub(r'\s+', '', text)
    # pattern = re.compile('[。，,.:：]')
    pattern = re.compile('[。?!！？.]')
    sentence_segments = pattern.sub(' ', text).split()
    sentences_merge = []
    for i in range(len(sentence_segments) // 2):
        tmp_merge = sentence_segments[2*i] + sentence_segments[2*i+1]
        sentences_merge.append(tmp_merge)
    return sentences_merge

def build_connect_graph_by_text_rank(text, title):
    """
    构建图
    :param text:
    :param title:
    :return:
    """
    # 图
    sentence_graph = nx.Graph()

    sentences = split_sentence(text)
    stopwords = get_stopwords(stopwords_file)
    sentences_cut = [cut("".join(token(sentence))) for sentence in sentences]

    sentences_cut_del_stopwords = [] # 包含标题

    # 文本内容去除停用词
    for s in sentences_cut:
        sentences_cut_del_stopwords.append([w for w in s if w not in stopwords])

    # 标题去除停用词
    if title:
        title_cut_del_stopwords = [w for w in cut("".join(title)) if w not in stopwords]
        # 合并到sentences, 方便计算
        if title_cut_del_stopwords != []:
            sentences_cut_del_stopwords.insert(0, title_cut_del_stopwords)

    print(sentences_cut_del_stopwords)


    for i, sentence in enumerate(sentences_cut_del_stopwords):
        for connect_id  in range(i+1, len(sentences_cut_del_stopwords)):
            # 获得两个句子相同词的数量
            same_words_len = len(set(sentence).intersection(set(sentences_cut_del_stopwords[connect_id])))
            print(same_words_len)

            if title and (i == 0 or i == 1):
                # 若是标题和第一句，则双倍权重
                sim = 2 * same_words_len / (
                        math.log(len(sentence)) + math.log(len(sentences_cut_del_stopwords[connect_id])))
            elif not title and i == 0:
                # 若没有标题，就看第一句
                sim = 2 * same_words_len / (
                        math.log(len(sentence)) + math.log(len(sentences_cut_del_stopwords[connect_id])))
            else:
                sim = same_words_len / (
                        math.log(len(sentence)) + math.log(len(sentences_cut_del_stopwords[connect_id])))

            # 将带权句子连接关系加入到图
            sentence_graph.add_edges_from([(i, connect_id)], weight=sim)

    return sentences, sentence_graph


def sims_ranking(text, title):
    """
    排序
    :param text:
    :param title:
    :return:
    """
    sentences, sentence_graph = build_connect_graph_by_text_rank(text, title)
    ranking_sims = nx.pagerank(sentence_graph) # 使用TextRank
    if title:
        ranking_sims.pop(0) # 去除标题

    ranking_sims = sorted(ranking_sims.items(), key=lambda x: x[1], reverse=True)
    return ranking_sims, sentences

def get_summarization_by_textrank(text, title, ratio=summary_ratio):
    """
    获得文本摘要by TextRank
    :param text:
    :param title:
    :param ratio:
    :return:
    """
    ranking_sims, sentences = sims_ranking(text, title)

    # 摘要比例
    max_len = int(len(sentences) * ratio)

    candidate_sentences_idx = [s[0] for s in ranking_sims[:max_len]]
    candidate_sentences_idx = sorted(candidate_sentences_idx)

    return ''.join(sentences[idx] for idx in candidate_sentences_idx)

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
    title_2 = "林肯公园主唱查斯特·贝宁顿 Chester Bennington自杀"

    summary1 = get_summarization_by_textrank(text_1, title_1)
    print(summary1)

    summary2 = get_summarization_by_textrank(text_2, title_2)
    print(summary2)




