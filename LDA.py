from gensim.models import LdaModel
from gensim.corpora import Dictionary
from bin.build_lda_model import get_train_set
from utils.util import get_stopwords, token, cut, split_sentence
from sklearn.metrics.pairwise import cosine_similarity
from setting import lda_model_path, summary_ratio

def get_model():
    """
    加载LDA 模型
    :param corpus_path:
    :param stopwords_path:
    :param model_path:
    :return:
    """
    train_set = get_train_set()
    stopwords = get_stopwords()

    lda = LdaModel.load(lda_model_path)
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


def get_sims_with_text(text, title):
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

    stopwords, dictionary, lda = get_model()
    sentences, flags = split_sentence(text)
    if sentences == []:
        raise NameError
    sims = {}

    if title:
        text += title # 文本加上标题

    # 文本加标题的主题
    content_topic_inference = get_topic_inference(cut(''.join(token(text))), stopwords, dictionary, lda).reshape(1, -1)

    # 比较文本与子句之间的主题相关性
    for i, sentence in enumerate(sentences):
        sims[i] = get_sim_with_content(sentence, content_topic_inference, stopwords, dictionary, lda )

    return sims, sentences, flags

def sims_ranking(text, title):
    """
    相似度排序
    :param text:
    :param title:
    :return:
    """
    sims, sentences, flags = get_sims_with_text(text, title)
    # sims[0] /= 2
    ranking_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)

    return ranking_sims, sentences, flags

def get_summarization_by_lda(text, title, ratio=summary_ratio):
    """
    获得文本摘要by LDA
    :param text:
    :param title:
    :param ratio: 获得的摘要与原文的比例
    :return:
    """

    ranking_sims, sentences, flags = sims_ranking(text, title)
    print("sentences", len(sentences), sentences)
    print("flags", len(flags), flags)
    # 摘要比例
    max_len = int(len(sentences) * ratio)
    summary = []
    if flags:
        for idx, sim in ranking_sims[:max_len]:
            summary.append(sentences[idx])
            summary.append(flags[idx])
    else:
        summary = [sentences[idx] for idx, sim in ranking_sims]

    return ''.join(summary)

if __name__ == "__main__":
    text1 = """虽然至今夏普智能手机在市场上无法排得上号，已经完全没落，并于 2013 年退出中国市场，
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
    title1 = "配骁龙660 全面屏鼻祖夏普新机酝酿中"

    text2 = """网易娱乐7月21日报道林肯公园主唱查斯特·贝宁顿 Chester Bennington于今天早上,
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

    title2 = "林肯公园主唱查斯特·贝宁顿 Chester Bennington自杀"

    summary1 = get_summarization_by_lda(text1, title1)
    print(summary1)

    summary2 = get_summarization_by_lda(text2, title2)
    print(summary2)




