import jieba
import re
import pandas as pd
from pyltp import SentenceSplitter

def cut(string):
    """切词"""
    return list(jieba.cut(string))

def token(string):
    """匹配文字"""
    return re.findall(r'[\d\w]+', string)

def get_stopwords(filename):
    """停用词"""
    with open(filename, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f])

def split_sentence(text):
    """分割文本"""
    text = re.sub(r'\s+', '', text)
    pattern = re.compile('[。，,.:：]')
    sentence_segments = pattern.sub(' ', text).split()
    return sentence_segments

# def split_sentence(text):
#     """分割文本"""
#     # pattern = re.compile('[。，,.]：')
#     # sentence_segments = pattern.sub(' ', sentence).split()
#     text = re.sub(r'\s+', '', text)
#     sentences = list(SentenceSplitter.split(text))
#     return sentences

def get_words_list(text, stopwords):
    """分词"""
    word_list = []
    for sentence in split_sentence(text):
        tmp = cut(''.join(token(sentence)))
        word_list.append([word for word in tmp if word not in stopwords])
    return word_list

def save_to_file(source_filename, save_filename, stopwords):
    """将content保存至文件"""
    data = pd.read_csv(source_filename, encoding='gb18030')
    data = data.fillna('')
    with open(save_filename, 'w', encoding='utf-8') as f:
        for content in data['content'].tolist():
            words_list = get_words_list(content, stopwords)
            content.append(' '.join([' '.join(words) for words in words_list]))
            f.write(str(content) + '\n')

if __name__ == "__main__":
    source_filename = '../data/sqlResult_1558435.csv'
    save_filename = '../data/news.txt'
    stopwords_filename = '../data/chinese_stopwords.txt'
    stopwords = get_stopwords(stopwords_filename)
    save_to_file(source_filename, save_filename, stopwords)

