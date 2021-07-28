import math
import os
import jieba
import functools

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))


def get_stopword_list():
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stop_word_path = _get_abs_path('data/stop_words_TFIDF.txt')
    stopword_list = [sw.strip() for sw in open(stop_word_path).readlines()]
    return stopword_list


def seg_to_list(sentence):
    ''' 分词方法，调用结巴接口 '''
    seg_list = jieba.cut(sentence)
    return seg_list


def word_filter(seg_list):
    '''
        1. 根据分词结果对干扰词进行过滤；
        2. 再判断是否在停用词表中，长度是否大于等于2等；
    '''
    stopword_list = get_stopword_list()  # 获取停用词表
    filter_list = []  # 保存过滤后的结果
    #  下面代码： 根据pos参数选择是否词性过滤
    ## 下面代码： 如果不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        word = seg  # 单词
        # 过滤停用词表中的词，以及长度为<2的词
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)

    return filter_list


def load_data(corpus_path=_get_abs_path('data/corpus/corpus2021.txt')):
    '''
        目的：
            调用上面方法对数据集进行处理，处理后的每条数据仅保留非干扰词
        参数：
            1. 数据加载
            2. corpus_path: 数据集路径
    '''
    doc_list = []  # 结果
    stop_word_path = _get_abs_path('data/stop_words_TFIDF.txt')
    stopword_list = [sw.strip() for sw in open(stop_word_path).readlines()]
    print(str(stopword_list))
    for line in open(corpus_path, 'r', encoding="utf-8"):
        content = line.strip()  # 每行的数据
        seg_list = seg_to_list(content)  # 分词
        filter_list = [s for s in seg_list if not s in stopword_list and len(s) > 1]
        doc_list.append(filter_list)  # 将处理后的结果保存到doc_list
    return doc_list


# TF-IDF的训练主要是根据数据集生成对应的IDF值字典，后续计算每个词的TF-IDF时，直接从字典中读取。

def train_idf():
    if os.access(_get_abs_path('data/tfidf/idf.txt'), os.F_OK):
        file = open(_get_abs_path('data/tfidf/idf.txt'), mode='r')
        file2 = open(_get_abs_path('data/tfidf/default_idf.txt'), mode='r')
        idf_dic = eval(file.read())
        default_idf = eval(file2.read())
    else:
        doc_list = load_data()
        idf_dic = {}  # idf对应的字典
        file = open(_get_abs_path('data/tfidf/idf.txt'), mode='w')  # 打开文件，没有则创建
        file2 = open(_get_abs_path('data/tfidf/default_idf.txt'), mode='w')
        tt_count = len(doc_list)  # 总文档数
        # 每个词出现的文档数
        for doc in doc_list:
            for word in set(doc):
                idf_dic[word] = idf_dic.get(word, 0.0) + 1.0
                print(idf_dic[word])
        # 按公式转换为idf值，分母加1进行平滑处理
        for k, v in idf_dic.items():
            idf_dic[k] = math.log(tt_count / (1.0 + v))
        # 对于没有在字典中的词，默认其尽在一个文档出现，得到默认idf值
        default_idf = math.log(tt_count / (1.0))
        file.write(str(idf_dic))
        file2.write(str(default_idf))
    return idf_dic, default_idf


# 为了输出top关键词时，先按照关键词的计算分值排序，在得分相同时，根据关键词进行排序
def cmp(e1, e2):
    ''' 排序函数，用于topK关键词的按值排序 '''
    import numpy as np
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


class TfIdf(object):
    # 四个参数分别是：训练好的idf字典，默认idf字典，处理后的待提取文本， 关键词数量
    def __init__(self, idf_dic, default_idf, word_list):
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.word_list = word_list
        self.tf_dic = self.get_tf_dic()  # 统计tf值

    def get_tf_dic(self):
        # 统计tf值
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0
        tt_count = len(self.word_list)
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count  # 根据tf求值公式

        return tf_dic

    def get_tfidf(self):
        # 计算tf-idf值
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)

            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        tfidf_dic.items()
        # 根据tf-idf排序，去排名前keyword_num的词作为关键词
        i = 0
        Li = []
        for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True):
            Li.append(k)
        return Li


def tfidf_extract(temp):
    if os.path.exists(temp):
        text = open(temp, 'r', encoding='utf-8').read()
    else:
        text = temp
    seg_list = seg_to_list(text)
    filter_list = word_filter(seg_list)
    word_list = filter_list
    idf_dic, default_idf = train_idf()
    tfidf_model = TfIdf(idf_dic, default_idf, word_list)
    li = []
    li = tfidf_model.get_tfidf()
    return li

