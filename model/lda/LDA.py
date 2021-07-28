import math
import numpy as np
import os
import functools
import warnings
warnings.filterwarnings('ignore')  # 警告扰人，手动封存
from gensim import corpora,models
import joblib
import linecache
import jieba

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

# jieba分词的版本
# 算法负责人张晴阳
# 算法需要两个文件夹来保存数据，一个是make文件保存语料库分词洗词后的结果，一个是model文件保存算法训练好的模型

# get_stopwords获取停用词，用以后面的过滤
def get_stopwords():
    stopwords = [line.strip() for line in open(_get_abs_path('data/stopwords_LDA.txt')).readlines()]
    return stopwords

# 定义一个分词方法
def seg_to_list(sentence):
    seg_list=jieba.lcut(sentence)
    return seg_list

# 定义干扰词过滤方法：根据分词结果对干扰词进行过滤,
def word_filter(lists):
    stopwords=get_stopwords()
    filter_list=[s for s in lists if not s in stopwords and len(s) > 1]
    return filter_list

def cmp(e1, e2):
    # 排序函数，用于topK关键词的按值排序
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

def load_data():
    doc_list=[]
    for i in range(64):
        path=_get_abs_path("data/lda/dic")+str(i)+".txt"
        f=open(path,"r",encoding="utf-8")
        li=eval(f.read())
        doc_list+=li
    return doc_list

def data_make():
    path= _get_abs_path("data/corpus/corpus2021.txt")
    stopwords = [line.strip() for line in open(_get_abs_path('data/stopwords_LDA.txt')).readlines()]
    for k in range(0,64):
        doc_list = []
        for i in range(1000):
            number = (k * 1000) + i
            line = linecache.getline(path, number)
            seg_list = seg_to_list(line)
            filter_list=[s for s in seg_list if not s in stopwords and len(s) > 1]
            doc_list.append(filter_list)
        file_name="/dic"+str(k)+".txt"
        f=open(_get_abs_path("data/lda/")+file_name,"w",encoding="utf-8")
        f.write(str(doc_list))
        f.close()

# doc_list：加载数据集方法的返回结果
# model：主题模型的具体算法
# num_topics：主题模型的主题数量
class TopicModel(object):
    def __init__(self,num_topics=4):
        self.num_topics = num_topics
        if os.access(_get_abs_path("data/lda/ldaMode.pkl"), os.F_OK):
            self.get_model()
            self.tfidf_model=joblib.load(_get_abs_path("data/lda/tfidf_model.pkl"))
            self.dictionary=corpora.Dictionary.load(_get_abs_path('data/lda/dictionary.dict'))

            f=open(_get_abs_path("data/lda/wordtopic.txt"), 'r')
            self.wordtopic_dic=eval(f.read())
            f.close()
        else:
            if os.access(_get_abs_path("data/lda/dic0.txt"), os.F_OK):
                doc_list = load_data()
            else:
                data_make()
                doc_list = load_data()
            # 使用gensim的接口，将文本转换为向量化的表示
            self.dictionary = corpora.Dictionary(doc_list)
            self.dictionary.save(_get_abs_path('data/lda/dictionary.dict'))
            # 使用BOW模型向量化
            corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
            # 对每个词，根据TF-IDF进行加权，得到加权后的向量表示
            self.tfidf_model = models.TfidfModel(corpus)
            joblib.dump(self.tfidf_model, _get_abs_path("data/lda/tfidf_model.pkl"))
            self.corpus_tfidf = self.tfidf_model[corpus]

            self.model = self.train_lda()

            # 得到数据集的 主题-词分布
            word_dic = self.word_dictionary(doc_list)
            self.wordtopic_dic = self.get_wordtopic(word_dic)
            # 保存数据，提高之后运行的效率
            f = open(_get_abs_path("data/lda/wordtopic.txt"), 'w')
            f.write(str(self.wordtopic_dic))
            f.close()
    # LDA的训练时根据现有的数据集生成文档-主题分布矩阵和主题-词分布矩阵，Gensim中有实现好的方法，可以直接调用
    def train_lda(self):
        # iterations模型训练迭代的次数,passes扫描文章的次数
        lda=models.LdaModel(self.corpus_tfidf,id2word=self.dictionary,num_topics=self.num_topics)
        # 进行model保存
        joblib.dump(lda, _get_abs_path("data/lda/ldaMode.pkl"))
        return lda
    # 获得保存好的模型
    def get_model(self):
        lda=joblib.load(_get_abs_path("data/lda/ldaMode.pkl"))
        self.model=lda

    # 根据计算出的主题分布，通过lda模型对输入文本进行提取关键词
    def get_wordtopic(self, word_dic):
        wordTopic_dic={}
        for word in word_dic:
            single_list=[word]
            wordcorpus=self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic=self.model[wordcorpus]
            wordTopic_dic[word]=wordtopic
        return wordTopic_dic

    def get_simword(self, word_list):
        # 计算词的分布和文档的分布的相似度，去相似度最高的keyword_num个词作为关键词
        sent_corpus=self.tfidf_model[self.dictionary.doc2bow(word_list)]
        sent_topic=self.model[sent_corpus]
        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1*x1
                b += x1*x1
                c += x2*x2
            sim=a/math.sqrt(b*c) if not (b*c)==0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        # 列表保存关键词提取的结果
        li=[]
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, sent_topic)
            sim_dic[k] = sim
        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True):
            li.append(k)
        return li

    def word_dictionary(self, doc_list):
        # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
        dictionary=[]
        for doc in doc_list:
            dictionary.extend(doc)
        dictionary=list(set(dictionary))
        return dictionary

    def doc2bowvec(self, word_list):
        vec_list=[1 if word in word_list else 0 for word in self.dictionary]
        return vec_list

# 对外提供的函数
def keyword_extraction(temp):
    if os.path.exists(temp):
        f=open(temp,"r",encoding="utf-8")
        text=f.read()
        f.close()
    else:
        text=temp
    seg_list = seg_to_list(text)
    filter_list = word_filter(seg_list)
    topic_model = TopicModel()
    li = topic_model.get_simword(filter_list)
    return li



