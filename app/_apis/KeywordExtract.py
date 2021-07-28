from app.base import BaseResource
from flask import Flask,request,jsonify
from gevent import pywsgi
import model.rakun.rakun as ra
import model.tfidf.TFIDF as ti
import model.lda.LDA as lda
import os
import jieba.analyse


def jieba_method(input):
    tags = jieba.analyse.extract_tags(input, 20)
    return tags

class KeywordExtract(BaseResource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser.add_argument('name', required=True, type=str,
                                 location=['form', 'json', 'args', 'files', 'values', 'headers'])

    def post(self):
        res = []
        my_json = request.get_json()
        text = my_json.get("text")
        method = my_json.get("method")

        if len(method) == 0:
            res = jieba_method(text)

        if len(method) == 1:
            if method[0] == 0:
                # jieba method
                res = jieba_method(text)
            elif method[0] == 1:
                # LDA
                res = lda.keyword_extraction(text)
            elif method[0] == 2:
                # RaKUn method
                res = ra.rakun(text, 'file')
            elif method[0] == 3:
                # TF-IDF
                res = ti.tfidf_extract(text)
            else:
                print('2nd param should be 0~3')
        else:
            print('there should be only 2 params')

        print(res)
        return jsonify({'keyword': res})



