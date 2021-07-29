'''
import json
from flask import Flask,request,jsonify
from annoy import AnnoyIndex




class search(BaseResource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser.add_argument('name', required=True, type=str,
                                 location=['form', 'json', 'args', 'files', 'values', 'headers'])

    def post(self):
        print("方法进入")
        res = []
        my_json = request.get_json()
        keyword = my_json.get("keyword")
        keyword = str(keyword)

        with open('./res/5m_tc_word_index.json', 'r', encoding='gbk') as fp:
            word_index = json.load(fp)
        tc_index = AnnoyIndex(200)
        tc_index.load('./res/5m_tc_index_build.index')
        reverse_word_index = dict([(index, word) for (word, index) in word_index.items()])

        index = word_index.get(keyword + '\n')
        print(index)
        if (index):
            result = tc_index.get_nns_by_item(index, 5, include_distances=True)
            sim_keywords = [(str(reverse_word_index[idx]).strip()) for idx, distance in zip(result[0], result[1]) if
                            distance < 0.85]
            res = sim_keywords
        else:
            res = 'KEYWORD MISS!'

        print("结果：", res)

        if res == None:
            res = ['WRONG!']
        return jsonify({'sim_words': res})

'''