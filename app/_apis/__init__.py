from app.extensions import api
from .firstApi import FirstApi
from .KeywordExtract import KeywordExtract
# from .search import sim_word_search

# 此处注册api
api.add_resource(FirstApi, '/hello')
api.add_resource(KeywordExtract,'/kwe')
# api.add_resource(sim_word_search,'/search')
