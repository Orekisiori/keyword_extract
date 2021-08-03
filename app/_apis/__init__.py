from app.extensions import api
from .firstApi import FirstApi
from .KeywordExtract import KeywordExtract

# 此处注册api
api.add_resource(FirstApi, '/hello')
api.add_resource(KeywordExtract,'/kwe')
