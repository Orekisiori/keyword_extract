from app.extensions import api
from .firstApi import FirstApi
from .rawTest import RawTest

# 此处注册api
api.add_resource(FirstApi, '/hello')
api.add_resource(RawTest,'/kwpu')
api.add_resource(KeywordExtract,'/kwe')
