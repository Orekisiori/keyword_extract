import os
import jieba

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))
_get_module_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(__file__), path))

def jieba_method(input):
    print(_get_abs_path(input))
    if os.path.isfile(_get_abs_path(input)):
        f = open(_get_abs_path(input), 'rb')
        content = f.read()
        tags = jieba.analyse.extract_tags(content, 20)
        f.close()
        return tags
    else:
        tags = jieba.analyse.extract_tags(input, 20)
        return tags