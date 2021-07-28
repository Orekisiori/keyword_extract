# encoding=utf-8
import os

import jieba

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))



def precess_data(text, input_type) -> str:
    """
    :param text: 同rakun()
    :param input_type: 同rakun()
    :return: 文件路径
    """

    if os.access(text, os.F_OK):
        with open(text, "r", encoding='utf-8') as f:  # 打开文件
            data = f.read()  # 读取文件
            words = jieba.cut(data)
            str = " ".join(words)
            res_path = _get_abs_path('data/rakun/rakun_temp.txt')
            with open(res_path, "w", encoding='utf-8') as f:
                f.write(str)
        return res_path

    else:
        words = jieba.cut(text)
        str = " ".join(words)
        res_path = _get_abs_path('data/rakun/rakun_temp.txt')
        with open(res_path, "w", encoding='utf-8') as f:
            f.write(str)
        return res_path
