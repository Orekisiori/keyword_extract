from flask import request
from flask.json import jsonify

from app.base import BaseResource



class RawTest(BaseResource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser.add_argument('name', required=True, type=str,
                                 location=['form', 'json', 'args', 'files', 'values', 'headers'])

    def get(self):
        return 'POST request only'

    def post(self):
        my_json = request.get_json()
        my_string = my_json.get("String")
        keyword = my_string[:4]
        keyword = keyword + '，' + my_string[9:15]
        keyword = keyword + '，' + my_string[25:28]
        keyword = keyword + '，' + my_string[42:46]
        keyword = keyword + '，' + my_string[-6:]
        print(keyword)
        return jsonify({'keyword': keyword})
