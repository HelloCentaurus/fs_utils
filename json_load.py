# coding=utf-8

import json
from fs_utils.log import fsDecryption

def json_decrypt(fp):
    file_content = fp.read()
    return fsDecryption(file_content)

def json_load(fp):
    file_content = json_decrypt(fp)
    return json.loads(file_content, encoding='utf-8')
