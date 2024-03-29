#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: author
@Date: 2020-07-13 20:16:37
@LastEditTime: 2020-07-18 17:28:41
@LastEditors: Please set LastEditors
@Description: Process a raw dataset into a sample file.
@FilePath: /project_2/data/process.py
'''

import sys
import os
import pathlib
import json
import jieba

from data_utils import write_samples, partition

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
curPath = os.path.abspath(os.path.dirname(__file__)) + '/'


samples = set()
# Read json file.
json_path = os.path.join(abs_path, '../files/服饰_50k.json')
with open(json_path, 'r', encoding='utf8') as file:
    jsf = json.load(file)

for jsobj in jsf.values():
    title = jsobj['title'] + ' '  # Get title.
    kb = dict(jsobj['kb']).items()  # Get attributes.
    kb_merged = ''
    for key, val in kb:
        kb_merged += key+' '+val+' '  # Merge attributes.

    ocr = ' '.join(list(jieba.cut(jsobj['ocr'])))  # Get OCR text.
    texts = []
    texts.append(title + ocr + kb_merged)  # Merge them.
    reference = ' '.join(list(jieba.cut(jsobj['reference'])))
    for text in texts:
        sample = text+'<sep>'+reference  # Seperate source and reference.
        samples.add(sample)
write_path = os.path.join(abs_path, '../files/samples.txt')
write_samples(samples, write_path)
partition(samples)
