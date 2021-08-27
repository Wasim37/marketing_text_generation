#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Author: frank
@Date: 2020-08-08 22:40:43
@File: back_translate.py
"""
# !pip3 install jieba==0.36.2
import jieba
import http.client
import hashlib
import urllib
import random
import json
import time
from data_utils import write_samples

import os

curPath = os.path.abspath(os.path.dirname(__file__)) + "/"


def translate(q, source, target):
    """translate q from source language to target language
    Please refer to the official documentation   https://api.fanyi.baidu.com/ 通用翻译API
    There are demo on the website ,  register on the web site ,and get AppID, key, python3 demo.

    Args:
        q (str): sentence
        source(str): The language code
        target(str): The language code
    Returns:
        (str): result of translation
    """
    ###########################################
    #          TODO: module 2 task 1          #
    ###########################################
    appid = '20201019000593790'  # Fill in your AppID
    secretKey = 'meLMnlr5lBpDjpL5kLIr'  # Fill in your key

    httpClient = None
    myurl = '/api/trans/vip/translate'

    fromLang = source  # 原文语种
    toLang = target  # 译文语种
    salt = random.randint(32768, 65536)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
            salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)
        return result
    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()


def back_translate(q):
    """back_translate

    Args:
        q (str): sentence

    Returns:
        (str): result of back translation
    """
    ###########################################
    #          TODO: module 2 task 2          #
    ###########################################
    en = translate(q, "zh", "en")['trans_result'][0]['dst']
    time.sleep(2)
    target = translate(en, "en", "zh")['trans_result'][0]['dst']
    time.sleep(2)
    return target


def translate_continue(sample_path, translate_path):
    """translate  original file to new file

    Args:
        sample_path (str): original file path
        translate_path (str): target file path
    Returns:
        (str): result of back translation
    """
    ###########################################
    #          TODO: module 2 task 3          #
    ###########################################
    if os.path.exists(translate_path):
        with open(translate_path, 'r+', encoding='urf-8') as file:
            exit_len = len(list(file))
    else:
        # with open(translate_path, 'w', encoding='urf-8') as file:
        exit_len = 0

    translated = []
    count = 0
    with open(curPath + sample_path, 'r', encoding='utf-8') as file:
        for line in file:
            count += 1
            print(count)
            if count <= exit_len or count == 21585:
                continue
            source, ref = tuple(line.strip().split('<sep>'))
            source = back_translate(source.strip())
            ref = back_translate(ref.strip())
            source = ' '.join(list(jieba.cut(source)))
            ref = ' '.join(list(jieba.cut(ref)))
            translated.append(source + ' <sep> ' + ref)
            if count % 10 == 0:
                print(count)
                write_samples(translated, translate_path, 'a')
                translated = []
                if count == 1000:
                    break


if __name__ == '__main__':
    sample_path = 'output/train.txt'
    translate_path = 'output/translated.txt'
    translate_continue(sample_path, translate_path)
