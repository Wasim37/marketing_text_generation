#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: author
@Date: 2020-07-13 20:16:37
LastEditTime: 2021-08-27 17:21:59
@FilePath: /project_2/data/embed_replace.py
'''

from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
from data_utils import read_samples, isChinese, write_samples
import os
from gensim import matutils
from itertools import islice
import numpy as np


class EmbedReplace():
    def __init__(self, sample_path, wv_path):
        print("read sample file ...")
        self.samples = read_samples(sample_path)
        self.refs = [sample.split('<sep>')[1].split() for sample in self.samples]
        print("load word_vectors file ...")
        self.wv = KeyedVectors.load_word2vec_format(wv_path, binary=False)

        if os.path.exists('saved/tfidf.model'):
            print("load tfidf model..")
            self.tfidf_model = TfidfModel.load('saved/tfidf.model')
            self.dct = Dictionary.load('saved/tfidf.dict')
            self.corpus = [self.dct.doc2bow(doc) for doc in self.refs]
        else:
            # 训练tfidf，后续单词替换时，用来排除核心词汇
            print("train tfidf model..")
            self.dct = Dictionary(self.refs)
            self.corpus = [self.dct.doc2bow(doc) for doc in self.refs]
            self.tfidf_model = TfidfModel(self.corpus)
            self.dct.save('saved/tfidf.dict')
            self.tfidf_model.save('saved/tfidf.model')
            self.vocab_size = len(self.dct.token2id)

    def vectorize(self, docs, vocab_size):
        '''
        docs :: iterable of iterable of (int, number)
        '''
        return matutils.corpus2dense(docs, vocab_size)

    def extract_keywords(self, dct, tfidf, threshold=0.2, topk=5):

        """find high TFIDF socore keywords
        根据TFIDF确认需要排除的核心词汇.
        注意：为了防止将体现关键卖点的词给替换掉，导致核心语义丢失，
        所以通过 tfidf 权重对词表的词进行排序，然后替换排序靠后的词

        Args:
            dct (Dictionary): gensim.corpora Dictionary  a reference Dictionary
            tfidf (list of tfidf):  model[doc]  [(int, number)]
            threshold (float) : high TFIDF socore must be greater than the threshold
            topk(int): num of highest TFIDF socore
        Returns:
            (list): A list of keywords
        """

        ###########################################
        #          TODO: module 1 task 1          #
        ###########################################
        # 降序。sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作
        tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)
        # islice()获取迭代器的切片，消耗迭代器. islice(iterable, [start, ] stop [, step])
        return list(islice([dct[w] for w, score in tfidf if score > threshold], topk))

    def replace(self, token_list, doc):
        """replace token by another token which is similar in wordvector
        在 embedding 的词向量空间中寻找语义最接近的词进行替换

        Args:
            token_list (list): reference token list
            doc (list): A reference represented by a word bag model
        Returns:
            (str):  new reference str
        """

        ###########################################
        #          TODO: module 1 task 2          #
        ###########################################
        keywords = self.extract_keywords(self.dct, self.tfidf_model[doc])
        num = int(len(token_list) * 0.3)
        new_tokens = token_list.copy()
        while num == int(len(token_list) * 0.3):
            indexes = np.random.choice(len(token_list), num)
            for index in indexes:
                token = token_list[index]
                if isChinese(token) and token not in keywords and token in self.wv:
                    new_tokens[index] = self.wv.most_similar(positive=token, negative=None, topn=1)[0][0]
            num -= 1
        return ' '.join(new_tokens)

    def generate_samples(self, write_path):
        """generate new samples file
        通过替换reference中的词生成新的reference样本

        Args:
            write_path (str):  new samples file path

        """
        ###########################################
        #          TODO: module 1 task 3          #
        ###########################################
        replaced = []
        count = 0
        for sample, token_list, doc in zip(self.samples, self.refs, self.corpus):
            replaced.append(
                sample.split('<sep>')[0] + ' <sep> ' +
                self.replace(token_list, doc))
            count += 1
            if count % 100 == 0:
                print(count)
                write_samples(replaced, write_path, 'a')
                replaced = []


if __name__ == '__main__':
    sample_path = 'output/train.txt'
    wv_path = 'word_vectors/merge_sgns_bigram_char300.txt'
    replacer = EmbedReplace(sample_path, wv_path)
    replacer.generate_samples('output/replaced.txt')
