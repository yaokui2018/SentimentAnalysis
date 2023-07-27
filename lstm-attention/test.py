#! /bin/env python
# -*- coding: utf-8 -*-
"""
预测
"""
import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import tensorflow.python.keras.preprocessing.sequence as sequence
import json
from tensorflow.python.keras.models import model_from_json

from utils import AttentionLayer

np.random.seed(1337)  # For Reproducibility
import sys

sys.setrecursionlimit(1000000)

# define parameters
maxlen = 100


def create_dictionaries(model=None, combined=None):
    '''
        Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        vocab = list(model.wv.key_to_index.keys())
        gensim_dict.doc2bow(vocab, allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined):  # 闭包-->临时使用
            """
            Words become integers
            """
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)  # freqxiao10->0
                data.append(new_txt)
            return data  # word=>index

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load('../model/Word2vec_model.pkl')
    _, _, combined = create_dictionaries(model, words)
    return combined

model = None
def lstm_predict(string):
    global model
    if not model:
        print('loading model......')
        custom_objects = {'AttentionLayer': AttentionLayer}
        with open('../model/lstm-attention.json', 'r') as f:
            json_string = json.load(f)
        model = model_from_json(json_string, custom_objects=custom_objects)

        print('loading weights......')
        model.load_weights('../model/lstm-attention.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    data = input_transform(string)
    data.reshape(1, -1)
    # print data
    # result = model.predict_classes(data)
    result = model.predict(data)
    predicted_classes = result.argmax(axis=1)

    # print result # [[1]]
    if predicted_classes[0] == 1:
        print(string, ' positive')
    elif predicted_classes[0] == 0:
        print(string, ' neural')
    else:
        print(string, ' negative')


if __name__ == '__main__':
    # string='酒店的环境非常好，价格也便宜，值得推荐'
    # string='手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
    # string = "这是我看过文字写得很糟糕的书，因为买了，还是耐着性子看完了，但是总体来说不好，文字、内容、结构都不好"
    # string = "虽说是职场指导书，但是写的有点干涩，我读一半就看不下去了！"
    # string = "书的质量还好，但是内容实在没意思。本以为会侧重心理方面的分析，但实际上是婚外恋内容。"
    # string = "不是太好"
    # string = "不错不错"
    string = "真的一般，没什么可以学习的"

    lstm_predict(string)

    while True:
        lstm_predict(input(">> "))
