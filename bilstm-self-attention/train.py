#! /bin/env python
# -*- coding: utf-8 -*-
"""
训练网络，并保存模型，其中LSTM的实现采用Python中的keras库
"""
import json
from tensorflow import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import jieba
import multiprocessing

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, LSTM, Dropout, Dense, Input, Activation, Bidirectional
from tensorflow.python.keras.preprocessing import sequence

from utils import AttentionLayer, SelfAttentionLayer

np.random.seed(1337)  # For Reproducibility
import sys

sys.setrecursionlimit(1000000)

# set parameters:
cpu_count = multiprocessing.cpu_count()  # 4
vocab_dim = 100
n_iterations = 1  # ideally more..
n_exposures = 10  # 所有频数超过10的词语
window_size = 7
n_epoch = 100
input_length = 100
maxlen = 100

batch_size = 1024


def loadfile(datapath="../data/"):
    # datapath += "ChnSentiCorp/"
    neg = pd.read_csv(datapath + 'neg.csv', header=None, index_col=None, sep="\t")
    pos = pd.read_csv(datapath + 'pos.csv', header=None, index_col=None, error_bad_lines=False, sep="\t")
    neu = pd.read_csv(datapath + 'neutral.csv', header=None, index_col=None)
    # neu = pd.DataFrame(["null"])

    combined = np.concatenate((pos[0], neu[0], neg[0]))
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neu), dtype=int),
                        -1 * np.ones(len(neg), dtype=int)))

    return combined, y


# 对句子经行分词，并去掉换行符
def tokenizer(text):
    """
    Simple Parser converting each document to lower-case, then
    removing the breaks for new lines and finally splitting on the
    whitespace
    """
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


def create_dictionaries(model=None, combined=None):
    """
    Function does are number of Jobs:
    1- Creates a word to index mapping
    2- Creates a word to vector mapping
    3- Transforms the Training and Testing Dictionaries
    """
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        vocab = list(model.wv.key_to_index.keys())
        gensim_dict.doc2bow(vocab, allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined):  # 闭包-->临时使用
            ''' Words become integers
            '''
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


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    model = Word2Vec(vector_size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     epochs=n_iterations)

    model.build_vocab(combined)  # input: list
    model.train(combined, total_examples=model.corpus_count, epochs=n_iterations)
    model.save('../model/Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2, shuffle=True)
    y_train = keras.utils.to_categorical(y_train, num_classes=3)
    y_test = keras.utils.to_categorical(y_test, num_classes=3)
    # print x_train.shape,y_train.shape
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


##定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple Keras Model...')

    # 输入层
    inputs = Input(shape=(input_length,))
    # Embedding层
    embedding_layer = Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=True,
                                weights=[embedding_weights], input_length=input_length)(inputs)
    # LSTM层并返回序列和状态
    lstm_layer, forward_h, forward_c, backward_h, backward_c = Bidirectional(
        LSTM(units=50, activation='tanh', return_sequences=True, return_state=True))(embedding_layer)
    # 注意力层
    attention_layer = SelfAttentionLayer()(lstm_layer)
    # # Dropout层
    dropout_layer = Dropout(0.5)(attention_layer)
    # 全连接层,输出维度=3
    dense_layer = Dense(3, activation='softmax')(dropout_layer)

    # 创建模型
    model = Model(inputs=inputs, outputs=dense_layer)

    # 创建EarlyStopping回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

    # 创建训练集和验证集
    train_size = int(0.9 * len(x_train))
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    print("Train...")  # batch_size=32
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, callbacks=[early_stopping],
              validation_data=(x_val, y_val))

    print("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('Test score:', score)

    print("Save model...")
    save_path = '../model/bilstm-self-attention.h5'
    model.save(save_path)  # 保存模型
    # 转换为JSON格式
    json_string = model.to_json()
    with open('../model/bilstm-self-attention.json', 'w') as outfile:
        json.dump(json_string, outfile)

    print("model saved!", save_path)

    # result = model.predict(x_test)
    # print(result)


# 训练模型，并保存
print('Loading Data...')
combined, y = loadfile()
print(len(combined), len(y))
print('Tokenising...')
combined = tokenizer(combined)
print('Training a Word2vec model...')
index_dict, word_vectors, combined = word2vec_train(combined)

print('Setting up Arrays for Keras Embedding Layer...')
n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
print("x_train.shape and y_train.shape:")
print(x_train.shape, y_train.shape)
train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)
