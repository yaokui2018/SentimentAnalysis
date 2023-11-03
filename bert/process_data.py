import numpy as np
import random
import json
import pandas as pd

random.seed(233)


def save_data(path, data_list):
    json_list = []
    for stc, label in data_list:
        sample = {}
        sample['text'] = stc
        sample['label'] = int(label)
        json_list.append(json.dumps(sample))
    json_list = '\n'.join(json_list)
    with open(path, 'w') as w:
        w.write(json_list)


datapath = '../data/'
neg = pd.read_csv(datapath + 'neg.csv', header=None, index_col=None, sep="\t")
pos = pd.read_csv(datapath + 'pos.csv', header=None, index_col=None, error_bad_lines=False, sep="\t")
neu = pd.read_csv(datapath + 'neutral.csv', header=None, index_col=None)

stc_list = np.concatenate((pos[0], neu[0], neg[0]))

# 0: neg, 1: neutral, 2: pos
label_list = np.concatenate(
    (2 * np.ones(len(pos), dtype=int), np.ones(len(neu), dtype=int), np.zeros(len(neg), dtype=int)))
c = list(zip(stc_list, label_list))
random.shuffle(c)
stc_list[:], label_list[:] = zip(*c)

# train:valid:test = 8:1:1
train_len = int(0.8 * len(stc_list))
test_len = int(0.9 * len(stc_list))
train_data = c[:train_len]
valid_data = c[train_len:test_len]
test_data = c[test_len:]

save_data('data/train.txt', train_data)
save_data('data/valid.txt', valid_data)
save_data('data/test.txt', test_data)
