# -*- coding: utf-8 -*-

def get_data(filename):
    data = []
    with open(filename, encoding='utf8') as f:
        for index, line in enumerate(f.readlines()):
            if index == 0:
                continue
            data.append(line.split("\t"))
    return data


def write_file(data_list, negfile="neg.csv", posfile="pos.csv"):
    neg_text = []
    pos_text = []
    for data in data_list:
        if len(data) == 3:
            data[0] = data[1]
            data[1] = data[2]
        label = data[0].strip()
        text = data[1].strip() + "\n"
        if label == "1":
            pos_text.append(text)
        elif label == "0":
            neg_text.append(text)
        else:
            print(label, text)
            assert 1 == 2

    with open(posfile, 'a+', encoding="utf8") as f:
        f.writelines(pos_text)
    with open(negfile, 'a+', encoding="utf8") as f:
        f.writelines(neg_text)

    print(f"文件写入成功，正样本：{len(pos_text)}，负样本：{len(neg_text)}")

if __name__ == '__main__':
    files = ['train.tsv', 'dev.tsv']
    for file in files:
        write_file(get_data(file))