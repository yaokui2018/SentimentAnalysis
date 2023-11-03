# -*- coding: utf-8 -*-
# Author: 薄荷你玩
# Date: 2023/11/03
"""
加载ONNX模型推理
"""
import numpy as np
from transformers import BertTokenizer
from onnxruntime import InferenceSession

print("loading model...")
# 需要将 vocab.txt 文件复制到 onnx 目录)
tokenizer = BertTokenizer.from_pretrained("onnx/")
session = InferenceSession("onnx/model.onnx")


def predict(text):
    inputs = tokenizer(text, return_tensors="np")
    for key in inputs:
        inputs[key] = inputs[key].astype(np.int64)
    outputs = session.run(output_names=None, input_feed=dict(inputs))[0][0]
    # result = torch.nn.functional.softmax(torch.tensor(np.array(outputs)), dim=-1)
    result = np.exp(outputs) / np.sum(np.exp(outputs))
    predicted_classes = result.argmax(axis=-1)
    print(result.tolist())
    if predicted_classes == 2:
        print(text, ' positive')
    elif predicted_classes == 1:
        print(text, ' neural')
    elif predicted_classes == 0:
        print(text, ' negative')


if __name__ == '__main__':
    while True:
        predict(input(">> "))
