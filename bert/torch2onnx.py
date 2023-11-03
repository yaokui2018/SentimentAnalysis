# -*- coding: utf-8 -*-
# Author: 薄荷你玩
# Date: 2023/11/03
"""
导出为ONNX模型
"""
import os
import shutil
import torch
from transformers import BertConfig, BertForSequenceClassification

# 加载checkpoint
model_checkpoint = "checkpoints"
# onnx 模型保存路径
onnx_path = "onnx/model.onnx"

config = BertConfig.from_pretrained(model_checkpoint)
model = BertForSequenceClassification.from_pretrained(model_checkpoint, config=config)

# 将classifier权重传递给导出ONNX模型的过程
# classifier_weight = model.classifier.weight
# classifier_bias = model.classifier.bias

dummy_input = torch.zeros(1, 120, dtype=torch.long)  # 假设输入的句子有120个token

os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
print("正在导出 onnx 模型...")
torch.onnx.export(
    model=model,
    args=(dummy_input, dummy_input, dummy_input),
    f=onnx_path,
    opset_version=12,
    verbose=False,
    export_params=True,  # store the trained parameter weights inside the model file
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],  # 需要注意顺序！不可随意改变, 否则结果与预期不符
    output_names=['last_hidden_state'],
    do_constant_folding=True,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
    },
)

print("ONNX模型已保存到:", onnx_path)

# 复制 vocab.txt
print(f"复制 {model_checkpoint}/vocab.txt -> {os.path.dirname(onnx_path)}/vocab.txt")
shutil.copy(model_checkpoint + "/vocab.txt", os.path.dirname(onnx_path) + "/vocab.txt")
