from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch

# download from https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment
# or https://modelscope.cn/models/Fengshenbang/Erlangshen-RoBERTa-330M-Sentiment
MODEL_PATH = "D:\code\SentimentAnalysis\Erlangshen-RoBERTa-330M-Sentiment"

print("loading model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)


def predict(text):
    output = model(torch.tensor([tokenizer.encode(text)]))
    result = torch.nn.functional.softmax(output.logits, dim=-1)
    predicted_classes = result.argmax(axis=1)
    if predicted_classes[0] == 1:
        print(text, ' positive')
    elif predicted_classes[0] == 0:
        print(text, ' negative')


if __name__ == '__main__':
    while True:
        predict(input(">> "))

# >> jintxian  positive
# >> 今天心情不好  negative
# >> 你是啊  negative
# >> 历史谁  positive
# >> 你是谁  negative
# >> 你的名字  positive
# >> 不好  negative
# >> 挺好用的  positive
# >> 这个商品有什么与哦漂亮  positive
# >> 这个商品没啥用  negative
