# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, PYTORCH_PRETRAINED_BERT_CACHE

from bert.config import get_args
from train import convert_examples_to_features
from train import MyPro
import json

args = get_args()


def init_model(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            print("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    print("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    processor = MyPro()
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(
        args.output_dir, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE + '/distributed_{}'.format(args.local_rank),
        num_labels=len(label_list)
    )
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # if not torch.cuda.is_available():
    #     model.load_state_dict(torch.load(args.model_save_pth, map_location='cpu')['state_dict'])
    # else:
    #     model.load_state_dict(torch.load(args.model_save_pth)['state_dict'])

    return model, processor, args, label_list, tokenizer, device


class ParseHandler:
    def __init__(self, model, processor, args, label_list, tokenizer, device, label_map, return_text=True):
        self.model = model
        self.processor = processor
        self.args = args
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.device = device
        self.label_map = label_map
        self.return_text = return_text

        self.model.eval()

    def parse(self, text_list):
        result = []
        test_examples = self.processor.get_ifrn_examples(text_list)
        test_features = convert_examples_to_features(
            test_examples, self.label_list, self.args.max_seq_length, self.tokenizer, show_exp=False)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.args.eval_batch_size)

        for idx, (input_ids, input_mask, segment_ids) in enumerate(test_dataloader):
            item = {}
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            text = test_examples[idx].text_a
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
                logits = outputs.logits
                logits = F.softmax(logits, dim=1)
                pred = logits.max(1)[1]
                logits = logits.detach().cpu().numpy()[0].tolist()
                if self.return_text:
                    item['text'] = text
                item['label'] = self.label_map[pred.item()]
                item['scores'] = logits
                result.append(item)
        return json.dumps(result)

    def predict(self, text=None):
        text_list = (text or "好好好,不好不好,大家好才是真的好,你好嗄").split(",")
        print(f'接收到数据: {text_list}')
        if len(text_list) > 32:
            return json.dumps({"error,list length must less than 32"})
        text_list = [text[:args.max_seq_length] for text in text_list]
        out = self.parse(text_list)
        return out


if __name__ == "__main__":
    print('init model started.')
    model, processor, args, label_list, tokenizer, device = init_model(args)
    print('init model finished.')

    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    parse_handler = ParseHandler(model, processor, args, label_list, tokenizer, device, label_map)
    while True:
        for res in json.loads(parse_handler.predict(input(">> "))):
            print(res)
