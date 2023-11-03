# -*- coding: utf-8 -*-
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # dir
    parser.add_argument("--data_dir", default='./data', type=str, help="数据集路径")
    parser.add_argument("--bert_model", default='bert-base-chinese', type=str, help="Bert模型名，可在线仓库名/本地路径")
    parser.add_argument("--output_dir", default='checkpoints/', type=str, help="模型 checkpoints 保存目录")
    # parser.add_argument("--model_save_pth", default='checkpoints/bert.pth', type=str, help="模型文件保存路径")
    # other parameters
    parser.add_argument("--max_seq_length", default=120, type=int, help="字符串最大长度")
    parser.add_argument("--do_train", default=True, action='store_true', help="训练模式")
    parser.add_argument("--do_eval", default=True, action='store_true', help="验证模式")
    parser.add_argument("--do_lower_case", default=False, action='store_true', help="英文字符的大小写转换，对于中文来说没啥用")
    parser.add_argument("--train_batch_size", default=32, type=int, help="训练时batch大小")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="验证时batch大小")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Adam初始学习步长")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="训练的epochs次数")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for."
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", default=False, action='store_true', help="是否不使用CUDA")
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank for distributed training on gpus.")
    parser.add_argument("--seed", default=777, type=int, help="初始化时的随机数种子")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimize_on_cpu", default=False, action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU.")
    parser.add_argument("--fp16", default=False, action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")
    parser.add_argument("--loss_scale", default=128, type=float,
                        help="Loss scaling, positive power of 2 values can improve fp16 convergence.")

    args = parser.parse_args()

    return args
