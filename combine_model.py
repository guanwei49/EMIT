import argparse
import os
import torch
from model.internvl_chat import (InternVLChatConfig, CustomizedInternVLChatModel)
from transformers import AutoTokenizer

## Combine PEFT model (Lora) with base model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--peft', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--save_path', type=str,
                        default='/mnt/vlr/laishi/code/EMIT/EMIT-8B')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)

    model = CustomizedInternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()

    if args.peft is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.peft, is_trainable=False, torch_dtype=torch.bfloat16).eval()

    print(model)
    model = model.merge_and_unload()
    print(model)

    save_directory = args.save_path
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 保存 tokenizer
    tokenizer.save_pretrained(save_directory)

    # 保存模型
    model.save_pretrained(save_directory)

    print(f"Merged model and tokenizer are saved in {save_directory}")
