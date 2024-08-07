# load model and compress
# parallelized with deepspeed
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from tqdm import tqdm
import pandas as pda
import argparse

def compute_compressed_length(LLM, load_in_8bit, load_in_4bit, dataset, lora):
    no = dataset[-5]
    #dataset = pda.read_csv(dataset)['segments'].tolist()
    dataset = pda.read_csv(dataset)['text'].tolist()

    device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    #init tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(LLM, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token 
    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(LLM, load_in_8bit=True, device_map=device_map,torch_dtype=torch.float16, use_flash_attention_2=True)
    elif load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(LLM, load_in_4bit=True, device_map=device_map,torch_dtype=torch.float16, use_flash_attention_2=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(LLM, device_map=device_map)
    if lora != None:
        model = PeftModel.from_pretrained(model, lora, device_map={"": 0})
    
    model.eval()

    compressed_lengths = []
    checkpoint = 512
    cnt = 1
    for segment in tqdm(dataset):
        
        if cnt % checkpoint == 0:
            # 64:128K byte is inferred
            check_cb = sum(compressed_lengths[:cnt-1])
            raw = cnt * 2048
            compressed = check_cb / 8
            print('raw:{} Bytes; compressed {} Bytes; Ratio: {}.\n'.format(raw, compressed, raw / compressed))
        cnt += 1

        tokens = tokenizer(segment, return_tensors="pt")
        tokens['input_ids'] = tokens['input_ids'].to(device_map)
        tokens['attention_mask'] = tokens['attention_mask'].to(device_map)
        with torch.no_grad():
            logits = model(**tokens).logits

        pd = torch.gather(logits.softmax(-1), -1, torch.roll(tokens["input_ids"], -1, 1).unsqueeze(-1))
        neg_log2_pd = -1 * torch.log2(pd).squeeze(-1)
        compressed_lengths.append(sum(neg_log2_pd.tolist()[0]))

    return compressed_lengths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('LLM', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--lora', type=str, default=None)
                        
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--load_in_8bit', action='store_true')
    group.add_argument('--load_in_4bit', action='store_true')

    args = parser.parse_args()
    cl = compute_compressed_length(args.LLM, args.load_in_8bit, args.load_in_4bit, args.dataset, args.lora)
    df = pda.DataFrame(cl, columns=['cl'])
    df.to_csv('result_{}.csv'.format(args.dataset[-5]), index=False)
