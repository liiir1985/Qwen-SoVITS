from tokenizers import Tokenizer
import torch
import base64
import numpy as np
from tqdm import tqdm
import random
import glob

LOCAL_PHONEME_TOKENIZER = "./pretrained_models/phoneme_tokenizer/phoneme_tokenizer.json"

class Qwen3AWQEvalDataset:
    dataset:list
    tokenizer:any
    phoneme_tokenizer:Tokenizer
    t2s_token_start:int
    def __init__(self,
             semantic_path: str,
             tokenizer):
        self.tokenizer = tokenizer
        self.dataset = []
        eos = torch.tensor(tokenizer.eos_token_id, dtype=torch.int64).unsqueeze(0)
        think_start = torch.tensor(tokenizer.convert_tokens_to_ids("<think>"), dtype=torch.int64).unsqueeze(0)
        think_end = torch.tensor(tokenizer.convert_tokens_to_ids("</think>"), dtype=torch.int64).unsqueeze(0)
        self.t2s_token_start = tokenizer.convert_tokens_to_ids("<t2s_0>")
        self.ph_token_start = tokenizer.convert_tokens_to_ids("<ph_0>")
        self.phoneme_tokenizer = Tokenizer.from_file(LOCAL_PHONEME_TOKENIZER)
        self.lang_tokens = {
            "en" : torch.tensor(tokenizer.convert_tokens_to_ids("<lang_en>"), dtype=torch.int64).unsqueeze(0),
            "zh" : torch.tensor(tokenizer.convert_tokens_to_ids("<lang_zh>"), dtype=torch.int64).unsqueeze(0),
            "ja" : torch.tensor(tokenizer.convert_tokens_to_ids("<lang_ja>"), dtype=torch.int64).unsqueeze(0),        
        }
        f_cnt = 0
        files = glob.glob(f"{semantic_path}/*.txt")        
        self.column_names=[
            "input_ids",
            "input_ids_full", 
        ]
        for i in tqdm(files,desc="Loading dataset"):
            with open(i, 'r', encoding='utf-8') as f:
                for line in f:
                    arr = line.split("\t")
                    prompt = f"<|im_start|>user\n文字转语音任务：{{{arr[0]}}}<|im_end|>\n<|im_start|>assistant\n"
                    input_ids = tokenizer([prompt], return_tensors="pt").to('cpu')
                    input_ids = input_ids.data['input_ids'].flatten()
                    ph_ids = torch.tensor(self.phoneme_tokenizer.encode(arr[2]).ids, dtype=torch.long).to('cpu') + self.ph_token_start
                    lang_token = self.lang_tokens.get(arr[1])
                    txt_ids = torch.cat([think_start, lang_token, ph_ids, think_end], dim=0)
                    txt_ids_full = torch.cat([input_ids, think_start, lang_token, ph_ids, think_end], dim=0)

                    # prompt_str = self.tokenizer.decode(txt_ids_full, skip_special_tokens=False)
                    # print(prompt_str)
                    # print(breakbreak)
                    
                    buffer = base64.b64decode(arr[3])
                    if random.randrange(100) < 50:
                        semantic_np = np.frombuffer(buffer, dtype=np.int16).copy()
                        semantic_ids = torch.from_numpy(semantic_np).to(torch.int64)
                        semantic_ids = semantic_ids + self.t2s_token_start
                        random_semantic = random.randrange(int((semantic_ids.shape[0]) / 1.5))
                        semantic_ids = semantic_ids[random_semantic:]
                        final = torch.cat([txt_ids, semantic_ids], dim=0)
                        final_full = torch.cat([txt_ids_full, semantic_ids], dim=0)
                    else:
                        final = txt_ids
                        final_full = txt_ids_full
                    
                    self.dataset.append({
                        "input_ids":final,
                        "input_ids_full":final_full
                        })
                    
            f_cnt+=1

        print(f"Dataset loaded with {len(self.dataset)} records")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]
    
    def get(self, key:str) -> dict:
        if key == "calibration":
            return self
        
    def shuffle(self):
        random.shuffle(self.dataset)
        return self

    def select(self, num):
        return [self.dataset[i] for i in num]