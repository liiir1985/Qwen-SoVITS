from torch.utils.data import Dataset
from tokenizers import Tokenizer
import base64
import numpy as np
import torch
from tqdm import tqdm
import random
import glob

LOCAL_MODEL_PATH = "./pretrained_models/qwen3" 
LOCAL_PHONEME_TOKENIZER = "./pretrained_models/phoneme_tokenizer/phoneme_tokenizer.json"

class Qwen3Text2SemanticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""
    dataset:list
    tokenizer:any
    phoneme_tokenizer:Tokenizer
    t2s_token_start:int
    random_mask_semantic:bool
    def __init__(
             self,
             semantic_path: str,
             tokenizer,
             random_mask_semantic=True,
             max_tokens_allowed = 1024,
             min_tokens_allowed = 0
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = []
        self.random_mask_semantic = random_mask_semantic
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

        files = glob.glob(f"{semantic_path}/*.txt")
        f_cnt = 1
        max_token_cnt = 0
        max_line=""
        skip_cnt = 0
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

                    prompt_phoneme = f'{{ "text": "{arr[0]}",\n"phoneme" : "{self.tokenizer.decode(ph_ids, skip_special_tokens=False)}"}}'
                    input_ids_ph = tokenizer([prompt_phoneme], return_tensors="pt").to('cpu')
                    input_ids_ph = input_ids_ph.data['input_ids'].flatten()

                    # prompt_str = self.tokenizer.decode(txt_ids_full, skip_special_tokens=False)
                    # print(prompt_str)
                    # print(breakbreak)
                    
                    buffer = base64.b64decode(arr[3])
                    semantic_np = np.frombuffer(buffer, dtype=np.int16).copy()
                    semantic_ids = torch.from_numpy(semantic_np).to(torch.int64)
                    semantic_ids = semantic_ids + self.t2s_token_start
                    final = torch.cat([txt_ids, semantic_ids, eos], dim=0)
                    final_full = torch.cat([txt_ids_full, semantic_ids, eos], dim=0)
                    #attention_mask = (final != tokenizer.pad_token_id).long()
                    # Create labels (copy of input_ids)
                    #labels = final.clone()
                    # Mask out prompt part (all tokens up to and including "### Response:\n")
                    #labels[0, :txt_ids.shape[0]] = -100
                    tokenCnt = final.shape[0]
                    if tokenCnt > max_tokens_allowed:
                        skip_cnt +=1
                        continue
                    if tokenCnt < min_tokens_allowed:
                        skip_cnt +=1
                        continue
                    if tokenCnt > max_token_cnt:
                        max_token_cnt = tokenCnt
                        #max_line = f"({line})({i})"
                    self.dataset.append({
                        "input_ids": final,
                        "input_ids_full":final_full, 
                        "input_ids_ph":input_ids_ph,
                        "prompt_len": txt_ids.shape[0], 
                        "prompt_len_full": txt_ids_full.shape[0], 
                        "lang": arr[1]
                    })
            f_cnt+=1
        

        print(f"Dataset loaded with {len(self.dataset)} records, max_tokens:{max_token_cnt}, skip_cnt:{skip_cnt}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]
    
    def collate(self, batch: list[dict]) -> dict:
        should_sft = random.randrange(100) < 5
        should_phoneme_sft = not should_sft and random.randrange(100) < 95
        if (not should_sft) and (not should_phoneme_sft):
            rn = random.randrange(100)
            if rn < 10:
                input_ids_list = [b["input_ids_ph"] for b in batch]
            elif rn < 30:
                input_ids_list = [b["input_ids"][b["prompt_len"]:] for b in batch]                
            else:
                input_ids_list = [b["input_ids"][1:b["prompt_len"]-1] for b in batch]
        else:
            input_ids_list = [b["input_ids_full"] if should_sft else b["input_ids"] for b in batch]
        print(self.tokenizer.decode(input_ids_list[0],skip_special_tokens=False))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)
        # Create labels (copy of input_ids)
        labels = input_ids.clone()
        # Mask out prompt part (all tokens up to and including "### Response:\n")
        for i, b in enumerate(batch):
            total_len = input_ids_list[i].shape[0]           
            if should_sft or should_phoneme_sft:
                prompt_len = b["prompt_len_full"] if should_sft else b["prompt_len"] # length of prompt in tokens            
                if self.random_mask_semantic and random.randrange(100) < 70:
                    max_val = input_ids_list[i].shape[0] - prompt_len - 10
                    random_semantic = min(random.randrange(int((input_ids_list[i].shape[0] - prompt_len) / 1.0)), max_val)
                else:
                    random_semantic = 0
                labels[i, :prompt_len + random_semantic] = -100  # ignore prompt tokens in loss
                
            labels[i, total_len:] = -100
            attention_mask[i, :total_len] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

