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

def add_ending_punctuation_by_lang(text:str, lang:str)->str:
    if lang == 'ja' or lang =='zh':
        if text[-1] != "。":
            return text + "。"
        else:
            return text
    else:
        if text[-1] != ".":        
            return text + "."
        else:
            return text

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
             random_mask_semantic=True
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = []
        self.random_mask_semantic = random_mask_semantic
        eos = torch.tensor(tokenizer.eos_token_id, dtype=torch.int64).unsqueeze(0)
        self.t2s_token_start = tokenizer.convert_tokens_to_ids("<t2s_0>")
        self.phoneme_tokenizer = Tokenizer.from_file(LOCAL_PHONEME_TOKENIZER)    

        files = glob.glob(f"{semantic_path}/*.txt")
        f_cnt = 1
        for i in tqdm(files,desc="Loading dataset"):
            with open(i, 'r', encoding='utf-8') as f:
                for line in f:
                    arr = line.split("\t")
                    prompt = f"<|im_start|>user\n语音转文本任务：{{{add_ending_punctuation_by_lang(arr[0], arr[1])}}}<|im_end|>\n<|im_start|>assistant\n"
                    txt_ids = tokenizer([prompt], return_tensors="pt").to('cpu')
                    txt_ids = txt_ids.data['input_ids'].flatten()
                    buffer = base64.b64decode(arr[3])
                    semantic_np = np.frombuffer(buffer, dtype=np.int16).copy()
                    semantic_ids = torch.from_numpy(semantic_np).to(torch.int64)
                    semantic_ids = semantic_ids + self.t2s_token_start
                    final = torch.cat([txt_ids, semantic_ids, eos], dim=0)
                    #attention_mask = (final != tokenizer.pad_token_id).long()
                    # Create labels (copy of input_ids)
                    #labels = final.clone()
                    # Mask out prompt part (all tokens up to and including "### Response:\n")
                    #labels[0, :txt_ids.shape[0]] = -100
                    self.dataset.append({
                        "input_ids": final, "prompt_len": txt_ids.shape[0], "lang": arr[1]
                    })
            f_cnt+=1
        

        print(f"Dataset loaded with {len(self.dataset)} records")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]
    
    def collate(self, batch: list[dict]) -> dict:
        input_ids_list = [b["input_ids"] for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)
        # Create labels (copy of input_ids)
        labels = input_ids.clone()
        # Mask out prompt part (all tokens up to and including "### Response:\n")
        for i, b in enumerate(batch):
            prompt_len = b["prompt_len"]  # length of prompt in tokens
            total_len = b["input_ids"].shape[0]
            if self.random_mask_semantic and random.randrange(100) < 50:
                random_semantic = random.randrange(int((input_ids_list[i].shape[0] - prompt_len) / 1.5))
            else:
                random_semantic = 0
            labels[i, :prompt_len + random_semantic] = -100  # ignore prompt tokens in loss
            labels[i, total_len:] = -100
            attention_mask[i, :total_len] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

