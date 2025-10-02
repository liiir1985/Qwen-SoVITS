from torch.utils.data import Dataset
from transformers import AutoTokenizer
import base64
import numpy as np
import torch
from tqdm import tqdm

LOCAL_MODEL_PATH = "./pretrained_models/qwen3" 

class Qwen3Text2SemanticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""
    dataset:list
    tokenizer:any
    t2s_token_start:int
    def __init__(
             self,
             semantic_path: str,
             tokenizer
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = []
        eos = torch.tensor(tokenizer.eos_token_id, dtype=torch.int64).unsqueeze(0)
        self.t2s_token_start = tokenizer.convert_tokens_to_ids("<t2s_0>")
        with open(semantic_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
                arr = line.split("\t")
                prompt = f"<|im_start|>user\n语音转文本任务：{{{arr[0]}}}<|im_end|>\n<|im_start|>assistant\n"
                txt_ids = tokenizer([prompt], return_tensors="pt").to('cpu')
                txt_ids = txt_ids.data['input_ids'].flatten()
                buffer = base64.b64decode(arr[1])
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
                    "input_ids": final, "prompt_len": txt_ids.shape[0]
                })

        print(f"Dataset loaded with {len(self.dataset)} records")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]
    
    def collate(self, batch: list[dict]) -> dict:
        input_ids_list = [b["input_ids"] for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        # Create labels (copy of input_ids)
        labels = input_ids.clone()
        # Mask out prompt part (all tokens up to and including "### Response:\n")
        for i, b in enumerate(batch):
            prompt_len = b["prompt_len"]  # length of prompt in tokens
            labels[i, :prompt_len] = -100  # ignore prompt tokens in loss
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

