from torch.utils.data import Dataset
from transformers import AutoTokenizer
import base64
import numpy as np
import torch

LOCAL_MODEL_PATH = "./pretrained_models/qwen3" 

class Qwen3Text2SemanticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""
    dataset:list
    def __init__(
             self,
             semantic_path: str
    ) -> None:
        super().__init__()

        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH,
            trust_remote_code=True # 尽管是从本地加载，但 Qwen 模型建议保留此参数
        )
        padding = torch.tensor(tokenizer.pad_token_id, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        self.dataset = []
        with open(semantic_path, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.split("\t")
                txt_ids = tokenizer([arr[0]], return_tensors="pt").to('cpu')
                txt_ids = txt_ids.data['input_ids']
                buffer = base64.b64decode(arr[1])
                semantic_np = np.frombuffer(buffer, dtype=np.int16).copy()
                semantic_ids = torch.from_numpy(semantic_np).unsqueeze(0).to(torch.int64)
                final = torch.cat([txt_ids, padding, semantic_ids], dim=1)
                attention_mask = (final != tokenizer.pad_token_id).long()
                # Create labels (copy of input_ids)
                labels = final.clone()
                # Mask out prompt part (all tokens up to and including "### Response:\n")
                labels[0, :txt_ids.shape[1]] = -100
                self.dataset.append({
                    "input_ids": final, "attention_mask": attention_mask, "labels": labels
                })

