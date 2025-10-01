import os
import json
from datasets import load_dataset
from feature_extractor import cnhubert
import torchaudio
import torch
import numpy as np
import librosa

def load_existing_ids(file_path) -> dict:
    """加载已记录的ID列表（如果文件存在）"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_ids(file_path, ids):
    """保存ID列表到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

os.environ['HF_HOME'] = 'E:/hf_cache'

cnhubert.cnhubert_base_path = os.environ.get("cnhubert_base_dir")
maxx = 0.95
alpha = 0.5
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

path = "Emilia/EN/*.tar" # Same for Emilia-YODAS; just replace "Emilia/" with "Emilia-YODAS/"
split_name = "en"
dataset = load_dataset("amphion/Emilia-Dataset",data_files={split_name: path}, split=split_name, streaming=True)
processed_ids = load_existing_ids("processed_datas.json")
#dataset = dataset.shuffle(seed=11223, buffer_size=10000)

cur_list = processed_ids.get("Emilia", list())
cur_id_bucket=set()
for i in cur_list:
    cur_id_bucket.add(i)
idx = 0
for sample in dataset:
    meta = sample['json']
    id = meta['id']
    dnsmos = meta['dnsmos']
    text = meta['text']
    decoder = sample['mp3']
    data = decoder.get_all_samples()
    if data.sample_rate != 32000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=data.sample_rate,
            new_freq=32000)
        audio_data = resampler(data.data)
    else:
        audio_data = data.data
    
    final_data = audio_data.flatten().numpy()
    tmp_max = np.abs(final_data).max()
    tmp_audio32 = (final_data / tmp_max * (maxx * alpha * 32768)) + ((1 - alpha) * 32768) * final_data
    tmp_audio32b = (final_data / tmp_max * (maxx * alpha * 1145.14)) + ((1 - alpha) * 1145.14) * final_data
    tmp_audio = librosa.resample(tmp_audio32b, orig_sr=32000, target_sr=16000)  # 不是重采样问题
    tensor_wav16 = torch.from_numpy(tmp_audio)
    break
