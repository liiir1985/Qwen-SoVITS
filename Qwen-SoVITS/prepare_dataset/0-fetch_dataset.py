import os
import json
from datasets import load_dataset
from feature_extractor import cnhubert
import torchaudio
import torch
import numpy as np
import librosa
from scipy.io import wavfile

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
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

cnhubert.cnhubert_base_path = 'F:/myexe/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base'#os.environ.get("cnhubert_base_dir")
model = cnhubert.get_model()
model = model.to(device)
maxx = 0.95
alpha = 0.5


path = "Emilia/EN/*.tar" # Same for Emilia-YODAS; just replace "Emilia/" with "Emilia-YODAS/"
split_name = "en"
target_duration = 60#1 * 60 * 60

dataset = load_dataset("amphion/Emilia-Dataset",data_files={split_name: path}, split=split_name, streaming=True)
database_path = f"{dataset_dir}/processed_datas.json"
dataset_save_path = f"{dataset_dir}/{split_name}/"
os.makedirs(dataset_save_path, exist_ok=True)

processed_ids = load_existing_ids(database_path)
#dataset = dataset.shuffle(seed=11223, buffer_size=10000)

cur_list = processed_ids.get("Emilia", list())
total_secs = 0
cur_id_bucket={}
for i in cur_list:
    total_secs+=i['duration']    
    cur_id_bucket[i['id']] = i
    
for sample in dataset:
    meta = sample['json']
    id = meta['id']
    dnsmos = meta['dnsmos']
    text = meta['text']
    duration = meta['duration']
    if id in cur_id_bucket:
        continue
    
    if dnsmos < 3.1:
        continue

    total_secs += duration

    decoder = sample['mp3']
    data = decoder.get_all_samples()

    torchaudio.save(
        uri=f"{dataset_save_path}/{id}.flac", 
        src=data.data,
        sample_rate=data.sample_rate,
        format="flac")
    
    with open(f"{dataset_save_path}/{id}.txt", 'w', encoding='utf8') as f: f.write(text)
    cur_id_bucket[id] = {'id':id,'duration':duration}
    if total_secs > target_duration:
        break
processed_ids['Emilia'] = [i for i in cur_id_bucket.values()]
save_ids(database_path, processed_ids)
    
def compute_hubert(data):
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
    wavfile.write("test.wav", 32000, tmp_audio32)
    tmp_audio32b = (final_data / tmp_max * (maxx * alpha * 1145.14)) + ((1 - alpha) * 1145.14) * final_data
    tmp_audio = librosa.resample(tmp_audio32b, orig_sr=32000, target_sr=16000)  # 不是重采样问题
    tensor_wav16 = torch.from_numpy(tmp_audio)
    tensor_wav16 = tensor_wav16.to(device)
    ssl = model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1, 2).cpu()  # torch.Size([1, 768, 215])
    
