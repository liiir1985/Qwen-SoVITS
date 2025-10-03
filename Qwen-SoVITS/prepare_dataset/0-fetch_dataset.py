import os
import json
from datasets import load_dataset
#from feature_extractor import cnhubert
from torchcodec.encoders import AudioEncoder
import torch
import numpy as np
import librosa
import argparse
from tqdm import tqdm

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

def crawl_dataset(dataset_dir, dataset_source:str, lang:str,target_duration, subset):
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)    
    database_path = f"{dataset_dir}/processed_datas.json"
    dataset_save_path = f"{dataset_dir}/{dataset_source}/{lang}/"
    os.makedirs(dataset_save_path, exist_ok=True)

    processed_ids = load_existing_ids(database_path)

    cur_list = processed_ids.get(dataset_source, list())
    if len(cur_list)>0:
        lang_list = cur_list.get(lang)
        if lang_list is None:            
            lang_list = list()
            cur_list[lang]=lang_list
        cur_list = lang_list
    else:
        cur_list = list()
        processed_ids[dataset_source] = {lang:cur_list}
    total_secs = 0
    cur_id_bucket={}
    for i in cur_list:
        total_secs+=i['duration']    
        cur_id_bucket[i['id']] = i
    
    if "Emilia" in dataset_source:
        crawl_emilia(dataset_source, cur_id_bucket,lang, dataset_save_path, target_duration, total_secs)
    elif dataset_source == "Galgame":
        crawl_galgame(subset, cur_id_bucket,lang, dataset_save_path, target_duration, total_secs)
    
    processed_ids[dataset_source][lang] = [i for i in cur_id_bucket.values()]
    save_ids(database_path, processed_ids)
def crawl_galgame(subset, cur_id_bucket:dict, lang, dataset_save_path,target_duration, total_secs):
    os.environ['HF_HOME'] = 'E:/hf_cache'
    dataset = load_dataset("joujiboi/Galgame-VisualNovel-Reupload", subset, split="train", streaming=True)
    dataset = dataset.shuffle(seed=11223, buffer_size=10000)
    with tqdm(total=target_duration, desc="Dataset crawling", unit="Secs") as t:
        t.update(total_secs)
        for sample in dataset:
            id = sample['audio_ID']
            text = sample['text']
            if id in cur_id_bucket:
                continue
            decoder = sample['audio']
            data = decoder.get_all_samples()
            duration = data.duration_seconds
            total_secs += duration
            t.update(duration)

            sound_file_path = f"{dataset_save_path}/{id}.flac"
            if not os.path.exists(sound_file_path):                
                encoder = AudioEncoder(samples=data.data, sample_rate=data.sample_rate)
                encoder.to_file(
                    dest=sound_file_path
                )            
            with open(f"{dataset_save_path}/{id}.txt", 'w', encoding='utf8') as f: f.write(text)
            cur_id_bucket[id] = {'id':id,'duration':duration}
            if total_secs > target_duration:
                break

def crawl_emilia(dataset_source, cur_id_bucket:dict, lang, dataset_save_path,target_duration, total_secs):
    os.environ['HF_HOME'] = 'E:/hf_cache'
    if lang == "en":
        path = f"{dataset_source}/EN/*.tar" # Same for Emilia-YODAS; just replace "Emilia/" with "Emilia-YODAS/"
        split_name = "en"
    elif lang == "zh":
        path = f"{dataset_source}/ZH/*.tar" # Same for Emilia-YODAS; just replace "Emilia/" with "Emilia-YODAS/"
        split_name = "zh"
    elif lang == "ja":
        path = f"{dataset_source}/JA/*.tar" # Same for Emilia-YODAS; just replace "Emilia/" with "Emilia-YODAS/"
        split_name = "ja"
    dataset = load_dataset("amphion/Emilia-Dataset",data_files={split_name: path}, split=split_name, streaming=True)    
    dataset = dataset.shuffle(seed=11223, buffer_size=10000)
    
    with tqdm(total=target_duration, desc="Dataset crawling", unit="Secs") as t:
        t.update(total_secs)
        for sample in dataset:
            meta = sample['json']
            id = meta.get('id', None)
            if id is None:
                id = meta["_id"]
            dnsmos = meta['dnsmos']
            text = meta['text']
            duration = meta['duration']
            if id in cur_id_bucket:
                continue
            
            if dnsmos < 3.1:
                continue
            
            total_secs += duration
            t.update(duration)

            sound_file_path = f"{dataset_save_path}/{id}.flac"
            if not os.path.exists(sound_file_path):
                decoder = sample['mp3']
                data = decoder.get_all_samples()

                encoder = AudioEncoder(samples=data.data, sample_rate=data.sample_rate)
                encoder.to_file(
                    dest=sound_file_path
                )            
            with open(f"{dataset_save_path}/{id}.txt", 'w', encoding='utf8') as f: f.write(text)
            cur_id_bucket[id] = {'id':id,'duration':duration}
            if total_secs > target_duration:
                break
    
    
def compute_hubert(data):    
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    #cnhubert.cnhubert_base_path = 'F:/myexe/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base'#os.environ.get("cnhubert_base_dir")
    #model = cnhubert.get_model()
    model = model.to(device)
    maxx = 0.95
    alpha = 0.5

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
    #wavfile.write("test.wav", 32000, tmp_audio32)
    tmp_audio32b = (final_data / tmp_max * (maxx * alpha * 1145.14)) + ((1 - alpha) * 1145.14) * final_data
    tmp_audio = librosa.resample(tmp_audio32b, orig_sr=32000, target_sr=16000)  # 不是重采样问题
    tensor_wav16 = torch.from_numpy(tmp_audio)
    tensor_wav16 = tensor_wav16.to(device)
    ssl = model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1, 2).cpu()  # torch.Size([1, 768, 215])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Dataset crawler for Qwen-Sovits"
    )
    parser.add_argument(
        "-o", 
        "--output_dir", 
        type=str, 
        default="./dataset", 
        help="Path to save the dataset"
    )
    parser.add_argument(
        "-s", 
        "--dataset_source", 
        type=str, 
        default="Emilia-YODAS", 
        help="Dataset source"
    )
    parser.add_argument(
        "-ss", 
        "--dataset_subset", 
        type=str, 
        default="0verflow_Shiny_Days", 
        help="Dataset sub set"
    )
    parser.add_argument(
        "-l", 
        "--lang", 
        type=str, 
        default="ja", 
        help="Dataset Language"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=120*60, 
        help="Dataset Language"
    )
    args = parser.parse_args()

    crawl_dataset(args.output_dir, args.dataset_source, args.lang, args.duration, args.dataset_subset)
