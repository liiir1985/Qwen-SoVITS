import os
import json
from datasets import load_dataset
#from feature_extractor import cnhubert
from torchcodec.encoders import AudioEncoder
from torchcodec.decoders import AudioDecoder
import torch
import numpy as np
import librosa
import argparse
from tqdm import tqdm
import glob
import re
import zipfile
import io

PROXY_ADDRESS = "http://127.0.0.1:10808" 

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

    os.environ['http_proxy'] = PROXY_ADDRESS
    os.environ['https_proxy'] = PROXY_ADDRESS

    if "Emilia" in dataset_source:
        crawl_emilia(dataset_source, cur_id_bucket,lang, dataset_save_path, target_duration, total_secs)
    elif dataset_source == "Galgame":
        crawl_galgame(subset, cur_id_bucket,lang, dataset_save_path, target_duration, total_secs)
    
    del os.environ['http_proxy']
    del os.environ['https_proxy']
    
    processed_ids[dataset_source][lang] = [i for i in cur_id_bucket.values()]
    save_ids(database_path, processed_ids)

def initialize_zip_count(base_fn):
    """
    在程序启动时调用，查找现有文件中最大的序号，并设置全局 zip_count。
    """    
    # 查找所有符合 BASE_FILENAME_XXX.zip 模式的文件
    search_pattern = f"{base_fn}_*.zip"
    existing_files = glob.glob(search_pattern)
    
    max_count = 0
    
    # 正则表达式用于提取文件名中的数字部分
    # 例如：从 "llm_streaming_dataset_015.zip" 中提取 "015"
    pattern = re.compile(f"{base_fn}_(\\d+)\\.zip")
    is_max_file_small = False
    
    for filename in existing_files:
        match = pattern.search(filename.replace("\\","/"))
        if match:
            # 将匹配到的数字字符串转换为整数
            current_count = int(match.group(1))
            if current_count > max_count:
                max_count = current_count
                is_max_file_small = os.path.exists(filename) and os.path.getsize(filename) < MAX_SIZE_BYTES
                
    # 从找到的最大序号基础上开始计数
    zip_count = max_count
    if is_max_file_small:
        zip_count = zip_count -1
    return zip_count

MAX_SIZE_MB = 100
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

def open_new_zip_file(zip_file, current_zip_path, zip_count, base_fn):
    """关闭旧文件（如果有）并打开一个新的 ZIP 文件"""
    
    if zip_file:
        zip_file.close()
    # 使用 :03d 确保数字部分是三位数 (例如 001, 010, 100)
    base_name = f"{base_fn}_{zip_count:03d}"
    current_zip_path = f"{base_name}.zip"
    current_txt_path = f"{base_name}.txt"
    zip_file = zipfile.ZipFile(current_zip_path, 'a', zipfile.ZIP_DEFLATED)
    return zip_file, current_zip_path, current_txt_path


def pack_dataset(dataset_dir, dataset_source:str, lang:str, subset):
    database_path = f"{dataset_dir}/processed_datas.json"
    dataset_save_path = f"{dataset_dir}/{dataset_source}/{lang}/"
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
    
    base_fn = f"{dataset_save_path}{lang}"
    zip_cnt = initialize_zip_count(base_fn)
    zip_cnt+=1
    zip_file, current_zip_path, current_txt_path = open_new_zip_file(None, None, zip_cnt, base_fn)
    txt_file = open(current_txt_path ,'w', encoding='utf-8')
    for i in tqdm(cur_list, desc="Packing dataset"):
        id = i['id']
        old_txt_path = f"{dataset_save_path}{id}.txt"
        old_audio_path = f"{dataset_save_path}{id}.flac"
        if os.path.exists(old_audio_path):
            i['zip_file'] = current_txt_path
            with open(old_txt_path, 'r', encoding='utf-8') as f:
                txt_file.write(f"{id}\t{f.read().replace("\n", "\\n")}\n")
            zip_file.write(old_audio_path, f"{id}.flac")
            os.remove(old_audio_path)
            os.remove(old_txt_path)

        if os.path.exists(current_zip_path) and os.path.getsize(current_zip_path) > MAX_SIZE_BYTES:
            zip_cnt+=1
            zip_file, current_zip_path, current_txt_path = open_new_zip_file(zip_file, current_zip_path, zip_cnt, base_fn)
            if txt_file is not None:
                txt_file.close()
            txt_file = open(current_txt_path ,'w', encoding='utf-8')
    if zip_file is not None:
        zip_file.close()
    if txt_file is not None:
        txt_file.close()  
    save_ids(database_path, processed_ids)

def repack_genshin(dataset_dir, chara:str):    
    dataset_save_path = f"{dataset_dir}/{chara}/"
    base_fn = f"{dataset_save_path}{chara}"
    zip_cnt = initialize_zip_count(base_fn)
    zip_cnt+=1
    zip_file, current_zip_path, current_txt_path = open_new_zip_file(None, None, zip_cnt, base_fn)
    txt_file = open(current_txt_path ,'a', encoding='utf-8')
    files = glob.glob(f"{dataset_save_path}*.lab")
    for i in tqdm(files, desc="Packing dataset"):
        base_name, ext = os.path.splitext(i)
        id = os.path.basename(base_name)
        old_audio_path = f"{dataset_save_path}{id}.wav"
        if os.path.exists(old_audio_path):
            decoder = AudioDecoder(old_audio_path)

            with open(i, 'r', encoding='utf-8') as f:
                txt_file.write(f"{id}\t{f.read().replace("\n", "\\n")}\n")
            data = decoder.get_all_samples()
            buffer = io.BytesIO()
            encoder = AudioEncoder(samples=data.data, sample_rate=data.sample_rate)
            encoder.to_file_like(buffer, "flac")
            buffer.seek(0)
            zip_file.writestr(f"{id}.flac", buffer.read())   
            buffer.close()
            del decoder
            del encoder
            os.remove(old_audio_path)
            os.remove(i)

        if os.path.exists(current_zip_path) and os.path.getsize(current_zip_path) > MAX_SIZE_BYTES:
            zip_cnt+=1
            zip_file, current_zip_path, current_txt_path = open_new_zip_file(zip_file, current_zip_path, zip_cnt, base_fn)
            if txt_file is not None:
                txt_file.close()
            txt_file = open(current_txt_path ,'w', encoding='utf-8')
    if zip_file is not None:
        zip_file.close()
    if txt_file is not None:
        txt_file.close()  

def crawl_galgame(subset, cur_id_bucket:dict, lang, dataset_save_path,target_duration, total_secs):
    os.environ['HF_HOME'] = 'E:/hf_cache'
    dataset = load_dataset("joujiboi/Galgame-VisualNovel-Reupload", subset, split="train", streaming=True)
    #dataset = dataset.shuffle(seed=11223, buffer_size=10000)
    base_fn = f"{dataset_save_path}{subset}"
    zip_cnt = initialize_zip_count(base_fn)
    zip_cnt+=1
    zip_file, current_zip_path, current_txt_path = open_new_zip_file(None, None, zip_cnt, base_fn)
    txt_file = open(current_txt_path ,'a', encoding='utf-8')
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

            buffer = io.BytesIO()

            sound_file_path = f"{id}.flac"
            decoder = sample['audio']
            data = decoder.get_all_samples()

            encoder = AudioEncoder(samples=data.data, sample_rate=data.sample_rate)
            encoder.to_file_like(buffer, "flac")
            buffer.seek(0)
            txt_file.write(f"{id}\t{text.replace("\n", "\\n")}\n")
            zip_file.writestr(sound_file_path, buffer.read())   

            cur_id_bucket[id] = {'id':id,'duration':duration, 'subset': subset, 'zip_file': os.path.relpath(current_txt_path, start=dataset_save_path)}
            if os.path.exists(current_zip_path) and os.path.getsize(current_zip_path) > MAX_SIZE_BYTES:
                zip_cnt+=1
                zip_file, current_zip_path, current_txt_path = open_new_zip_file(zip_file, current_zip_path, zip_cnt, base_fn)
                if txt_file is not None:
                    txt_file.close()
                txt_file = open(current_txt_path ,'a', encoding='utf-8')

            #with open(f"{dataset_save_path}/{id}.txt", 'w', encoding='utf8') as f: f.write(text)
            if total_secs > target_duration:
                break
    if zip_file is not None:
        zip_file.close()
    if txt_file is not None:
        txt_file.close()  

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
    #dataset = dataset.shuffle(seed=11223, buffer_size=10000)
    base_fn = f"{dataset_save_path}{lang}"
    zip_cnt = initialize_zip_count(base_fn)
    zip_cnt+=1
    zip_file, current_zip_path, current_txt_path = open_new_zip_file(None, None, zip_cnt, base_fn)
    txt_file = open(current_txt_path ,'a', encoding='utf-8')

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
            buffer = io.BytesIO()

            sound_file_path = f"{id}.flac"
            decoder = sample['mp3']
            data = decoder.get_all_samples()

            encoder = AudioEncoder(samples=data.data, sample_rate=data.sample_rate)
            encoder.to_file_like(buffer, "flac")
            buffer.seek(0)
            txt_file.write(f"{id}\t{text.replace("\n", "\\n")}\n")
            zip_file.writestr(sound_file_path, buffer.read())   
            
            cur_id_bucket[id] = {'id':id,'duration':duration, 'zip_file': os.path.relpath(current_txt_path, start=dataset_save_path)}
            if os.path.exists(current_zip_path) and os.path.getsize(current_zip_path) > MAX_SIZE_BYTES:
                zip_cnt+=1
                zip_file, current_zip_path, current_txt_path = open_new_zip_file(zip_file, current_zip_path, zip_cnt, base_fn)
                if txt_file is not None:
                    txt_file.close()
                txt_file = open(current_txt_path ,'a', encoding='utf-8')
            #with open(f"{dataset_save_path}/{id}.txt", 'w', encoding='utf8') as f: f.write(text)
            if total_secs > target_duration:
                break
    if zip_file is not None:
        zip_file.close()
    if txt_file is not None:
        txt_file.close()  
    
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
        default="Z:/sata11-18612520532/AI/TTS/dataset", 
        help="Path to save the dataset"
    )
    parser.add_argument(
        "-s", 
        "--dataset_source", 
        type=str, 
        default="Galgame", 
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
        default=60*60*60, 
        help="Dataset Language"
    )
    parser.add_argument(
        '--repack',
        type=int,
        default=2,
        help='Repack the dataset to use zip file to store the files'
    )
    args = parser.parse_args()
    if args.repack == 1:
        pack_dataset(args.output_dir, args.dataset_source, args.lang, args.dataset_subset)
    elif args.repack == 2:
        repack_genshin(args.output_dir, args.dataset_subset)
    else:
        crawl_dataset(args.output_dir, args.dataset_source, args.lang, args.duration, args.dataset_subset)
