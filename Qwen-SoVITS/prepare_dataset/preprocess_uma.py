import argparse
import os
from tqdm import tqdm
import glob
import re
import io
import json
import zipfile

from torchcodec.encoders import AudioEncoder
from torchcodec.decoders import AudioDecoder
from fetch_dataset import load_existing_ids, save_ids,initialize_zip_count,open_new_zip_file, MAX_SIZE_BYTES

def preprocess_uma(dataset_dir, target_duration):
    dataset_source = "Umamusume"
    lang = "ja"
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
    base_fn = f"{dataset_save_path}{dataset_source}"
    zip_cnt = initialize_zip_count(base_fn)
    zip_cnt+=1
    zip_file, current_zip_path, current_txt_path = open_new_zip_file(None, None, zip_cnt, base_fn)
    txt_file = open(current_txt_path ,'a', encoding='utf-8')
    zip = zipfile.ZipFile(f"dataset/Umamusume/umavoice.zip", 'r')
    with zip.open("output.txt") as z_file:
        index_file = io.TextIOWrapper(z_file, encoding="utf-8")
        for i in tqdm(index_file, desc="Packing dataset"):
            arr = i.split("|")
            base_name, ext = os.path.splitext(arr[0])
            id = f"{os.path.basename(os.path.dirname(base_name))}_{os.path.basename(base_name)}"
            audio_bytes = zip.read(arr[0])
            audio_buffer = io.BytesIO(audio_bytes)
            decoder = AudioDecoder(audio_buffer)

            txt_file.write(f"{id}\t{arr[2]}\n")
            data = decoder.get_all_samples()
            buffer = io.BytesIO()
            encoder = AudioEncoder(samples=data.data, sample_rate=data.sample_rate)
            encoder.to_file_like(buffer, "flac")
            buffer.seek(0)
            zip_file.writestr(f"{id}.flac", buffer.read())   
            buffer.close()
            del decoder
            del encoder
            audio_buffer.close()            
            #os.remove(old_audio_path)
            #os.remove(i)

            if os.path.exists(current_zip_path) and os.path.getsize(current_zip_path) > MAX_SIZE_BYTES:
                zip_cnt+=1
                zip_file, current_zip_path, current_txt_path = open_new_zip_file(zip_file, current_zip_path, zip_cnt, base_fn)
                if txt_file is not None:
                    txt_file.close()
                txt_file = open(current_txt_path ,'w', encoding='utf-8')
    zip.close()
    if zip_file is not None:
        zip_file.close()
    if txt_file is not None:
        txt_file.close()  

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
        "--duration", 
        type=int, 
        default=10*60*60, 
        help="Dataset Language"
    )
    args = parser.parse_args()

    preprocess_uma(args.output_dir, args.duration)