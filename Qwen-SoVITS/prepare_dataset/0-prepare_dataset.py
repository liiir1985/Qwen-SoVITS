import argparse
import glob
import os
from feature_extractor import cnhubert
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm
import torchaudio
import numpy as np
import librosa
import torch
import utils
from utils import load_audio, get_audio_hubert
from module.models import SynthesizerTrn
import base64
import zipfile
import io
import shutil

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def prepare(output_dir, src_dir, dataset, lang, model_dir, sr=32000):
    cnhubert.cnhubert_base_path = model_dir
    model = cnhubert.get_model()
    model = model.to(device)
    txt_folder = f"{output_dir}/1-txts"
    os.makedirs(txt_folder, exist_ok=True)
    hubert_folder = f"{output_dir}/2-huberts"
    os.makedirs(hubert_folder, exist_ok=True)
    files = glob.glob(f"{src_dir}/{dataset}/{lang}/*.txt")

    for i in tqdm(files, desc="Processing audios"):
        base_name, ext = os.path.splitext(i)
        fn = os.path.basename(base_name)

        dst_txt_path = f"{txt_folder}/{dataset}_{lang}_{fn}.txt"
        if os.path.exists(dst_txt_path):
            continue
        src_txt_file = open(i, 'r', encoding='utf-8')
        src_zip_file = zipfile.ZipFile(f"{base_name}.zip", 'r')
        dst_zip_file = zipfile.ZipFile(f"{hubert_folder}/{dataset}_{lang}_{fn}.zip", 'w', zipfile.ZIP_DEFLATED)
        dst_txt_file = open(dst_txt_path, 'w', encoding='utf-8')
        namelist = src_zip_file.namelist()
        for line in src_txt_file:
            arr = line.split('\t')
            if f"{arr[0]}.flac" not in namelist:
                continue
            audio_bytes = src_zip_file.read(f"{arr[0]}.flac")
            audio_buffer = io.BytesIO(audio_bytes)
            hubert_path = f"{arr[0]}.pth"            
            
            try:
                final_data = load_audio(audio_buffer, sr)
                tmp_max = np.abs(final_data).max()
                if tmp_max > 2.2:
                    print("%s-filtered,%s" % (i, tmp_max))
                    continue
                ssl = get_audio_hubert(model, final_data, sr)
                if np.isnan(ssl.detach().numpy()).sum() != 0:
                    print("nan filtered:%s" % i)
                    continue
                buffer = io.BytesIO()
                torch.save(ssl, buffer)
                buffer.seek(0)

                dst_zip_file.writestr(hubert_path, buffer.read())

                dest_txt_path = f"{txt_folder}/{fn}.txt"
                dst_txt_file.write(f"{arr[0]}\t{lang}\t{arr[1]}")
            except Exception as ex:
                print(f"Error while processing {base_name}.flac\n{ex}")
            audio_buffer.close()
        
        src_txt_file.close()
        src_zip_file.close()
        dst_txt_file.close()
        dst_zip_file.close()
def process_semantic(output_dir, pretrained_s2G = "./pretrained_models/s2Gv2ProPlus.pth"):
    hps = utils.get_hparams_from_file("./configs/s2v2ProPlus.json")
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        version="v2ProPlus",
        **hps.model,
    )
    vq_model = vq_model.to(device)
    vq_model.eval()
    print(
        vq_model.load_state_dict(
            torch.load(pretrained_s2G, map_location="cpu", weights_only=False)["weight"], strict=False
        )
    )
    txt_folder = f"{output_dir}/1-txts"
    hubert_folder = f"{output_dir}/2-huberts"
    semantic_folder = f"{output_dir}/3-semantic_pairs"
    os.makedirs(semantic_folder, exist_ok=True)
    files = glob.glob(f"{txt_folder}/*.txt")
    max_code = 0
    min_code = 10000
    for i in tqdm(files, desc="Processing audios"):
        base_name, ext = os.path.splitext(i)
        fn = os.path.basename(base_name)

        dst_txt_path = f"{semantic_folder}/{fn}.txt"
        if os.path.exists(dst_txt_path):
            continue

        src_txt_file = open(i, 'r', encoding='utf-8')
        src_zip_file = zipfile.ZipFile(f"{hubert_folder}/{fn}.zip", 'r')
        with open(dst_txt_path, 'w', encoding='utf-8') as fw:        
            for line in src_txt_file:
                arr = line.split('\t')
                if arr[2][-1] == "\n":
                    arr[2] = arr[2][:-1]
                hubert_file = src_zip_file.open(f"{arr[0]}.pth")
                ssl_content = torch.load(hubert_file, map_location="cpu")
                ssl_content = ssl_content.to(device)
                hubert_file.close()
                codes = vq_model.extract_latent(ssl_content)
                cmax = codes.max()
                cmin = codes.min()
                if cmax > max_code:
                    max_code = cmax
                if cmin < min_code:
                    min_code = cmin
                i16_codes = codes.cpu().to(torch.int16).numpy()
                base64_str = base64.b64encode(i16_codes.tobytes()).decode('utf-8')
                fw.write(f"{arr[2]}\t{arr[1]}\t{base64_str}\n")

        src_txt_file.close()
        src_zip_file.close()
    print(f"min code: {min_code}, max code:{max_code}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Dataset preprocessor for Qwen-Sovits"
    )
    parser.add_argument(
        "-o", 
        "--output_dir", 
        type=str, 
        default="Y:/AI/TTS/", 
        help="Path to save the dataset"
    )
    parser.add_argument(
        "-src", 
        "--source_dir", 
        type=str, 
        default="Y:/AI/TTS/dataset", 
        help="Dataset source"
    )
    parser.add_argument(
        "-l", 
        "--lang", 
        type=str, 
        default="ja", 
        help="Dataset Language"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="Genshin", 
        help="Dataset Source"
    )
    parser.add_argument(
        "-s", 
        "--step", 
        type=int, 
        default=1, 
        help="Process step"
    )
    parser.add_argument(
        "-m", 
        "--pretrained_model", 
        type=str, 
        default="./pretrained_models/chinese-hubert-base", 
        help="Path for pretrained model"
    )
    args = parser.parse_args()
    if args.step == 0:
        prepare(args.output_dir, args.source_dir, args.dataset, args.lang, args.pretrained_model)
    elif args.step == 1:
        process_semantic(args.output_dir)
