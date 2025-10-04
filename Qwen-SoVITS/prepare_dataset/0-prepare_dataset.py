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
        hubert_path = f"{hubert_folder}/{fn}.pth"
        
        if not os.path.exists(hubert_path):
            try:
                audio_path = f"{base_name}.flac"        
                final_data = load_audio(audio_path, sr)
                tmp_max = np.abs(final_data).max()
                if tmp_max > 2.2:
                    print("%s-filtered,%s" % (i, tmp_max))
                    continue
                ssl = get_audio_hubert(model, final_data, sr)
                if np.isnan(ssl.detach().numpy()).sum() != 0:
                    print("nan filtered:%s" % i)
                    continue        
                torch.save(ssl, hubert_path)

                dest_txt_path = f"{txt_folder}/{fn}.txt"
                with open(i, 'r', encoding='utf8') as f:
                    txt = f.read()
                    with open(dest_txt_path, 'w', encoding='utf8') as f2:
                        f2.write(f"[{lang}]{txt}")
            except:
                print(f"Error while processing {base_name}.flac")
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
    files = glob.glob(f"{txt_folder}/*.txt")
    max_code = 0
    min_code = 10000
    with open(f"{output_dir}/semantic_pairs.txt", 'w', encoding='utf-8') as fw:
        for i in tqdm(files, desc="Processing audios"):
            base_name, ext = os.path.splitext(i)
            fn = os.path.basename(base_name)

            ssl_content = torch.load(f"{hubert_folder}/{fn}.pth", map_location="cpu")
            ssl_content = ssl_content.to(device)
            codes = vq_model.extract_latent(ssl_content)
            cmax = codes.max()
            cmin = codes.min()
            if cmax > max_code:
                max_code = cmax
            if cmin < min_code:
                min_code = cmin
            i16_codes = codes.cpu().to(torch.int16).numpy()
            base64_str = base64.b64encode(i16_codes.tobytes()).decode('utf-8')
            with open(i,'r', encoding='utf8') as f:
                txt = f.read()
            fw.write("%s\t%s\n" % (txt, base64_str))
    print(f"min code: {min_code}, max code:{max_code}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Dataset preprocessor for Qwen-Sovits"
    )
    parser.add_argument(
        "-o", 
        "--output_dir", 
        type=str, 
        default="./logs", 
        help="Path to save the dataset"
    )
    parser.add_argument(
        "-src", 
        "--source_dir", 
        type=str, 
        default="./dataset", 
        help="Dataset source"
    )
    parser.add_argument(
        "-l", 
        "--lang", 
        type=str, 
        default="en", 
        help="Dataset Language"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="Emilia", 
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
