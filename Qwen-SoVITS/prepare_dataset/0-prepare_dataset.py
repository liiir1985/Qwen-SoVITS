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

def prepare(output_dir, src_dir, lang, model_dir, sr=32000):
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    cnhubert.cnhubert_base_path = model_dir
    model = cnhubert.get_model()
    model = model.to(device)
    maxx = 0.95
    alpha = 0.5
    txt_folder = f"{output_dir}/1-txts"
    os.makedirs(txt_folder, exist_ok=True)
    hubert_folder = f"{output_dir}/2-huberts"
    os.makedirs(hubert_folder, exist_ok=True)
    files = glob.glob(f"{src_dir}/{lang}/*.txt")

    for i in tqdm(files, desc="Processing audios"):
        base_name, ext = os.path.splitext(i)
        fn = os.path.basename(base_name)
        
        audio_path = f"{base_name}.flac"
        decoder = AudioDecoder(audio_path)
        data = decoder.get_all_samples()
        if data.sample_rate != sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=data.sample_rate,
                new_freq=sr)
            audio_data = resampler(data.data)
        else:
            audio_data = data.data
        
        final_data = audio_data.flatten().numpy()
        tmp_max = np.abs(final_data).max()
        if tmp_max > 2.2:
            print("%s-filtered,%s" % (i, tmp_max))
            continue
        #tmp_audio32 = (final_data / tmp_max * (maxx * alpha * 32768)) + ((1 - alpha) * 32768) * final_data
        #wavfile.write("test.wav", 32000, tmp_audio32)
        tmp_audio32b = (final_data / tmp_max * (maxx * alpha * 1145.14)) + ((1 - alpha) * 1145.14) * final_data
        tmp_audio = librosa.resample(tmp_audio32b, orig_sr=sr, target_sr=16000)  # 不是重采样问题
        tensor_wav16 = torch.from_numpy(tmp_audio)
        tensor_wav16 = tensor_wav16.to(device)
        ssl = model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1, 2).cpu() 
        if np.isnan(ssl.detach().numpy()).sum() != 0:
            print("nan filtered:%s" % i)
            continue

        dest_txt_path = f"{txt_folder}/{fn}.txt"
        with open(i, 'r', encoding='utf8') as f:
            txt = f.read()
            with open(dest_txt_path, 'w', encoding='utf8') as f2:
                f2.write(f"[{lang}]{txt}")
        
        torch.save(ssl, f"{hubert_folder}/{fn}.pth")

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
        "-m", 
        "--pretrained_model", 
        type=str, 
        default="./pretrained_models/chinese-hubert-base", 
        help="Dataset source"
    )
    args = parser.parse_args()

    prepare(args.output_dir, args.source_dir, args.lang, args.pretrained_model)
