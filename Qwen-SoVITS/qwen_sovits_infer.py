from module.qwen_t2s import Qwen3Text2SemanticModel
from module.models import SynthesizerTrn
import utils
import torch
from feature_extractor import cnhubert
from utils import load_audio, get_audio_hubert
from sovits_infer import SovitsSemantic2AudioModel
import os
import soundfile as sf

t2s_model = Qwen3Text2SemanticModel("./pretrained_models/qwen3")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnhubert.cnhubert_base_path = "./pretrained_models/chinese-hubert-base"
hubert_model = cnhubert.get_model()
hubert_model = hubert_model.to(device)

s2a_model = SovitsSemantic2AudioModel("./pretrained_models/s2Gv2ProPlus.pth", hubert_model, t2s_model)

# pretrained_s2G = "./pretrained_models/s2Gv2ProPlus.pth"
# hps = utils.get_hparams_from_file("./configs/s2v2ProPlus.json")
# vq_model = SynthesizerTrn(
#     hps.data.filter_length // 2 + 1,
#     hps.train.segment_size // hps.data.hop_length,
#     n_speakers=hps.data.n_speakers,
#     version="v2ProPlus",
#     **hps.model,
# )
# vq_model = vq_model.to(device)
# vq_model.eval()
# print(
#     vq_model.load_state_dict(
#         torch.load(pretrained_s2G, map_location="cpu", weights_only=False)["weight"], strict=False
#     )
# )



audio_path="./dataset/01-00-a00-0090__0verflow_Shiny_Days.flac"

synthesis_result = s2a_model.get_tts_wav(audio_path, "はぁ……ご飯食べた？おかゆでも作ろうか？", "ja", "目の前がぐるぐるしてるぅ……あぅ？キッチン借りるね","ja")
print(synthesis_result)
result_list = list(synthesis_result)

if result_list:
    last_sampling_rate, last_audio_data = result_list[-1]
    output_wav_path = os.path.join(".", "output.wav")
    sf.write(output_wav_path, last_audio_data, last_sampling_rate)
    print(f"Audio saved to {output_wav_path}")
# samples = load_audio(audio_path)
# hubert = get_audio_hubert(hubert_model, samples)
# hubert = hubert.to(device)
# codes = vq_model.extract_latent(hubert)
# i16_codes = codes.cpu().squeeze(dim=1).to(torch.int64)

# t2s_model.infer("電話をいただきましょうか?電話をいただきましょうか?","[ja]私はとても感動しています私はとても感動しています",i16_codes)

