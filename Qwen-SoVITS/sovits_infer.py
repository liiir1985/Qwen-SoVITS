from process_sovits_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from module.models import Generator, SynthesizerTrn
import json
import torch
import re
from time import time as ttime
import numpy as np
import librosa
import torchaudio
import traceback
import tools.audio_sr
from text.cleaner import clean_text, cleaned_text_to_sequence

from text.LangSegmenter import LangSegmenter
from module.qwen_t2s import Qwen3Text2SemanticModel
from module.mel_processing import spectrogram_torch
from sv import SV

is_half = False
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dtype = torch.float16 if is_half == True else torch.float32

def i18n(str):
    return str

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}
punctuation = set(["!", "?", "…", ",", ".", "-", " "])

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut4(inp):
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)

def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError(i18n("请输入有效文本"))
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text

def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_phones_and_bert(text, language, version, final=False):
    text = re.sub(r' {2,}', ' ', text)
    textlist = []
    langlist = []
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text,"ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text,"ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                langlist.append(language)
            textlist.append(tmp["text"])
    print(textlist)
    print(langlist)
    phones_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    return phones, norm_text

def audio_sr(audio, sr):
    global sr_model
    if sr_model == None:
        from tools.audio_sr import AP_BWE

        try:
            sr_model = AP_BWE(device, DictToAttrRecursive)
        except FileNotFoundError:
            print(i18n("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载好"))
            return audio.cpu().detach().numpy(), sr
    return sr_model(audio, sr)

class SovitsSemantic2AudioModel:
    vq_model:SynthesizerTrn
    hps:any
    version:any
    model_version:any
    if_lora_v3:any
    ssl_model:any
    t2s_model:Qwen3Text2SemanticModel
    sv_cn_model:SV
    resample_transform_dict:dict

    def __init__(self, sovits_path, ssl_model, t2s_model:Qwen3Text2SemanticModel):
        self.ssl_model = ssl_model
        self.t2s_model = t2s_model
        self.change_sovits_weights(sovits_path)
        self.sv_cn_model = SV(device, is_half)
        self.cache = {}
        self.resample_transform_dict={}
        
    def resample(self, audio_tensor, sr0, sr1, device):
        key = "%s-%s-%s" % (sr0, sr1, str(device))
        if key not in self.resample_transform_dict:
            self.resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
        return self.resample_transform_dict[key](audio_tensor)
        
    def get_spepc(self, hps, filename, dtype, device, is_v2pro=False):
        # audio = load_audio(filename, int(hps.data.sampling_rate))

        # audio, sampling_rate = librosa.load(filename, sr=int(hps.data.sampling_rate))
        # audio = torch.FloatTensor(audio)

        sr1 = int(hps.data.sampling_rate)
        audio, sr0 = torchaudio.load(filename)
        if sr0 != sr1:
            audio = audio.to(device)
            if audio.shape[0] == 2:
                audio = audio.mean(0).unsqueeze(0)
            audio = self.resample(audio, sr0, sr1, device)
        else:
            audio = audio.to(device)
            if audio.shape[0] == 2:
                audio = audio.mean(0).unsqueeze(0)

        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
        spec = spectrogram_torch(
            audio,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = spec.to(dtype)
        if is_v2pro == True:
            audio = self.resample(audio, sr1, 16000, device).to(dtype)
        return spec, audio
        
    def change_sovits_weights(self, sovits_path):
        self.version, self.model_version, self.if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
        print(sovits_path, self.version, self.model_version, self.if_lora_v3)

        dict_s2 = load_sovits_new(sovits_path)
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
            hps.model.version = "v2"  # v3model,v2sybomls
        elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            hps.model.version = "v1"
        else:
            hps.model.version = "v2ProPlus"
        self.hps = hps
        self.version = hps.model.version
        self.vq_model = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **hps.model,
            )
        if "pretrained" not in sovits_path:
            try:
                del self.vq_model.enc_q
            except:
                pass
        if is_half == True:
            self.vq_model = self.vq_model.half().to(device)
        else:
            self.vq_model = self.vq_model.to(device)
        self.vq_model.eval()
        
        print("loading sovits_%s" % self.model_version, self.vq_model.load_state_dict(dict_s2["weight"], strict=False))
        
        # with open("./weight.json") as f:
        #     data = f.read()
        #     data = json.loads(data)
        #     data["SoVITS"][self.version] = sovits_path
        # with open("./weight.json", "w") as f:
        #     f.write(json.dumps(data))
            
    def get_tts_wav(self,
        ref_wav_path,
        prompt_text,
        prompt_language,
        text,
        text_language,
        how_to_cut=i18n("不切"),
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        ref_free=False,
        speed=1,
        if_freeze=False,
        inp_refs=None,
        sample_steps=8,
        if_sr=False,
        pause_second=0.3,
    ):
        model_version = "v2ProPlus"
        t = []
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
        if_sr = False
        t0 = ttime()
        
        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in splits:
                prompt_text += "。" if prompt_language != "en" else "."
            print(i18n("实际输入的参考文本:"), prompt_text)
        text = text.strip("\n")
        # if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

        print(i18n("实际输入的目标文本:"), text)
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * pause_second),
            dtype=np.float16 if is_half == True else np.float32,
        )
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            zero_wav_torch = zero_wav_torch.to(device)
        if not ref_free:
            with torch.no_grad():
                wav16k, sr = librosa.load(ref_wav_path, sr=16000)
                if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                    raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
                wav16k = torch.from_numpy(wav16k)
                if is_half == True:
                    wav16k = wav16k.half().to(device)
                else:
                    wav16k = wav16k.to(device)
                wav16k = torch.cat([wav16k, zero_wav_torch])
                ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
                codes = self.vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                prompt = prompt_semantic.unsqueeze(0).to("cpu")

        t1 = ttime()
        t.append(t1 - t0)

        if how_to_cut == i18n("凑四句一切"):
            text = cut1(text)
        elif how_to_cut == i18n("凑50字一切"):
            text = cut2(text)
        elif how_to_cut == i18n("按中文句号。切"):
            text = cut3(text)
        elif how_to_cut == i18n("按英文句号.切"):
            text = cut4(text)
        elif how_to_cut == i18n("按标点符号切"):
            text = cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        print(i18n("实际输入的目标文本(切句后):"), text)
        texts = text.split("\n")
        texts = process_text(texts)
        texts = merge_short_text_in_array(texts, 5)
        audio_opt = []

        for i_text, text in enumerate(texts):
            # 解决输入目标文本的空行导致报错的问题
            if len(text.strip()) == 0:
                continue
            if text[-1] not in splits:
                text += "。" if text_language != "en" else "."
            print(i18n("实际输入的目标文本(每句):"), text)
            phones2, norm_text2 = get_phones_and_bert(text, text_language, self.version)
            print(i18n("前端处理后的文本(每句):"), norm_text2)
            t2 = ttime()
            # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
            # print(cache.keys(),if_freeze)
            if i_text in self.cache and if_freeze == True:
                pred_semantic = self.cache[i_text]
            else:
                pred_semantic = self.t2s_model.infer(text, prompt_text, prompt).unsqueeze(0).unsqueeze(0)
                self.cache[i_text] = pred_semantic
                
            t3 = ttime()
            is_v2pro = model_version in {"v2Pro", "v2ProPlus"}
            # print(23333,is_v2pro,model_version)
            ###v3不存在以下逻辑和inp_refs
            refers = []
            if is_v2pro:
                sv_emb = []
            if inp_refs:
                for path in inp_refs:
                    try:  #####这里加上提取sv的逻辑，要么一堆sv一堆refer，要么单个sv单个refer
                        refer, audio_tensor = self.get_spepc(self.hps, path.name, dtype, device, is_v2pro)
                        refers.append(refer)
                        if is_v2pro:
                            sv_emb.append(self.sv_cn_model.compute_embedding3(audio_tensor))
                    except:
                        traceback.print_exc()
            if len(refers) == 0:
                refers, audio_tensor = self.get_spepc(self.hps, ref_wav_path, dtype, device, is_v2pro)
                refers = [refers]
                if is_v2pro:
                    sv_emb = [self.sv_cn_model.compute_embedding3(audio_tensor)]
            audio = self.vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed, sv_emb=sv_emb
            )[0][0]
            max_audio = torch.abs(audio).max()  # 简单防止16bit爆音
            if max_audio > 1:
                audio = audio / max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav_torch)  # zero_wav
            t4 = ttime()
            t.extend([t2 - t1, t3 - t2, t4 - t3])
            t1 = ttime()
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))
        audio_opt = torch.cat(audio_opt, 0)  # np.concatenate
        if model_version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
            opt_sr = 32000
        elif model_version == "v3":
            opt_sr = 24000
        else:
            opt_sr = 48000  # v4
        if if_sr == True and opt_sr == 24000:
            print(i18n("音频超分中"))
            audio_opt, opt_sr = audio_sr(audio_opt.unsqueeze(0), opt_sr)
            max_audio = np.abs(audio_opt).max()
            if max_audio > 1:
                audio_opt /= max_audio
        else:
            audio_opt = audio_opt.cpu().detach().numpy()
        yield opt_sr, (audio_opt * 32767).astype(np.int16)