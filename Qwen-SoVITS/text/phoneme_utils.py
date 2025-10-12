from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from text.LangSegmenter import LangSegmenter
import re
import os
import json
from text.cleaner import clean_text, cleaned_text_to_sequence
from text import symbols_v2

def make_phoneme_tokenizer(output_path):
    _symbol_to_id_v2 = {s: i for i, s in enumerate(symbols_v2.symbols_full)}
    vocab_file = f"{output_path}/phoneme_vocab.json"
    with open(vocab_file,'w',encoding='utf-8') as f:
        f.write(json.dumps(_symbol_to_id_v2, ensure_ascii=False, indent=2))
    # 1. 初始化模型：基于 WordLevel 模型，传入您的词表文件路径
    #    WordLevel 模型直接将词表文件中的符号映射为 ID
    tokenizer = Tokenizer(WordLevel.from_file(vocab_file,unk_token="UNK"))
    # 2. 设置分词预处理器 (Pre-tokenizer)
    #    音素通常是空格分隔的，所以使用 Whitespace 分隔
    tokenizer.pre_tokenizer = Whitespace()
    # 4. 设置其他特殊标记
    tokenizer.enable_padding(direction="right", pad_id=tokenizer.token_to_id("_"), pad_token="_")

    # 保存分词器
    tokenizer.save(f"{output_path}/phoneme_tokenizer.json") 


def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_phones(text, language, version, final=False):
    text = re.sub(r' {2,}', ' ', text).replace("．",".")#处理全角点
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
    phones_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        l = lang.replace("all_", "")
        phones, word2ph, norm_text = clean_text(text, l, version)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones("." + text, language, version, final=True)

    return " ".join(phones), norm_text

if __name__ == "__main__":
    path = "./pretrained_models/phoneme_tokenizer"
    os.makedirs(path, exist_ok=True)
    #make_phoneme_tokenizer(path)
    phone, txt = get_phones("……うん", "ja", "v2")
    tokenizer:Tokenizer = Tokenizer.from_file(f"{path}/phoneme_tokenizer.json")
    tokens = tokenizer.encode(phone)

    print(tokenizer.decode(tokens.ids))
    
