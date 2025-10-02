from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments, Trainer
import torch
from data.qwen_t2s_dataset import Qwen3Text2SemanticDataset
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def start_train(output_dir, model_path, batch_size, epoch, save_epoch, max_ckpt):
    print(f"Loading model on device: {device}")

    # 2. 加载 Tokenizer (分词器)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True # 尽管是从本地加载，但 Qwen 模型建议保留此参数
    )

    dataset = Qwen3Text2SemanticDataset("./logs/semantic_pairs.txt", tokenizer)

    # 3. 加载 Model (模型权重)
    # 将本地路径作为第一个参数传入 from_pretrained()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype='auto',
        device_map="auto", # 自动分配模型到可用的 GPU/CPU
        trust_remote_code=True 
    )

    # 确保模型被加载到正确的设备上（虽然 device_map="auto" 会处理，但这是好习惯）
    model.to(device)

    print(f"✅ Qwen3 0.6B 模型已从本地路径 {model_path} 成功加载。")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epoch,
        per_device_train_batch_size=batch_size,    # use batch size 2 per GPU
        gradient_accumulation_steps=1,    # no grad accumulation (since batch 2 is fine)
        logging_steps=20,                 # log every 20 steps
        save_strategy="epoch", 
        save_steps=save_epoch,                     # no checkpoints (not needed for demo)
        save_total_limit=max_ckpt, 
        report_to=[],                     # no W&B or HF logging
        bf16=True,                        # Qwen3 is using bf16 training
        disable_tqdm=False,               # ← re-enable tqdm bars
        remove_unused_columns=False,      # <— keep extra columns like prompt_len
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=dataset.collate,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=True)

class Qwen3Text2SemanticModel:
    tokenizer:any
    model:any
    def __init__(self, model_path):
        print(f"Loading model on device: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='auto',
            device_map="auto",
            trust_remote_code=True 
        )

        self.model.to(device)
    
    def infer(self, prompt:str, ref_txt:str, ref_semantic:torch.Tensor):
        text = f"语音转文本任务：{{{ref_txt}.{prompt}}}"
        txt_ids = self.tokenizer([text], return_tensors="pt").to('cpu')
        # input_ids = txt_ids.data['input_ids']
        # attention_mask = txt_ids.data['attention_mask']
        # attention_mask_ref = (ref_semantic != self.tokenizer.pad_token_id).long()
        # input_ids = torch.cat([input_ids,ref_semantic], dim=1)
        # attention_mask = torch.cat([attention_mask, attention_mask_ref], dim=1)
        # txt_ids.data['input_ids'] = input_ids
        # txt_ids.data['attention_mask'] = attention_mask
        txt_ids = txt_ids.to(device)
        generated_ids = self.model.generate(
            **txt_ids, 
            max_new_tokens=256
        )

        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\n--- Model Response ---")
        print(response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="T2S model trainer for Qwen-Sovits"
    )
    parser.add_argument(
        "-o", 
        "--output_dir", 
        type=str, 
        default="./logs/s1", 
        help="Path to save the checkpoint"
    )
    parser.add_argument(
        "-b", 
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size"
    )
    parser.add_argument(
        "-e", 
        "--epoch", 
        type=int, 
        default=5, 
        help="Epochs to train"
    )
    parser.add_argument(
        "--save_epoch", 
        type=int, 
        default=5, 
        help="Save ckpt every n epochs"
    )
    parser.add_argument(
        "--max_ckpt", 
        type=int, 
        default=3, 
        help="Maximum ckpt to keep"
    )
    parser.add_argument(
        "-m", 
        "--pretrained_model", 
        type=str, 
        default="./pretrained_models/qwen3", 
        help="Path for pretrained model"
    )
    args = parser.parse_args()
    start_train(args.output_dir, args.pretrained_model, args.batch_size, args.epoch,args.save_epoch,args.max_ckpt)
# --- 4. 运行推理测试 ---
# text = "how are you"
# model_inputs = tokenizer([text], return_tensors="pt").to(device)
# generated_ids = model.generate(
#     **model_inputs, 
#     max_new_tokens=256
# )

# response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# print("\n--- Model Response ---")
# print(response)