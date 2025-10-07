from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments, Trainer
import torch
from data.qwen_t2s_dataset import Qwen3Text2SemanticDataset
import argparse
import os
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def start_train(output_dir, model_path, batch_size, gradient_acc, epoch, save_epoch, max_ckpt, random_mask):
    print(f"Loading model on device: {device}")

    # 2. 加载 Tokenizer (分词器)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True # 尽管是从本地加载，但 Qwen 模型建议保留此参数
    )

    dataset = Qwen3Text2SemanticDataset("./logs/3-semantic_pairs", tokenizer, random_mask)

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
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["WANDB_PROJECT"] = "Qwen-Sovits"
    run_name = f"qwen_sovits_run_{current_time}" # 例如: my_model_run_20251006_230530
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epoch,
        learning_rate=4e-5,  # 适用于全参数微调 (或 LoRA微调可尝试 1e-4)    
        # 【核心调整 2：使用 Cosine 调度器】
        lr_scheduler_type="cosine",    
        # 【核心调整 3：设置 Warmup】
        warmup_ratio=0.05, # 前 5% 的步数用于学习率爬升
        per_device_train_batch_size=batch_size,    # use batch size 2 per GPU
        gradient_accumulation_steps=gradient_acc,    # no grad accumulation (since batch 2 is fine)
        logging_steps=10,                 # log every 20 steps
        logging_dir=f'./logs/tensorboard/{run_name}',
        run_name=run_name,
        save_strategy="epoch", 
        save_steps=save_epoch,                     # no checkpoints (not needed for demo)
        save_total_limit=max_ckpt, 
        report_to=["wandb"],                     # no W&B or HF logging
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
    checkpoint_exists = any(
        os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint-")
        for d in os.listdir(output_dir)
    )
    if checkpoint_exists:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

def modify_base_model(output_dir, model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype='auto',
        device_map="auto",
        trust_remote_code=True 
    )

    NUM_SEMANTIC_TOKENS = 2048
    semantic_tokens = [f"<t2s_{i}>" for i in range(NUM_SEMANTIC_TOKENS)]   
    special_tokens_dict = {"additional_special_tokens": semantic_tokens} 
    tokenizer.add_special_tokens(special_tokens_dict)
    new_vocab_size = len(tokenizer)
    model.resize_token_embeddings(new_vocab_size)
    assert model.get_input_embeddings().weight.size(0) == new_vocab_size
    assert model.get_output_embeddings().weight.size(0) == new_vocab_size
    
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

class Qwen3Text2SemanticModel:
    tokenizer:any
    model:AutoModelForCausalLM
    t2s_token_start:any
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
        self.t2s_token_start = self.tokenizer.convert_tokens_to_ids("<t2s_0>")
        self.model.to(device)
    
    def infer(self, prompt:str, ref_txt:str, ref_semantic:torch.Tensor):
        text = f"<|im_start|>user\n语音转文本任务：{{{ref_txt}、{prompt}}}<|im_end|>\n<|im_start|>assistant\n"
        ref_semantic = ref_semantic + self.t2s_token_start
        txt_ids = self.tokenizer([text], return_tensors="pt").to('cpu')
        input_ids = txt_ids.data['input_ids']
        attention_mask = txt_ids.data['attention_mask']
        attention_mask_ref = (ref_semantic != self.tokenizer.pad_token_id).long()
        input_ids = torch.cat([input_ids,ref_semantic], dim=1).to(device)
        attention_mask = torch.cat([attention_mask, attention_mask_ref], dim=1).to(device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1000,
            max_length=1000,
            temperature=0.6,
            top_p=0.5,               # Top-P 采样
            top_k=10,                # Top-K 采样（安全网）
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )

        result = generated_ids[0][input_ids.shape[1]:]
        if result[-1] == self.tokenizer.eos_token_id:
            result = result[:-1]

        response = self.tokenizer.decode(result, skip_special_tokens=True)
        print("\n--- Model Response ---")
        print(response)
        result = result - self.t2s_token_start
        return result[result >= 0]

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
        default=2, 
        help="Batch size"
    )
    parser.add_argument(
        "-e", 
        "--epoch", 
        type=int, 
        default=6, 
        help="Epochs to train"
    )
    parser.add_argument(
        "-ga", 
        "--gradient_acc", 
        type=int, 
        default=1, 
        help="Gradient accumulation steps to train"
    )
    parser.add_argument(
        "--save_epoch", 
        type=int, 
        default=400, 
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
    parser.add_argument(
        "-bm", 
        '--modify_base_model',
        action='store_true',
        default=False,
        help='Modify the base model to adapt the semantic tokens'
    )
    parser.add_argument(
        '--random_mask',
        action='store_true',
        default=False,
        help='Random mask the semantic tokens to act as part of the prompt'
    )
    args = parser.parse_args()
    if args.modify_base_model:
        modify_base_model(args.output_dir, args.pretrained_model)
    else:
        start_train(args.output_dir, args.pretrained_model, args.batch_size, args.gradient_acc, args.epoch,args.save_epoch,args.max_ckpt, args.random_mask)
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