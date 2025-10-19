from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments, Trainer, AutoConfig
from tokenizers import Tokenizer
from transformers.models.qwen3 import Qwen3ForCausalLM
import torch
import torch.nn as nn
from data.qwen_t2s_dataset import Qwen3Text2SemanticDataset
import argparse
from tqdm import tqdm
import os
import datetime
from safetensors.torch import load_file
from text.phoneme_utils import get_phones
from transformers.generation import TextStreamer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Qwen3Text2SemanticModelForTraining(Qwen3ForCausalLM):
    def __init__(self, config, phoneme_vocab_size):
        super().__init__(config)
        # 假设 Qwen 的 Transformer 模块输出的隐藏维度是 config.hidden_size
        hidden_size = config.hidden_size 
        
        # 1. 音素识别分类头 (Phoneme Recognition Head)
        self.pr_head = nn.Linear(hidden_size, phoneme_vocab_size)
        
        # 2. 预留参数用于PR Loss权重
        self.pr_loss_weight = 0.2 
        if config.torch_dtype is not None:
            # 将模型的 dtype 转换为 config 中指定的类型 (例如 bfloat16)
            self.to(config.torch_dtype) 

    def forward(self, input_ids=None, labels=None, phoneme_target_ids=None, **kwargs):
        # 1. 运行原始 Qwen Transformer 模块
        outputs = super().forward(input_ids=input_ids, labels=labels, **kwargs, output_hidden_states=True)
        # 提取最后一个Transformer块的隐藏状态 (Hidden States)
        hidden_states = outputs.hidden_states[-1] 

        # 2. 计算 Semantic Token (主任务) Loss
        lm_loss = outputs.loss # 原始 CausalLM Loss，即 Semantic Token预测损失
        
        # 3. 计算 PR Loss (辅助任务)
        pr_loss = torch.tensor(0.0, device=lm_loss.device)
        if phoneme_target_ids is not None:
            # PR Head进行预测
            pr_logits = self.pr_head(hidden_states)
            # 展平以便于交叉熵计算 (忽略填充 ID)
            pr_loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 
            pr_loss = pr_loss_fct(
                pr_logits.view(-1, pr_logits.size(-1)), 
                phoneme_target_ids.view(-1)
            )

        # 4. 总损失：主 Loss + 辅助 Loss (加权求和)
        total_loss = lm_loss + self.pr_loss_weight * pr_loss
        
        # 返回总损失和其他必要的输出
        return {"loss": total_loss, "lm_loss": lm_loss.detach(), "pr_loss": pr_loss.detach()}
    
    def from_pretrained(model_path):
        config = AutoConfig.from_pretrained(
            model_path
        )
        state_dict = load_file(os.path.join(model_path, 'model.safetensors'))
        model = Qwen3Text2SemanticModelForTraining(
            config=config,  # 传入 Qwen 的基础配置
            phoneme_vocab_size=250 # 传入您的新参数
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, 
            strict=False 
        )
        return model

def start_train(output_dir, model_path, batch_size, gradient_acc, epoch, save_epoch, max_ckpt, random_mask):
    print(f"Loading model on device: {device}")

    # 2. 加载 Tokenizer (分词器)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True # 尽管是从本地加载，但 Qwen 模型建议保留此参数
    )

    dataset = Qwen3Text2SemanticDataset("./logs/3-semantic_pairs", tokenizer, random_mask)
    # split_datasets = dataset.train_test_split(test_size=0.1) 

    # # 训练集
    # train_dataset = split_datasets['train']
    # # 验证集
    # eval_dataset = split_datasets['test'] 
    # 3. 加载 Model (模型权重)
    # 将本地路径作为第一个参数传入 from_pretrained()
    model = Qwen3Text2SemanticModelForTraining.from_pretrained(model_path)

    # 确保模型被加载到正确的设备上（虽然 device_map="auto" 会处理，但这是好习惯）
    model.to(device)

    print(f"✅ Qwen3 0.6B 模型已从本地路径 {model_path} 成功加载。")
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["WANDB_PROJECT"] = "Qwen-Sovits"
    run_name = f"qwen_sovits_run_{current_time}" # 例如: my_model_run_20251006_230530
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epoch,
        # === 关键的验证设置 ===
        #evaluation_strategy="steps",         # 设置在每隔一定步数后进行验证
        # evaluation_strategy="epoch",       # 或者在每个 epoch 结束时进行验证
        #eval_steps=30,
        learning_rate=4e-5,  # 适用于全参数微调 (或 LoRA微调可尝试 1e-4)    
        # 【核心调整 2：使用 Cosine 调度器】
        lr_scheduler_type="cosine",    
        # 【核心调整 3：设置 Warmup】
        warmup_ratio=0.05, # 前 5% 的步数用于学习率爬升
        per_device_train_batch_size=batch_size,    # use batch size 2 per GPU
        per_device_eval_batch_size=batch_size,
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
        #train_dataset=train_dataset,      # 训练集 (必须)
        #eval_dataset=eval_dataset, 
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
    config = model.config
    state_dict = model.state_dict()
    model = Qwen3Text2SemanticModelForTraining(
        config=config,  # 传入 Qwen 的基础配置
        phoneme_vocab_size=250 # 传入您的新参数
    )
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, 
        strict=False 
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

class SemanticTokenStreamer(TextStreamer):
    
    def __init__(self, max_token=1024):
        self.token_list = None
        self.tqdm = tqdm(total=max_token, desc="Generating Semantic tokens", unit="token")

    def put(self, input_ids):
        self.tqdm.update(1)
        if self.token_list is None:
            self.token_list = input_ids
        else:
            self.token_list = torch.cat([self.token_list, input_ids.unsqueeze(0)], dim =1)
        
    def end(self):
        self.tqdm.close()
        print("[Generation Complete]\n")

LOCAL_PHONEME_TOKENIZER= "./pretrained_models/phoneme_tokenizer/phoneme_tokenizer.json"
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
        self.think_start = torch.tensor(self.tokenizer.convert_tokens_to_ids("<think>"), dtype=torch.int64).unsqueeze(0)
        self.think_end = torch.tensor(self.tokenizer.convert_tokens_to_ids("</think>"), dtype=torch.int64).unsqueeze(0)
        self.ph_token_start = self.tokenizer.convert_tokens_to_ids("<ph_0>")
        self.phoneme_tokenizer = Tokenizer.from_file(LOCAL_PHONEME_TOKENIZER)
        self.model.to(device)
    
    def infer(self, prompt:str, ref_txt:str, ref_semantic:torch.Tensor, max_tokens = 1024):
        text = f"<|im_start|>user\n文字转语音任务：{{{ref_txt}。{prompt}}}<|im_end|>\n<|im_start|>assistant\n"
        #text = f"<|im_start|>user\n语音转文本任务：{{{ref_txt}}}<|im_end|>\n<|im_start|>assistant\n"
        txt_ids = self.tokenizer([text], return_tensors="pt").to('cpu')
        txt_ids = txt_ids.data['input_ids'].squeeze(0)

        ph, norm_text = get_phones(f"{ref_txt}。{prompt}", "all_ja", "v2", final=True)
        print(f"音素结果：{ph}")
        ph_ids = torch.tensor(self.phoneme_tokenizer.encode(ph).ids, dtype=torch.long).to('cpu') + self.ph_token_start
        input_ids = torch.cat([self.think_start, ph_ids, self.think_end], dim=0).unsqueeze(0)
        
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        ref_semantic = ref_semantic + self.t2s_token_start        
        attention_mask_ref = (ref_semantic != self.tokenizer.pad_token_id).long()
        input_ids = torch.cat([input_ids,ref_semantic], dim=1).to(device)
        attention_mask = torch.cat([attention_mask, attention_mask_ref], dim=1).to(device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            max_length=max_tokens,
            temperature=0.9,
            top_p=0.95,               # Top-P 采样
            top_k=20,                # Top-K 采样（安全网）
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer = SemanticTokenStreamer(max_tokens)
        )

        result = generated_ids[0][input_ids.shape[1]:]
        if result[-1] == self.tokenizer.eos_token_id:
            result = result[:-1]

        response = self.tokenizer.decode(result, skip_special_tokens=False)
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