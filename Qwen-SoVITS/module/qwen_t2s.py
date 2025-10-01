from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from data.qwen_t2s_dataset import Qwen3Text2SemanticDataset
LOCAL_MODEL_PATH = "./pretrained_models/qwen3" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on device: {device}")

dataset = Qwen3Text2SemanticDataset("./logs/semantic_pairs.txt")

# 2. 加载 Tokenizer (分词器)
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True # 尽管是从本地加载，但 Qwen 模型建议保留此参数
)

# 3. 加载 Model (模型权重)
# 将本地路径作为第一个参数传入 from_pretrained()
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype='auto',
    device_map="auto", # 自动分配模型到可用的 GPU/CPU
    trust_remote_code=True 
)

# 确保模型被加载到正确的设备上（虽然 device_map="auto" 会处理，但这是好习惯）
model.to(device)

print(f"✅ Qwen3 0.6B 模型已从本地路径 {LOCAL_MODEL_PATH} 成功加载。")

# --- 4. 运行推理测试 ---

text = "how are you"
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(
    **model_inputs, 
    max_new_tokens=256
)

response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\n--- Model Response ---")
print(response)