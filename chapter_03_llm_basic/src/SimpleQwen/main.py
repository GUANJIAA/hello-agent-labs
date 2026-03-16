from email import message
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型ID
model_id = "Qwen/Qwen1.5-0.5B-chat"

# 设置设备，优先使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_id)
model.to(device)

print("模型和分词器加载完成")

message = [
    {"role": "system", "content": "你是一个专业的助手"},
    {"role": "user", "content": "你好"}
]

# 使用分词器的模板格式化输入
text = tokenizer.apply_chat_template(
    message, 
    tokenize=False,
    add_generation_prompt=True
)

# 编码输入文本
inputs = tokenizer(text, return_tensors="pt").to(device)

print("编码后输入:", inputs)

# 使用模型生成回答
# max_new_tokens 控制了模型最多能生成多少个新的token
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

# 将生成的 Token ID 截取掉输入部分
# 这样我们只解码模型生成的部分
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码生成的文本
output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print("模型生成的回答:", output_text)