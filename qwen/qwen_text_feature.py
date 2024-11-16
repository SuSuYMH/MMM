from transformers import AutoTokenizer, AutoModel
import torch

# 指定缓存目录
cache_tokenizer_dir = '/data_ssd2/ymh/MMM/qwen/qwen_tokenizer'
cache_model_dir = '/data_ssd2/ymh/MMM/qwen/qwen_model'

# 加载 Qwen 的分词器和模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", cache_dir=cache_tokenizer_dir)
model = AutoModel.from_pretrained("Qwen/Qwen-7B", cache_dir=cache_model_dir)

# 输入文本
text = "你好，世界！This is Qwen tokenizer in action."

# 使用分词器将文本转换为模型输入
inputs = tokenizer(text, return_tensors="pt")  # 返回 PyTorch 张量
print("Tokenized Inputs:", inputs)

# 使用模型提取特征
with torch.no_grad():  # 禁用梯度计算（推理时加速）
    outputs = model(**inputs)

# 提取最后隐藏层的特征 (hidden states)
hidden_states = outputs.last_hidden_state
print("Hidden States Shape:", hidden_states.shape)

# 提取每个句子的特征向量（比如使用池化后的 [CLS] 向量）
cls_embedding = hidden_states[:, 0, :]  # 第一个 token ([CLS]) 的表示
print("CLS Embedding Shape:", cls_embedding.shape)