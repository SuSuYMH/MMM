from transformers import AutoTokenizer

# 指定缓存目录
cache_dir = '/data_ssd2/ymh/MMM/qwen/qwen_tokenizer'

# 加载Qwen分词器到自定义目录
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", cache_dir=cache_dir, local_files_only=True)

# 示例文本
text = "你好，世界！This is Qwen tokenizer in action."

# 分词（Tokenize）
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# 将分词结果转换为 Token ID
token_ids = tokenizer.encode(text, add_special_tokens=True)
print("Token IDs:", token_ids)

# 将 Token ID 解码回原始文本
decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
print("Decoded Text:", decoded_text)