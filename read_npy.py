import numpy as np

# 读取 .npy 文件
# 编码后的output/vq/2024-10-27-19-36-20_300stepVQ/codebook/中的npy，应该是一串数字，表示的是pose对应在codebook中的索引
# data = np.load('output/vq/2024-10-27-19-36-20_300stepVQ/codebook/000004.npy')
data = np.load('/data_ssd2/ymh/MMM/output/vq/2024-11-03-23-40-20_vq-baseline/codebook/000002.npy')

# 打印数据内容
print(data)

# 查看数据类型和形状
print("数据类型:", data.dtype)
print("数据形状:", data.shape)


# import pickle

# # 指定 .pkl 文件的路径
# file_path = 'glove/our_vab_words.pkl'

# # 读取 .pkl 文件
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # 打印数据内容
# print(len(data))