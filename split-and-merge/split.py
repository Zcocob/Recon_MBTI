import torch

# 加载模型权重文件
model_state_dict = torch.load('pytorch_model.bin')

# 初始化变量
part1, part2 = {}, {}
total_size = sum(param.numel() * param.element_size() for param in model_state_dict.values())
split_threshold = total_size // 2  # 将大小均分
current_size = 0

# 遍历模型权重字典，拆分为两部分
for key, value in model_state_dict.items():
    size = value.numel() * value.element_size()
    if current_size + size <= split_threshold:
        part1[key] = value
        current_size += size
    else:
        part2[key] = value

# 保存拆分后的文件
torch.save(part1, 'pytorch_model_part1.bin')
torch.save(part2, 'pytorch_model_part2.bin')

print("拆分完成，生成了两个文件：pytorch_model_part1.bin 和 pytorch_model_part2.bin")
