import torch

# 加载拆分的模型文件
part1 = torch.load('pytorch_model_part1.bin')
part2 = torch.load('pytorch_model_part2.bin')

# 合并字典
merged_state_dict = {**part1, **part2}

# 保存合并后的模型
torch.save(merged_state_dict, 'reconstructed_pytorch_model.bin')

print("合并完成，生成了文件：reconstructed_pytorch_model.bin")