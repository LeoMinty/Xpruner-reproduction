# create_and_save_blank_model.py
import torch
from deit_modified import deit_small_patch16_224 # 确保从修改过的文件导入

NUM_CLASSES = 10

print("正在创建模型结构...")
# 创建模型实例，不加载任何预训练权重
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)

output_filename = "test_save.pth"
print(f"正在测试保存到: {output_filename} ...")

# 执行保存操作
torch.save(model.state_dict(), output_filename)

print("测试保存成功！您的模型和保存代码工作正常。")
print("现在请修改 main_phase1.py 并重新开始训练。")