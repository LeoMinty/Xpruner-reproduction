# main_phase2.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.data import Subset
import numpy as np

# 关键：从你修改过的本地文件导入模型
from deit_modified import deit_small_patch16_224
from vision_transformer_modified import MaskedAttention # 导入用于类型检查

# --- 1. 定义超参数和配置 ---
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 10 # 减少epochs用于测试
ALPHA_TARGET = 0.2 # 目标总剪枝率

# 模型状态文件路径
MODEL_STATE_PATH = "deit_small_phase1_masks_cifar10.pth"

# --- 2. 准备数据集 (与第一阶段相同) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 首先，加载完整的训练集
full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# --- 新增：创建数据集的子集 ---
subset_percentage = 0.1 # 使用10%的数据进行快速调试
num_train = len(full_train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices) # 打乱索引
split = int(np.floor(subset_percentage * num_train))
subset_indices = indices[:split]

# 使用Subset创建子集
train_subset = Subset(full_train_dataset, subset_indices)
print(f"使用 {len(train_subset)} / {num_train} 个样本进行快速调试...")

# 在DataLoader中使用子集
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. 加载模型并切换到剪枝模式 ---
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)
print(f"正在从 {MODEL_STATE_PATH} 加载模型状态...")
model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location=device), strict=False)
model.to(device)
print("加载成功！")

# 关键：激活所有MaskedAttention模块的剪枝模式
num_prunable_elements = 0
for module in model.modules():
    if isinstance(module, MaskedAttention):
        module.is_pruning_phase = True
        num_prunable_elements += module.explainability_mask.numel()

# --- 4. 设置损失函数和优化器 ---
ce_loss_fn = nn.CrossEntropyLoss()

# 新增：用于L_R损失的可学习参数
# beta = nn.Parameter(torch.tensor(0.0, device=device))
# gamma = nn.Parameter(torch.tensor(0.0, device=device))

def calculate_pruning_loss_simple(model, alpha_target, total_prunable_elements):
    """计算一个简单的、稳定的二次惩罚剪枝损失"""
    current_R = torch.tensor(0.0, device=device)
    for module in model.modules():
        if isinstance(module, MaskedAttention):
            r = torch.sigmoid(module.r_logit)
            num_elements_in_module = module.explainability_mask.numel()
            current_R += ((r * num_elements_in_module) / total_prunable_elements).sum()

    # 核心修改：使用二次惩罚
    loss_r = (current_R - alpha_target)**2
    return loss_r, current_R # 同时返回current_R用于监控

# --- 关键：为不同参数组设置不同的优化器和学习率 ---
# a. 冻结第一阶段学到的掩码分数
pruning_params = []
model_weights = []
for name, param in model.named_parameters():
    if "explainability_mask" in name:
        param.requires_grad = False
    elif "r_logit" in name or "theta" in name:
        pruning_params.append(param)
    else:
        model_weights.append(param)

# 添加beta和gamma到模型权重组
# model_weights.append(beta)
# model_weights.append(gamma)
        
optimizer_weights = torch.optim.AdamW(model_weights, lr=5e-4)
optimizer_pruning = torch.optim.AdamW(pruning_params, lr=0.02)

print(f"模型权重参数组大小: {len(model_weights)}")
print(f"剪枝参数组大小: {len(pruning_params)}")


# --- 5. 第二阶段训练循环 ---
model.train()
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 梯度清零
        optimizer_weights.zero_grad()
        optimizer_pruning.zero_grad()
        
        # 前向传播
        outputs = model(images, y_labels=labels)
        
        # 计算损失
        loss_ce = ce_loss_fn(outputs, labels)
        loss_r, current_R_val = calculate_pruning_loss_simple(model, ALPHA_TARGET, num_prunable_elements)
        
        # 引入一个超参数 lambda_prune 来放大剪枝损失的权重
        lambda_prune = 50.0 # 可以从1.0, 10.0, 100.0开始尝试
        total_loss = loss_ce + lambda_prune * loss_r
        # total_loss = loss_ce + loss_r
        
        # 反向传播
        total_loss.backward()
        
        # 更新参数
        optimizer_weights.step()
        optimizer_pruning.step()
        
        if i % 50 == 0:
            
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Total Loss: {total_loss.item():.4f}, CE Loss: {loss_ce.item():.4f}, Pruning Loss: {loss_r.item():.4f}, Current R: {current_R_val.item():.4f}")


print("第二阶段训练完成!")
# 保存最终的剪枝模型
output_filename = "deit_small_phase2_pruned_test.pth"
print(f"正在将模型状态保存到: {output_filename} ...")
torch.save(model.state_dict(), output_filename)
print("保存成功！")