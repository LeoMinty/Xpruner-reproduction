# main_phase2.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 关键：从你修改过的本地文件导入模型
from deit_modified import deit_small_patch16_224
from vision_transformer_modified import MaskedAttention # 导入用于类型检查

# --- 1. 定义超参数和配置 ---
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 80 # 论文中DeiT的剪枝训练轮数
ALPHA_TARGET = 0.5 # 目标总剪枝率, e.g., 剪掉50%

# 模型状态文件路径
MODEL_STATE_PATH = "deit_small_phase1_masks_cifar10.pth"

# --- 2. 准备数据集 (与第一阶段相同) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. 加载模型并切换到剪枝模式 ---
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)
print(f"正在从 {MODEL_STATE_PATH} 加载模型状态...")
model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location=device))
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
beta = nn.Parameter(torch.tensor(0.0)).to(device)
gamma = nn.Parameter(torch.tensor(0.0)).to(device)

def calculate_pruning_loss(model, alpha_target, total_prunable_elements, beta, gamma):
    """计算剪枝率正则化损失 L_R (Eq. 10, 11)"""
    current_R = torch.tensor(0.0, device=device)
    for module in model.modules():
        if isinstance(module, MaskedAttention):
            r = torch.sigmoid(module.r_logit)
            num_elements_in_module = module.explainability_mask.numel()
            current_R += (r * num_elements_in_module) / total_prunable_elements
    
    loss_r = beta * (alpha_target - current_R)**2 + gamma * (alpha_target - current_R)
    return loss_r

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
model_weights.append(beta)
model_weights.append(gamma)
        
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
        loss_r = calculate_pruning_loss(model, ALPHA_TARGET, num_prunable_elements, beta, gamma)
        total_loss = loss_ce + loss_r
        
        # 反向传播
        total_loss.backward()
        
        # 更新参数
        optimizer_weights.step()
        optimizer_pruning.step()
        
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Total Loss: {total_loss.item():.4f}, CE Loss: {loss_ce.item():.4f}, Pruning Loss: {loss_r.item():.4f}")

print("第二阶段训练完成!")
# 保存最终的剪枝模型
output_filename = "deit_small_phase2_pruned.pth"
print(f"正在将模型状态保存到: {output_filename} ...")
torch.save(model.state_dict(), output_filename)
print("保存成功！")