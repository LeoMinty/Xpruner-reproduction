# main_phase1.py
import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ！！！关键：从你修改过的本地文件导入模型！！！
from deit_modified import deit_small_patch16_224

# --- 1. 定义超参数和配置 ---
NUM_CLASSES = 10  # 先以CIFAR-10为例
BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 50 # 论文中DeiT的训练轮数
LAMBDA_SP = 0.01 # 稀疏性损失权重 (需要调试)
LAMBDA_SM = 0.01 # 平滑性损失权重 (需要调试)

# --- 2. 准备数据集 (CIFAR-10) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. 加载模型并载入预训练权重 ---
# 这是将预训练权重加载到修改后模型的标准方法
print("正在加载模型...")
# a. 创建一个标准的、未经修改的预训练模型
base_model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=NUM_CLASSES)

# b. 创建我们修改过的模型实例
# 注意：你需要将num_classes传递进去
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)

# c. 加载权重 (忽略我们新增的mask)
model.load_state_dict(base_model.state_dict(), strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("模型加载完毕。")


# --- 4. 设置损失函数和优化器 ---
# 从我上次回复中复制 calculate_total_loss_phase1 和 get_all_masks 函数到这里

# 冻结模型原始权重
for name, param in model.named_parameters():
    if "explainability_mask" not in name:
        param.requires_grad = False

# 确认只有掩码是可训练的
print("以下参数将被训练:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# 优化器只包含掩码参数
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, momentum=0.9)
ce_loss_fn = nn.CrossEntropyLoss()

# --- 5. 训练循环 ---
model.train()
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 将标签传递给模型
        outputs = model(images, y_labels=labels)
        
        loss = calculate_total_loss_phase1(model, outputs, labels, ce_loss_fn, LAMBDA_SP, LAMBDA_SM)
        
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

print("第一阶段训练完成!")
# 保存训练好的掩码权重
# torch.save(model.state_dict(), 'model_with_trained_masks.pth')