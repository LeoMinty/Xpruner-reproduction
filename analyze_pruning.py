# analyze_pruning.py
import torch
import matplotlib.pyplot as plt
from deit_modified import deit_small_patch16_224

# --- 配置 ---
MODEL_STATE_PATH = "deit_small_phase2_pruned.pth"
NUM_CLASSES = 10
NUM_BLOCKS = 12 # deit-small 有12个block

# --- 加载模型 ---
# 创建一个与第二阶段完全相同的模型结构，以便加载state_dict
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_STATE_PATH, map_location='cpu')
model.load_state_dict(state_dict, strict=False) # strict=False以忽略beta, gamma等
model.eval()

print("--- 学到的逐层剪枝率 (r) ---")
layer_pruning_rates = []
for i in range(NUM_BLOCKS):
    # 从state_dict中直接读取r_logit的值
    r_logit = state_dict.get(f'blocks.{i}.attn.r_logit')
    if r_logit is not None:
        # 使用sigmoid将其转换为0到1之间的剪枝率
        r = torch.sigmoid(r_logit).item()
        layer_pruning_rates.append(r)
        print(f"Block {i}: 剪枝率 r = {r:.4f}")

# --- 可视化 ---
if layer_pruning_rates:
    plt.figure(figsize=(10, 6))
    plt.bar(range(NUM_BLOCKS), layer_pruning_rates, color='skyblue')
    plt.xlabel('Transformer Block Index')
    plt.ylabel('Pruning Rate (r)')
    plt.title('Learned Layer-wise Pruning Rates for DeiT-Small')
    plt.xticks(range(NUM_BLOCKS))
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--')
    plt.savefig('learned_pruning_rates.png')
    print("\n剪枝率可视化图已保存为 learned_pruning_rates.png")