# prune_model.py
import torch
from torch import nn
from deit_modified import deit_small_patch16_224
from vision_transformer_modified import VisionTransformer, Block, Attention # 导入原始结构

# --- 配置 ---
PHASE2_MODEL_PATH = "deit_small_phase2_pruned_test.pth"
FINAL_MODEL_PATH = "deit_small_final_pruned_model.pth"
NUM_CLASSES = 10

# 1. 加载第二阶段训练好的模型
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)
state_dict = torch.load(PHASE2_MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

# 2. 计算每层要保留的头的索引
print("--- 正在计算要保留的注意力头 ---")
pruning_config = {}
for i, block in enumerate(model.blocks):
    attn_module = block.attn.attn # 原始的Attention模块
    mask_scores = block.attn.explainability_mask # (num_classes, num_heads, head_dim)
    r_logit = block.attn.r_logit
    
    num_heads = attn_module.num_heads
    r = torch.sigmoid(r_logit).item()
    num_heads_to_keep = int(num_heads * (1.0 - r))
    
    # 计算每个头的重要性分数（跨类别求和）
    head_importance = mask_scores.sum(dim=[0, 2]) # 形状: (num_heads,)
    
    # 找到分数最高的头的索引
    _, top_k_indices = torch.topk(head_importance, k=num_heads_to_keep)
    top_k_indices = sorted(top_k_indices.tolist()) # 排序以方便权重复制
    
    pruning_config[f'blocks.{i}.attn'] = top_k_indices
    print(f"Block {i}: 总共 {num_heads} 个头, 保留 {num_heads_to_keep} 个. 保留索引: {top_k_indices}")

# 3. 创建一个新的、更小的模型并复制权重 (这一步非常繁琐，此处提供一个简化的逻辑)
# 注意：一个完整的实现需要您重写Attention和VisionTransformer类，使其能够根据config构建
# 这里我们用一个简化的方法：直接修改原模型的权重和结构
print("\n--- 正在创建并填充剪枝后的模型 ---")
pruned_model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)
pruned_state_dict = pruned_model.state_dict()

for name, param in state_dict.items():
    if "attn" not in name:
        # 如果不是注意力模块的参数，直接复制
        if name in pruned_state_dict and pruned_state_dict[name].shape == param.shape:
            pruned_state_dict[name] = param
    else:
        # --- 处理注意力模块的权重 ---
        # 这是一个简化的演示，实际操作更复杂
        # 需要根据保留的头的索引来切片qkv.weight, proj.weight等
        # 例如: W_qkv_kept = W_qkv_original.view(...).index_select(dim_for_heads, indices_to_keep).reshape(...)
        # 由于代码非常复杂，建议先理解逻辑。核心是根据pruning_config对权重张量进行切片。
        # 暂时我们还是复制所有权重，后续微调时，被剪掉的头将不会被使用。
        if name in pruned_state_dict and pruned_state_dict[name].shape == param.shape:
            pruned_state_dict[name] = param

pruned_model.load_state_dict(pruned_state_dict)

# 附加剪枝配置到模型，以便微调时使用
pruned_model.pruning_config = pruning_config
torch.save(pruned_model.state_dict(), FINAL_MODEL_PATH)
print(f"\n剪枝后的模型权重已保存到 {FINAL_MODEL_PATH}")
print("注意：以上过程仅为逻辑演示，未进行物理上的参数删除。")
print("下一步，我们将在微调脚本中加载这个模型，并只使用保留的头。")