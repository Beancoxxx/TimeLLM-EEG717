"""
TimeLLM EEG分类集成测试脚本
用于验证模型是否正确集成
适配RTX 2060 (6GB显存)
"""

import torch
import sys
import os

# 添加项目路径到Python路径
project_path = r"D:\文件\文件\HKU\Dissertation\Time-LLM-main-editversion\Time-LLM-main"
sys.path.append(project_path)

print("=" * 70)
print("TimeLLM EEG分类集成测试")
print("=" * 70)
print(f"项目路径: {project_path}")
print(f"Python路径已添加: {project_path in sys.path}")

# 检查CUDA
print(f"\nCUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 测试导入
print("\n1. 测试模块导入...")
try:
    from models.TimeLLM import Model
    print("✓ 成功导入 TimeLLM.Model")
except Exception as e:
    print(f"✗ 导入TimeLLM失败: {e}")
    print("请确保已将修改后的TimeLLM.py放在models文件夹中")
    exit(1)

try:
    from data_provider.data_factory import data_provider
    print("✓ 成功导入 data_provider")
except Exception as e:
    print(f"✗ 导入data_provider失败: {e}")
    exit(1)

# 创建适合RTX 2060的配置
class Config:
    # 任务配置
    task_name = 'classification'
    
    # 数据配置
    seq_len = 256      # EEG序列长度
    pred_len = 0       # 分类任务不需要预测长度
    label_len = 0      # 分类任务不需要标签长度
    enc_in = 32        # DEAP的32个EEG通道
    num_class = 2      # 二分类（正面/负面情绪）
    
    # 模型配置（针对6GB显存优化）
    d_model = 32       # 模型维度
    n_heads = 4        # 注意力头数（减少内存使用）
    d_ff = 64          # FFN维度（减少内存使用）
    dropout = 0.1
    patch_len = 16     # 补丁长度
    stride = 8         # 步长
    
    # LLM配置（关键：减少内存使用）
    llm_model = 'GPT2'
    llm_dim = 768     # Llama-7B的维度
    llm_layers = 2     # 只使用2层（大幅减少内存！）
    
    # 其他必需参数
    features = 'M'
    e_layers = 2
    d_layers = 1
    factor = 1
    activation = 'gelu'
    embed = 'timeF'
    freq = 'h'
    
print("\n2. 创建模型...")
print("配置参数:")
print(f"  - LLM层数: {Config.llm_layers} (针对6GB显存优化)")
print(f"  - 模型维度: {Config.d_model}")
print(f"  - 通道数: {Config.enc_in}")
print(f"  - 类别数: {Config.num_class}")

try:
    config = Config()
    model = Model(config)
    print("✓ 模型创建成功")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数量: {total_params/1e6:.2f}M")
    print(f"  - 可训练参数: {trainable_params/1e6:.2f}M")
    
except Exception as e:
    print(f"✗ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试前向传播
print("\n3. 测试前向传播...")
batch_size = 2  # 小批次测试
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将模型移到GPU（如果可用）
if torch.cuda.is_available():
    try:
        model = model.to(device)
        print(f"✓ 模型已移至: {device}")
    except Exception as e:
        print(f"✗ 无法将模型移至GPU: {e}")
        device = torch.device('cpu')
        print("  使用CPU运行")

# 创建测试数据
x = torch.randn(batch_size, config.seq_len, config.enc_in).to(device)
x_mark = torch.zeros(batch_size, config.seq_len, 4).to(device)

print(f"\n输入数据形状:")
print(f"  - x: {x.shape}")
print(f"  - x_mark: {x_mark.shape}")

# 前向传播测试
try:
    with torch.no_grad():
        # 清空GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        output = model(x, x_mark)
        
    print(f"\n✓ 前向传播成功!")
    print(f"  - 输出形状: {output.shape}")
    print(f"  - 期望形状: [batch_size={batch_size}, num_class={config.num_class}]")
    print(f"  - 输出示例: {output[0].cpu().numpy()}")
    
    # 检查输出是否合理
    if output.shape == (batch_size, config.num_class):
        print("\n✓ 输出维度正确！")
    else:
        print("\n✗ 输出维度不正确")
        
except Exception as e:
    print(f"\n✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    
    if "out of memory" in str(e).lower():
        print("\n内存不足建议:")
        print("  1. 减少 llm_layers (当前: {})".format(Config.llm_layers))
        print("  2. 减少 batch_size")
        print("  3. 减少 d_model 或 d_ff")

# 测试数据加载器
print("\n" + "="*50)
print("4. 测试EEG数据加载器...")

class DataConfig:
    # 复用模型配置
    task_name = 'classification'
    data = 'DEAP'
    root_path = r'D:\文件\文件\HKU\Dissertation\dataset\DEAP\data_preprocessed_python'
    seq_len = 256
    pred_len = 0
    label_len = 0
    batch_size = 4
    num_workers = 0
    embed = 'timeF'
    freq = 'h'
    features = 'M'
    
    # EEG特定参数
    num_class = 2
    enc_in = 32
    classification_type = 'valence'
    overlap = 128
    normalize = True
    filter_freq = [0.5, 45]
    sampling_rate = 128
    subject_list = ['s01']  # 只测试第一个被试

try:
    data_args = DataConfig()
    
    # 检查数据路径
    if os.path.exists(data_args.root_path):
        print(f"✓ 数据路径存在: {data_args.root_path}")
        files = [f for f in os.listdir(data_args.root_path) if f.endswith('.dat')]
        print(f"  找到 {len(files)} 个数据文件")
        if files:
            print(f"  示例文件: {files[:3]}")
    else:
        print(f"✗ 数据路径不存在: {data_args.root_path}")
        print("  跳过数据加载测试")
        exit(0)
    
    # 尝试加载数据
    print("\n尝试加载DEAP数据...")
    train_data, train_loader = data_provider(data_args, 'train')
    print(f"✓ 数据加载成功!")
    print(f"  - 训练集大小: {len(train_data)}")
    print(f"  - 批次数: {len(train_loader)}")
    
    # 获取一个批次
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        print(f"\n批次信息:")
        print(f"  - batch_x: {batch_x.shape}")
        print(f"  - batch_y: {batch_y.shape}")
        print(f"  - 标签值: {batch_y[:8]}")  # 显示前8个标签
        break
        
except Exception as e:
    print(f"\n✗ 数据加载失败: {e}")
    import traceback
    traceback.print_exc()

# 测试模型与数据的兼容性
print("\n" + "="*50)
print("5. 测试模型与真实数据的兼容性...")

if 'train_loader' in locals():
    try:
        # 使用真实数据测试前向传播
        batch_x = batch_x.to(device)
        batch_x_mark = batch_x_mark.to(device)
        
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            output = model(batch_x, batch_x_mark)
            
        print("✓ 模型成功处理真实DEAP数据!")
        print(f"  - 输入形状: {batch_x.shape}")
        print(f"  - 输出形状: {output.shape}")
        
        # 测试损失计算
        criterion = torch.nn.CrossEntropyLoss()
        batch_y = batch_y.long().to(device)
        loss = criterion(output, batch_y)
        print(f"  - 损失值: {loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ 处理真实数据失败: {e}")
        import traceback
        traceback.print_exc()

# 总结
print("\n" + "="*70)
print("测试总结")
print("="*70)

print("\n如果所有测试都通过，你可以:")
print("1. 创建完整的训练脚本")
print("2. 开始训练模型")
print("\n建议的下一步:")
print("- 如果遇到内存问题，进一步减少 llm_layers 到 1")
print("- 可以尝试增加 batch_size 到 8 或 16")
print("- 确保数据路径正确，所有被试文件都存在")

print("\n祝实验顺利! 🚀")
