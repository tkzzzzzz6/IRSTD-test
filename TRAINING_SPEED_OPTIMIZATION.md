# 训练加速优化指南

## 快速开始（推荐配置）

### 1. 提速命令（保留精度）
```bash
# 增加数据加载线程 + 增大batch size
python train.py --model_names ACM ALCNet --dataset_names IRSTD-1K \
  --batchSize 32 --threads 8 --intervals 20
```

### 2. 极速模式（稍损精度但快3-4倍）
```bash
# 减小patch大小 + 增加batch size + 多线程
python train.py --model_names ACM ALCNet --dataset_names IRSTD-1K \
  --batchSize 64 --patchSize 128 --threads 8 --nEpochs 200 --intervals 20
```

---

## 详细优化清单

| 配置参数 | 默认值 | 推荐值 | 效果 | 影响 |
|---------|-------|-------|------|------|
| `--threads` | 1 | 8 | **快速** | 数据加载 2-4x |
| `--batchSize` | 16 | 32-64 | **快速** | GPU吞吐 1.5-2x |
| `--patchSize` | 256 | 128-192 | **较快** | 内存 & 速度 1.5x |  
| `--intervals` | 10 | 20-50 | **轻微** | 减少日志开销 |
| `--nEpochs` | 400 | 200-300 | **快速** | 总耗时直接减少 |

---

## 按场景的优化建议

### 场景1：快速验证模型（1小时内训完）
```bash
python train.py --model_names ACM --dataset_names NUAA-SIRST \
  --batchSize 64 --patchSize 128 --threads 8 --nEpochs 50
```
预期：基线速度的 8-10 倍，但精度可能下降 2-5%

### 场景2：生产训练（保留精度，快 2 倍）  
```bash
python train.py --model_names ACM ALCNet --dataset_names IRSTD-1K \
  --batchSize 32 --patchSize 256 --threads 8 --nEpochs 300 --intervals 20
```
预期：快 2 倍，精度基本持平

### 场景3：多卡训练（如有 2 张 GPU）
```bash
# train.py 已集成 DataParallel，自动启用多 GPU
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_names ACM ALCNet \
  --batchSize 64 --threads 8 --nEpochs 300
```
预期：快 1.8-2.0 倍（由于通信开销）

---

## 硬件瓶颈诊断

### CPU 瓶颈（数据加载慢）
- 症状：GPU 利用率 < 50%，数据加载打满 CPU
- 解决：增加 `--threads` 至 8-16（取决于 CPU 核心数）

### GPU 显存瓶颈  
- 症状：OOM 错误
- 解决：降低 `--batchSize` 或 `--patchSize`

### GPU 计算不饱和
- 症状：GPU 利用率低但无 OOM
- 解决：增加 `--batchSize` 直到达到显存极限的 80%

---

## 进阶优化（需修改代码）

### 1. 启用混合精度训练（AMP）→ 快 20-30%
编辑 `train.py`，在 optimizer 后添加：
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# In training loop:
with autocast():
    output = net(img)
    loss = net.loss(output, gt_mask)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. 启用 pin_memory（快 10-15%）
编辑 `train.py` 的 DataLoader：
```python
train_loader = DataLoader(
    dataset=train_set, 
    num_workers=opt.threads, 
    batch_size=opt.batchSize, 
    shuffle=True,
    pin_memory=True  # ← 加这一行
)
```

### 3. 启用梯度检查点（节省显存 30-40%，但计算稍慢）
需要模型支持，编辑对应的 model 文件

---

## 当前系统信息

运行下列命令查看你的 GPU 信息：
```bash
# Windows 显示 GPU 用量  
nvidia-smi

# 或者实时监控（需要 nvitop）
nvitop
```

---

## 总体预期提速效果

| 优化组合 | 相对加速 | 内存增加 | 精度影响 |
|---------|--------|--------|--------|
| threads=8 only | 1.5x | ~5% | 0% |
| batch=32 + threads=8 | 2.0x | ~15% | 0-1% |
| batch=64 + patch=128 + threads=8 | 3-4x | ~40% | 2-5% |
| +AMP (mixed precision) | 1.2-1.5x | -40% | <1% |

---

## 监测训练速度

在命令后加 `| tee train_log.txt` 保存日志，用于后续分析：
```bash
python train.py --model_names ACM --dataset_names IRSTD-1K \
  --batchSize 32 --threads 8 | tee train_$(date +%s).txt
```

查看首个 epoch 的时间（排除 model 初始化开销）。
