# train.py 核心优化改动

## 改动位置 1：增加 pin_memory（快 10-15%）

**原始（第 53 行）：**
```python
train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
```

**改为：**
```python
train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, pin_memory=True, drop_last=True)
```

**说明：**
- `pin_memory=True`：锁定内存页，CPU → GPU 数据传输快 10-20%
- `drop_last=True`：丢弃最后不完整的 batch，避免 GPU 混在统计里

---

## 改动位置 2：修改默认参数（推荐值）

**原始（第 26-35 行）：**
```python
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
...
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--intervals", type=int, default=10, help="Intervals for print loss")
```

**改为：**
```python
parser.add_argument("--batchSize", type=int, default=32, help="Training batch sizse")  # 16 → 32
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
...
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use")  # 1 → 8
parser.add_argument("--intervals", type=int, default=20, help="Intervals for print loss")  # 10 → 20
```

**说明：**
- `batchSize 16→32`：GPU 吞吐 +1.5x（如显存允许）
- `threads 1→8`：数据加载 +4x（避免 CPU 成瓶颈）
- `intervals 10→20`：减少日志 I/O 开销

---

## 改动位置 3：（可选）启用混合精度训练 amp（快 20-30%）

**在第 12 行后添加：**
```python
from torch.cuda.amp import autocast, GradScaler
```

**在 optimizer 创建后（第 92 行后）添加：**
```python
scaler = GradScaler()
```

**在训练循环中（第 104-107 行），将：**
```python
output = net(img)
loss = net.loss(output, gt_mask)
loss.backward()
optimizer.step()
```

**改为：**
```python
with autocast():
    output = net(img)
    loss = net.loss(output, gt_mask)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**说明：**
- 自动混合精度（FP16），显存 -40%，速度 +20-30%
- 精度下降 < 1%

---

## 快速应用（不改代码，只用命令行参数）

如果你不想编辑 train.py，直接用这个命令：

```bash
# 快 2 倍（保留精度）
python train.py --model_names ACM ALCNet --dataset_names IRSTD-1K \
  --batchSize 32 --threads 8 --intervals 20

# 快 3-4 倍（稍损精度）
python train.py --model_names ACM ALCNet --dataset_names IRSTD-1K \
  --batchSize 64 --patchSize 128 --threads 8 --nEpochs 200 --intervals 20
```

---

## 验证优化是否生效

运行训练时，观察：
1. **首个 epoch 时间**（第 2-3 个 iteration 开始后记录）
2. **GPU 利用率**（用 `nvidia-smi -l 1` 实时监控，应该 > 80%）
3. **CPU 利用率**（应该 < 50%，说明不是 I/O 瓶颈）

如果 GPU 利用率还很低，说明还有优化空间（可以继续摸大 batchSize）。

---

## 自动化脚本（run_optimized_train.sh / .ps1）

### Windows PowerShell（保存为 run_train_fast.ps1）
```powershell
# 快速验证模式
python train.py --model_names ACM --dataset_names NUAA-SIRST `
  --batchSize 64 --patchSize 128 --threads 8 --nEpochs 50 --intervals 20

# 完整训练模式  
# python train.py --model_names ACM ALCNet --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST `
#   --batchSize 32 --threads 8 --intervals 20
```

运行：`.\run_train_fast.ps1`

### Linux/Mac（保存为 run_train_fast.sh）
```bash
#!/bin/bash
# 快速验证模式
python train.py --model_names ACM --dataset_names NUAA-SIRST \
  --batchSize 64 --patchSize 128 --threads 8 --nEpochs 50 --intervals 20

# 完整训练模式
# python train.py --model_names ACM ALCNet --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST \
#   --batchSize 32 --threads 8 --intervals 20
```

运行：`bash run_train_fast.sh`
