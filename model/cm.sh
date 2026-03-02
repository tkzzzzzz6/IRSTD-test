# train
# python train.py --model_names ACM ALCNet --dataset_names SIRST-v1

python train.py --model_names ACM ALCNet  --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST

# test
# python test.py --model_names ACM ALCNet --dataset_names SIRST-v1
python test.py --model_names ACM ALCNet --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST

# inference
# python inference.py --model_names ACM --dataset_names SIRST-v1
python inference.py --model_names ACM --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST

# evaluate
# python evaluate.py --model_names ACM --dataset_names SIRST-v1
python evaluate.py --model_names ACM --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST

# 评估更多模型
# python evaluate.py --model_names ACM ALCNet DNANet UIUNet RDIAN ISTDU-Net RISTDnet U-Net \
#   --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST

# ====================  PARAMETERS/FLOPs CALCULATION  ====================
python cal_params.py --model_names ACM ALCNet

# 计算更多模型的参数
# python cal_params.py --model_names ACM ALCNet DNANet UIUNet RDIAN ISTDU-Net RISTDnet U-Net


# ====================  TRAIN  ====================
# 快速模式：快 2 倍，精度基本一致
python train.py --model_names ACM ALCNet --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST \
  --batchSize 32 --threads 8 --nEpochs 300 --intervals 20

# 完整模型对标：添加更多 SOTA 模型
python train.py --model_names ACM ALCNet DNANet UIUNet RDIAN ISTDU-Net RISTDnet U-Net --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST --batchSize 32 --threads 8 --nEpochs 10 --intervals 1

python train.py --model_names  --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST --batchSize 32 --threads 8 --nEpochs 10 --intervals 1
# 原始配置（保留参考）
# python train.py --model_names ACM ALCNet --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST

# ====================  TEST  ====================
python test.py --model_names ACM ALCNet --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST \
  --threads 8

# 测试更多模型（需要先训练）
# python test.py --model_names ACM ALCNet DNANet UIUNet RDIAN ISTDU-Net RISTDnet U-Net \
#   --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST --threads 8

# ====================  INFERENCE  ====================
python inference.py --model_names ACM ALCNet --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST \
  --threads 8

# 推理更多模型
# python inference.py --model_names ACM ALCNet DNANet UIUNet RDIAN \
#   --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST --threads 8

# ====================  EVALUATE  ====================
python evaluate.py --model_names ACM ALCNet --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST

# 评估更多模型
# python evaluate.py --model_names ACM ALCNet DNANet UIUNet RDIAN ISTDU-Net RISTDnet U-Net \
#   --dataset_names IRSTD-1K NUAA-SIRST NUDT-SIRST

# ====================  PARAMETERS/FLOPs CALCULATION  ====================
python cal_params.py --model_names ACM ALCNet

# 计算更多模型的参数
# python cal_params.py --model_names ACM ALCNet DNANet UIUNet RDIAN ISTDU-Net RISTDnet U-Net