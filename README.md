# ATO
It highlights a key bottleneck in knowledge distillation (KD)—the tradeoff between KL loss magnitude and student underfitting. By introducing Adaptive Threshold Optimization (ATO), the paper offers a principled way to learn temperature values, improving knowledge transfer from teacher to student models.
### Dependencies

Python 3.9

PyTorch 2.0

torch==2.0.0

tqdm

yacs

tensorboardX



### CIFAR-100

Download CIFAR-100 dataset to ./data

Download the cifar_teachers.tar and untar it to ./download_ckpts via tar xvf cifar_teachers.tar.

### Training

```python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --adaptive-temperature --base-temp 2 --lambda_threshold 0.3```

### Results

| Dataset   | Teacher  | Student  | Acc@1  |
| --------- | -------- | -------- | ------ |
| CIFAR-100 | wrn_40_2 | wrn_16_2 | 75.62% |

### 
