# ATO

## Abstract
The temperature plays a pivotal role in knowledge distillation (KD) by controlling the smoothness of outputs for both teacher and student networks, hence paving the way to modulate the Kullback–Leibler (KL) loss for their soft outputs to distill the student.  Most of the existing arts treat temperature as a fixed hyperparameter to minimize the KL loss, such strategy, however, is suboptimal owing to the mismatch between fixed temperature and evolving student during the distillation process; following that, we further revisit the role of temperature upon KL loss, while reveal that both excessively high and low temperatures lead to gradient vanishing or explosion for student's parameters regarding KL loss, resulting in underfitting output for student. In this paper, we investigate how to learn temperature to yield a desirable KL loss for student's distillation. Technically, we propose an **A**daptive **T**hreshold **O**ptimization method, named **ATO**, to comprise two alternating stages: learning temperature and student's parameters, where an adaptive threshold regarding KL loss is intuitively obtained as per the upper and lower bounds of the KL loss, serving as the stopping criterion during the temperature optimization process, to learn an appropriate temperature to alleviate student's underfitting. We theoretically and empirically demonstrate the effectiveness of our proposed ATO and showcase the significant performance gains over state-of-the-art KD models in terms of classification and object detection across various benchmarks over CIFAR-100, ImageNet and MS-COCO. 

### Dependencies

Python 3.9

PyTorch 2.0

torch==2.0.0

tqdm

yacs

tensorboardX



### CIFAR-100

Download CIFAR-100 dataset to ./data

Download the [cifar_teachers.tar](https://github.com/megvii-research/mdistiller/releases/tag/checkpoints) and untar it to ./download_ckpts via tar xvf cifar_teachers.tar.

### Training

```python tools/train.py --cfg configs/cifar100/kd/vgg13_MobileNetV2.yaml --adaptive-temperature --base-temp 2.5 --lambda_threshold 0.6```

### Results

| Dataset   | Teacher  |   Student   | Acc@1  |
| --------- | -------- | ------------| ------ |
| CIFAR-100 |  vgg13   | MobileNetV2 | 68.75% |

### 
