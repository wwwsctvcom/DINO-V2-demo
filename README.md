# DINO-V2-demo
lane detection based DINO V2 model, offering a simple reading and easy usage code for training and predict.


# Dataset
数据集使用kaggle中的数据集`https://www.kaggle.com/datasets/lyhue1991/lanelines`


# Train
训练使用默认参数，基座模型使用facebook/dinov2-base，可以从transformers下载，或者自动通过网络下载，16G内存下
全参训练的batch size设置为5，如果不是全参训练的话，batch size可以尽量设置大一点，训练出来的效果会更好；
```
python train.py
```
如下是训练了9个epoch之后的损失值变化，在代码中可以修改train.py中的代码，可以在每个step计算出iou和acc，但是会导致
训练速度慢很多，可以根据自身情况进行调整；
```
Epoch: 1/9: 100% 750/750 [16:30<00:00, 1.30s/it, lr=8.54e-5, train average loss=0.0611, train loss=0.0456]
Epoch: 2/9: 100% 750/750 [14:56<00:00, 1.19s/it, lr=5e-5, train average loss=0.0375, train loss=0.0329]
Epoch: 3/9: 100% 750/750 [15:00<00:00, 1.20s/it, lr=1.46e-5, train average loss=0.0334, train loss=0.0337]
Epoch: 4/9: 100% 750/750 [15:03<00:00, 1.21s/it, lr=0, train average loss=0.0312, train loss=0.0348]
Epoch: 5/9: 100% 750/750 [15:05<00:00, 1.22s/it, lr=1.46e-5, train average loss=0.0296, train loss=0.0262]
Epoch: 6/9: 100% 750/750 [15:03<00:00, 1.21s/it, lr=5e-5, train average loss=0.0283, train loss=0.025]
Epoch: 7/9: 100% 750/750 [15:06<00:00, 1.21s/it, lr=8.54e-5, train average loss=0.0271, train loss=0.0282]
Epoch: 8/9: 100% 750/750 [15:05<00:00, 1.21s/it, lr=0.0001, train average loss=0.0262, train loss=0.0267]
Epoch: 9/9: 100% 750/750 [15:09<00:00, 1.22s/it, lr=8.54e-5, train average loss=0.0249, train loss=0.0209]
```

# Predict
使用了测试集中的道路图片进行预测，并进行可视化，通过几轮训练，训练之后的模型具备检测道路线的能力，但是效果并不是特别理想，
需要适当的优化，和更多epoch的训练才能使得效果更好，这里只提供了demo，需要更深入的使用需要更多数据更充分的训练；
```
# 代码可以使用默认参数直接运行
python predict.py
```
