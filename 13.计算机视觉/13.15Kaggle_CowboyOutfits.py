# Kaggle竞赛
#   牛仔行头检测
#       数据集: https://www.kaggle.com/c/cowboyoutfits
#       结果提交: https://competitions.codalab.org/competitions/34338#results

# 数据重采样
#   当有类别样本严重不足时, 可以人工干预提升它们对模型的影响力
#   最简单的是将不足的类别样本复制多次
#   在随机采样小批量时对每个类使用不同的采样频率
#   在计算损失时增大不足类别样本的权重
#   使用SMOTE
#       在不足类样本中选择 相近的两个样本 做插值

# 模型
#   YOLOX: YOLOv3 + anchor free
#   YOLOv5: YOLOv3 Pytorch版本的改进版
#       YOLOv4和YOLOv5均是社区改进版, 命名有争议
#   Detectron2: Faster RCNN
#   大都采用了多模型, k则融合

# 总结
#   目标检测代码实现复杂, 训练代价大, 上手仍以找到容易上手的库为主
#   因为超参数多, 一般需要较长时间探索