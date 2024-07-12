# 多尺度目标检测

import torch
from d2l import torch as d2l

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
print(h, w)

# 在特征图(fmap)上生成锚框(anchors), 每个单位(像素)作为锚框的中心
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes, anchors[0] * bbox_scale)
    d2l.plt.show()

# 相当于先把原图抽象为4×4的图像，然后以这16个像素为中心画锚框，然后再将其映射回原图像大小
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
# 同上
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])

