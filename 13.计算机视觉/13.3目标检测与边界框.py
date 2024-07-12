# 目标检测
#   从图像中找到所有感兴趣的物体, 并用边缘框将其表示出来
# 边缘框(bounding box): 左上为原点(0,0), 向下y增加, 向右x增加
#   一个边缘框可以通过4个数字定义
#       左上x, 左上y, 右下x, 右下y
#       左上x, 左上y, 宽, 高
# 目标检测数据集
#   每行表示一个物体
#       图片文件名, 物体类别, 边缘框
#   COCO
#       80物体, 330K图片, 1.5M物体
# 总结
#   目标检测识别图片里的多个物体的类别和位置
#   位置通常用边缘框表示

# 边缘框的实现
import torch
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img)
d2l.plt.show()

# 定义边缘框两种表示之间进行转换的函数
# 从(左上, 右下)转换到(中间, 宽度, 高度)
#@save
def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes
# 从(中间, 宽度, 高度)转换到(左上, 右下)
#@save
def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# 定义图像中猫和狗的边界框
# bbox是边界框的英文缩写(左上和右下的坐标)
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

# 通过转换两次来验证边界框转换函数的正确性
boxes = torch.tensor((dog_bbox, cat_bbox))
print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

# 将边界框在图中画出
#@save
def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
d2l.plt.show()