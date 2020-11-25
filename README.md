# Pytorch-K-Means
### Homework : Using K-Means to handel pictures with GUI

## 环境
torch

numpy

cv2

matplotlib

PySimpleGUI

easygui


## 使用
python main.py即可

## 基本思想
1.读取图片，不论RGB，像素点本身二维压成一维

2.随机选取K个点，分别对应K类

3.每个点都和这K个点的RGB进行比较，找到最接近的那个，标记为同类

4.重复二三步，直到每个点的标签不再变动

5.对每一类的点的RGB取平均，得到结果


## 难点
主要难点在于如何取出同类的点并分别做取平均操作

## 解决方法
1.为了取出同类的点，将得到的label扩充三份，置换，这样就可以靠where取得某类别的RGB了

2.为了区分原本RGB中的0和被过滤而变成的0，将原本的0改为1e-9（10的-9次）

3.sum求和，nonzero配合len得到数量。由于做了区分，不会在求平均时漏算原本的0或多算过滤出的0

4.得到结果后，同类的点都换成均值（依然保持其他类别的点为0）。最后所有类别的结果累加即可

