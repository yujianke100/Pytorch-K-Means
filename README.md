# Pytorch-K-Means
### Homework : Using K-Means to handle pictures with GUI

## UPDATE 12.14.2020
自己笔记本是低压5代i5，跑pytorch太慢，一怒之下用numpy重写了一份

## 环境
torch（*with_pytorch需要*），numpy，cv2，matplotlib，PySimpleGUI，easygui

pytorch以外的库，可直接pip install -r requirements.txt安装。

[Pytorch官网安装](https://pytorch.org/get-started/locally/)

## 使用
python {name}.py即可，后者结果和比较图会存在result文件夹中

## 目标
1.利用 K means， 将随机生成的点按距离进行分类

2.利用 K means，图片中颜色相近的点变成相同的颜色，而最终的颜色就由这些点求均值得到

## 基本思想
### 以下都为img的实现思想，随机点的思路相同,不多赘述。

1.读取图片，不论RGB，像素点本身二维压成一维

2.随机选取K个中心点，分别对应K类

3.每个点都和这K个点的RGB进行比较，找到最接近的那个，标记为同类

4.对每类RGB求均值得K个新的中心点(平均RGB，并非图像中的点)，

5.重复三四步，直到每个点的标签不再变动

6.对每一类的点的RGB取平均（直接用最终的中心点赋值即可）

7.叠加所有类别的点，重新排成原长宽，得到结果图

8.每次类型聚合结束后都和上一次的结果进行比较。由于需要进度条来体现它在动，需要用一个具体的数值判断进度。所以前后标签矩阵比较一下sum一下得到相差的数量。算点的就直接.all()放if里跳出循环就行

## 细节
1.嫌麻烦，直接全员DoubleTensor(pytorch版本)

2.由于不需要更改原矩阵的数值，与中心点做差的矩阵用expand扩充得到就行(numpy只能expand_dims配合repeat)

3.利用tensor的性质，不需要对中心点的矩阵做扩充。适当位置加个维度就行（pytorch版本）

4.距离没有用正常的欧氏距离，直接做差绝对值求和以减少计算量（其实没啥区别）

5.为了让进度条更直观，只在最后百分之五的进度上进行显示

6.既然where没法1取3，那就把label扩充三份转置一下再取

7.偷懒，统计循环次数的变量直接global，呵呵

8.直接imread没法读中文路径，改用imdecode

## 可能是BUG
easygui调用路径选择框偶尔会造成explorer未响应

## 为什么在展示时不直接plt.show()
找不到不影响show的同时还能在关闭窗口后继续运行的方法。应该是和GUI的库有什么冲突，因为打点的demo可以这么干。不多折腾了就这样吧。

## 为什么有俩GUI的库
懒
