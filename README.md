# Pytorch-K-Means
### Homework : Using K-Means to handel pictures with GUI

## 环境
torch，numpy，cv2，matplotlib，PySimpleGUI，easygui

## 使用
python main.py即可

## 目标
利用 K means，让图片中颜色相近的点变成相同的颜色，而最终的颜色就由这些点求均值得到

## 基本思想
1.读取图片，不论RGB，像素点本身二维压成一维

2.随机选取K个点，分别对应K类

3.每个点都和这K个点的RGB进行比较，找到最接近的那个，标记为同类

4.对每类求均值得新K点(平均RGB，并非图像中的点)，回到第三步，直到每个点的标签不再变动

5.对每一类的点的RGB取平均（直接用最终的K点赋值即可）

6.叠加所有类别的点，重新排成原长宽，得到结果图

## 难点
1.如何高效地判断每个点于K点之间地色差

2.如何取出同类的点并分别做取平均操作

## 解决方法
1.将每个K点的值复制出一整张完整的矩阵，这样得到的K张矩阵直接和原图做差求均值，就能批量得到色差了

2.将得到的label扩充三份，这样就可以靠where取得某类别的点了。K点及均值，直接赋值即可

## 可能是BUG
easygui调用路径选择框偶尔会造成explorer未响应

## 为什么在展示时不直接plt.show()
找不到不影响show的同时还能在关闭窗口后继续运行的方法

## 为什么有俩GUI的库
懒
