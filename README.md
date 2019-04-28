# vgg
vgg模型进行遥感影像场景分类
其中vgg19.py为模型搭建文件

1.使用时需要先下载预训练模型 [imagenet-vgg-verydeep-19.mat](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)

2.然后利用prepare_data.py制作数据集

3.在train.py中修改相关文件的路径

如下：

'''
pre_vgg19_model = r"imagenet-vgg-verydeep-19.mat"  # 预训练模型

image_pkl = r"image.pkl"  # 图像矩阵

label_pkl = r"label.pkl"  # 标签矩阵
'''

4.训练时直接运行train.py

5.训练结束后，模型文件保存至model文件夹，tensorboard日志文件在logs文件夹，利用tensorboard可以查看loss曲线等


该项目使用的数据集是开放的遥感数据集，可以自行下载
------------------
1.UC Merced Land-Use Data Set
contains 21 scene classes and 100 samples of size 256x256 in each class.

图像像素大小为256*256，总包含21类场景图像，每一类有100张，共2100张

下载地址：http://weegee.vision.ucmerced.edu/datasets/landuse.html
-------------------------------------------------------------------------
2.WHU-RS19 Data Set 
has 19 different scene classes and 50 samples of size 600x600 in each class.

图像像素大小为600*600，总包含19类场景图像，每一类大概50张，共1005张

下载地址：http://dsp.whu.edu.cn/cn/staff/yw/HRSscene.html
-------------------------------------------------------------------------
3.RSSCN7 Data Set

contains 7 scene classes and 400 samples of size 400x400 in each class.

图像像素大小为400*400，总包含7类场景图像，每一类有400张，共2800张

下载地址：https://sites.google.com/site/qinzoucn/documents
-------------------------------------------------------------------------
