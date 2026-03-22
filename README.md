# HiDDen:利用深度网络隐藏数据论文复现

代码地址https://github.com/ando-khachatryan/HiDDeN

论文地址https://arxiv.org/abs/1807.09937

数据集下载 http://cocodataset.org/#download

数据集目录

```
<data_root>/
  train/
    train_class/
      train_image1.jpg
      train_image2.jpg
      ...
  val/
    val_class/
      val_image1.jpg
      val_image2.jpg
      ...
```

本项目训练共使用10000张图片，其中测试集7000，验证集3000

由于代码比较老，一些库的用法已废弃，该项目对其进行了修改