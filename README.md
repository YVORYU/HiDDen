# HiDDen:利用深度网络隐藏数据论文复现

代码地址https://github.com/ando-khachatryan/HiDDeN

论文地址https://arxiv.org/abs/1807.09937

数据集下载 http://cocodataset.org/#download

由于代码比较老，一些库的用法已废弃，该项目对其进行了修改

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

本项目训练共使用10000张图片，其中训练集7000，验证集3000。

## 训练:

**new**

```
python main.py new --name <experiment_name> --data-dir <dataset> --batch-size <b>
```

**continue**

```
python main.py continue  --folder <run_folder>
```

## 测试

```
python test_model.py -o <options-and-config.pickle> -c <checkpoint> -s <source_image> -m <message>
```



