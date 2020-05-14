# pedestrain-detection-towncentre
基于Tensorflow1的行人检测，使用TownCentre数据集。

# 1. 数据集介绍
该实验提供一个小型行人检测数据集TownCentre，该数据集包含一个视频TownCentreXVID.avi和标签文件TownCentre-groundtruth.top。

- 数据集网站：[link](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html#datasets)

- TownCentreXVID.avi下载：[link](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi)

- TownCentre-groundtruth.top下载：[link](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentre-groundtruth.top)

其中TownCentreXVID.avi一共5 min，每1 sec包含25帧图像（1920*1080），因此一共包含7500帧图像；

TownCentre-groundtruth.top包含前4500帧图像中行人的位置信息，每一行信息组织格式如下：

personNumber, **frameNumber**, headValid, bodyValid, headLeft, headTop, headRight, headBottom, **bodyLeft, bodyTop, bodyRight, bodyBottom**

- personNumber - A unique identifier for the individual person 
- **frameNumber - The frame number (counted from 0)**
- headValid - 1 if the head region is valid, 0 otherwise
- bodyValid - 1 if the body region is valid, 0 otherwise
- headLeft,headTop,headRight,headBottom - The head bounding box in pixels
- **bodyLeft,bodyTop,bodyRight,bodyBottom - The body bounding box in pixels**

对于行人检测，我们主要需要上述加粗的数据。

# 2. 实验环境设置
该实验基于TensorFlow Object Detection API实现对行人的检测，实验环境配置:
- 操作系统：Windows10
- GPU：NVIDIA GTX 1060
- CUDA+CUDNN：10.0 + 7.6.5
- Tensorflow版本：gpu-1.13.1

使用下面的语句安装依赖库：

```
pip3 install pillow lxml matplotlib contextlib2 cython opencv-python
```

该实验需要下载TensorFlow Object Detection API，直接在tensorflow的github上clone的是最新版本的models（ > Tensorflow2），在链接[link](https://github.com/tensorflow/models/archive/v1.13.0.zip)可以下载1.13对应版本的models。之后会使用models/research/目录下的object_detection和slim目录。

编译COCO API，由于cocodataset自带的coco是不支持Windows的，在链接[link](https://github.com/jmhIcoding/pycocoapi.git)内下载pycocotools文件夹，将其拷贝到models/research目录下。

Windows 10下可以直接下载已经编译好的protoc可执行文件，一定要安装3.4.0版本的protoc才不会报错，protoc的下载链接见[link](https://github.com/protocolbuffers/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip)。

使用protoc编译接口,将下载的protoc.exe文件置于models\research目录下，运行：

```
protoc.exe .\object_detection\protos\*.proto --python_out=.
```

新建一个用户变量：PYTHONPATH，它的值是models\research 和models\research\slim所在的目录。

# 3. 数据预处理

### (1)从视频中提取图像
参考代码avi2jpg.py。

将视频TownCentreXVID.avi的每一帧提取出来，共7500帧，前4500帧作为训练集，存储在train_images文件夹中；后3000帧作为测试集，存储在test_images文件夹中。

视频中的图像尺寸是1920×1080，提取时缩小为1/2。

### (2)从标注数据中导出所需数据
参考代码top2csv.py。

将TownCentre-groundtruth.top中的标注数据导出成csv格式。

4500张训练集中，训练集、验证集的划分比例为95：5。

得到文件“pedestrain_train.csv”和“pedestrain_valid.csv”。

### (3)把数据转换成record格式
把数据“pedestrain_train.csv”和“pedestrain_valid.csv”转换成record格式，得到文件“train.record”和“valid.record”。

# 4. 训练模型

### (1)下载预训练模型
使用在MS COCO数据集上预训练的Faster R-CNN + Inceptionv2模型，模型下载见链接[link](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)。

### (2)编写label_map.pbtxt
用记事本将下列代码保存为“label_map.pbtxt”：

```
item {
id :1
name : ‘pedestrain’
}
```

### (3)编写config文件
见“pipeline.config”文件。

需要修改的五个路径如下：

```
......
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "D:\\Python\\pedestrain_detection\\pedestrain_detection\\pretrained\\faster_rcnn_inception_v2_coco_2018_01_28\\model.ckpt"
  from_detection_checkpoint: true
  num_steps: 2000
}
train_input_reader {
  label_map_path: "D:\\Python\\pedestrain_detection\\pedestrain_detection\\anotation\\label_map.pbtxt"
  tf_record_input_reader {
    input_path: "D:\\Python\\pedestrain_detection\\pedestrain_detection\\train.record"
  }
}
eval_config {
  num_examples: 250
  max_evals: 10
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "D:\\Python\\pedestrain_detection\\pedestrain_detection\\anotation\\label_map.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "D:\\Python\\pedestrain_detection\\pedestrain_detection\\valid.record"
  }
}
```

### (4)模型训练
在“\models\research”路径下命令行窗口输入下列语句进入训练：

```
python object_detection\model_main.py --pipeline_config_path=D:\Python\pedestrain_detection\pedestrain_detection\pipeline.config --model_dir=D:\Python\pedestrain_detection\pedestrain_detection\train --num_train_steps=2000 --sample_1_of_n_eval_eval_examples=1 --alsologtostderr
```

### (5)导出模型
本实验使用第1241次训练的结果导出模型。

在“\models\research”路径下命令行窗口输入下列语句导出模型：

```
python object_detection\export_inference_graph.py --pipeline_config_path=D:\Python\pedestrain_detection\pedestrain_detection\pipeline.config --input_type=image_tensor --trained_checkpoint_prefix=D:\Python\pedestrain_detection\pedestrain_detection\train\model.ckpt-1241 --output_directory=D:\Python\pedestrain_detection\pedestrain_detection\train\save_model\\
```

# 5. 预测结果
参考代码“inference.py”。预测视频链接：[B站](https://www.bilibili.com/video/BV19K4y147Ke)
