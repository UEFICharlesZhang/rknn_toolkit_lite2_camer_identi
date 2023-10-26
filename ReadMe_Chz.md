Step:
# 1.install rknn_toolkit_lite2
安装与python 版本对应的 whl
```
pip3 install rknn_toolkit_lite2-1.5.2-cp311-cp311-linux_aarch64.whl
```
# 2.下载模型对应的Label
参考https://pytorch.org/hub/pytorch_vision_resnet/
```
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```
下载之后可以翻译为中文。
# 3.修改test脚本，通过摄像头获取图片
