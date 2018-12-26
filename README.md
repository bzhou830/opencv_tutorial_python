## **<OpenCV轻松学> Python版**

## 1. 安装OpenCV package

直接使用pip包管理工具

```python
pip install numpy
pip install matplotlib
pip install python-OpenCV
```

准备一张图片，试试安装的是否成功

```python
import cv2
import numpy as np

from matplotlib import pyplot as plt

img =cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## 2. OpenCV中的GUI相关操作

- 读入图像，视频文件，摄像头
- 显示图像，播放视频文件，播放摄像头
- 保存图像，保存视频文件
cv2.imread()，cv2.imshow()，cv2.imwrite()
cv2.VideoCapture()，cv2.VideoWrite()


安装MatPlotlib
```python
pip install matplotlib
```



