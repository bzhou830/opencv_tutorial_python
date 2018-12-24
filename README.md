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
## 2. 使用Matplotlib显示图像





