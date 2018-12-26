## **<OpenCV轻松学> Python版**

## 1. 安装OpenCV package for Python

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

### 2.1 读取，显示，保存图像

```python
import numpy as np
import cv2

img = cv2.imread('messi5.jpg',0)
cv2.imshow('image', img)
k = cv2.waitKey(0) & 0xFF

if k == 27: # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png', img)

cv2.destroyAllWindows()
```

也可以使用matplotlib显示图像

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()
```
彩色图像使用 OpenCV 加载时是 BGR 模式。但是 Matplotib 是 RGB
模式。所以彩色图像如果已经被 OpenCV 读取，那它将不会被 Matplotib 正
确显示。这时候需要对通道的顺序进行调整。

### 2.2 摄像头图像获取和显示

```python
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
```








