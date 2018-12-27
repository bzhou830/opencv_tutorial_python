## **<OpenCV轻松学> Python版**

## 0. 准备

### 1. 使用jupyter + RISE制作课件

```python
pip install jupyter

pip install RISE

jupyter-nbextension install rise --py --sys-prefix
jupyter-nbextension enable rise --py --sys-prefix

```

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

VideoCapture 是 opencv 中封装的一个视频/摄像头操作的对象。他的参数可以是
设备的索引号，也可以是一个视频文件。设备索引号就是指定要使用的摄像头的ID。
一般的笔记本电脑都有内置摄像头, 那么这个摄像头的ID就是 0。也可以通过设置成 1 或
者其他的来选择别的摄像头。

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
想要读取文件只需要将上述中cv2.VideoCapture(0)中传入的参数改成视频文件的路径即可。

### 2.3 视频保存

关于读取摄像头，并且将获取图像写到视频文件中，可以使用如下的方法：
```python
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,0)
        # write the flipped frame
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
```

## 3. OpenCV中绘图

- 使用OpenCV中绘制几何图形的相关汗
- cv2.line(), cv2.circle(), cv2.rectangle(), cv2.ellipse(), cv2.putText()
- 上述函数使用相似的参数：
  - img：你想要绘制图形的那幅图像。
  - color：形状的颜色。以 RGB 为例，需要传入一个元组，例如： （255,0,0 ）
代表蓝色。对于灰度图只需要传入灰度值。
  - thickness：线条的粗细。如果给一个闭合图形设置为 -1，那么这个图形
就会被填充。默认值是 1.
  - linetype：线条的类型，8 连接，抗锯齿等。默认情况是 8 连接。cv2.LINE_AA
为抗锯齿，这样看起来会非常平滑。

### 3.1 画线

```python
import numpy as np
import cv2

# Create a black image
img=np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
cv2.line(img,(0,0),(511,511),(255,0,0),5)
```

### 3.2 画矩形

```python
import numpy as np
import cv2

# Create a black image
img=np.zeros((512,512,3), np.uint8)

cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
```

### 3.3 画圆
```python
import numpy as np
import cv2

# Create a black image
img=np.zeros((512,512,3), np.uint8)

cv2.circle(img,(447,63), 63, (0,0,255), -1)
```

### 3.4 画椭圆
```python
import numpy as np
import cv2

# Create a black image
img=np.zeros((512,512,3), np.uint8)

cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
```

### 3.5 画多边形
```python
import numpy as np
import cv2

# Create a black image
img=np.zeros((512,512,3), np.uint8)

pts=np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts=pts.reshape((-1,1,2))
# 这里 reshape 的第一个参数为 -1, 表明这一维的长度是根据后面的维度的计算出来的。
```

### 3.6 画多边形

要在图片上绘制文字，你需要设置下列参数：
- 你要绘制的文字
- 你要绘制的位置
- 字体类型（通过查看 cv2.putText() 的文档找到支持的字体）
- 字体的大小
- 文字的一般属性如颜色，粗细，线条的类型等。为了更好看一点推荐使用
linetype=cv2.LINE_AA。

在图像上绘制白色的 OpenCV。
```python
import numpy as np
import cv2

# Create a black image
img = np.zeros((512,512,3), np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV', (10, 500), font, 4, (255, 255, 255), 2)
# 这里 reshape 的第一个参数为 -1, 表明这一维的长度是根据后面的维度的计算出来的。
```
