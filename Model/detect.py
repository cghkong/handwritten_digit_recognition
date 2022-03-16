import torch
import cv2
from digitsrecognitionAPP.Model.model import DeepConvNet
import numpy as np
from digitsrecognitionAPP.common.functions import softmax
import torchvision.transforms as T
from PIL import Image
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Recongizer:
    # 1. 初始化操作
    def __init__(self, model_file="deep_convnet_params.pkl"):
        super(Recongizer, self).__init__()
        # 1. 实例化模型
        self.network = DeepConvNet()
        self.network.load_params(model_file)

    # 2.定义一个图像预处理函数
    def pre_img(self, img):
        # 2. 颜色转换BGR2Gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 3. 进行类型的转换
        img = img.astype('float32')
        # 5. 去噪
        img[img >= 95] = 255
        # cv2.imshow("img",img.astype('unit8'))
        # cv2.waitKey(0)
        cv2.imwrite("test1.jpg", img.astype('uint8'))


    def accessPiexl(self,img):
        height = img.shape[0]
        width = img.shape[1]
        for i in range(height):
            for j in range(width):
                img[i][j] = 255 - img[i][j]
        return img

    # 反相二值化图像
    def accessBinary(self,img, threshold=128):
        img = self.accessPiexl(img)
        # 边缘膨胀，不加也可以
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
        return img

    def findBorderContours(self,img,maxArea=60):
        contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        borders = []
        for contour in contours:
            # 将边缘拟合成一个边框
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > maxArea:
                border = [(x, y), (x + w, y + h)]
                borders.append(border)
        borders = sorted(borders,key=lambda t: t[0][0])
        return borders

    def transMNIST(self,img,borders, size=(28, 28)):
        imgData = np.zeros((len(borders), size[0], size[0]), dtype='uint8')
        for i, border in enumerate(borders):
            borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
            # 根据最大边缘拓展像素，扩大边缘边框，固定值0填充，变成正方形
            # top, bottom, left, right：上下左右要扩展的像素数
            extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
            # 扩大多少由你的原图数字大小决定，使数字处在图片中间，图片越大，扩大的就大一些
            targetImg = cv2.copyMakeBorder(borderImg, 30, 30, extendPiexl + 30, extendPiexl + 30, cv2.BORDER_CONSTANT)
            targetImg = cv2.resize(targetImg, size)
            #targetImg = np.expand_dims(targetImg, axis=-1)
            imgData[i] = targetImg
        return imgData

    # 3. 定义识别图像函数
    def recongizer(self,filename):
        image = cv2.imread(filename)
        self.pre_img(image)
        img = cv2.imread('test1.jpg', cv2.IMREAD_GRAYSCALE)
        pre_img = self.accessBinary(img)
        cv2.imwrite("test2.jpg", pre_img.astype('uint8'))
        # 分割处理
        borders = self.findBorderContours(pre_img)
        imgData = self.transMNIST(pre_img,borders)
        # 2. 预测图像
        imgdata = torch.Tensor(imgData).view(imgData.shape[0],1,imgData.shape[1],imgData.shape[2])
        class_id = []
        for i in range(len(imgdata)):
            img_data = np.array(imgdata[i],dtype=np.float32)
            img_newdata = torch.from_numpy(img_data).clone()
            y = self.network.predict(np.array(img_newdata.view(1,1,28,28)))
            __result = softmax(y)
            res = np.argmax(__result)
            class_id.append(res)
        str_res = [str(i) for i in class_id]
        result = "".join(str_res)
        return result

'''
rec = Recongizer()
#image = cv2.imread('photo/2157.jpg',cv2.IMREAD_GRAYSCALE)
class_id= rec.recongizer()
print(class_id)
'''