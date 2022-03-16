from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QPixmap,QImage
from digitsrecognitionAPP.digitsforms.digits_ui import Ui_Dialog
from digitsrecognitionAPP.digitsdevs.video import CameraVideo
from digitsrecognitionAPP.Model.detect import Recongizer
import numpy as np
import cv2


class MainForm(QDialog):
    def __init__(self):
        super(MainForm,self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.is_cap = False
        self.rec = Recongizer("deep_convnet_params.pkl")

    def openvideo(self):
        self.dev = CameraVideo()
        self.dev.start()
        self.dev.sig_video.connect(self.show_video)

    def capture(self):
        self.is_cap = True

    def rec(self):
        class_id= self.rec.recongizer("test.jpg")
        result = F"{class_id}"
        self.ui.label.setText(result)

    def show_video(self,h,w,c,data):
        img = QImage(data,w,h,w*c,QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        if  self.is_cap:
            # self.cap_img = np.X(data,np.uint8)
            # self.cap_img = cv2.imdecode(self.cap_img,cv2.IMREAD_COLOR)
            self.cap_img = np.ndarray(
                (h,w,c),
                np.uint8,
                data
            )
            cv2.imwrite("test.jpg", self.cap_img.astype('uint8'))
            self.ui.img_label.setPixmap(pix)
            self.ui.img_label.setScaledContents(True)
            self.is_cap = False
        self.ui.cap_label.setPixmap(pix)#将图像数据产给标签
        self.ui.cap_label.setScaledContents(True)

    def closeEvent(self,e):
        self.dev.close()
        self.ui.cap_label.clear()