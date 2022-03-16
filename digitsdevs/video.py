from PyQt5.QtCore import QThread, pyqtSignal
import cv2


class CameraVideo(QThread):
    sig_video = pyqtSignal(int, int, int, bytes)

    def __init__(self):
        super(CameraVideo, self).__init__()
        # 1.获取设备
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # 2.定义一个是否获取图像的标识
        self.is_over = False

    def run(self):
        while not self.is_over:
            # 获取视频中每一帧
            status, fream = self.cap.read()
            shape = fream.shape  # 获取图像的h，w,c
            # 图片颜色顺序的转换
            img = cv2.cvtColor(fream, cv2.COLOR_BGR2RGB)
            self.sig_video.emit(shape[0], shape[1], shape[2], img.tobytes())
            self.msleep(100)

    def close(self):
        self.is_over = True
        if self.cap.isOpened():
            self.cap.release()




