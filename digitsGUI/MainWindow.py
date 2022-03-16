from PyQt5.QtWidgets import QDialog,QMainWindow,QApplication
from PyQt5.QtGui import QPixmap,QImage
import sys, os
import numpy as np
from digitsrecognitionAPP.digitsforms.handwritingdigits import Ui_MainWindow
from digitsrecognitionAPP.digitsGUI.mnist_cnn_gui_main import MainWindow
from digitsrecognitionAPP.digitsGUI.digitsrecognitionfream import MainForm


class MainWindows(QMainWindow,Ui_MainWindow):

    def __init__(self):
        super(MainWindows,self).__init__()
        self.Ui = Ui_MainWindow()
        self.Ui.setupUi(self)

    def chooseone(self):
        Gui = MainWindow()
        Gui.exec()

    def choosetwo(self):
        Gui = MainForm()
        Gui.exec()
