from PyQt5.QtWidgets import QApplication
from digitsrecognitionAPP.digitsGUI.MainWindow import MainWindows
import sys

class APP():
    def __init__(self):
        super(APP, self).__init__()
        app = QApplication(sys.argv)
        window = MainWindows()
        window.show()
        sys.exit(app.exec_())
