from PyQt5 import QtWidgets as qt
import sys

from mainwindow import Ui_MainWindow

class Window(qt.QMainWindow,Ui_MainWindow):

    def __init__(self,parent=None):
        super(Window,self).__init__(parent)
        self.setupUi(self)

        self.newAutoEncoderSaveButton.clicked.connect(lambda: qt.QFileDialog.getSaveFileName(self, "Veuilez"))
        self.addImagesButton.clicked.connect(self.openImages)
        self.newAutoEncoderSaveButton.clicked.connect(self.saveAE)

    def openImages(self):
        filePaths,_=qt.QFileDialog.getOpenFileNames(self,caption="Veuillez choisir une ou plusieurs images")
        print(filePaths)

    def saveAE(self):
        self.aeFilePath = qt.QFileDialog.getSaveFileName(self,"Veuillez choisir un nom")

def main():
    app = qt.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
