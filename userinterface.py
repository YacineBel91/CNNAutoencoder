import sys

import torch
from PyQt5 import QtWidgets as qt

from mainwindow import Ui_MainWindow


class AdamOptim:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, params):
        return torch.optim.Adam(params, **self.kwargs)


class ASGDOptim:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, params):
        return torch.optim.ASGD(params, **self.kwargs)


class RMSPropOptim:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, params):
        return torch.optim.RMSprop(params, **self.kwargs)


class Window(qt.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.setupUi(self)

        self.newAutoEncoderSaveButton.clicked.connect(self.saveAE)

        # The goal of this dictionary is to properly link a user's choice (the optimizer) with the data to be fed
        # Each choice has a tuple associated with it. First element of this tuple is a lambda returning a dict with the latest input data needed for the selected optimizer.
        # Second element is the tab (widget) that needs to be shown to the user according to his choice of optimizer
        # Third element is the optimizer factory that must be created when the user confirms he wants to use the given optimizer
        self.optimizers = {"Adam": (lambda:
                                    {
                                        "lr": self.AdamLearningRate.value()
                                    },
                                    self.AdamTab, AdamOptim),
                           "ASGD": (lambda:
                                    {
                                        "lr": self.ASGDLearningRate.value(),
                                        "lambd": self.ASGDlambd.value()
                                    },
                                    self.ASGDTab, ASGDOptim),
                           "RMSProp": (lambda:
                                       {
                                           "lr": self.RMSPropLearningRate.value(),
                                           "momentum": self.RMSPropMomentum.value()
                                       },
                                       self.RMSPropTab, RMSPropOptim)
                           }

        self.optimizerCombobox.clear()
        self.optimizerCombobox.addItems(self.optimizers.keys())# Fill the combobox with the exact optimizer name (the names give access to the tab and data)

        # Whenever the user selects another optimizer through the combobox, we show him the corresponding tab
        self.optimizerCombobox.currentTextChanged.connect(lambda s:self.optimizerSelectTab.setCurrentWidget(self.optimizers[s][1]))

        #Whenever the user selects another optimizer tab, we make sure the combobox displays the related information
        self.optimizerSelectTab.currentChanged.connect(lambda i:self.optimizerCombobox.setCurrentIndex(i))

    def saveAE(self):
        self.aeFilePath = qt.QFileDialog.getSaveFileName(self, "Veuillez choisir un nom")

    def trainAE(self):
        optimOptions = self.optimizers[self.optimizerCombobox.currentText()][0]()
        optimFactory = self.optimizers[self.optimizerCombobox.currentText()][2](**optimOptions)
        return


def main():
    app = qt.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
