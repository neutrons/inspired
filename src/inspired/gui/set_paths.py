import os
import sys

from qtpy.QtWidgets import QDialog, QFileDialog

from inspired.gui.ui_set_paths import Ui_SetPaths


class SetPaths(QDialog):
    def __init__(self):
        super(SetPaths, self).__init__()
        self.ui = Ui_SetPaths()
        self.ui.setupUi(self)

    def init_paths(self, predictor_path, dft_database_path):
        self.predictor_path = predictor_path
        self.dft_database_path = dft_database_path

        self.ui.lineEdit_dp_path.setText(predictor_path)
        self.ui.lineEdit_dft_path.setText(dft_database_path)

    def browse_dftdb(self):
        wd = QFileDialog.getExistingDirectory(self, "Path to DFT database", "")
        if wd:
            self.dft_database_path = wd
            self.ui.lineEdit_dft_path.setText(wd)

    def browse_dp_model(self):
        wd = QFileDialog.getExistingDirectory(self, "Path to predictor models", "")
        if wd:
            self.predictor_path = wd
            self.ui.lineEdit_dp_path.setText(wd)

    def save_paths(self):
        self.predictor_path = self.ui.lineEdit_dp_path.displayText()
        self.dft_database_path = self.ui.lineEdit_dft_path.displayText()
        with open(os.path.join(os.path.expanduser("~"), ".config", "inspired", "config"), "w") as f:
            f.write("predictor_path:" + self.predictor_path + "\n")
            f.write("dft_database_path:" + self.dft_database_path + "\n")
        self.close()

    def quit_inspired(self):
        sys.exit()
