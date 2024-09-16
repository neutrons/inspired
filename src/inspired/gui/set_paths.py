from qtpy.QtWidgets import (QFileDialog, QDialog)
from inspired.gui.ui_set_paths import Ui_SetPaths
import os
import sys

class SetPaths(QDialog):
    def __init__(self):
        super(SetPaths, self).__init__()
        self.ui = Ui_SetPaths()
        self.ui.setupUi(self)

    def init_paths(self,predictor_path,dft_database_path,mace_model_file,m3gnet_model_path):
        self.predictor_path = predictor_path
        self.dft_database_path = dft_database_path
        self.mace_model_file = mace_model_file
        self.m3gnet_model_path = m3gnet_model_path

        self.ui.lineEdit_dp_path.setText(predictor_path)
        self.ui.lineEdit_dft_path.setText(dft_database_path)
        self.ui.lineEdit_mace_path.setText(mace_model_file)
        self.ui.lineEdit_m3gnet_path.setText(m3gnet_model_path)

    def browse_dftdb(self):
        wd = QFileDialog.getExistingDirectory(self, 'Path to DFT database', '')
        if wd:
            self.dft_database_path = wd
            self.ui.lineEdit_dft_path.setText(wd)

    def browse_dp_model(self):
        wd = QFileDialog.getExistingDirectory(self, 'Path to predictor models', '')
        if wd:
            self.predictor_path = wd
            self.ui.lineEdit_dp_path.setText(wd)

    def browse_mace(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Choose file',
                '',"MACE model file (*)")
        if fname:
            self.mace_model_file = fname
            self.ui.lineEdit_mace_path.setText(fname)

    def browse_m3gnet(self):
        wd = QFileDialog.getExistingDirectory(self, 'Path to M3GNet models', '')
        if wd:
            self.m3gnet_model_path = wd
            self.ui.lineEdit_m3gnet_path.setText(wd)

    def save_paths(self):
        self.predictor_path = self.ui.lineEdit_dp_path.displayText()
        self.dft_database_path = self.ui.lineEdit_dft_path.displayText()
        self.mace_model_file = self.ui.lineEdit_mace_path.displayText()
        self.m3gnet_model_path = self.ui.lineEdit_m3gnet_path.displayText()
        with open(os.path.join(os.path.expanduser('~'),'.config','inspired','config'), 'w') as f:
             f.write('predictor_path:'+self.predictor_path+'\n')
             f.write('dft_database_path:'+self.dft_database_path+'\n')
             f.write('mace_model_file:'+self.mace_model_file+'\n')
             f.write('m3gnet_model_path:'+self.m3gnet_model_path+'\n')
        self.close()

    def quit_inspired(self):
        sys.exit()
