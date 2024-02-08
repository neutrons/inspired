import sys
import os
import warnings

warnings.simplefilter("ignore")

from qtpy.QtWidgets import (QApplication, QMainWindow, QFileDialog, QHeaderView, QDialog, QAbstractItemView)
from ui_inspired import Ui_INSPIRED
from ui_help_dp import Ui_Help_DP
from ui_help_dft import Ui_Help_DFT
from ui_help_mlff import Ui_Help_MLFF
from dp_worker import DPWorker
from dft_worker import DFTWorker
from mlff_worker import MLFFWorker
from pandas_model import PandasModel
from ase.io import read
from ase.spacegroup import get_spacegroup
from ase.formula import Formula
import torch

class INSPIRED(QMainWindow):
    def __init__(self, parent=None):
        super(INSPIRED, self).__init__(parent)
        self.ui = Ui_INSPIRED()
        self.ui.setupUi(self)
        self.dp_worker = DPWorker()
        self.dft_worker = DFTWorker()
        self.mlff_worker = MLFFWorker()

        self.atoms = None
        self.pred_type = "Phonon DOS"
        self.show_df = self.dft_worker.get_initial_search_df()
        self.ui.tableView.setModel(PandasModel(self.show_df))
        for i in range(3):
            self.ui.tableView.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.ui.tableView.verticalHeader().setVisible(False)
        self.ui.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ui.radioButton_database.setChecked(True)
        self.ui.lineEdit_lmin_mlff.setText('12')
        self.ui.lineEdit_fmax_mlff.setText('0.01')
        self.ui.lineEdit_nmax_mlff.setText('100')
        self.ui.lineEdit_delta_mlff.setText('0.03')

        self.root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(self.root_path, 'dftdb')
        self.cwd_path = os.getcwd()
        self.ui.label_cwd.setText("Current Working Directory: "+self.cwd_path)
        self.userdft_path = self.cwd_path


    def setup_working_directory(self):
        wd = QFileDialog.getExistingDirectory(self, 'Working directory', '')
        if wd:
            self.cwd_path = wd
        self.ui.statusbar.showMessage(f"Working directory is set to {self.cwd_path}.", 1000)
        self.ui.label_cwd.setText("Current Working Directory: "+self.cwd_path)
        os.chdir(self.cwd_path)


    def select_structure_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', 
                '',"Structure files (*)")
              #  '',"CIF files (*.cif)")
        if fname:
            return fname
        else:
            return None

    def read_structure_file(self,filename):
        if filename.endswith('.cif'):
            self.atoms = read(filename, format='cif')
        elif 'POSCAR' in filename or 'CONTCAR' in filename:
            self.atoms = read(filename, format='vasp')
        else:
            print('ERROR: File format not supported yet (currently support cif and VASP format)')
        if self.atoms:
            sg = get_spacegroup(self.atoms)
            form = self.atoms.get_chemical_formula()
            cell = self.atoms.cell
            print('INFO: Crystal structure loaded successfully')
            print('INFO: Cell lengths: ',cell.lengths())
            print('INFO: Cell angles: ',cell.angles())
            print('INFO: Cell volume: ',cell.volume)
            print('INFO: Number of atoms in cell: ',len(self.atoms.numbers))
            print('INFO: Chemical formula: ',Formula(form).reduce()[0])
            print('INFO: Space group: ',sg.no,sg.symbol)
    
    # Predictor tab

    def upload_structure_dp(self):
        fname = self.select_structure_file()
        if fname:
            try:
                self.read_structure_file(fname)
                #torch.set_default_dtype(torch.float64)
                self.ui.label_structure_dp.setText(f"Structure file: {fname}")
                self.ui.statusbar.showMessage("")
            except Exception as e:
                self.ui.statusbar.showMessage(f"Error: {e.__repr__()} when loading {fname}")

    def predict_selection_changed(self, *args, **kwargs):
        sender_name = self.sender().objectName()
        if sender_name == 'radioButton_predict_phonon_dos':
            self.pred_type = "Phonon DOS"
        elif sender_name == 'radioButton_predict_vision':
            self.pred_type = "VISION spectrum"
        else:
            self.pred_type = "S(|Q|, E)"

    def run_prediction(self):
        if self.atoms:
            print('INFO: Prediction started...')
            self.dp_worker.predict(struc=self.atoms, pred_type=self.pred_type, partial_dos=self.ui.checkBox_partial_dos.isChecked())
            print('INFO: Prediction finished.')
            #self.ui.statusbar.showMessage("Prediction done")
        else:
            print('ERROR: Select structure file first')

    def plot_spec_dp(self):
        self.dp_worker.savenplot(cwd=self.cwd_path,partial_dos=self.ui.checkBox_partial_dos.isChecked(),unit=self.ui.comboBox_plot_unit_dp.currentIndex(),
                                 setrange=self.ui.checkBox_range_dp.isChecked(),interactive=self.ui.checkBox_interactive_dp.isChecked(),
                                 xmin=self.ui.lineEdit_xmin_dp.displayText(),xmax=self.ui.lineEdit_xmax_dp.displayText(),
                                 ymin=self.ui.lineEdit_ymin_dp.displayText(),ymax=self.ui.lineEdit_ymax_dp.displayText(),
                                 zmin=self.ui.lineEdit_zmin_dp.displayText(),zmax=self.ui.lineEdit_zmax_dp.displayText())


    # DFT tab

    def cal_with_dftdb(self):
        self.ui.comboBox_search_type.setEnabled(True)
        self.ui.lineEdit_search_value.setEnabled(True)
        self.ui.pushButton_search.setEnabled(True)
        self.ui.tableView.setEnabled(True)
        self.ui.pushButton_dft_folder.setEnabled(False)

    def cal_with_userfile(self):
        self.ui.comboBox_search_type.setEnabled(False)
        self.ui.lineEdit_search_value.setEnabled(False)
        self.ui.pushButton_search.setEnabled(False)
        self.ui.tableView.setEnabled(False)
        self.ui.pushButton_dft_folder.setEnabled(True)

    def choose_dft_folder(self):
        self.userdft_path = QFileDialog.getExistingDirectory(self, 'User DFT directory', '')
        self.ui.label_dft_folder.setText("DFT folder: "+self.userdft_path)

    def search_crystal_structure(self):
        search_type = self.ui.comboBox_search_type.currentText()
        search_keyword = self.ui.lineEdit_search_value.displayText()
        match_exact = self.ui.checkBox_exact.isChecked()
        self.show_df = self.dft_worker.search_structure(search_type, search_keyword, match_exact)
        self.ui.tableView.setModel(PandasModel(self.show_df))

    def copy_files(self):
        if self.ui.radioButton_database.isChecked():
            mp_dfindex = self.ui.tableView.currentIndex().row()
            mp_id = self.show_df.iat[mp_dfindex, 0]
            self.dft_worker.prepare_dft_files(self.cwd_path, self.data_path, mp_id)
        elif self.ui.radioButton_user_dft.isChecked():
            self.dft_worker.prepare_dft_files(self.cwd_path, self.userdft_path)

    def plot_dos_disp_dft(self):
        self.copy_files()
        self.dft_worker.plot_dos_dispersion()

    def setup_oclimax_dft(self):
        self.copy_files()
        self.dft_worker.oclimax.set_oclimax_cwd(self.cwd_path)
        self.dft_worker.oclimax.exec()
        self.ui.label.setText(f"Parameter file: {self.dft_worker.oclimax.params_file_name}")
        
    def run_oclimax_dft(self):
        self.dft_worker.oclimax.run_oclimax()

    def plot_spec_dft(self):
        self.dft_worker.oclimax.plot_spec(unit=self.ui.comboBox_plot_unit_dft.currentIndex(),
                                 setrange=self.ui.checkBox_range_dft.isChecked(),interactive=self.ui.checkBox_interactive_dft.isChecked(),
                                 xmin=self.ui.lineEdit_xmin_dft.displayText(),xmax=self.ui.lineEdit_xmax_dft.displayText(),
                                 ymin=self.ui.lineEdit_ymin_dft.displayText(),ymax=self.ui.lineEdit_ymax_dft.displayText(),
                                 zmin=self.ui.lineEdit_zmin_dft.displayText(),zmax=self.ui.lineEdit_zmax_dft.displayText())


    # MLFF tab

    def upload_structure_mlff(self):
        fname = self.select_structure_file()
        if fname:
            try:
                self.read_structure_file(fname)
                #torch.set_default_dtype(torch.float32)
                self.ui.label_structure_mlff.setText(f"Structure file: {fname}")
                self.ui.statusbar.showMessage("")
            except Exception as e:
                self.ui.statusbar.showMessage(f"Error: {e.__repr__()} when loading {fname}")

    def opt_and_dos_cal_mlff(self):
        if not self.atoms:
            print('ERROR: No structure loaded.')
            return
        self.mlff_worker.run_opt_and_dos(self.atoms,potential_index=self.ui.comboBox_mlff_model.currentIndex(),
                                         lmin=self.ui.lineEdit_lmin_mlff.displayText(),
                                         fmax=self.ui.lineEdit_fmax_mlff.displayText(),
                                         nmax=self.ui.lineEdit_nmax_mlff.displayText(),
                                         delta=self.ui.lineEdit_delta_mlff.displayText())

    def setup_oclimax_mlff(self):
        qden = 35.0
        abc = self.atoms.cell.cellpar()[0:3]
        nx = int(qden/abc[0])+1
        ny = int(qden/abc[1])+1
        nz = int(qden/abc[2])+1

        self.mlff_worker.oclimax.set_oclimax_cwd(self.cwd_path)
        self.mlff_worker.generate_initial_mesh_file([nx,ny,nz])
        self.mlff_worker.oclimax.exec()
        self.ui.label_param_file_mlff.setText(f"Parameter file: {self.mlff_worker.oclimax.params_file_name}")

    def run_oclimax_mlff(self):
        self.mlff_worker.oclimax.run_oclimax()

    def plot_spec_mlff(self):
        self.mlff_worker.oclimax.plot_spec(unit=self.ui.comboBox_plot_unit_mlff.currentIndex(),
                                 setrange=self.ui.checkBox_range_mlff.isChecked(),interactive=self.ui.checkBox_interactive_mlff.isChecked(),
                                 xmin=self.ui.lineEdit_xmin_mlff.displayText(),xmax=self.ui.lineEdit_xmax_mlff.displayText(),
                                 ymin=self.ui.lineEdit_ymin_mlff.displayText(),ymax=self.ui.lineEdit_ymax_mlff.displayText(),
                                 zmin=self.ui.lineEdit_zmin_mlff.displayText(),zmax=self.ui.lineEdit_zmax_mlff.displayText())


    # help dialogs
    def open_help_dp(self):
        help_dp = QDialog()
        self.setup_help = Ui_Help_DP()
        self.setup_help.setupUi(help_dp)
        help_dp.exec()

    def open_help_dft(self):
        help_dft = QDialog()
        self.setup_help = Ui_Help_DFT()
        self.setup_help.setupUi(help_dft)
        help_dft.exec()

    def open_help_mlff(self):
        help_mlff = QDialog()
        self.setup_help = Ui_Help_MLFF()
        self.setup_help.setupUi(help_mlff)
        help_mlff.exec()


if __name__ == "__main__":
    print('********************************************************************************************************')
    print('* Inelastic Neutron Scattering Prediction for Instantaneous Results and Experimental Design (INSPIRED) *')
    print('********************************************************************************************************')
    print('INFO: Initializing ...')
    app = QApplication(sys.argv)
    main_window = INSPIRED()
    print('INFO: Starting GUI ...')
    main_window.show()
    print('INFO: Ready. Set your working directory in Menu, or use current directory by default.')
    print('INFO: To start with a new material, it is strongly recommended that a new/empty directory is created/assigned as working directory.')
    print('INFO: Existing files from a previous/different model in the working directory may interfere with the calculation and lead to incorrect results.')
    sys.exit(app.exec_())
