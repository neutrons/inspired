import sys
import os
import shutil
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
from set_paths import SetPaths
from pandas_model import PandasModel
from ase.io import read
from ase.formula import Formula
from ase.spacegroup import get_spacegroup
import torch

class INSPIRED(QMainWindow):
    def __init__(self, parent=None):
        super(INSPIRED, self).__init__(parent)
        self.ui = Ui_INSPIRED()
        self.ui.setupUi(self)
        self.dp_worker = DPWorker()
        self.dft_worker = DFTWorker()
        self.mlff_worker = MLFFWorker()
        self.set_paths = SetPaths()
        self.root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cwd_path = os.getcwd()
        self.ui.label_cwd.setText("Current Working Directory: "+self.cwd_path)
        self.userdft_path = self.cwd_path
        self.setup_paths()
        self.check_paths()

        self.atoms = None

        self.pred_type = "Phonon DOS"

        self.show_df = self.dft_worker.get_initial_search_df(dft_path=self.dft_database_path)
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


    def setup_working_directory(self):
        wd = QFileDialog.getExistingDirectory(self, 'Working directory', '')
        if wd:
            self.cwd_path = wd
        self.ui.statusbar.showMessage(f"Working directory is set to {self.cwd_path}.", 1000)
        self.ui.label_cwd.setText("Current Working Directory: "+self.cwd_path)
        os.chdir(self.cwd_path)


    def setup_paths(self):
        home_path = os.path.expanduser('~')
        user_config = os.path.join(home_path,'.config','inspired','config')
        sys_config = os.path.join(self.root_path, 'config')
        self.predictor_path = os.path.join(self.root_path, 'model')
        self.dft_database_path = os.path.join(self.root_path, 'dftdb')
        self.mace_model_file = os.path.join(self.root_path, 'mlff', '2023-08-14-mace-universal.model')
        self.m3gnet_model_path = os.path.join(self.root_path, 'mlff', 'M3GNet-MP-2021.2.8-PES')
        if not os.path.isfile(user_config) and not os.path.isfile(sys_config):
            print('INFO: config file not found, using default paths.')
            return
        elif not os.path.isfile(user_config) and os.path.isfile(sys_config):
            os.makedirs(os.path.join(home_path,'.config','inspired'))
            shutil.copyfile(sys_config,user_config)
            
        with open(user_config, 'r') as f:
            lines = f.readlines()
        path_dict = {}
        for line in lines:
            key, value = line.split(':')
            key = key.strip()
            value = value.strip()
            path_dict[key] = value
        if 'predictor_path' in path_dict and path_dict['predictor_path']!='default':
            self.predictor_path = path_dict['predictor_path']
        if 'dft_database_path' in path_dict and path_dict['dft_database_path']!='default':
            self.dft_database_path = path_dict['dft_database_path']
        if 'mace_model_file' in path_dict and path_dict['mace_model_file']!='default':
            self.mace_model_file = path_dict['mace_model_file']
        if 'm3gnet_model_path' in path_dict and path_dict['m3gnet_model_path']!='default':
            self.m3gnet_model_path = path_dict['m3gnet_model_path']



    def check_paths(self):           
        if not os.path.isdir(self.predictor_path):
            print('ERROR: Predictor model path does not exist. Please reset in config file or Preferences in the Menu.')
            self.set_preferences()
        for dp_file in ['crys_dos_predictor.torch','crys_vis_predictor.torch','latent_space_predictor.torch','decoder.pt']:
            if not os.path.isfile(os.path.join(self.predictor_path,dp_file)):
                print('ERROR: Predictor model path missing '+dp_file+'. Please reset in config file or Preferences in the Menu.')
                self.set_preferences()
        if not os.path.isdir(self.dft_database_path):
            print('ERROR: DFT database path does not exist. Please reset in config file or Preferences in the Menu.')
            self.set_preferences()
        if not os.path.isfile(os.path.join(self.dft_database_path,'crystals.dat')):
            print('ERROR: DFT database file (crystals.dat) could not be found in the specified path. Please reset in config file or Preferences in the Menu.')
            self.set_preferences()
        if not os.path.isfile(self.mace_model_file):
            print('ERROR: MACE model file cannot be found. Please reset in config file or Preferences in the Menu.')
            self.set_preferences()
        if not os.path.isdir(self.m3gnet_model_path):
            print('ERROR: M3GNet model path does not exist. Please reset in config file or Preferences in the Menu.')
            self.set_preferences()


    def set_preferences(self):

        self.set_paths.init_paths(self.predictor_path,self.dft_database_path,self.mace_model_file,self.m3gnet_model_path)
        self.set_paths.exec()
        self.setup_paths()
        self.check_paths()


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
            occ = self.atoms.info['occupancy']
            fra = []
            for j in list(occ.values()):
                fra.append(float(list(j.values())[0]))
            if min(fra)<0.999:
                print('WARNING: The structure in the CIF file must be a physical representation of the actual atoms.') 
                print('WARNING: Statistical representation in the form of fractional/partial occupancies is not supported.')
                print('WARNING: A supercell may be created to account for disorder and remove partial occupancies.')
                print('WARNING: All partial occupancies will be treated as full atoms in the following calculations.')

        elif 'POSCAR' in filename or 'CONTCAR' in filename:
            self.atoms = read(filename, format='vasp')
        else:
            print('ERROR: File format not supported yet (currently support cif and VASP format)')
            return
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
            print('INFO: Prediction started.')
            self.dp_worker.predict(model_path=self.predictor_path, struc=self.atoms, pred_type=self.pred_type, partial_dos=self.ui.checkBox_partial_dos.isChecked())
            print('INFO: Prediction finished.')
            #self.ui.statusbar.showMessage("Prediction done")
        else:
            print('ERROR: Select structure file first.')

    def plot_spec_dp(self):
        self.dp_worker.savenplot(model_path=self.predictor_path,cwd=self.cwd_path,partial_dos=self.ui.checkBox_partial_dos.isChecked(),unit=self.ui.comboBox_plot_unit_dp.currentIndex(),
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
            self.dft_worker.prepare_dft_files(self.cwd_path, self.dft_database_path, mp_id)
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
                self.ui.label_structure_mlff.setText(f"Structure file: {fname}")
                self.ui.statusbar.showMessage("")
            except Exception as e:
                self.ui.statusbar.showMessage(f"Error: {e.__repr__()} when loading {fname}")

    def opt_and_dos_cal_mlff(self):
        if not self.atoms:
            print('ERROR: No structure loaded.')
            return
        self.mlff_worker.run_opt_and_dos(self.mace_model_file,self.m3gnet_model_path,self.atoms,potential_index=self.ui.comboBox_mlff_model.currentIndex(),
                                         lmin=self.ui.lineEdit_lmin_mlff.displayText(),
                                         fmax=self.ui.lineEdit_fmax_mlff.displayText(),
                                         nmax=self.ui.lineEdit_nmax_mlff.displayText(),
                                         delta=self.ui.lineEdit_delta_mlff.displayText())

    def setup_oclimax_mlff(self):
        if not self.atoms:
            print('ERROR: No structure loaded.')
            return
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
