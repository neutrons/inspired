import os
import re
import shutil
import pandas as pd
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
from oclimax import OCLIMAX

class DFTWorker():
    """calculate INS spectra from available entries in the DFT database (or user-supplied DFT results)
    """

    def __init__(self):
        self.initial_search_df = pd.DataFrame()
        self.oclimax = OCLIMAX()


    def get_initial_search_df(self,dft_path):
        """list crystals in the database
        """

        crystal_list = os.path.join(dft_path,'crystals.dat')
        df = pd.read_csv(crystal_list, sep=' ', skiprows=[1], skipinitialspace=True)
        df['Space group (number)'] = df['space'].astype(str) + df['group'].astype(str)
        df.rename(columns={'id':'MP ID', 'formula':'Formula'}, inplace=True)
        self.initial_search_df = df.copy()
        df_toshow = df[['MP ID', 'Formula', 'Space group (number)']]

        return df_toshow


    def search_structure(self, search_type, search_keyword, match_exact):
        """search for the crystal in various ways
        """

        initial_search_df = self.initial_search_df
        try:
            re.compile(search_keyword)
        except re.error:
            print('ERROR: Search value is not a valid regex, try again.')
            return initial_search_df
        if search_type == 'By composition':
            if match_exact:
                df = initial_search_df[initial_search_df['Formula'].str.fullmatch(search_keyword, case=True)]
            else:
                df = initial_search_df[initial_search_df['Formula'].str.contains(search_keyword, case=False)]
        elif search_type == 'By MP ID':
            if match_exact:
                df = initial_search_df[initial_search_df['MP ID'].str.contains('-'+search_keyword+'-', case=True, regex=False)]
            else:
                df = initial_search_df[initial_search_df['MP ID'].str.contains(search_keyword, case=False)]
        elif search_type == 'By space group':
            if match_exact:
                df = initial_search_df[initial_search_df['space'].str.fullmatch(search_keyword, case=True)]
            else:
                df = initial_search_df[initial_search_df['space'].str.contains(search_keyword, case=False)]
        elif search_type == 'By space group number':
            if match_exact:
                df = initial_search_df[initial_search_df['group'].str.fullmatch('\('+search_keyword+'\)', case=True)]
            else:
                df = initial_search_df[initial_search_df['group'].str.contains(search_keyword, case=False)]

        current_search_df = df[['MP ID', 'Formula', 'Space group (number)']]
        print('INFO: Search value '+search_keyword+' returns '+str(len(current_search_df))+' match(es).')
        return current_search_df


    def prepare_dft_files(self, cwd_path, data_path, mp_id=None):
        """copy DFT files to the current working directory
        """

        if mp_id:
            dft_path = os.path.join(data_path, mp_id)
        else:
            dft_path = data_path
        if dft_path != cwd_path:
            list_file_tocopy = ['FORCE_SETS', 'FORCE_CONSTANTS', 'POSCAR-unitcell', 'BORN', 'mesh.conf']
            for fn in list_file_tocopy:              # copy required files
                file_path = os.path.join(dft_path, fn)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, cwd_path)
                else:
                    try:
                        os.remove(os.path.join(cwd_path,fn))
                    except OSError:
                        pass
        self.oclimax.oclimax_params.get_default_mesh()
        self.oclimax.use_default_mesh()


    def plot_dos_dispersion(self):

        self.oclimax.oclimax_params.get_supercell()
        print('INFO: Plotting phonon dispersion and DOS. For large unitcells this may take a few moments.')
        print('INFO: Frequency unit in plot is THz. 1 THz = 4.136 meV = 33.356 cm-1')
        print('INFO: Phonon DOS data will be saved in total_dos.dat file')
        phonon = phonopy.load(supercell_matrix=self.oclimax.oclimax_params.supercell,
                      primitive_matrix=[[1,0,0], [0,1,0], [0,0,1]],
                      unitcell_filename="POSCAR-unitcell")

        bands, labels, path_connections = get_band_qpoints_by_seekpath(phonon._primitive, 1, is_const_interval=True)
        points = []
        for i in range(len(bands)):
            if i==0 or (bands[i-1][1]!=bands[i][0]).any():
                points.append(bands[i][0])
            points.append(bands[i][1])
        print('INFO: Labels for the special points in phonon dispersion:')
        for i in range(len(labels)):
            print(labels[i],points[i])
        

        # To plot DOS next to band structure
        phonon.auto_band_structure()
        phonon.auto_total_dos()
        phonon.write_total_dos()
        phonon.plot_band_structure_and_dos().show()


    def plot_spec_dft(self):
        self.oclimax.plot_spec()
