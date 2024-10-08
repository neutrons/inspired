import os
import subprocess

# Initialize and access oclimax parameters

class OclimaxParams():
    def __init__(self):

        self.dict_default_params = {'TASK': 0,
                                    'INSTR': 0,
                                    'TEMP': '0.00',
                                    'E_UNIT' : 1,
                                    'OUTPUT': '0',
                                    'MAXO': '10',
                                    'CONV': '2',
                                    'PHASE': '0',
                                    'MASK': '0',
                                    'NORM': '0',
                                    'ELASTIC': '-1 -1',
                                    'MINE': '0.1',
                                    'MAXE': '400.00',
                                    'dE': '0.1',
                                    'ECUT': '0.01',
                                    'ERES': '0.31  0.5E-02  0.8E-06',
                                    'MINQ': '0.10',
                                    'MAXQ': '10.00',
                                    'dQ': '0.10',
                                    'QRES': '-1',
                                    'THETA': '0.0  60.0',
                                    'Ef': '4.00',
                                    'Ei': '400.00',
                                    'L1': '11.60',
                                    'L2': '2.00',
                                    'L3': '3.00',
                                    'dt_m': '3.91',
                                    'dt_ch': '5.95',
                                    'dL3': '3.50',
                                    'MINQ_y': '1.00',
                                    'MAXQ_y': '2.00',
                                    'dQ_y': '0.02',
                                    'WING': '0',
                                    'A_ISO': '0.0350',
                                    'W_WIDTH': '20.0',
                                    'Q1': '1  0  0',
                                    'Q2': '0  1  0',
                                    'Q3': '0  0  1',
                                    'Q1bin': '2.0  0.01  5.0',
                                    'Q2bin': '-0.05  0.05  0.05',
                                    'Q3bin': '-0.05  0.05  0.05',
                                    'Ebin': '0.1  0.1  100.0',
                                    'x-axis': 0,
                                    'y-axis': 0,
                                    }

        self.dict_params = self.dict_default_params.copy()
        self.default_mesh = ['1','1','1']
        self.supercell = None

    def get_supercell(self):
        if os.path.isfile("mesh.conf"):
            sc = subprocess.run(["sed -n '/^DIM/p' mesh.conf"], shell=True, capture_output=True, text=True).stdout
            scm = list(map(int,sc.split('=')[1].split()))
            if len(scm)==9:
                self.supercell = [[scm[0],scm[1],scm[2]],[scm[3],scm[4],scm[5]],[scm[6],scm[7],scm[8]]]
            elif len(scm)==3:
                self.supercell = scm
            else:
                print('Check supercell dimension in mesh.conf file.')
        else:
            print('Could not open/read mesh.conf file, make sure it is in current working directory.')
    
    def get_default_mesh(self):
        if os.path.isfile("mesh.conf"):
            mesh = subprocess.run(["sed -n '/^MP/p' mesh.conf"], shell=True, capture_output=True, text=True).stdout
            self.default_mesh = mesh.split('=')[1].split()[-3:]
        else:
            print('Could not open/read mesh.conf file, make sure it is in current working directory.')
    
    def generate_mesh_file(self, mesh_list=['1', '1', '1']):
        if os.path.isfile("mesh.conf"):
            newmesh = ' '.join(mesh_list)
            subprocess.run(["sed -i 's/.*MP.*/MP = "+newmesh+"/' mesh.conf"], shell=True)
        else:
            print('Could not open/read mesh.conf file, make sure it is in current working directory.')
        
    def load_parameters(self, params_file_path = "oclimax.params"):
        try:
            pf = open(params_file_path, "r")
        except OSError:
            print('Could not open/read params file, make sure it is in current working directory.')
            return
        para_lines = pf.readlines()
        pf.close()
        for s in para_lines:
            if s.startswith('#!'):
                s = s[2:]
            param = [p.strip() for p in s.split('#',1)[0].split('=',1)]
            if len(param)==2 and param[0] in list(self.dict_params.keys()):
                if param[0] in ['TASK','INSTR','E_UNIT','x-axis','y-axis']:
                    self.dict_params[param[0]] = int(param[1])
                else:
                    self.dict_params[param[0]] = param[1]

    def write_new_parameter_file(self, save_params_file_path = "oclimax.params"):
        pf = open(save_params_file_path, 'w')
        list_file_lines = []
        list_file_lines.append('# This file was generated by INSPIRED and should be used within INSPIRED.')
        list_file_lines.append('# Using this file directly with the standalone OCLIMAX is discouraged and may not produce the intended results.')
        list_file_lines.append('TASK    =   ' + str(self.dict_params['TASK']))
        list_file_lines.append('INSTR   =   ' + str(self.dict_params['INSTR']))
        list_file_lines.append('TEMP    =   ' + self.dict_params['TEMP'])
        list_file_lines.append('E_UNIT  =   ' + str(self.dict_params['E_UNIT']))
        list_file_lines.append('OUTPUT  =   ' + self.dict_params['OUTPUT'])
        list_file_lines.append('MAXO    =   ' + self.dict_params['MAXO'])
        list_file_lines.append('CONV    =   ' + self.dict_params['CONV'])
        list_file_lines.append('PHASE   =   ' + self.dict_params['PHASE'])
        list_file_lines.append('MASK    =   ' + self.dict_params['MASK'])
        list_file_lines.append('NORM    =   ' + self.dict_params['NORM'])
        list_file_lines.append('ELASTIC =   ' + self.dict_params['ELASTIC'])
        list_file_lines.append('MINE    =   ' + self.dict_params['MINE'])
        list_file_lines.append('MAXE    =   ' + self.dict_params['MAXE'])
        list_file_lines.append('dE      =   ' + self.dict_params['dE'])
        list_file_lines.append('ECUT    =   ' + self.dict_params['ECUT'])
        list_file_lines.append('ERES    =   ' + self.dict_params['ERES'])
        list_file_lines.append('MINQ    =   ' + self.dict_params['MINQ'])
        list_file_lines.append('MAXQ    =   ' + self.dict_params['MAXQ'])
        list_file_lines.append('dQ      =   ' + self.dict_params['dQ'])
        list_file_lines.append('QRES    =   ' + self.dict_params['QRES'])
        list_file_lines.append('THETA   =   ' + self.dict_params['THETA'])
        list_file_lines.append('Ef      =   ' + self.dict_params['Ef'])
        list_file_lines.append('Ei      =   ' + self.dict_params['Ei'])
        list_file_lines.append('L1      =   ' + self.dict_params['L1'])
        list_file_lines.append('L2      =   ' + self.dict_params['L2'])
        list_file_lines.append('L3      =   ' + self.dict_params['L3'])
        list_file_lines.append('dt_m    =   ' + self.dict_params['dt_m'])
        list_file_lines.append('dt_ch   =   ' + self.dict_params['dt_ch'])
        list_file_lines.append('dL3     =   ' + self.dict_params['dL3'])
        list_file_lines.append('MINQ_y  =   ' + self.dict_params['MINQ_y'])
        list_file_lines.append('MAXQ_y  =   ' + self.dict_params['MAXQ_y'])
        list_file_lines.append('dQ_y    =   ' + self.dict_params['dQ_y'])
        list_file_lines.append('WING    =   ' + self.dict_params['WING'])
        list_file_lines.append('A_ISO   =   ' + self.dict_params['A_ISO'])
        list_file_lines.append('W_WIDTH =   ' + self.dict_params['W_WIDTH'])
        list_file_lines.append('#! Q1 =   ' + self.dict_params['Q1'])
        list_file_lines.append('#! Q2 =   ' + self.dict_params['Q2'])
        list_file_lines.append('#! Q3 =   ' + self.dict_params['Q3'])
        list_file_lines.append('#! Q1bin =   ' + self.dict_params['Q1bin'])
        list_file_lines.append('#! Q2bin =   ' + self.dict_params['Q2bin'])
        list_file_lines.append('#! Q3bin =   ' + self.dict_params['Q3bin'])
        list_file_lines.append('#! Ebin =   ' + self.dict_params['Ebin'])
        list_file_lines.append('#! x-axis =   ' + str(self.dict_params['x-axis']))
        list_file_lines.append('#! y-axis =   ' + str(self.dict_params['y-axis']))

        file_content = '\n'.join(list_file_lines)
        pf.write(file_content)

        pf.close()

    def update_param(self, param_key, param):
        self.dict_params[param_key] = param

    def get_param(self, param_key):
        return self.dict_params[param_key]
    
    def reset_params_default(self):
        self.dict_params = self.dict_default_params.copy()

