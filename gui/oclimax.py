from qtpy.QtWidgets import (QFileDialog, QDialog)
from ui_oclimax import Ui_OCLIMAX
from ui_help_oclimax import Ui_Help_OCLIMAX
from oclimax_params import OclimaxParams
import subprocess
import os
import numpy as np
import glob
import time

import csv
import matplotlib.pyplot as plt

class OCLIMAX(QDialog):
    def __init__(self):
        super(OCLIMAX, self).__init__()
        self.ui = Ui_OCLIMAX()
        self.ui.setupUi(self)
        self.oclimax_params = OclimaxParams()
        self.params_file_name = "oclimax.params"
        self.csv_file_name = None
        self.oclimax_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'oclimax')

        self.enable_disable_single_crystal_parameters(False)

        self.display_parameters()

    def set_oclimax_cwd(self, path):
        os.chdir(path)
    
    def use_default_mesh(self):
        self.ui.lineEdit_bscale.setText('1.0')
        self.ui.spinBox_mesh1.setValue(int(self.oclimax_params.default_mesh[0]))
        self.ui.spinBox_mesh2.setValue(int(self.oclimax_params.default_mesh[1]))
        self.ui.spinBox_mesh3.setValue(int(self.oclimax_params.default_mesh[2]))
        self.oclimax_params.generate_mesh_file(mesh_list=self.oclimax_params.default_mesh)

    def save_mesh(self):
        mesh_list = []
        mesh_list.append(str(self.ui.spinBox_mesh1.value()))
        mesh_list.append(str(self.ui.spinBox_mesh2.value()))
        mesh_list.append(str(self.ui.spinBox_mesh3.value()))
        self.oclimax_params.generate_mesh_file(mesh_list=mesh_list)

    def load_parameters(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', 
                '',"OCLIMAX parameter files (*.params)")
        if fname:
            self.oclimax_params.load_parameters(fname)
            self.display_parameters()
            self.params_file_name = fname

    def use_default_parameters(self):
        self.oclimax_params.reset_params_default()
        self.display_parameters()
        self.oclimax_params.write_new_parameter_file()
        self.params_file_name = "oclimax.params"

    def save_parameters(self):
        self.get_params_from_display()
        fname, _ = QFileDialog.getSaveFileName(self, 'Save file',
                                               '', "OCLIMAX parameter files (*.params)")
        if fname:
            if not fname.endswith('.params'):
                fname += '.params'
            self.params_file_name = fname
            self.oclimax_params.write_new_parameter_file(fname)
        else:
            self.oclimax_params.write_new_parameter_file()

        self.close()

    def get_task_index(self):
        """translate oclimax task/instr to task index in inspired
        """

        task = self.oclimax_params.get_param('TASK')
        instrument = self.oclimax_params.get_param('INSTR')
        if task == 0 and instrument == 0:
            ind = 0
        elif task == 0 and instrument == 3:
            ind = 1
        elif task == 1 and instrument == 0:
            ind = 2
        elif task == 1 and instrument == 3:
            ind = 3
        elif task == 2:
            ind = 4
        elif task == 3:
            ind = 5
        else:
            ind = 6
        return ind

    def set_task_instr(self,ind):
        """translate task index in inspired to oclimax task/instr
        """

        if ind == 0:
            self.oclimax_params.update_param('TASK', 0)
            self.oclimax_params.update_param('INSTR', 0)
        elif ind == 1:
            self.oclimax_params.update_param('TASK', 0)
            self.oclimax_params.update_param('INSTR', 3)
        elif ind == 2:
            self.oclimax_params.update_param('TASK', 1)
            self.oclimax_params.update_param('INSTR', 0)
        elif ind == 3:
            self.oclimax_params.update_param('TASK', 1)
            self.oclimax_params.update_param('INSTR', 3)
        elif ind == 4:
            self.oclimax_params.update_param('TASK', 2)
            self.oclimax_params.update_param('INSTR', 3)
        elif ind == 5:
            self.oclimax_params.update_param('TASK', 3)
            self.oclimax_params.update_param('INSTR', 3)
           

    def display_parameters(self):
        self.ui.comboBox_task.setCurrentIndex(self.get_task_index())
        self.ui.lineEdit_output.setText(self.oclimax_params.get_param('OUTPUT'))
        self.ui.lineEdit_temp.setText(self.oclimax_params.get_param('TEMP'))
        self.ui.lineEdit_maxo.setText(self.oclimax_params.get_param('MAXO'))
        self.ui.lineEdit_mask.setText(self.oclimax_params.get_param('MASK'))
        self.ui.lineEdit_theta.setText(self.oclimax_params.get_param('THETA'))
        self.ui.comboBox_Eunit.setCurrentIndex(self.oclimax_params.get_param('E_UNIT'))
        self.ui.lineEdit_norm.setText(self.oclimax_params.get_param('NORM'))
        self.ui.lineEdit_Ei.setText(self.oclimax_params.get_param('Ei'))
        self.ui.lineEdit_Ecut.setText(self.oclimax_params.get_param('ECUT'))
        self.ui.lineEdit_Emin.setText(self.oclimax_params.get_param('MINE'))
        self.ui.lineEdit_Emax.setText(self.oclimax_params.get_param('MAXE'))
        self.ui.lineEdit_Qmin.setText(self.oclimax_params.get_param('MINQ'))
        self.ui.lineEdit_Qmax.setText(self.oclimax_params.get_param('MAXQ'))
        self.ui.lineEdit_dE.setText(self.oclimax_params.get_param('dE'))
        self.ui.lineEdit_dQ.setText(self.oclimax_params.get_param('dQ'))
        self.ui.lineEdit_Eres.setText(self.oclimax_params.get_param('ERES'))
        self.ui.lineEdit_Qres.setText(self.oclimax_params.get_param('QRES'))
        self.ui.lineEdit_Q1.setText(self.oclimax_params.get_param('Q1'))
        self.ui.lineEdit_Q2.setText(self.oclimax_params.get_param('Q2'))
        self.ui.lineEdit_Q3.setText(self.oclimax_params.get_param('Q3'))
        self.ui.lineEdit_Q1bin.setText(self.oclimax_params.get_param('Q1bin'))
        self.ui.lineEdit_Q2bin.setText(self.oclimax_params.get_param('Q2bin'))
        self.ui.lineEdit_Q3bin.setText(self.oclimax_params.get_param('Q3bin'))
        self.ui.lineEdit_Ebin.setText(self.oclimax_params.get_param('Ebin'))
        self.ui.comboBox_xaxis.setCurrentIndex(self.oclimax_params.get_param('x-axis'))
        self.ui.comboBox_yaxis.setCurrentIndex(self.oclimax_params.get_param('y-axis'))

    def get_params_from_display(self):
        self.set_task_instr(self.ui.comboBox_task.currentIndex())
        self.oclimax_params.update_param('OUTPUT', self.ui.lineEdit_output.displayText())
        self.oclimax_params.update_param('TEMP', self.ui.lineEdit_temp.displayText())
        self.oclimax_params.update_param('MAXO', self.ui.lineEdit_maxo.displayText())
        self.oclimax_params.update_param('MASK', self.ui.lineEdit_mask.displayText())
        self.oclimax_params.update_param('THETA', self.ui.lineEdit_theta.displayText())
        self.oclimax_params.update_param('E_UNIT', self.ui.comboBox_Eunit.currentIndex())
        self.oclimax_params.update_param('NORM', self.ui.lineEdit_norm.displayText())
        self.oclimax_params.update_param('Ei', self.ui.lineEdit_Ei.displayText())
        self.oclimax_params.update_param('ECUT', self.ui.lineEdit_Ecut.displayText())
        self.oclimax_params.update_param('MINE', self.ui.lineEdit_Emin.displayText())
        self.oclimax_params.update_param('MAXE', self.ui.lineEdit_Emax.displayText())
        self.oclimax_params.update_param('MINQ', self.ui.lineEdit_Qmin.displayText())
        self.oclimax_params.update_param('MAXQ', self.ui.lineEdit_Qmax.displayText())
        self.oclimax_params.update_param('dE', self.ui.lineEdit_dE.displayText())
        self.oclimax_params.update_param('dQ', self.ui.lineEdit_dQ.displayText())
        self.oclimax_params.update_param('ERES', self.ui.lineEdit_Eres.displayText())
        self.oclimax_params.update_param('QRES', self.ui.lineEdit_Qres.displayText())
        self.oclimax_params.update_param('Q1', self.ui.lineEdit_Q1.displayText())
        self.oclimax_params.update_param('Q2', self.ui.lineEdit_Q2.displayText())
        self.oclimax_params.update_param('Q3', self.ui.lineEdit_Q3.displayText())
        self.oclimax_params.update_param('Q1bin', self.ui.lineEdit_Q1bin.displayText())
        self.oclimax_params.update_param('Q2bin', self.ui.lineEdit_Q2bin.displayText())
        self.oclimax_params.update_param('Q3bin', self.ui.lineEdit_Q3bin.displayText())
        self.oclimax_params.update_param('Ebin', self.ui.lineEdit_Ebin.displayText())
        self.oclimax_params.update_param('x-axis', self.ui.comboBox_xaxis.currentIndex())
        self.oclimax_params.update_param('y-axis', self.ui.comboBox_yaxis.currentIndex())

    def enable_disable_single_crystal_parameters(self, enable=False):
        self.ui.lineEdit_Q1.setEnabled(enable)
        self.ui.lineEdit_Q2.setEnabled(enable)
        self.ui.lineEdit_Q3.setEnabled(enable)
        self.ui.lineEdit_Q1bin.setEnabled(enable)
        self.ui.lineEdit_Q2bin.setEnabled(enable)
        self.ui.lineEdit_Q3bin.setEnabled(enable)
        self.ui.lineEdit_Ebin.setEnabled(enable)
        self.ui.comboBox_xaxis.setEnabled(enable)
        self.ui.comboBox_yaxis.setEnabled(enable)

        self.ui.lineEdit_Emin.setEnabled(not enable)
        self.ui.lineEdit_Emax.setEnabled(not enable)
        self.ui.lineEdit_dE.setEnabled(not enable)
        self.ui.lineEdit_Qmin.setEnabled(not enable)
        self.ui.lineEdit_Qmax.setEnabled(not enable)
        self.ui.lineEdit_dQ.setEnabled(not enable)
        self.ui.lineEdit_maxo.setEnabled(not enable)

    def task_changed(self):
        if self.ui.comboBox_task.currentIndex() == 4 or self.ui.comboBox_task.currentIndex() == 5:
            self.oclimax_params.update_param('MAXO', '1')
            self.ui.lineEdit_maxo.setText(self.oclimax_params.get_param('MAXO'))
            self.enable_disable_single_crystal_parameters(True)
            if self.ui.comboBox_task.currentIndex() == 4:
                self.ui.comboBox_yaxis.setCurrentIndex(0)
            elif self.ui.comboBox_task.currentIndex() == 5:
                self.ui.comboBox_yaxis.setCurrentIndex(2)

        else:
            self.oclimax_params.update_param('MAXO', '10')
            self.ui.lineEdit_maxo.setText(self.oclimax_params.get_param('MAXO'))
            self.enable_disable_single_crystal_parameters(False)

    def unit_changed(self):  # No action for now, reserved for future development
        if self.ui.comboBox_Eunit.currentIndex() == 1:
            return

    def run_oclimax(self):
        if not os.path.isfile("POSCAR-unitcell"):
            print('ERROR: POSCAR-unitcell file in working directory is required for this calculation.')
            return
        if not os.path.isfile("mesh.conf"):
            print('ERROR: mesh.conf file in working directory is required for this calculation.')
            return
        if not (os.path.isfile("FORCE_SETS") or os.path.isfile("FORCE_CONSTANTS")):
            print('ERROR: FORCE_SETS or FORCE_CONSTANTS file is required for this calculation.')
            return
        if os.path.isfile("FORCE_SETS") and os.path.isfile("FORCE_CONSTANTS"):
            print('WARNING: Found both FORCE_SETS and FORCE_CONSTANTS files. Inconsistency issue possible.')
        task = self.oclimax_params.get_param('TASK')
        if task == 0 or task == 1:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            print('INFO: Running phonopy calculation. Press Ctrl+C in terminal to abort.')
            try:
                subprocess.run(["phonopy", "-c", "POSCAR-unitcell", "mesh.conf"], text=True) # run phononpy
            except:
                print('ERROR: Phonopy calculation failed or terminated.')
                return
            print('INFO: Phonopy calculation finished. Running file format conversion. Press Ctrl+C in terminal to abort.')
            try:
                subprocess.run([os.path.join(self.oclimax_path,"oclimax_convert"), "-yaml", "mesh.yaml", "-o", "oclimax_"+timestr], text=True)
            except:
                print('ERROR: File format conversion failed or terminated.')
                return
            print('INFO: File converted. Running OCLIMAX simulation. Press Ctrl+C in terminal to abort.')
            try:
                subprocess.run([os.path.join(self.oclimax_path,'oclimax_run'), 'oclimax_'+timestr+'.oclimax', self.params_file_name], text=True)
            except:
                print('ERROR: OCLIMAX simulation failed or terminated.')
                return
            generated_csv = glob.glob('oclimax_'+timestr+'*K.csv')
            self.csv_file_name = generated_csv[0]
            print('INFO: OCLIMAX simulation finished. Output file: '+self.csv_file_name)

        else:
            self.run_oclimax_sc()

    def run_oclimax_sc(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        task = self.oclimax_params.get_param('TASK')
        Q1 = self.oclimax_params.get_param('Q1').split()
        Q2 = self.oclimax_params.get_param('Q2').split()
        Q3 = self.oclimax_params.get_param('Q3').split()
        Qb1 = self.oclimax_params.get_param('Q1bin').split()
        Qb2 = self.oclimax_params.get_param('Q2bin').split()
        Qb3 = self.oclimax_params.get_param('Q3bin').split()
        Eb = self.oclimax_params.get_param('Ebin').split()
        for param in [Q1,Q2,Q3,Qb1,Qb2,Qb3]:
            if len(param)!=3:
                print('ERROR: Check Q and Q_bin parameters (each should have three numbers, click "Help" for more information)')
                return
        for param in [Qb1,Qb2,Qb3]:
            if float(param[0])>float(param[2]):
                print('ERROR: Check Qbin parameters ([Qmin,Qstep,Qmax])')
                return
            if float(param[1])<0:
                print('ERROR: Check Qbin parameters (Qstep cannot be negative)')
                return
            if float(param[1])==0:
                if float(param[0])!=float(param[2]):
                    print('ERROR: Check Qbin parameters (Qstep can be zero only if Qmin=Qmax)')
                    return
                else:
                    param[1]=1
        if len(Eb)!=3:
            if task==3 and len(Eb)==2:
                if float(Eb[1])>float(Eb[0]):
                    Eb = np.array([float(Eb[0]),(float(Eb[1])-float(Eb[0]))/2.0,float(Eb[1])])
                else:
                    print('ERROR: Check E_bin parameters (Emax must be greater than Emin)')
                    return
            else:
                print('ERROR: Check E_bin parameters ([Emin,Estep,Emax] for the task)')
                return
        else:
            if float(Eb[2])<=float(Eb[0]):
                print('ERROR: Check E_bin parameters (Emax must be greater than Emin)')
                return
            if float(Eb[1])<=0:
                print('ERROR: Check E_bin parameters (Estep must be greater than zero)')
                return
            Eb = np.array([float(Eb[0]),float(Eb[1]),float(Eb[2])])
        Q = np.array([[float(Q1[0]),float(Q1[1]),float(Q1[2])],[float(Q2[0]),float(Q2[1]),float(Q2[2])],[float(Q3[0]),float(Q3[1]),float(Q3[2])]])
        Qb = np.array([[float(Qb1[0]),float(Qb1[1]),float(Qb1[2])],[float(Qb2[0]),float(Qb2[1]),float(Qb2[2])],[float(Qb3[0]),float(Qb3[1]),float(Qb3[2])]])
        bscale = float(self.ui.lineEdit_bscale.displayText())
        print('INFO: BAND_POINTS scaling factor set to '+str(bscale))
        dim = subprocess.run(["sed -n '/^DIM/p' mesh.conf"], shell=True, capture_output=True, text=True).stdout
        # S(Q,E)
        print('INFO: Running OCLIMAX simulation for single crystals, expect multiple calculation cycles. Press Ctrl+C in terminal to abort.')
        if task == 2:
            xaxis = self.oclimax_params.get_param('x-axis')
            s = [0,1,2]
            s.remove(xaxis)

            Qx = Q[xaxis]
            Qxb = Qb[xaxis]
            qQ1 = Q[s[0]]
            qQ2 = Q[s[1]]

            try:
                result=subprocess.run(["phonopy", "-c", "POSCAR-unitcell", "mesh.conf"], text=True) # run phononpy
            except:
                print('ERROR: Phonopy calculation failed or terminated.')
                return
            print('INFO: Converting data format')
            try:
                result=subprocess.run([os.path.join(self.oclimax_path,"oclimax_convert"),"-yaml","mesh.yaml","-o","mesh"], text=True)
            except:
                print('ERROR: File format conversion failed or terminated.')
                return
            q1=np.linspace(Qb[s[0]][0],Qb[s[0]][2],int((Qb[s[0]][2]-Qb[s[0]][0])/Qb[s[0]][1])+1)
            q2=np.linspace(Qb[s[1]][0],Qb[s[1]][2],int((Qb[s[1]][2]-Qb[s[1]][0])/Qb[s[1]][1])+1)

            count = 0
            npoint = max(int(bscale*1.0/Qxb[1])+1,int(bscale*(Eb[2]-Eb[0])/Eb[1])+1)
            for i in range(len(q1)):
                for j in range(len(q2)):
                    count += 1
                    print('INFO: Running cycle '+str(count)+' of '+str(len(q1)*len(q2)))
                    f = open('band.conf','w')
                    print(dim, file=f)
                    p1 = -0.5*Qx+q1[i]*qQ1+q2[j]*qQ2
                    p2 = q1[i]*qQ1+q2[j]*qQ2
                    p3 = 0.5*Qx+q1[i]*qQ1+q2[j]*qQ2
                    print('BAND = ',p1[0],p1[1],p1[2],p2[0],p2[1],p2[2],p3[0],p3[1],p3[2], file=f)
                    print('BAND_POINTS = ',npoint, file=f)
                    print('# bscale '+str(bscale), file=f)
                    print('FC_SYMMETR= TRUE', file=f)
                    print('EIGENVECTORS = .true.', file=f)
                    if os.path.isfile("FORCE_CONSTANTS"):
                        print('FORCE_CONSTANTS = read', file=f)
                    if os.path.isfile("BORN"):
                        print('NAC = .true.', file=f)
                    f.close()
                    try:
                        result=subprocess.run(["phonopy", "-c", "POSCAR-unitcell", "band.conf"], text=True)
                    except:
                        print('ERROR: Phonopy calculation failed or terminated.')
                        return
                    print('INFO: Converting data format')
                    try:
                        result=subprocess.run([os.path.join(self.oclimax_path,"oclimax_convert"),"-yamld","band.yaml","-o","band_"+timestr], text=True)
                    except:
                        print('ERROR: File format conversion failed or terminated.')
                        return

                    f = open('oclimax-sc.params','w')
                    print('TASK    =         2', file=f)
                    print('INSTR   =         3', file=f)
                    print('TEMP    =     ', self.oclimax_params.get_param('TEMP'), file=f)
                    print('E_UNIT  =     ', self.oclimax_params.get_param('E_UNIT'), file=f)
                    print('OUTPUT  =     ', self.oclimax_params.get_param('OUTPUT'), file=f)
                    print('MAXO    =     ', self.oclimax_params.get_param('MAXO'), file=f)
                    print('MASK    =     ', self.oclimax_params.get_param('MASK'), file=f)
                    print('NORM    =     ', self.oclimax_params.get_param('NORM'), file=f)
                    print('MINE    =     ', Eb[0], file=f)
                    print('MAXE    =     ', Eb[2], file=f)
                    print('dE      =     ', Eb[1], file=f)
                    print('ECUT    =     ', self.oclimax_params.get_param('ECUT'), file=f)
                    print('ERES    =     ', self.oclimax_params.get_param('ERES'), file=f)
                    print('MINQ    =     ', Qxb[0], file=f)
                    print('MAXQ    =     ', Qxb[2], file=f)
                    print('dQ      =     ', Qxb[1], file=f)
                    print('QRES    =     ', self.oclimax_params.get_param('QRES'), file=f)
                    print('THETA   =     ', self.oclimax_params.get_param('THETA'), file=f)
                    print('Ei      =     ', self.oclimax_params.get_param('Ei'), file=f)
                    print('HKL     =     ', p2[0],p2[1],p2[2], file=f)
                    print('Q_vec   =     ', Qx[0],Qx[1],Qx[2], file=f)
                    f.close()
                    outfile='cut_qe_'+timestr+'_'+str(i)+'_'+str(j)+'.csv'
                    try:
                        result=subprocess.run([os.path.join(self.oclimax_path,"oclimax_run"),"mesh.oclimax","oclimax-sc.params","band_"+timestr+".oclimax"], text=True)
                    except:
                        print('ERROR: OCLIMAX simulation failed or terminated.')
                        return
                    generated_csv = glob.glob("band_"+timestr+"_2Dmesh_scqw_*K.csv")
                    result=subprocess.run(["mv",generated_csv[0],outfile])

            all_cut_csv = glob.glob("cut_qe_"+timestr+"*.csv")
            print('INFO: Integrating cuts')
            try:
                result=subprocess.run(["python", os.path.join(self.oclimax_path,"merge_csv.py"), "oclimax_sc_qe_"+timestr] + all_cut_csv, text=True)
            except:
                print('ERROR: Merging files failed or terminated.')
                return
            generated_csv = glob.glob('oclimax_sc_qe_'+timestr+'*.csv')

        # S(Q1,Q2)
        if task == 3:
            xaxis = self.oclimax_params.get_param('x-axis')
            yaxis = self.oclimax_params.get_param('y-axis') - 1
            s = [0,1,2]
            s.remove(xaxis)
            s.remove(yaxis)

            Qx = Q[xaxis]
            Qy = Q[yaxis]
            Qxb = Qb[xaxis]
            Qyb = Qb[yaxis]
            qQ1 = Q[s[0]]

            try:
                result=subprocess.run(["phonopy", "-c", "POSCAR-unitcell", "mesh.conf"], text=True) # run phononpy
            except:
                print('ERROR: Phonopy calculation failed or terminated.')
                return
            print('INFO: Converting data format')
            try:
                result=subprocess.run([os.path.join(self.oclimax_path,"oclimax_convert"),"-yaml","mesh.yaml","-o","mesh"], text=True)
            except:
                print('ERROR: File format conversion failed or terminated.')
                return
            q1=np.linspace(Qb[s[0]][0],Qb[s[0]][2],round((Qb[s[0]][2]-Qb[s[0]][0])/Qb[s[0]][1])+1)
            #print(q1)

            for i in range(len(q1)):
                print('INFO: Running cycle '+str(i+1)+' of '+str(len(q1)))
                f = open('QPOINTS','w')
                qx=np.linspace(-0.5,0.5,int(bscale*1.0/Qxb[1])+1)
                qy=np.linspace(-0.5,0.5,int(bscale*1.0/Qyb[1])+1)
                print(len(qx)*len(qy), file=f)
                shft = q1[i]*qQ1
                for j in range(len(qx)):
                    for k in range(len(qy)):
                        qp = qx[j]*Qx+qy[k]*Qy+shft
                        print(qp[0],qp[1],qp[2], file=f)
                f.close()

                f = open('band.conf','w')
                print(dim, file=f)
                print('QPOINTS = .TRUE.', file=f)
                print('FC_SYMMETR= TRUE', file=f)
                print('EIGENVECTORS = .true.', file=f)
                if os.path.isfile("FORCE_CONSTANTS"):
                    print('FORCE_CONSTANTS = read', file=f)
                if os.path.isfile("BORN"):
                    print('NAC = .true.', file=f)
                f.close()
                try:
                    result=subprocess.run(["phonopy", "-c", "POSCAR-unitcell", "band.conf"], text=True)
                except:
                    print('ERROR: Phonopy calculation failed or terminated.')
                    return
                print('INFO: Converting data format')
                try:
                    result=subprocess.run([os.path.join(self.oclimax_path,"oclimax_convert"),"-vpd","POSCAR-unitcell","qpoints.yaml","-o","band_"+timestr], text=True)
                except:
                    print('ERROR: File format conversion failed or terminated.')
                    return

                f = open('oclimax-sc.params','w')
                print('TASK    =         3', file=f)
                print('INSTR   =         3', file=f)
                print('TEMP    =     ', self.oclimax_params.get_param('TEMP'), file=f)
                print('E_UNIT  =     ', self.oclimax_params.get_param('E_UNIT'), file=f)
                print('OUTPUT  =     ', self.oclimax_params.get_param('OUTPUT'), file=f)
                print('MAXO    =     ', self.oclimax_params.get_param('MAXO'), file=f)
                print('MASK    =     ', self.oclimax_params.get_param('MASK'), file=f)
                print('NORM    =     ', self.oclimax_params.get_param('NORM'), file=f)
                print('MINE    =     ', Eb[0], file=f)
                print('MAXE    =     ', Eb[2], file=f)
                print('dE      =     ', Eb[1], file=f)
                print('ECUT    =     ', self.oclimax_params.get_param('ECUT'), file=f)
                print('ERES    =     ', self.oclimax_params.get_param('ERES'), file=f)
                print('MINQ    =     ', Qxb[0], file=f)
                print('MAXQ    =     ', Qxb[2], file=f)
                print('dQ      =     ', Qxb[1], file=f)
                print('QRES    =     ', self.oclimax_params.get_param('QRES'), file=f)
                print('THETA   =     ', self.oclimax_params.get_param('THETA'), file=f)
                print('Ei      =     ', self.oclimax_params.get_param('Ei'), file=f)
                print('HKL     =     ', shft[0],shft[1],shft[2], file=f)
                print('Q_vec   =     ', Qx[0],Qx[1],Qx[2], file=f)
                print('Q_vec_y =     ', Qy[0],Qy[1],Qy[2], file=f)
                print('MINQ_y  =     ', Qyb[0], file=f)
                print('MAXQ_y  =     ', Qyb[2], file=f)
                print('dQ_y    =     ', Qyb[1], file=f)
                f.close()

                outfile='cut_qq_'+timestr+'_'+str(i)+'.csv'
                try:
                    result=subprocess.run([os.path.join(self.oclimax_path,"oclimax_run"),"mesh.oclimax","oclimax-sc.params","band_"+timestr+".oclimax"], text=True)
                except:
                    print('ERROR: OCLIMAX simulation failed or terminated.')
                    return
                generated_csv = glob.glob("band_"+timestr+"_2Dmesh_scqq_*K.csv")
                result=subprocess.run(["mv",generated_csv[0],outfile])

            all_cut_csv = glob.glob("cut_qq_"+timestr+"*.csv")
            print('INFO: Integrating cuts')
            try:
                result=subprocess.run(["python", os.path.join(self.oclimax_path,"merge_csv.py"), "oclimax_sc_qq_"+timestr] + all_cut_csv, text=True)
            except:
                print('ERROR: Merging files failed or terminated.')
                return
            generated_csv = glob.glob('oclimax_sc_qq_'+timestr+'*.csv')
        self.csv_file_name = generated_csv[0]
        print('INFO: OCLIMAX simulation finished. Output file: '+self.csv_file_name)

    def plot_spec(self,unit=0,setrange=False,interactive=False,xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None):

        if not self.csv_file_name:
            print('ERROR: No result to plot. Please run INS simulation first.')
            return

        sim = list(csv.reader(open(self.csv_file_name,'r')))
        
        TASK = sim[2][0].split('#')[1]
        INSTR = sim[2][1]
        E_UNIT = sim[2][4]
        T = float(sim[2][2])
        
        if int(E_UNIT) == 0:
            csv_unit = r'cm$^{-1}$'
        elif int(E_UNIT) == 1:
            csv_unit = 'meV'
        elif int(E_UNIT) == 2:
            csv_unit = 'THz'
        if int(unit) == 1:
            plt_unit = r'cm$^{-1}$'
        elif int(unit) == 0:
            plt_unit = 'meV'
        elif int(unit) == 2:
            plt_unit = 'THz'
        
        if plt_unit==csv_unit:
            scalex = 1.0
        elif plt_unit==r'cm$^{-1}$' and csv_unit=='meV':
            scalex = 8.06554
        elif plt_unit==r'cm$^{-1}$' and csv_unit=='THz':
            scalex = 33.35641
        elif plt_unit=='meV' and csv_unit==r'cm$^{-1}$':
            scalex = 0.12399
        elif plt_unit=='meV' and csv_unit=='THz':
            scalex = 4.13567
        elif plt_unit=='THz' and csv_unit==r'cm$^{-1}$':
            scalex = 0.02998
        elif plt_unit=='THz' and csv_unit=='meV':
            scalex = 0.24180
        
        
        i=0
        while i<len(sim):
            if not sim[i]:
                del sim[i]
                continue
            try:
                float(sim[i][0])
            except ValueError:
                del sim[i]
                continue
            i=i+1
        
        for row in range(len(sim)):
            for col in range(len(sim[0])):
                try:
                    float(sim[row][col])
                except ValueError:
                    sim[row][col]='0.0'
                    continue

 
        if int(INSTR) in [0,1,2]:
            fig, ax = plt.subplots(figsize=(6, 5))
            tsim=[[float(sim[row][col]) for row in range(len(sim))] for col in range(len(sim[0]))]
            tsim[0]=[tsim[0][i]*scalex for i in range(len(tsim[0]))]
            ns = len(sim[0])-1
            x_min = min(tsim[0])
            x_max = max(tsim[0])
            y_min = min(tsim[1])
            y_max = max(tsim[1])
            if setrange:
                try:
                    x_min = float(xmin)  #x_min = max([float(xmin),x_min])
                except:
                    pass
                try:
                    x_max = float(xmax)  #x_max = min([float(xmax),x_max])
                except:
                    pass
                try:
                    y_min = float(ymin)  #y_min = max([float(ymin),y_min])
                except:
                    pass
                try:
                    y_max = float(ymax)  #y_max = min([float(ymax),y_max])
                except:
                    pass
            ax.set_xlim(x_min,x_max)
            ax.set_ylim(y_min,y_max)
            for i in range(ns):
                tsim[i+1]=[tsim[i+1][j] for j in range(len(tsim[i+1]))]
                ax.plot(tsim[0],tsim[i+1],label='Spec'+str(i+1))
            ax.set_xlabel('Energy transfer ('+plt_unit+')')
            ax.set_ylabel('Intensity (a.u.)')
            ax.legend(frameon=False)
            fig.show()
        
        elif int(INSTR)==3:
            tsim=[[float(sim[row][col]) for row in range(len(sim))] for col in range(len(sim[0]))]
            x=[tsim[0][i] for i in range(len(tsim[0]))]
            if int(TASK)==3:
                y=[tsim[1][i] for i in range(len(tsim[1]))]
            else:
                y=[tsim[1][i]*scalex for i in range(len(tsim[1]))]
            z=[tsim[2][i] for i in range(len(tsim[2]))]
            xi=np.sort(np.asarray(list(set(x))))
            yi=np.sort(np.asarray(list(set(y))))
            zi = np.zeros((len(yi),len(xi)))
            for i in range(len(yi)):
                for j in range(len(xi)):
                    k = i+j*len(yi)
                    if z[k]>=0.0:
                        zi[i][j]=z[k]
                    elif z[k]<=-1:
                        zi[i][j]=np.nan
                    else:
                        zi[i][j]=0.0
            x_min = min(x)
            x_max = max(x)
            y_min = min(y)
            y_max = max(y)
            z_min = 0
            z_max = max(z)
            if setrange:
                try:
                    x_min = max([float(xmin),x_min])
                except:
                    pass
                try:
                    x_max = min([float(xmax),x_max])
                except:
                    pass
                try:
                    y_min = max([float(ymin),y_min])
                except:
                    pass
                try:
                    y_max = min([float(ymax),y_max])
                except:
                    pass
                try:
                    z_min = float(zmin)  #z_min = max([float(zmin),z_min])
                except:
                    pass
                try:
                    z_max = float(zmax)  #z_max = min([float(zmax),z_max])
                except:
                    pass
            x1=np.where(np.logical_and(xi>=x_min,xi<=x_max))[0][0]
            x2=np.where(np.logical_and(xi>=x_min,xi<=x_max))[0][-1]
            y1=np.where(np.logical_and(yi>=y_min,yi<=y_max))[0][0]
            y2=np.where(np.logical_and(yi>=y_min,yi<=y_max))[0][-1]
            Z = zi[y1:y2+1,x1:x2+1]
            asp = (x_max-x_min)/(y_max-y_min)*1.0

            if interactive:
                fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharex="col", sharey="row",
                                         gridspec_kw=dict(height_ratios=[1, 3],
                                                          width_ratios=[3, 1]))
                axs[0,1].set_visible(False)
                axs[0,0].set_box_aspect(1/3)
                axs[1,0].set_box_aspect(1)
                axs[1,1].set_box_aspect(3/1)
                
                axs[0,0].set_ylabel('Z profile')
                axs[1,1].set_xlabel('Z profile')
                axs[1,1].set_xlim(right=z_max)
                axs[0,0].set_ylim(top=z_max)
                
                
                axs[1,0].imshow(Z,origin='lower',extent=[x_min,x_max,y_min,y_max],vmin=z_min,vmax=z_max,interpolation='gaussian',cmap='jet')
                axs[1,0].set_aspect(aspect=asp, share=False)
                
                
                v_line = axs[1,0].axvline(np.nan, color='r')
                h_line = axs[1,0].axhline(np.nan, color='g')
                v_prof, = axs[1,1].plot(np.zeros(len(Z[:,0])),np.linspace(y_min,y_max,len(Z[:,0])), 'r-')
                h_prof, = axs[0,0].plot(np.linspace(x_min,x_max,len(Z[0,:])),np.zeros(len(Z[0,:])), 'g-')


                dx = xi[1]-xi[0]
                dy = yi[1]-yi[0]

                def on_move(event):
                    if event.inaxes is axs[1,0]:
                        cur_x = event.xdata
                        cur_y = event.ydata
                
                        v_line.set_xdata([cur_x,cur_x])
                        h_line.set_ydata([cur_y,cur_y])
                        v_prof.set_xdata(Z[:,int((cur_x-x_min)/dx)])
                        h_prof.set_ydata(Z[int((cur_y-y_min)/dy),:])
                
                        fig.canvas.draw_idle()
                
                fig.canvas.mpl_connect('motion_notify_event', on_move)
                ax = axs[1,0]

            else:
                fig, ax = plt.subplots(figsize=(6, 5))
                pos = ax.imshow(Z,origin='lower',extent=[x_min,x_max,y_min,y_max],vmin=z_min,vmax=z_max,aspect=asp,interpolation='gaussian',cmap='jet')
                fig.colorbar(pos, ax=ax)

            if int(TASK)==2:
                ax.set_xlabel(r'Q (r.l.u.)')
                ax.set_ylabel('E ('+plt_unit+')')
            elif int(TASK)==3:
                ax.set_xlabel(r'Qx (r.l.u.)')
                ax.set_ylabel(r'Qy (r.l.u.)')
            else:
                ax.set_xlabel(r'|Q| (${\rm \AA}^{-1}$)')
                ax.set_ylabel('E ('+plt_unit+')')
            fig.show()
        



    def open_help_oclimax(self):
        help_oclimax= QDialog()
        self.setup_help = Ui_Help_OCLIMAX()
        self.setup_help.setupUi(help_oclimax)
        help_oclimax.exec()


