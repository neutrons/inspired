from ase import Atoms
#from ase.phonons import Phonons
from ase.io import read, write
from ase.optimize import FIRE
import matgl
from matgl.ext.ase import M3GNetCalculator, Relaxer
from chgnet.model import StructOptimizer, CHGNetCalculator
from mace.calculators import MACECalculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
import numpy as np
import math
import os
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
from phonopy.file_IO import write_FORCE_CONSTANTS
import torch
from inspired.gui.oclimax import OCLIMAX


class MLFFWorker():
    def __init__(self):
        self.oclimax = OCLIMAX()
        self.nx = self.ny = self.nz = None

    def run_opt_and_dos(self, mace_file, m3gnet_path, struc=None, potential_index=0,lmin=12.0,fmax=0.01,nmax=100,delta=0.03):
        """structure optimization and phonon calculation with MLFF
        """

        try:
            lmin = float(lmin)
        except:
            lmin = 12.0
        try:
            fmax = float(fmax)
        except:
            fmax = 0.001
        try:
            nmax = int(nmax)
        except:
            nmax = 100
        try:
            delta = float(delta)
        except:
            delta = 0.03
        abc = struc.cell.cellpar()[0:3]
        nx = math.ceil(lmin/abc[0])        # calculate default mesh in BZ based on cell size
        ny = math.ceil(lmin/abc[1])
        nz = math.ceil(lmin/abc[2])
        self.nx = nx
        self.ny = ny
        self.nz = nz
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.set_default_dtype(torch.float32)
        print('INFO: Running structural optimization...')
        if potential_index == 0:  # MACE
            calculator = MACECalculator(model_paths=mace_file, device=device)
            struc.set_calculator(calculator)
            dyn = FIRE(struc)
            dyn.run(fmax=fmax, steps=nmax)
            atoms_relaxed = dyn.atoms.copy()
        elif potential_index == 1:  # CHGNet
            calculator = CHGNetCalculator()
            relaxer = StructOptimizer()
            relax_results = relaxer.relax(struc, fmax=fmax, steps=nmax, relax_cell=False)
            final_structure = relax_results['final_structure']
            atoms_relaxed = AseAtomsAdaptor().get_atoms(final_structure)
        elif potential_index == 2:  # M3GNet
            pot = matgl.load_model(m3gnet_path)
            calculator = M3GNetCalculator(pot)
            relaxer = Relaxer(potential=pot,relax_cell=False)
            relax_results = relaxer.relax(struc, fmax=fmax, steps=nmax, verbose=True)
            final_structure = relax_results['final_structure']
            atoms_relaxed = AseAtomsAdaptor().get_atoms(final_structure)
        write('POSCAR-unitcell', atoms_relaxed, direct=True, format='vasp')
        print('INFO: Structural optimization finished.')

        # Phonon calculator
        npc = len(struc.numbers)
        nsc = npc*nx*ny*nz
        print('INFO: Number of atoms in unit cell: '+str(npc))
        print('INFO: Supercell dimension: '+' '.join(map(str,[nx,ny,nz])))
        print('INFO: Total number of atoms in supercell: '+str(nsc))

        unitcell, _ = read_crystal_structure("POSCAR-unitcell", interface_mode='vasp')
        phonon = Phonopy(unitcell,
                         supercell_matrix=[self.nx, self.ny, self.nz],
                         primitive_matrix=np.array([[1,0,0], [0,1,0], [0,0,1]]))

        # Phonon calculation with ASE
        """ 
        supercell = (self.nx, self.ny, self.nz)
        ph = Phonons(atoms_relaxed, calculator, supercell=supercell, delta=delta)
        print('INFO: Running phonon calculation. Expect '+str(nsc)+'x6 cache.json files in ./phonon when completed.')
        print('INFO: Check phonon folder in working directory for progress. Press Ctrl+C in terminal to abort.')
        try:
            ph.run()
        except:
            print('ERROR: Phonon calculation failed or terminated.')
            return

        # Read forces and assemble the dynamical matrix
        ph.read(acoustic=True)
        ph.clean()

        fcs = ph.get_force_constant()

        #print(len(fcs),len(fcs[0]),len(fcs[0][0]))

        force_constants = np.zeros((npc,nsc,3,3))
        for i in range(npc):
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        k = ix+iy*nx+iz*nx*ny
                        for j in range(npc):
                            force_constants[i,j*nx*ny*nz+k] = fcs[k,3*i:3*i+3,3*j:3*j+3]
        
        phonon.force_constants = force_constants
        """

        # Phonon calculation with phonopy

        phonon.generate_displacements(distance=delta)
        supercells = phonon.supercells_with_displacements
        ns = len(supercells)
        print('INFO: Total number of displacements: '+str(ns))

        sets_of_forces=[]
        for i in range(ns):
            supercell = supercells[i]
            sc = Atoms(symbols=supercell.get_chemical_symbols(),
                       scaled_positions=supercell.get_scaled_positions(),
                       cell=supercell.get_cell(),
                       pbc=True)
            calculator.calculate(atoms=sc,properties='forces')
            sets_of_forces.append(calculator.results['forces'])
            if i<ns-1:
                print('INFO: '+str(i+1)+' of '+str(ns)+' displacements finished', end ='\r')
            else:
                print('INFO: '+str(i+1)+' of '+str(ns)+' displacements finished')

        phonon.forces = sets_of_forces
        phonon.produce_force_constants()


        try:
            os.remove('BORN')
        except OSError:
            pass
        try:
            os.remove('FORCE_SETS')
        except OSError:
            pass
        write_FORCE_CONSTANTS(phonon.get_force_constants(),p2s_map=phonon.primitive.get_primitive_to_supercell_map())
        print('INFO: Phonon calculation finished.')
        print('INFO: Plotting phonon dispersion and DOS. For large unitcells this may take a few moments.')
        print('INFO: Frequency unit in plot is THz. 1 THz = 4.136 meV = 33.356 cm-1')
        print('INFO: Phonon DOS data will be saved in total_dos.dat file')
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


    def generate_initial_mesh_file(self, mesh_list=[1,1,1]):
        """generate mesh.conf file for MLFF calculation
        """

        if self.nx and self.ny and self.nz:
            try:
                os.remove('BORN')
            except OSError:
                pass
            try:
                os.remove('FORCE_SETS')
            except OSError:
                pass
            mf = open("mesh.conf",'w')
            mf.write('DIM = ' + str(self.nx) + ' 0 0 0 ' + str(self.ny) + ' 0 0 0 ' + str(self.nz) + '\n')
            mf.write('\n')
            mf.write(' '.join(['MP =',' '.join(list(map(str,mesh_list))),'\n']))
            mf.write('FC_SYMMETRY = .TRUE. \n')
            mf.write('GAMMA_CENTER = .TRUE. \n')
            mf.write('EIGENVECTORS = .TRUE. \n')
            mf.write('FORCE_CONSTANTS = READ \n')
            mf.close()
            self.oclimax.oclimax_params.get_default_mesh()
            self.oclimax.use_default_mesh()
        else:
            print('INFO: Please run phonon calculation first before setting up INS calculation.')
