import contextlib
import io
import math
import os

import numpy as np
import torch
from ase import Atoms
from ase.filters import ExpCellFilter

# from ase.phonons import Phonons
from ase.io import write
from ase.optimize import FIRE
from phonopy import Phonopy
from phonopy.file_IO import parse_BORN, write_FORCE_CONSTANTS
from phonopy.interface.calculator import read_crystal_structure
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath

from inspired.gui.oclimax import OCLIMAX


class MLFFWorker:
    def __init__(self):
        self.oclimax = OCLIMAX()
        self.nx = self.ny = self.nz = None

    def run_opt_and_dos(
        self,
        struc=None,
        potential_index=0,
        use_specific_model=False,
        mlff_model_name=None,
        lmin=12.0,
        fmax=0.01,
        nmax=100,
        delta=0.03,
        relax_cell=False,
    ):
        """Structure optimization and phonon calculation with MLFF"""
        try:
            lmin = float(lmin.strip())
        except:
            try:
                lmin = list(map(int, lmin.strip().split()))
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
        if not isinstance(lmin, list):
            nx = math.ceil(lmin / abc[0])  # calculate default mesh in BZ based on cell size
            ny = math.ceil(lmin / abc[1])
            nz = math.ceil(lmin / abc[2])
        elif len(lmin) == 3:
            nx = lmin[0]
            ny = lmin[1]
            nz = lmin[2]
        else:
            print("ERROR: Check Lmin/Dim. Must be one float number or three integers.")
            return
        self.nx = nx
        self.ny = ny
        self.nz = nz
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("INFO: Running structural optimization...")
        if potential_index == 0:  # MatterSim
            from mattersim.forcefield.potential import MatterSimCalculator

            torch.set_default_dtype(torch.float32)
            if use_specific_model and mlff_model_name is not None:
                with contextlib.redirect_stderr(io.StringIO()) as f:
                    calculator = MatterSimCalculator(load_path=mlff_model_name, device=device)
            else:
                with contextlib.redirect_stderr(io.StringIO()) as f:
                    calculator = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)
        elif potential_index == 1:  # SevenNet
            from sevenn.calculator import SevenNetCalculator

            torch.set_default_dtype(torch.float32)
            if use_specific_model and mlff_model_name is not None:
                with contextlib.redirect_stderr(io.StringIO()) as f:
                    calculator = SevenNetCalculator(model=mlff_model_name, device=device)
            else:
                with contextlib.redirect_stderr(io.StringIO()) as f:
                    calculator = SevenNetCalculator(model="7net-mf-ompa", modal="mpa", device=device)
        '''
        elif potential_index == 2:  # ORB v3
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator

            with contextlib.redirect_stderr(io.StringIO()) as f:
                orbff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
            calculator = ORBCalculator(orbff, device=device)
        elif potential_index == 3:  # MACE
            from mace.calculators import mace_mp

            if use_specific_model and mlff_model_name is not None:
                with contextlib.redirect_stderr(io.StringIO()) as f:
                    calculator = mace_mp(model=mlff_model_name, default_dtype="float64", device=device)
            else:
                with contextlib.redirect_stderr(io.StringIO()) as f:
                    calculator = mace_mp(model="medium-mpa-0", default_dtype="float64", device=device)
        elif potential_index == 4:  # MACE-OFF
            from mace.calculators import mace_off

            if use_specific_model and mlff_model_name is not None:
                with contextlib.redirect_stderr(io.StringIO()) as f:
                    calculator = mace_off(model=mlff_model_name, default_dtype="float64", device=device)
            else:
                with contextlib.redirect_stderr(io.StringIO()) as f:
                    calculator = mace_off(model="medium", default_dtype="float64", device=device)
        '''
        struc.set_calculator(calculator)
        if relax_cell:
            ecf = ExpCellFilter(struc)
            dyn = FIRE(ecf)
        else:
            dyn = FIRE(struc)
        try:
            dyn.run(fmax=fmax, steps=nmax)
        except Exception as error:
            print("ERROR: Simulation failed.", error)
            return
        atoms_relaxed = dyn.atoms.copy()
        write("POSCAR-unitcell", atoms_relaxed, direct=True, format="vasp")
        print("INFO: Structural optimization finished.")

        # Phonon calculator
        npc = len(struc.numbers)
        nsc = npc * nx * ny * nz
        print("INFO: Number of atoms in unit cell: " + str(npc))
        print("INFO: Supercell dimension: " + " ".join(map(str, [nx, ny, nz])))
        print("INFO: Total number of atoms in supercell: " + str(nsc))

        unitcell, _ = read_crystal_structure("POSCAR-unitcell", interface_mode="vasp")
        phonon = Phonopy(
            unitcell,
            supercell_matrix=[self.nx, self.ny, self.nz],
            primitive_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        )

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
        print("INFO: Total number of displacements: " + str(ns))

        sets_of_forces = []
        for i in range(ns):
            supercell = supercells[i]
            sc = Atoms(
                symbols=supercell.get_chemical_symbols(),
                scaled_positions=supercell.get_scaled_positions(),
                cell=supercell.get_cell(),
                pbc=True,
            )
            calculator.calculate(atoms=sc, properties="forces")
            sets_of_forces.append(calculator.results["forces"])
            if i < ns - 1:
                print(
                    "INFO: " + str(i + 1) + " of " + str(ns) + " displacements finished",
                    end="\r",
                )
            else:
                print("INFO: " + str(i + 1) + " of " + str(ns) + " displacements finished")

        phonon.forces = sets_of_forces
        phonon.produce_force_constants()

        if os.path.isfile("BORN"):
            print("INFO: BORN file found in the current folder.")
            print("INFO: Unless it is there on purpose to include NAC, please remove it.")
            nac_params = parse_BORN(phonon.primitive, filename="BORN")
            nac_params["factor"] = 14.4
            phonon.set_nac_params(nac_params)
        try:
            os.remove("FORCE_SETS")
        except OSError:
            pass
        write_FORCE_CONSTANTS(
            phonon.get_force_constants(),
            p2s_map=phonon.primitive.get_primitive_to_supercell_map(),
        )
        print("INFO: Phonon calculation finished.")
        print("INFO: Plotting phonon dispersion and DOS. For large unitcells this may take a few moments.")
        print("INFO: Frequency unit in plot is THz. 1 THz = 4.136 meV = 33.356 cm-1")
        print("INFO: Phonon DOS data will be saved in total_dos.dat file")
        with contextlib.redirect_stderr(io.StringIO()) as f:
            bands, labels, path_connections = get_band_qpoints_by_seekpath(phonon.primitive, 1, is_const_interval=True)
        points = []
        for i in range(len(bands)):
            if i == 0 or (bands[i - 1][1] != bands[i][0]).any():
                points.append(bands[i][0])
            points.append(bands[i][1])
        print("INFO: Labels for the special points in phonon dispersion:")
        for i in range(len(labels)):
            print(labels[i], points[i])

        # To plot DOS next to band structure
        phonon.auto_band_structure()
        phonon.auto_total_dos()
        phonon.write_total_dos()
        phonon.plot_band_structure_and_dos().show()

    def generate_initial_mesh_file(self, mesh_list=[1, 1, 1]):
        """Generate mesh.conf file for MLFF calculation"""
        if self.nx and self.ny and self.nz:
            if os.path.isfile("BORN"):
                print("INFO: BORN file found in the current folder.")
                print("INFO: Unless it is there on purpose to include NAC, please remove it.")
            try:
                os.remove("FORCE_SETS")
            except OSError:
                pass
            mf = open("mesh.conf", "w")
            mf.write("DIM = " + str(self.nx) + " 0 0 0 " + str(self.ny) + " 0 0 0 " + str(self.nz) + "\n")
            mf.write("\n")
            mf.write(" ".join(["MP =", " ".join(list(map(str, mesh_list))), "\n"]))
            mf.write("FC_SYMMETRY = .TRUE. \n")
            mf.write("GAMMA_CENTER = .TRUE. \n")
            mf.write("EIGENVECTORS = .TRUE. \n")
            mf.write("FORCE_CONSTANTS = READ \n")
            mf.close()
            self.oclimax.oclimax_params.get_default_mesh()
            self.oclimax.use_default_mesh()
        else:
            print("INFO: Please run phonon calculation first before setting up INS calculation.")
