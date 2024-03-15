# Inelastic Neutron Scattering Prediction for Instantaneous Results and Experimental Design (INSPIRED)

## Introduction:

INSPIRED is a PyQt GUI that performs rapid prediction/calculation/visualization of inelastic neutron scattering (INS) spectra from a given structural model. This can be done in three different ways, each with its advantages and limitations:

1. Direct prediction from a symmetry-aware neural network

* Pros: any structure, minimum parameter setting, very fast (seconds)

* Cons: powder samples only, predefined Q and E ranges, less accurate

2. INS simulation based on existing DFT-calculated forces

* Pros: both powder and single crystal samples, very accurate

* Cons: only for crystals with DFT results found in the database (or provided by the users)

3. INS simulation using pre-trained universal machine learning force fields

* Pros: any structure, both powder and single crystal samples

* Cons: can be slow for large or low-symmetry systems, accuracy varies (generally between 1 and 2)


## Installation:
INSPIRED currently only runs on Linux operating systems. The neutron data server ([analysis cluster](https://analysis.sns.gov)) at ORNL will work if you can access it. We are working on deploying INSPIRED on the analysis cluster so that all users can use it easily. Before the deployment is completed, you may install INSPIRED in your home directory on the analysis cluster (or any other Linux machine) by following the steps below ([Option 1](https://github.com/cyqjh/inspired#option-1)). Alternatively, you may download a pre-compiled VirtualBox image and run INSPIRED as a virtual machine (VM) on any platform, including Windows, MacOS, and Linux ([Option 2](https://github.com/cyqjh/inspired#option-2)).

### Option 1
1. Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/) or [Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) for Linux if it is not already installed. If you are unsure which one to choose, check [here](https://docs.anaconda.com/free/distro-or-miniconda/). Note that for the purpose of running INSPIRED, either one will work. 

2. With conda initiated, run the following commands:

    `conda create -n inspired python=3.9`

    `conda activate inspired`

4. Go to a location where you would like to install the program (e.g., $HOME/software), run:
   
    `git clone https://github.com/cyqjh/inspired.git`
   
    After the download, you should see a folder named “inspired”. Go to the folder by running:
   
    `cd inspired`
   
    (Note: to get updated code in the future, you can simply run “git pull” in this folder)

4. To download the latest DFT database and ML models from Zenodo and extract the files, run:
   
    `wget https://zenodo.org/records/10723108/files/dftdb.tar.gz`

    `wget https://zenodo.org/records/10723108/files/model.tar.gz`

    `tar zxvf dftdb.tar.gz`

    `tar zxvf model.tar.gz`

5. Determine if your computer is CPU-only or equipped with GPU/CUDA. If you are not sure, choose the CPU-only option. Run:

    `cd ./install`

    `bash conda_env_cpu.sh` (or `bash conda_env_gpu.sh` for GPU machines).

    (Note: It may take a while for the installation to be completed. If you encounter errors, you may try running the commands in the sh file one by one to diagnose.)

6. If all packages are installed successfully, you may now go to a working directory of your choice and start INSPIRED by running:
   
     `inspired`

### Option 2:
   If you do not have access to a Linux computer, or you cannot install INSPIRED properly by following the steps in Option 1, you may consider running INSPIRED as a VM.
1. Install [VirtualBox for your operating system](https://www.virtualbox.org/wiki/Downloads).
2. To download the VirtualBox image from Zenodo, run:

   `wget https://zenodo.org/records/10723108/files/inspired.ova`

3. Start VirtualBox, [import the inspired.ova file as an appliance](https://docs.oracle.com/cd/E26217_01/E26796/html/qs-import-vm.html).
4. Run the “inspired” VM. Within the VM, start a terminal, run:

   `inspired`

Note: You can use the “shared folder” within the VM to access (read and write) files on your host computer. You can set your INSPIRED working directory in the shared folder. More information can be found [here](https://docs.oracle.com/en/virtualization/virtualbox/6.0/user/sharedfolders.html).


## Additional notes:

When using INSPIRED, it is important to ensure the "Current Working Directory' (CWD) is correctly set, as this is where the program read/write all its input/output files. The default CWD is where you started INSPIRED from, and once this program is running, the CWD can be changed in the Menu. It is strongly recommended that a new folder is created for each material to avoid mixed input/output files and potential errors.

There is a "Help" button at the bottom right corner of each window. Click on the button for instructions.

If you have any questions, please contact YQ Cheng at chengy@ornl.gov for help.


## Citation:

INSPIRED paper is in preparation.


## References:

1.	Cheng, Y.;  Wu, G.;  Pajerowski, D. M.;  Stone, M. B.;  Savici, A. T.;  Li, M.; Ramirez-Cuesta, A. J., Direct prediction of inelastic neutron scattering spectra from the crystal structure. Machine Learning: Science and Technology 2023, 4 (1), 015010.
2.	Cheng, Y.;  Stone, M. B.; Ramirez-Cuesta, A. J., A database of synthetic inelastic neutron scattering spectra from molecules and crystals. Scientific Data 2023, 10 (1), 54.
3.	Chen, Z.;  Andrejevic, N.;  Smidt, T.;  Ding, Z.;  Xu, Q.;  Chi, Y.-T.;  Nguyen, Q. T.;  Alatas, A.;  Kong, J.; Li, M., Direct Prediction of Phonon Density of States With Euclidean Neural Networks. Advanced Science 2021, 8 (12), 2004214.
4.	Togo, A. http://phonondb.mtl.kyoto-u.ac.jp/. (accessed 08/30/2022).
5.	Togo, A.; Tanaka, I., First principles phonon calculations in materials science. Scripta Materialia 2015, 108, 1-5.
6.	Cheng, Y.;  Daemen, L.;  Kolesnikov, A.; Ramirez-Cuesta, A., Simulation of inelastic neutron scattering spectra using OCLIMAX. Journal of chemical theory and computation 2019, 15 (3), 1974-1982.
7.	Hjorth Larsen, A.;  Jørgen Mortensen, J.;  Blomqvist, J.;  Castelli, I. E.;  Christensen, R.;  Dułak, M.;  Friis, J.;  Groves, M. N.;  Hammer, B.;  Hargus, C.;  Hermes, E. D.;  Jennings, P. C.;  Bjerre Jensen, P.;  Kermode, J.;  Kitchin, J. R.;  Leonhard Kolsbjerg, E.;  Kubal, J.;  Kaasbjerg, K.;  Lysgaard, S.;  Bergmann Maronsson, J.;  Maxson, T.;  Olsen, T.;  Pastewka, L.;  Peterson, A.;  Rostgaard, C.;  Schiøtz, J.;  Schütt, O.;  Strange, M.;  Thygesen, K. S.;  Vegge, T.;  Vilhelmsen, L.;  Walter, M.;  Zeng, Z.; Jacobsen, K. W., The atomic simulation environment—a Python library for working with atoms. Journal of Physics: Condensed Matter 2017, 29 (27), 273002.
8.	Batatia, I.;  Kovacs, D. P.;  Simm, G.;  Ortner, C.; Csányi, G., MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. Advances in Neural Information Processing Systems 2022, 35, 11423-11436.
9.	Batatia, I.;  Batzner, S.;  Kovács, D. P.;  Musaelian, A.;  Simm, G. N.;  Drautz, R.;  Ortner, C.;  Kozinsky, B.; Csányi, G., The design space of E (3)-equivariant atom-centered interatomic potentials. arXiv preprint arXiv:2205.06643 2022.
10.	Batatia, I.;  Benner, P.;  Chiang, Y.;  Elena, A. M.;  Kovács, D. P.;  Riebesell, J.;  Advincula, X. R.;  Asta, M.;  Baldwin, W. J.; Bernstein, N., A foundation model for atomistic materials chemistry. arXiv preprint arXiv:2401.00096 2023.
11.	Deng, B.;  Zhong, P.;  Jun, K.;  Riebesell, J.;  Han, K.;  Bartel, C. J.; Ceder, G., CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling. Nature Machine Intelligence 2023, 5 (9), 1031-1041.
12.	Chen, C.; Ong, S. P., A universal graph deep learning interatomic potential for the periodic table. Nature Computational Science 2022, 2 (11), 718-728.
