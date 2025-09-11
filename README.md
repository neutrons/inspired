# Inelastic Neutron Scattering Prediction for Instantaneous Results and Experimental Design (INSPIRED)

## Introduction:

INSPIRED is a PyQt GUI that performs rapid prediction/calculation/visualization of inelastic neutron scattering (INS) spectra from a given structural model. This can be done in three different ways, each with its advantages and limitations:


1. INS simulation based on existing DFT-calculated forces

* Pros: both powder and single crystal samples, more accurate

* Cons: only for crystals with DFT results found in the database (or provided by the users)

  ![dft_tab](https://github.com/user-attachments/assets/572dd5e5-6441-4ff8-9086-856eef6a729b)


2. INS simulation using pre-trained universal machine learning force fields (MLFFs)

* Pros: any structure, both powder and single crystal samples

* Cons: can be slow for large or low-symmetry systems, accuracy varies significantly from case to case

  ![mlff_tab](https://github.com/user-attachments/assets/47ff98ba-3c7c-4755-97f4-2f0d5720e5ae)


3. Direct prediction from a symmetry-aware neural network

* Pros: any structure, minimum parameter setting, very fast (seconds), good for complex systems when explicit modeling is not possible

* Cons: powder samples only, predefined Q and E ranges, less accurate, accuracy varies significantly from case to case

  ![dp_tab](https://github.com/user-attachments/assets/7cc13f86-3cff-4514-8444-5322cebbdf9e)


In general, one should follow the above order in choosing the tools. If DFT results are available for the system of interest, they should be used. If not, try the MLFFs to see if you can calculate phonons in reasonable time. If not, the Predictor can be used to obtain a quick guess.


## Installation:

INSPIRED is available on the [Analysis cluster](https://analysis.sns.gov/). To run INSPIRED, open a terminal and type:

  ```bash
  inspired
  ```

Alternatively, you could also use the Desktop Menu (Applications->Analysis->INSPIRED) to open the main window.

Please be sure to `Set Working Directory` in the INSPIRED Menu before running any calculations to avoid writing the output files in unintended/unknown locations.

Occasionally, there can be updates/bug-fixes that are not yet available in the production version. To use the dev version on analysis, type the following command in a terminal:

  ```bash
  inspired --dev
  ```

INSPIRED can also be installed locally on your personal computer. The current version of INSPIRED only runs on Linux platforms. Windows users could take advantage of the [Windows Subsystem for Linux (WSL)](https://github.com/neutrons/inspired#wsl-installation-for-windows-users) and Mac users could install a Linux [virtual machine](https://www.virtualbox.org/wiki/Downloads). Once you have a Linux platform, INSPIRED can be installed by following the steps below:

### Option 1: Install INSPIRED as a pixi package ([v0.4.2](https://github.com/neutrons/inspired#package-versions) and later)

1. Install [pixi](https://pixi.sh/) for managing environments, dependencies, packaging, and task execution (if it is not already installed).

    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

    Also install [git](https://github.com/git-guides/install-git), if it is missing.

2. Go to a location where you would like to install the program (e.g., $HOME/software), run:

    ```bash
    git clone https://github.com/neutrons/inspired.git
    ```

    After the download, you should see a folder named “inspired”. Go to the folder:

    ```bash
    cd inspired
    ```

3. Download the repository. Setup/Update the environment

    ```bash
    pixi install
    ```

4. Enter the environment

    ```bash
    pixi shell
    ```

5. To download the latest DFT database and ML models from Zenodo and extract the files, go to (create) a folder where you want to keep these files, run:

    ```bash
    wget https://zenodo.org/records/11478889/files/dftdb.tar.gz
    wget https://zenodo.org/records/10723108/files/model.tar.gz
    tar zxvf dftdb.tar.gz
    tar zxvf model.tar.gz
    ```

6. If all packages are installed successfully, you may now go to a working directory of your choice and start INSPIRED by running:

    ```bash
    inspired
    ```

    when you run the program for the first time, it may ask you to specify the paths to the DFT database and ML models. Set them to where you downloaded/extracted the files, and the main user interface should pop up.


### Option 2: Install INSPIRED as a conda environment ([v0.4.1](https://github.com/neutrons/inspired#package-versions))

1. Install a conda environment manager for Linux, if not already installed. It can be [miniforge](https://github.com/conda-forge/miniforge), [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/), or [anaconda](https://docs.anaconda.com/free/anaconda/install/linux/). Also install [git](https://github.com/git-guides/install-git) if it is missing. 

2. Go to a location where you would like to install the program (e.g., $HOME/software), run:
   
    ```bash
    git clone -b conda041 https://github.com/neutrons/inspired.git
    ```
        
    After the download, you should see a folder named “inspired”. Go to the folder by running:
   
    ```bash
    cd inspired
    ```
   
4. With conda initiated, run the following commands in the project's root directory (inspired):

    ```bash
    conda env create
    conda activate inspired-dev
    pip install -e .
    ```

5. To download the latest DFT database and ML models from Zenodo and extract the files, go to (create) a folder where you want to keep these files, run:
   
    ```bash
    wget https://zenodo.org/records/11478889/files/dftdb.tar.gz
    wget https://zenodo.org/records/10723108/files/model.tar.gz
    tar zxvf dftdb.tar.gz
    tar zxvf model.tar.gz
    ```

6. If all packages are installed successfully, you may now go to a working directory of your choice and start INSPIRED by running:
   
    ```bash
    inspired
    ```

   when you run the program for the first time, it may ask you to specify the paths to the DFT database and ML models. Set them to where you downloaded/extracted the files, and the main user interface should pop up.


### WSL installation for Windows users

1. Make sure Hyper-V is enabled: search `Turn Windows features on and off` in Windows search bar, click to open the control panel, if `Hyper-V` if not checked, check the box and restart the computer.

2. Install WSL: search `Command Prompt` in Windows search bar, right click, choose `Run as administrator`. Type `wsl.exe --install` and run. This will install WSL on your computer.

3. Install Ubuntu: Most users do not use Administrator as default account. Therefore, open a new `Command Prompt` window (this time NOT as administrator), and run `wsl.exe --install` again. It will install [Ubuntu Linux](https://ubuntu.com/) under your account. 

4. Start Ubuntu: search 'Ubuntu' in Windows search bar, click the Ubuntu App. The pop-up window will be a Linux terminal.

5. Additional setup you may need in the Ubuntu terminal: 

    ```bash
    sudo apt-get update
    sudo apt-get install git-all
    sudo apt install libsxb-cursor0 libxcb-xinerama0 libqt5x11extras5
    chmod 0700 /run/user/1000/
    ```

6. Now you can install INSPIRED following Option 1 or Option 2 above.


### Package versions

MLFF models in v0.4.2 and later (Option 1) only include MatterSim and SevenNet. To use other MLFF models such as MACE, MACE-OFF, and ORB, install v0.4.1 with conda (Option 2).


## Additional notes:

When using INSPIRED, it is important to ensure the "Current Working Directory" (CWD) is correctly set, as this is where the program reads/writes all its input/output files. The default CWD is where you started INSPIRED, and once the program is running, you can change the CWD in the Menu. It is strongly recommended that a new folder be created for each project to avoid mixed input/output files and potential errors.

There is a "Help" button at the bottom right corner of each window. Click on the button for instructions.

Finally, but **very importantly**, INSPIRED is a GUI that provides streamlined access to various tools for users to obtain simulated INS spectra easily and quickly from a structure model. The tools included in INSPIRED may be under active development by their developers, such as the MLFFs. With growing computing power, larger training datasets, and more sophisticated models, the scope of the data and the accuracy of the models will keep improving. While we will try to keep the models included in INSPIRED up to date, we do not and cannot guarantee the accuracy of the results. Please check the references (1,8-12) for more details about these tools and their limitations. User discretion is extremely important while using INSPIRED, and please check the results carefully before using them for your analyses. Here are some additional resources on the benchmark of the MLFFs:

https://matbench-discovery.materialsproject.org/

https://iopscience.iop.org/article/10.1088/2632-2153/adfa68


## Contact:

If you have any questions, please contact YQ Cheng at chengy@ornl.gov for help.


## Citation:

Bowen Han, Andrei T. Savici, Mingda Li, Yongqiang Cheng, [INSPIRED: Inelastic Neutron Scattering Prediction for Instantaneous Results and Experimental Design](https://doi.org/10.1016/j.cpc.2024.109288), Computer Physics Communications, 304, 109288 (2024).


## References:

1.	Cheng, Y.;  Wu, G.;  Pajerowski, D. M.;  Stone, M. B.;  Savici, A. T.;  Li, M.; Ramirez-Cuesta, A. J., Direct prediction of inelastic neutron scattering spectra from the crystal structure. Machine Learning: Science and Technology 2023, 4 (1), 015010.
2.	Cheng, Y.;  Stone, M. B.; Ramirez-Cuesta, A. J., A database of synthetic inelastic neutron scattering spectra from molecules and crystals. Scientific Data 2023, 10 (1), 54.
3.	Chen, Z.;  Andrejevic, N.;  Smidt, T.;  Ding, Z.;  Xu, Q.;  Chi, Y.-T.;  Nguyen, Q. T.;  Alatas, A.;  Kong, J.; Li, M., Direct Prediction of Phonon Density of States With Euclidean Neural Networks. Advanced Science 2021, 8 (12), 2004214.
4.	(Togo18 database) Togo, A. http://phonondb.mtl.kyoto-u.ac.jp/. (accessed 08/30/2022).
5.	Togo, A.; Tanaka, I., First principles phonon calculations in materials science. Scripta Materialia 2015, 108, 1-5.
6.	Cheng, Y.;  Daemen, L.;  Kolesnikov, A.; Ramirez-Cuesta, A., Simulation of inelastic neutron scattering spectra using OCLIMAX. Journal of chemical theory and computation 2019, 15 (3), 1974-1982.
7.	Hjorth Larsen, A.;  Jørgen Mortensen, J.;  Blomqvist, J.;  Castelli, I. E.;  Christensen, R.;  Dułak, M.;  Friis, J.;  Groves, M. N.;  Hammer, B.;  Hargus, C.;  Hermes, E. D.;  Jennings, P. C.;  Bjerre Jensen, P.;  Kermode, J.;  Kitchin, J. R.;  Leonhard Kolsbjerg, E.;  Kubal, J.;  Kaasbjerg, K.;  Lysgaard, S.;  Bergmann Maronsson, J.;  Maxson, T.;  Olsen, T.;  Pastewka, L.;  Peterson, A.;  Rostgaard, C.;  Schiøtz, J.;  Schütt, O.;  Strange, M.;  Thygesen, K. S.;  Vegge, T.;  Vilhelmsen, L.;  Walter, M.;  Zeng, Z.; Jacobsen, K. W., The atomic simulation environment—a Python library for working with atoms. Journal of Physics: Condensed Matter 2017, 29 (27), 273002.
8.	(MACE) Batatia, I.;  Kovacs, D. P.;  Simm, G.;  Ortner, C.; Csányi, G., MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. Advances in Neural Information Processing Systems 2022, 35, 11423-11436.
9.	(MACE) Batatia, I.;  Batzner, S.;  Kovács, D. P.;  Musaelian, A.;  Simm, G. N.;  Drautz, R.;  Ortner, C.;  Kozinsky, B.; Csányi, G., The design space of E (3)-equivariant atom-centered interatomic potentials. arXiv preprint arXiv:2205.06643 2022.
10.	(MACE) Batatia, I.;  Benner, P.;  Chiang, Y.;  Elena, A. M.;  Kovács, D. P.;  Riebesell, J.;  Advincula, X. R.;  Asta, M.;  Baldwin, W. J.; Bernstein, N., A foundation model for atomistic materials chemistry. arXiv preprint arXiv:2401.00096 2023.
11.	(MACE-OFF) Kovács, D. P. et al. MACE-OFF: Transferable Short Range Machine Learning Force Fields for Organic Molecules. arXiv preprint arXiv:2312.15211 2023.
12.	(MatterSim) Yang, H. et al. MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures. arXiv preprint arXiv:2405.04967 2024. https://github.com/microsoft/mattersim.
13.	(ORB v3) Rhodes, B.; Vandenhaute, S.; Šimkus, V.; Gin, J.; Godwin, J.; Duignan, T.; Neumann, M; Orb-v3: atomistic simulation at scale. arXiv preprint arXiv:2504.06231 2025. https://github.com/orbital-materials/orb-models
14.	(SevenNet) Park, Y., Kim, J., Hwang, S., Han, S., Scalable Parallel Algorithm for Graph Neural Network Interatomic Potentials in Molecular Dynamics Simulations. J. Chem. Theory Comput. 2024, 20, 11, 4857–4868. https://github.com/MDIL-SNU/SevenNet.
15.	Hinuma, Y.; Pizzi, G.; Kumagai, Y.; Oba, F.; Tanaka, I., Band structure diagram paths based on crystallography. Comp. Mat. Sci. 2017, 128, 140.
16.	Togo, A.; Tanaka, I., Spglib: a software library for crystal symmetry search. 2018, arXiv:1808.01590
17.	Geiger, M.; Smidt, T., e3nn: Euclidean neural networks. 2022, doi:10.48550/ARXIV.2207.09453.
18.	Geiger, M.; et al., Euclidean neural networks: e3nn. 2022, doi:10.5281/zenodo.6459381.


[![CI](https://github.com/neutrons/inspired/actions/workflows/actions.yml/badge.svg?branch=next)](https://github.com/neutrons/inspired/actions/workflows/actions.yml)
