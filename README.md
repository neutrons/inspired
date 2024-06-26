# Inelastic Neutron Scattering Prediction for Instantaneous Results and Experimental Design (INSPIRED)

## Introduction:

INSPIRED is a PyQt GUI that performs rapid prediction/calculation/visualization of inelastic neutron scattering (INS) spectra from a given structural model. This can be done in three different ways, each with its advantages and limitations:

1. Direct prediction from a symmetry-aware neural network

* Pros: any structure, minimum parameter setting, very fast (seconds)

* Cons: powder samples only, predefined Q and E ranges, less accurate
  
  ![image](https://github.com/cyqjh/inspired/assets/105002220/2fd4288e-2739-4852-9d4e-4513c9cfc029)


2. INS simulation based on existing DFT-calculated forces

* Pros: both powder and single crystal samples, more accurate

* Cons: only for crystals with DFT results found in the database (or provided by the users)
  
  ![image](https://github.com/cyqjh/inspired/assets/105002220/c28a9c3a-1fae-4c38-9ca3-dfb03d19492b)


3. INS simulation using pre-trained universal machine learning force fields

* Pros: any structure, both powder and single crystal samples

* Cons: can be slow for large or low-symmetry systems, accuracy varies significantly from case to case (see Additional notes)
  
  ![image](https://github.com/cyqjh/inspired/assets/105002220/04571dc1-5dfb-469b-92e9-8e5f523b33d2)




## Installation:

For most users, the easiest way to use INSPIRED is to download a pre-installed VirtualBox image and run INSPIRED as a virtual machine (VM) on any platform, including Windows, MacOS, and Linux ([Option 1](https://github.com/cyqjh/inspired#option-1)). If you have acess to a Linux machine (such as the [Analysis cluster](https://analysis.sns.gov/)) and would like to have a native installation, you may also do so ([Option 2](https://github.com/cyqjh/inspired#option-2)).

### Option 1
1. Install [VirtualBox for your operating system](https://www.virtualbox.org/wiki/Downloads). Unfortunately, VirtualBox does not support Apple M3 chips. There was a [developer preview](https://download.virtualbox.org/virtualbox/7.0.8/VirtualBox-7.0.8_BETA4-156879-macOSArm64.dmg) from an older version of VirtualBox that may support M1/M2 chips, but we did not test it. We are working on a solution to address this issue.
2. Download the VirtualBox appliance file (inspired_vm.ova) from [Zenodo](https://doi.org/10.5281/zenodo.11478889). The file is over 6GB, and it may take a while to download (depending on the speed of the internet). On MacOS/Linux, you may download by command line:

   `wget https://zenodo.org/records/11478889/files/inspired_vm.ova`

3. Start VirtualBox, [import the inspired_vm.ova file as an appliance](https://docs.oracle.com/cd/E26217_01/E26796/html/qs-import-vm.html).
4. Run the “inspired_vm” VM. If prompted, use "inspired" for both user ID and password for authentication.
5. Set up the "shared folder" to access (read and write) files on your host computer. Click "Devices->Shared Folders->Shared folders settings" on the VM menu. Click the "add new shared folder" icon to the right. In "Folder Path", find the folder on your host computer you would like the guest VM to have access to. "Folder Name" is a label for this folder and can be anything you want (we use inspired_cwd as an example). "Mount point" is the path in the VM where the shared folder will be mounted (e.g., /home/inspired/cwd). Note that both "Folder Name" and "Mount point" must be consistent with the arguments used in the "sudo mount" command line in Step 6. You may check "Auto-mount" and "Make permanent" so you can skip this step in the future as long as you keep using this folder to share files between your host computer and the VM.
   
   ![image](https://github.com/cyqjh/inspired/assets/105002220/69cbfd29-71a8-43ba-adbf-6ea78f8b8a22)
   ![image](https://github.com/cyqjh/inspired/assets/105002220/bed6ba73-b2e1-457c-9d99-63f195afcfd9)

  
6. After finishing the setup, open a terminal in the VM (click the icon at the bottom left corner, "System Tools->QTerminal"), run (use "inspired" as password if prompted for authentication):

   `sudo mount -t vboxsf -o rw,uid=1000,gid=1000 inspired_cwd /home/inspired/cwd`

   ![image](https://github.com/cyqjh/inspired/assets/105002220/c7b2f0d7-6c0c-4b18-b734-01cf3e018ac7)
   ![image](https://github.com/cyqjh/inspired/assets/105002220/48df97f2-79da-4c68-b1a1-a601224bef51)

   To automatically run this command in the future when you start the VM, you may add it to the crontab file by running:

   `crontab -e`

   If asked to choose an editor and you are not sure, select nano. Add the following line to the end of the crontab file:

   `@reboot echo "inspired" | sudo -S mount -t vboxsf -o rw,uid=1000,gid=1000 inspired_cwd /home/inspired/cwd`

   Press Ctrl+S to save and Ctrl+X to exit (if using nano). This folder will now be automatically mounted when you run the VM on this computer.

7. Update the code/model to the newest version by running:

   `cd /home/inspired/inspired`
   
   `git pull`

     This step is important as the VM may not contain the newest version of INSPIRED. It is also recommended that you run this regularly to ensure you get the latest bug fixes. The database and the predictor model are not updated in this way. They will be updated on Zenodo when new versions are available and they can be obtained in the VM by following Step 4 in Option 2.

9. Go to the shared folder (create a subfolder if needed) and run:

   `inspired`

Note: The VM desktop resolution can be changed at "Preferences->LXQt Settings->Monitor settings" within the VM. The VM window size can be changed on the VirtualBox menu (under View).

   ![image](https://github.com/cyqjh/inspired/assets/105002220/89609ffa-38a4-41b4-82eb-dde7f7ddcfe0)



### Option 2
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
   
    `wget https://zenodo.org/records/11478889/files/dftdb.tar.gz`

    `wget https://zenodo.org/records/10723108/files/model.tar.gz`

    `tar zxvf dftdb.tar.gz`

    `tar zxvf model.tar.gz`

5. Determine if your computer is CPU-only or equipped with GPU/CUDA. If you are not sure, choose the CPU-only option. Run:

    `cd ./install`

    `bash conda_env_cpu.sh` (or `bash conda_env_gpu.sh` for GPU machines).

    (Note: It may take a while for the installation to be completed. If you encounter errors, you may try running the commands in the sh file one by one to diagnose. If you encounter an error associated with missing pydantic, please run `pip install pydantic` to install.)

6. If all packages are installed successfully, you may now go to a working directory of your choice and start INSPIRED by running:
   
     `inspired`

## Additional notes:

When using INSPIRED, it is important to ensure the "Current Working Directory" (CWD) is correctly set, as this is where the program reads/writes all its input/output files. The default CWD is where you started INSPIRED, and once this program is running, you can change it in the Menu. It is strongly recommended that a new folder be created for each project to avoid mixed input/output files and potential errors. If you prefer not to change the working directory, then please make sure to run through all steps in each calculation (do not skip steps) to avoid accidently picking up input files from a previous calculation.

There is a "Help" button at the bottom right corner of each window. Click on the button for instructions.

Finally, but **very importantly**, INSPIRED is a GUI that provides streamlined access to various resources for users to obtain simulated INS spectra easily and quickly from a structure model. Many of the tools included in INSPIRED are under active development by their developers, such as the MLFFs. With growing computing power, larger training datasets, and more sophisticated models, the scope of the data and the accuracy of the models will keep improving. While we will try to keep the models included in INSPIRED up to date, we do not and cannot guarantee the accuracy of the results. Please check the references (1,8-12) for more details about these tools and their limitations. User discretion is extremely important while using INSPIRED, and please check the results carefully before using them for your analyses. Note that in some cases, even though the calculated phonon dispersion from the MLFFs appears fine, there could be a systematic softening (or, less likely, hardening) of the modes. Here are some additional resources on the benchmark of the MLFFs:

https://matbench-discovery.materialsproject.org/

https://arxiv.org/abs/2405.07105


## Contact:

If you have any questions, please contact YQ Cheng at chengy@ornl.gov for help.


## Citation:

Bowen Han, Andrei T. Savici, Mingda Li, Yongqiang Cheng, INSPIRED: Inelastic Neutron Scattering Prediction for Instantaneous Results and Experimental Design, Computer Physics Communications, in press (2024)
.


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
13.	Hinuma, Y.; Pizzi, G.; Kumagai, Y.; Oba, F.; Tanaka, I., Band structure diagram paths based on crystallography. Comp. Mat. Sci. 2017, 128, 140.
14.	Togo, A.; Tanaka, I., Spglib: a software library for crystal symmetry search. 2018, arXiv:1808.01590
15.	Geiger, M.; Smidt, T., e3nn: Euclidean neural networks. 2022, doi:10.48550/ARXIV.2207.09453.
16.	Geiger, M.; et al., Euclidean neural networks: e3nn. 2022, doi:10.5281/zenodo.6459381.
