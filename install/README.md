Installation instructions:


1. Use a Linux computer (Windows and Mac are not supported yet).
   The Analysis cluster (analysis.sns.gov) will have INSPIRED installed and available to all users. After the deployment, you can simply open a terminal, go to your working directory, and run "inspired". 
   Before this is available, you can install INSPIRED in your home directory on the Analysis cluster by following Steps 2-6 below.
   If you do not have access to a Linux computer and would like to use it on a local computer, you may consider installing a Linux virtual machine with VirtualBox (https://www.virtualbox.org/). It is recommended that the virtual machine is allocated at least 4GB of memory and 50GB of disk space. Depending on the Linux distribution, it may also be necessary to install additional libraries if they are not installed by default (e.g., missing libxcb-xinerama0 has been found to be the culprit in some cases).

2. Install Anaconda/Miniconda for Linux (https://docs.anaconda.com/free/anaconda/install/linux/) if it is not already installed.

3. With conda initiated, run the following commands:

   conda create -n inspired python=3.9
   conda activate inspired

4. Determine if your computer is CPU-only or equipped with GPU/CUDA. If you are not sure, choose the CPU-only option.
   Go to the ./install folder (cd ./install). Run the command lines in the conda_env_cpu.sh file (or the conda_env_gpu.sh file for GPU machines).
   You may run them one by one in order (recommended for first-time installation) or as a bash script (recommended for re-installation when you know it will be error-free).
   It may take a while for the installation to be completed.


5. If all packages are installed successfully, you may now go to a working directory of your choice and start INSPIRED by running:

   inspired

6. It may take about 10-40 seconds (depending on the computer) for the program to initialize and for the main window to display.



When using INSPIRED, it is important to ensure the "Current Working Directory' (CWD) is correctly set, as this is where the program read/write all its input/output files. The default CWD is where you started INSPIRED from, and once this program is running, the CWD can be changed in the Menu. It is strongly recommended that a new folder is created for each material to avoid mixed input/output files and potential errors.

There is a "Help" button at the right bottom corner of each window. Click on the button for instructions.


If you have any questions, please contact YQ Cheng at chengy@ornl.gov for help.

