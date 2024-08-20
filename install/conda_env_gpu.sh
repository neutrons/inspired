#!/bin/bash

pip install https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp39-cp39-linux_x86_64.whl
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch_geometric
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install pandas
pip install pyqt5
pip install qtpy
pip install phonopy
pip install seekpath
pip install mace-torch
pip install matgl
pip install chgnet
bash ./install.sh
