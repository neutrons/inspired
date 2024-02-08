from ase import Atom, Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
import numpy as np
import pandas as pd
import torch
import torch_geometric as tg

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

def load_data(filename):
    df = pd.read_csv(filename)

    try:
        df['structure'] = df['structure'].apply(eval).progress_map(lambda x: Atoms.fromdict(x))

    except:
        species = []

    else:
        df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
        df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
        species = sorted(list(set(df['species'].sum())))

    return df, species

def build_data(entry, lsv=np.zeros(50), r_max=5.):
    # one-hot encoding atom type and mass
    type_encoding = {}
    specie_am = []
    for Z in range(1, 119):
        specie = Atom(Z)
        type_encoding[specie.symbol] = Z
        specie_am.append(specie.mass)

    type_onehot = torch.eye(len(type_encoding))
    am_onehot = torch.diag(torch.tensor(specie_am))

    symbols = list(entry.symbols).copy()
    positions = torch.from_numpy(entry.positions.copy())
    lattice = torch.from_numpy(entry.cell.array.copy()).unsqueeze(0)

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry, cutoff=r_max, self_interaction=True)

    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_vec = (positions[torch.from_numpy(edge_dst)]
                - positions[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)

    data = tg.data.Data(
        pos=positions, lattice=lattice, symbol=symbols,
        x=am_onehot[[type_encoding[specie] for specie in symbols]],  # atomic mass (node feature)
        z=type_onehot[[type_encoding[specie] for specie in symbols]],  # atom type (node attribute)
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec, edge_len=edge_len,
        lsv=torch.from_numpy(lsv).unsqueeze(0)
    )

    data = [data]

    return data
