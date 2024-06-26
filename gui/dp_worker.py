import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_geometric as tg
from ase import Atom
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset, DataLoader
from latent_space_model import PeriodicNetwork, Decoder


class DPWorker():
    """direct prediction of INS spectra from the crystal structure
    ML models have been pre-trained
    Part of the code in this section was modified from the following sources
    https://github.com/zhantaochen/phonondos_e3nn
    https://github.com/ninarina12/phononDoS_tutorial
    The autoencoder used to compress S(Q,E) was based on the work by Geoffery Wu (https://geoffreywu.me/)
    """

    def __init__(self):
        self.data = {}
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pdos = None
        self.timestr = None


    def build_data(self, entry, lsv=np.zeros(50), r_max=5.):
        """build a graph neural network from the crystal structure
        """

        default_dtype = torch.float64
        torch.set_default_dtype(default_dtype)

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

 
    def nnmodel(self,em_dim,out_dim,r_max,nneigh,model_file,pool):
        """define and load the pre-trained e3nn model
        """

        model = PeriodicNetwork(
            in_dim=118,  # dimension of one-hot encoding of atom type
            em_dim=em_dim,  # dimension of atom-type embedding
            irreps_in=str(em_dim) + "x0e",
            # em_dim scalars (L=0 and even parity) on each atom to represent atom type
            irreps_out=str(out_dim) + "x0e",  # out_dim scalars (L=0 and even parity) to output
            irreps_node_attr=str(em_dim) + "x0e",
            # em_dim scalars (L=0 and even parity) on each atom to represent atom type
            layers=2,  # number of nonlinearities (number of convolutions = layers + 1)
            mul=32,  # multiplicity of irreducible representations
            lmax=1,  # maximum order of spherical harmonics
            max_radius=r_max,  # cutoff radius for convolution
            num_neighbors=nneigh,  # scaling factor based on the typical number of neighbors
            reduce_output=True  # whether or not to aggregate features of all atoms at the end
        )

        # predict on all data
        model.load_state_dict(torch.load(model_file + '.torch', map_location=self.device)['state'])
        model.pool = pool

        model.to(self.device)
        model.eval()

        return model


    def predict(self, model_path, struc, pred_type, partial_dos=True):
        """make predictions
        """

        torch.set_default_dtype(torch.float64)
        if pred_type == "Phonon DOS":
            r_max = 6.0
            lsv = np.zeros(120)
            nneigh = 58.94458615577375      # for DOS prediction
            xaxis = np.arange(10, 1201, 10)
            model_file = os.path.join(model_path, 'crys_dos_predictor')

            self.input_data = self.build_data(struc, lsv, r_max)

            out_dim = len(lsv)
            em_dim = 64

            model = self.nnmodel(em_dim,out_dim,r_max,nneigh,model_file,pool=True)

            dataloader = tg.loader.DataLoader(self.input_data, batch_size=1)
            lsv_pred = np.empty((len(self.input_data), 0)).tolist()

            with torch.no_grad():
                i0 = 0
                for i, d in enumerate(dataloader):
                    d.to(self.device)
                    output = model(d)
                    lsv_pred[i0:i0 + len(d.lsv) - 1] = [[k] for k in output.cpu().numpy()]
                    i0 += len(d.lsv)

            self.data = {'x': xaxis,
                         'y': lsv_pred[0][0],
                         'type': 'Phonon DOS',
                         }
            
            if partial_dos:
                self.symbol = self.input_data[0].symbol
                model = self.nnmodel(em_dim,out_dim,r_max,nneigh,model_file,pool=False)
                with torch.no_grad():
                    for i, d in enumerate(dataloader):
                        d.to(self.device)
                        output = model(d).cpu().numpy()
                self.species = list(set(self.symbol))
                n = len(self.species)
                pdos = dict(zip(self.species, [np.zeros((output.shape[1])) for k in range(n)]))
                for j in range(output.shape[0]):
                    pdos[self.symbol[j]] += output[j,:]

                for j, s in enumerate(self.species):
                    pdos[s] /= self.symbol.count(s)

                self.pdos = pdos


        elif pred_type == "VISION spectrum":
            r_max = 6.0
            lsv = np.zeros(120)
            nneigh = 59.05215000696383    # for VISION spectrum prediction
            xaxis = np.arange(10, 1201, 10)
            model_file = os.path.join(model_path, 'crys_vis_predictor')

            self.input_data = self.build_data(struc, lsv, r_max)

            out_dim = len(lsv)
            em_dim = 64

            model = self.nnmodel(em_dim,out_dim,r_max,nneigh,model_file,pool=True)

            dataloader = tg.loader.DataLoader(self.input_data, batch_size=1)
            lsv_pred = np.empty((len(self.input_data), 0)).tolist()

            with torch.no_grad():
                i0 = 0
                for i, d in enumerate(dataloader):
                    d.to(self.device)
                    output = model(d)
                    lsv_pred[i0:i0 + len(d.lsv) - 1] = [[k] for k in output.cpu().numpy()]
                    i0 += len(d.lsv)

            self.data = {'x': xaxis,
                         'y': lsv_pred[0][0],
                         'type': 'VISION spectrum',
                         }

        elif pred_type == "S(|Q|, E)":
            r_max = 6.0
            model_file = os.path.join(model_path, 'latent_space_predictor')
            lsv = np.zeros(50)
            xaxis = np.arange(1, 51, 1)
            nneigh = 59.220755709410284   # for latent space vector prediction

            self.input_data = self.build_data(struc, lsv, r_max)

            out_dim = len(lsv)
            em_dim = 64

            model = self.nnmodel(em_dim,out_dim,r_max,nneigh,model_file,pool=True)

            dataloader = tg.loader.DataLoader(self.input_data, batch_size=1)
            lsv_pred = np.empty((len(self.input_data), 0)).tolist()

            with torch.no_grad():
                i0 = 0
                for i, d in enumerate(dataloader):
                    d.to(self.device)
                    output = model(d)
                    lsv_pred[i0:i0 + len(d.lsv) - 1] = [[k] for k in output.cpu().numpy()]
                    i0 += len(d.lsv)

            self.data = {'x': xaxis,
                         'y': lsv_pred,
                         'type': 'S(|Q|, E)',
                         }

        else:
            self.data = {}

        self.timestr = time.strftime("%Y%m%d-%H%M%S")   # each prediction is timestamped
    

    def savenplot(self,model_path,cwd,partial_dos=True,unit=0,setrange=False,interactive=False,xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None):
        """save prediction results to files and make plots
        """

        if not self.data:
            print('ERROR: No data to plot.')
            return

        wn2mev = 0.12398
        wn2thz = 0.02998
        mev2wn = 8.06554
        mev2thz = 0.24180

        torch.set_default_dtype(torch.float64)
        if self.data['type'] == 'Phonon DOS' or self.data['type'] == 'VISION spectrum':
            fig, ax = plt.subplots(figsize=(6, 5))
            if unit == 0:
                xaxis = self.data['x']*wn2mev
                xlabel = 'meV'
            elif unit == 2:
                xaxis = self.data['x']*wn2thz
                xlabel = 'THz'
            else:
                xaxis = self.data['x']
                xlabel = r'cm$^{-1}$'
            x_min = np.amin(xaxis)
            x_max = np.amax(xaxis)
            y_min = np.amin(self.data['y'])
            y_max = np.amax(self.data['y'])
            if setrange:
                try:
                    x_min = float(xmin)
                except:
                    pass
                try:
                    x_max = float(xmax)
                except:
                    pass
                try:
                    y_min = float(ymin)
                except:
                    pass
                try:
                    y_max = float(ymax)
                except:
                    pass
            ax.set_xlim(x_min,x_max)
            ax.set_ylim(y_min,y_max)
            ax.plot(xaxis, self.data['y'], label="Prediction", color='black')
            ax.legend(frameon=False)
            ax.set_xlabel('Energy ('+xlabel+')')
            if self.data['type'] == 'VISION spectrum':
                ax.set_ylabel('INS intensity')
                outfile = 'VISION_pred_'+self.timestr+'.csv'
            else:
                ax.set_ylabel('Phonon DOS')
                outfile = 'PhonDOS_pred_'+self.timestr+'.csv'
            fig.show()

            f = open(os.path.join(cwd,outfile),'w')
            print('# Energy ('+xlabel+'), Normalized spectrum',file=f)
            for i in range(len(xaxis)):
                print(xaxis[i], ',', self.data['y'][i], file=f)
            f.close()
            print('INFO: '+self.data['type']+' data saved to '+outfile)

            if self.data['type'] == 'Phonon DOS' and partial_dos and self.pdos:
                fontsize = 16
                N = len(self.species)
                fig, ax = plt.subplots(figsize=(6, 5))
                cm = plt.get_cmap('gist_rainbow')
                ax.set_prop_cycle('color', [cm(1.*i/N) for i in range(N)])
                for j, s in enumerate(self.species):
                    ax.plot(xaxis, self.pdos[s]/self.pdos[s].max(), lw=2, label=s)
                ax.legend(frameon=False)
                ax.set_xlim(x_min,x_max)
                ax.set_ylim(y_min,y_max)
                ax.set_xlabel('Energy ('+xlabel+')')
                ax.set_ylabel('Partial Phonon DOS')
                fig.show()
                outfile = 'PPDOS_pred_'+self.timestr+'.csv'
                f = open(os.path.join(cwd,outfile),'w')
                print('# Energy ('+xlabel+'), Normalized spectrum ('+','.join([s for s in self.species])+')',file=f)
                for i in range(len(self.data['x'])):
                    print(str(xaxis[i])+',', ','.join([str(self.pdos[s][i]) for s in self.species]), file=f)
                f.close()
                print('INFO: Partial phonon DOS data saved to '+outfile)
        

        elif self.data['type'] == 'S(|Q|, E)':

            x_min = 0
            x_max = x_max0 = 15
            y_min = 0
            z_min = 0
            z_max = 1.0
            if unit == 1:
                y_max = y_max0 = 150*mev2wn
                plt_unit = r'cm$^{-1}$'
            elif unit == 2:
                y_max = y_max0 = 150*mev2thz
                plt_unit = 'THz'
            else:
                y_max = y_max0 = 150
                plt_unit = 'meV'

            # Latent space vector is decoded in the following steps to reconstruct the 2D S(Q,E)
            DECODER_MODEL = os.path.join(model_path, 'decoder.pt')

            encoded_space_dim = 50
            dropout = 0
            downsample_ratio = 1  # Scale images to 300 / downsample_ratio x 300/downsample_ratio

            decoder = Decoder(encoded_space_dim, downsample_ratio, dropout)
            decoder.load_state_dict(torch.load(DECODER_MODEL, map_location=self.device))
            decoder.eval()
            decoder.to(self.device)

            data_loader = DataLoader(self.data['y'], batch_size=1, shuffle=False, num_workers=1)

            for i, batch in enumerate(data_loader):
                for j, code in enumerate(batch):
                    code = code.to(self.device)
                    newImage = decoder(code)
                    sqe = np.transpose(newImage.cpu().detach()[0][0])
                    [ne, nq] = np.shape(sqe)
            xi = np.linspace(x_max0/nq,x_max0,nq)
            yi = np.linspace(y_max0/ne,y_max0,ne)
            dx = xi[1]-xi[0]
            dy = yi[1]-yi[0]


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
                    z_min = float(zmin) 
                except:
                    pass
                try:
                    z_max = float(zmax) 
                except:
                    pass
                x1=np.where(np.logical_and(xi>=x_min,xi<=x_max))[0][0]
                x2=np.where(np.logical_and(xi>=x_min,xi<=x_max))[0][-1]
                y1=np.where(np.logical_and(yi>=y_min,yi<=y_max))[0][0]
                y2=np.where(np.logical_and(yi>=y_min,yi<=y_max))[0][-1]
                Z = sqe[y1:y2+1,x1:x2+1]
            else:
                Z = sqe
            asp = (x_max-x_min)/(y_max-y_min)*1.0

            if interactive:
                # Make interactive 2D plot, modified from the example by Diziet Asahi from the following link
                # https://stackoverflow.com/questions/59144464/plotting-two-cross-section-intensity-at-the-same-time-in-one-figure

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

            ax.set_xlabel(r'|Q|($\mathregular{\AA^{-1}}$)')
            ax.set_ylabel('E ('+plt_unit+')')
            fig.show()

            outfile = 'SQE_pred_'+self.timestr+'.csv'
            f = open(os.path.join(cwd,outfile),'w')
            print('# Key parameters for the calculation', file=f)
            print('# TASK,INSTR,TEMP,WING,E_UNIT', file=f)
            print('#    1,   3,   0,   0,   1', file=f)
            print('# Q, deltaE, Total', file=f)
            for iq in range(nq):
                for ie in range(ne):
                    print((iq+1)/nq*x_max0,',',(ie+1)/ne*y_max0,',',sqe[ie][iq].item(), file=f)
                print(' ', file=f)
            f.close()
            print('INFO: S(|Q|,E) spectra data saved to '+outfile)

