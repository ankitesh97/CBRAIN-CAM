import sys
sys.path.insert(1,"/home1/07064/tg863631/anaconda3/envs/CbrainCustomLayer/lib/python3.6/site-packages") #work around for h5py
from cbrain.imports import *
from cbrain.cam_constants import *
from cbrain.utils import *
from cbrain.layers import *
from cbrain.data_generator import DataGenerator
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
# import tensorflow_probability as tfp
import xarray as xr
import numpy as np
from cbrain.model_diagnostics import ModelDiagnostics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as imag
import scipy.integrate as sin
import matplotlib.ticker as mticker
import pickle
from tensorflow.keras import layers
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime
from cbrain.climate_invariant import *
import yaml



from cbrain.imports import *
from cbrain.utils import *
from cbrain.normalization import *
import h5py
from sklearn.preprocessing import OneHotEncoder



class DataGeneratorClassification(tf.keras.utils.Sequence):
    def __init__(self, data_fn, input_vars, output_vars, percentile_path, data_name,
                 norm_fn=None, input_transform=None, output_transform=None,
                 batch_size=1024, shuffle=True, xarray=False, var_cut_off=None, normalize_flag=True, bin_size=100):
        # Just copy over the attributes
        self.data_fn, self.norm_fn = data_fn, norm_fn
        self.input_vars, self.output_vars = input_vars, output_vars
        self.batch_size, self.shuffle = batch_size, shuffle
        self.bin_size = bin_size
        self.percentile_bins = load_pickle(percentile_path)['Percentile'][data_name]
        self.enc = OneHotEncoder(sparse=False)
        classes = np.arange(self.bin_size+2)
        self.enc.fit(classes.reshape(-1,1))
        # Open datasets
        self.data_ds = xr.open_dataset(data_fn)
        if norm_fn is not None: self.norm_ds = xr.open_dataset(norm_fn)
     # Compute number of samples and batches
        self.n_samples = self.data_ds.vars.shape[0]
        self.n_batches = int(np.floor(self.n_samples) / self.batch_size)

        # Get input and output variable indices
        self.input_idxs = return_var_idxs(self.data_ds, input_vars, var_cut_off)
        self.output_idxs = return_var_idxs(self.data_ds, output_vars)
        self.n_inputs, self.n_outputs = len(self.input_idxs), len(self.output_idxs)
        
                # Initialize input and output normalizers/transformers
        if input_transform is None:
            self.input_transform = Normalizer()
        elif type(input_transform) is tuple:
            ## normalize flag added by Ankitesh
            self.input_transform = InputNormalizer(
                self.norm_ds,normalize_flag, input_vars, input_transform[0], input_transform[1], var_cut_off)
        else:
            self.input_transform = input_transform  # Assume an initialized normalizer is passed
            
            
        if output_transform is None:
            self.output_transform = Normalizer()
        elif type(output_transform) is dict:
            self.output_transform = DictNormalizer(self.norm_ds, output_vars, output_transform)
        else:
            self.output_transform = output_transform  # Assume an initialized normalizer is passed

        # Now close the xarray file and load it as an h5 file instead
        # This significantly speeds up the reading of the data...
        if not xarray:
            self.data_ds.close()
            self.data_ds = h5py.File(data_fn, 'r')
    
    def __len__(self):
        return self.n_batches
    
    # TODO: Find a better way to implement this, currently it is the hardcoded way.
    def _transform_to_one_hot(self,Y):
        '''
            return shape = batch_size X 64 X bin_size
        '''

        Y_trans = []
        out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']
        var_dict = {}
        var_dict['PHQ'] = Y[:,:30]
        var_dict['TPHYSTND'] = Y[:,30:60]
        var_dict['FSNT'] = Y[:,60]
        var_dict['FSNS'] = Y[:,61]
        var_dict['FLNT'] = Y[:,62]
        var_dict['FLNS'] = Y[:,63]
        perc = self.percentile_bins
        for var in out_vars[:2]:
            all_levels_one_hot = []
            for ilev in range(30):
                bin_index = np.digitize(var_dict[var][:,ilev],perc[var][ilev])
                one_hot = self.enc.transform(bin_index.reshape(-1,1))
                all_levels_one_hot.append(one_hot)
            var_one_hot = np.stack(all_levels_one_hot,axis=1) 
            Y_trans.append(var_one_hot)
        for var in out_vars[2:]:
            bin_index = np.digitize(var_dict[var][:], perc[var])
            one_hot = self.enc.transform(bin_index.reshape(-1,1))[:,np.newaxis,:]
            Y_trans.append(one_hot)
        
        Y_concatenated = np.concatenate(Y_trans,axis=1)
        transformed = {}
        for i in range(64):
            transformed[f'output_{i}'] = Y_concatenated[:,i,:]
        return transformed
            
        
        
        
    def __getitem__(self, index):
        # Compute start and end indices for batch
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        # Grab batch from data
        batch = self.data_ds['vars'][start_idx:end_idx]

        # Split into inputs and outputs
        X = batch[:, self.input_idxs]
        Y = batch[:, self.output_idxs]
        # Normalize
        X = self.input_transform.transform(X)
        Y = self.output_transform.transform(Y) #shape batch_size X 64 
        Y = self._transform_to_one_hot(Y)
        return X, Y

    def on_epoch_end(self):
        self.indices = np.arange(self.n_batches)
        if self.shuffle: np.random.shuffle(self.indices)





scale_dict = load_pickle('/export/nfs0home/ankitesg/CBrain_project/CBRAIN-CAM/nn_config/scale_dicts/009_Wm2_scaling.pkl')

TRAINFILE = 'CI_SP_M4K_train_shuffle.nc'
VALIDFILE = 'CI_SP_M4K_valid.nc'
NORMFILE = 'CI_SP_M4K_NORM_norm.nc'
data_path = '/scratch/ankitesh/data/'


train_gen = DataGeneratorClassification(
    data_fn=f'{data_path}{TRAINFILE}', 
    input_vars=['QBP','TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX'], 
    output_vars=['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS'], 
    percentile_path='/export/nfs0home/ankitesg/data/percentile_data_bin_size_20.pkl', 
    data_name = 'M4K',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    norm_fn = f'{data_path}{NORMFILE}',
    batch_size=1024,
    bin_size=20
)


valid_gen = DataGeneratorClassification(
    data_fn=f'{data_path}{VALIDFILE}', 
    input_vars=['QBP','TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX'], 
    output_vars=['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS'], 
    percentile_path='/export/nfs0home/ankitesg/data/percentile_data_bin_size_20.pkl', 
    data_name = 'M4K',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    norm_fn = f'{data_path}{NORMFILE}',
    batch_size=1024,
    bin_size=20
)


bin_size = 20

#this defines a single branch out of 64 branches
def define_single_output_branch(densout,out_index):
    out = Dense(bin_size+2, activation='softmax',name=f"output_{out_index}")(densout)
    return out


inp = Input(shape=(64,))
densout = Dense(128, activation='linear')(inp)
densout = LeakyReLU(alpha=0.3)(densout)
for i in range (4):
    densout = Dense(128, activation='linear')(densout)
    densout = LeakyReLU(alpha=0.3)(densout)
densout = Dense(32, activation='linear')(densout)
densout = LeakyReLU(alpha=0.3)(densout)
all_outputs = [define_single_output_branch(densout,i) for i in range(64)]
model = tf.keras.models.Model(inputs=inp, outputs=all_outputs)



losses = {}
for i in range(64):
    losses[f'output_{i}'] = "categorical_crossentropy"


model.compile(tf.keras.optimizers.Adam(), loss=losses, metrics=["accuracy"])
path_HDF5 = '/scratch/ankitesh/models/'
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint(path_HDF5+'BF_Classification_bin_size_20.hdf5',save_best_only=True, monitor='val_loss', mode='min')


with tf.device('/gpu:1'):
    Nep = 5
    model.fit_generator(train_gen, epochs=Nep, validation_data=valid_gen,\
                       callbacks=[earlyStopping, mcp_save])
