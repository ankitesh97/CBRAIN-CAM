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


DATA_PATH = '/fast/ankitesh/data/'
TRAINFILE = 'CI_SP_M4K_train_shuffle.nc'
VALIDFILE = 'CI_SP_M4K_valid.nc'
percentile_path='/export/nfs0home/ankitesg/data/percentile_data.pkl'
data_name='M4K'
bin_size = 100
scale_dict = load_pickle('/export/nfs0home/ankitesg/CBrain_project/CBRAIN-CAM/nn_config/scale_dicts/009_Wm2_scaling.pkl')



percentile_bins = load_pickle(percentile_path)['Percentile'][data_name]
enc = OneHotEncoder(sparse=False)
classes = np.arange(bin_size+2)
enc.fit(classes.reshape(-1,1))



data_ds = xr.open_dataset(f"{DATA_PATH}{TRAINFILE}")
n = data_ds['vars'].shape[0]
print("end loading data")

coords = list(data_ds['vars'].var_names.values)
coords = coords + ['PHQ_BIN']*30+['TPHYSTND_BIN']*30+['FSNT_BIN','FSNS_BIN','FLNT_BIN','FLNS_BIN']



def _transform_to_one_hot(Y):
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
    perc = percentile_bins
    for var in out_vars[:2]:
        all_levels_one_hot = []
        for ilev in range(30):
            bin_index = np.digitize(var_dict[var][:,ilev],perc[var][ilev])
            one_hot = enc.transform(bin_index.reshape(-1,1))
            all_levels_one_hot.append(one_hot)
        var_one_hot = np.stack(all_levels_one_hot,axis=1) 
        Y_trans.append(var_one_hot)
    for var in out_vars[2:]:
        bin_index = np.digitize(var_dict[var][:], perc[var])
        one_hot = enc.transform(bin_index.reshape(-1,1))[:,np.newaxis,:]
        Y_trans.append(one_hot)

    Y_concatenated = np.concatenate(Y_trans,axis=1)
    return Y_concatenated



inp_vars = ['QBP','TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
inp_coords = coords[:64]
out_coords = coords[64:128]
bin_coords = list(range(bin_size+2))


print("starting processing")
all_data_arrays = []
batch_size = 32768
for i in range(0,n,batch_size):
    all_vars = data_ds['vars'][i:i+batch_size]
    inp_vals = all_vars[:,:64]
    out_vals = all_vars[:,64:128]
    one_hot = _transform_to_one_hot(out_vals)
    sample_coords = list(range(i,i+all_vars.shape[0]))
    x3 = xr.Dataset(
        {
            "X": (("sample", "inp_coords"),inp_vals),
            "Y_raw":(("sample","out_cords"),out_vals),
            "Y": (("sample", "out_coords","bin_index"), one_hot),
        },
        coords={"sample": sample_coords, "inp_coords": inp_coords,"out_coords":out_coords,"bin_index":bin_coords},
    )
    all_data_arrays.append(x3)
    print(int(i/batch_size), end='\r')


print("combining")
final_da = xr.combine_by_coords(all_data_arrays)


print("saving")
final_da.to_netcdf('/scratch/ankitesh/data/new_data_for_v2.nc')
