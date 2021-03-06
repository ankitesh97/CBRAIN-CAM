from cbrain.imports import *
from cbrain.data_generator import *
from cbrain.cam_constants import *
from cbrain.losses import *
from cbrain.utils import limit_mem
from cbrain.layers import *
from cbrain.data_generator import DataGenerator
import tensorflow as tf
from tensorflow import math as tfm
#import tensorflow_probability as tfp
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import xarray as xr
import numpy as np
from cbrain.model_diagnostics import ModelDiagnostics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as imag
import scipy.integrate as sin
#import cartopy.crs as ccrs
import matplotlib.ticker as mticker
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import sklearn
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from cbrain.imports import *
from cbrain.utils import *
from cbrain.normalization import *
import h5py
from sklearn.preprocessing import OneHotEncoder
from cbrain.climate_invariant import *
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# Load coordinates (just pick any file from the climate model run)
coor = xr.open_dataset("/DFS-L/DATA/pritchard/ankitesg/data/CESM2_f19_v13_updated_NN_pelayout01_ens_07.cam.h1.2003-01-22-00000.nc",\
                    decode_times=False)
lat = coor.lat; lon = coor.lon; lev = coor.lev;
DATA_DIR = '/DFS-L/DATA/pritchard/ankitesg/datav3/'
hyam = coor.hyam
hybm = coor.hybm
scale_dict = load_pickle('/export/nfs0home/ankitesg/tom/CBRAIN-CAM/nn_config/scale_dicts/2020_10_16_scale_dict_RG.pkl')['scale_dict_RG']


class DataGeneratorClimInvRealGeo(DataGenerator):

    def __init__(self, data_fn, input_vars, output_vars,
             norm_fn=None, input_transform=None, output_transform=None,
             batch_size=1024, shuffle=True, xarray=False, var_cut_off=None, normalize_flag=True,
             rh_trans=True,t2tns_trans=True,
             lhflx_trans=True,
             scaling=True,interpolate=True,
             hyam=None,hybm=None,
             inp_subRH=None,inp_divRH=None,
             inp_subTNS=None,inp_divTNS=None,
             lev=None, interm_size=40,
             lower_lim=6,
             is_continous=True,Tnot=5,
                mode='train', exp=None):
        self.scaling = scaling
        self.interpolate = interpolate
        self.rh_trans = rh_trans
        self.t2tns_trans = t2tns_trans
        self.lhflx_trans = lhflx_trans
        self.inp_shape = 64
        self.exp = exp
        self.mode=mode
        super().__init__(data_fn, input_vars,output_vars,norm_fn,input_transform,output_transform,
                        batch_size,shuffle,xarray,var_cut_off,normalize_flag) ## call the base data generator
        self.inp_sub = self.input_transform.sub
        self.inp_div = self.input_transform.div


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
        X_norm = self.input_transform.transform(X)
        Y = self.output_transform.transform(Y)
        return X_norm, Y
    
    
    
    
in_vars = ['QBP','TBP','CLDLIQBP','CLDICEBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars = ['QBCTEND','TBCTEND','CLDLIQBCTEND', 'CLDICEBCTEND', 'NN2L_FLWDS', 'NN2L_PRECC', 
            'NN2L_PRECSC', 'NN2L_SOLL', 'NN2L_SOLLD', 'NN2L_SOLS', 'NN2L_SOLSD', 'NN2L_NETSW']


TRAINFILE = 'RG_SP_M4K_train_shuffle.nc'
NORMFILE = 'RG_SP_M4K_NORM_norm.nc'
VALIDFILE = 'RG_SP_M4K_valid.nc'

train_gen_bf = DataGeneratorClimInv(
    data_fn = f'{DATA_DIR}{TRAINFILE}',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = f'{DATA_DIR}{NORMFILE}',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True,
    normalize_flag=True,
    lev=lev,
    hyam=hyam,hybm=hybm,
    rh_trans = False,t2tns_trans=False,
    lhflx_trans=False,
    scaling=False,
    interpolate=False
)


valid_gen_bf = DataGeneratorClimInv(
    data_fn = f'{DATA_DIR}{VALIDFILE}',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = f'{DATA_DIR}{NORMFILE}',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True,
    normalize_flag=True,
    lev=lev,
    hyam=hyam,hybm=hybm,
    rh_trans = False,t2tns_trans=False,
    lhflx_trans=False,
    scaling=False,
    interpolate=False
)


# model.compile(tf.keras.optimizers.Adam(), loss="mse")
path_HDF5 = '/DFS-L/DATA/pritchard/ankitesg/models/'
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint(path_HDF5+'BF_RGV5.h5',save_best_only=True, monitor='val_loss', mode='min')

config_file = 'CI_RG_M4K_CONFIG.yml' # Configuration file
data_file = ['RG_SP_M4K_valid.nc']  # Validation/test data sets
NNarray = ['BF_RGV3.h5'] # NN to evaluate 
path_HDF5 = '/DFS-L/DATA/pritchard/ankitesg/models/'
NNname = ['BF'] # Name of NNs for plotting
dict_lay = {'SurRadLayer':SurRadLayer,'MassConsLayer':MassConsLayer,'EntConsLayer':EntConsLayer,
            'QV2RH':QV2RH,'T2TmTNS':T2TmTNS,'eliq':eliq,'eice':eice,'esat':esat,'qv':qv,'RH':RH,
           'reverseInterpLayer':reverseInterpLayer,'ScaleOp':ScaleOp}


from kerastuner import HyperModel

from kerastuner.tuners import RandomSearch,BayesianOptimization

class RGModel(HyperModel):
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden
        
    def build(self, hp):   
        model = Sequential()
        model.add(Input(shape=(108,)))
        model.add(Dense(units=hp.Int(
                            'units',
                            min_value=32,
                            max_value=512,
                            step=32,
                            default=128
                        ),
                        activation=hp.Choice(
                            'dense_activation',
                            values=['relu', 'tanh', 'sigmoid'],
                            default='relu'
                        )
                    )
        )
        # model.add(LeakyReLU(alpha=0.3))
        for i in range(hp.Int('num_layers', 4, 8)):
            model.add(Dense(units=hp.Int(
                            'units',
                            min_value=32,
                            max_value=512,
                            step=32,
                            default=128
                        ),
                        activation=hp.Choice(
                            'dense_activation',
                            values=['relu', 'tanh', 'sigmoid'],
                            default='relu'
                        )
                    )
        )   

        model.add(Dense(112, activation='linear'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='mse',
            metrics=['mse']
        )
        return model

    
    
hypermodel = RGModel(n_hidden=2)

HYPERBAND_MAX_EPOCHS = 40
MAX_TRIALS = 40
EXECUTION_PER_TRIAL = 4


tuner = BayesianOptimization(
    hypermodel,
    objective='val_mean_squared_error',
    seed=1,
    max_trials=MAX_TRIALS,
    executions_per_trial=EXECUTION_PER_TRIAL,
    directory='random_search',
    project_name='RGBFV8'
)

print(tuner.search_space_summary())

N_EPOCH_SEARCH = 10
# train_generator, steps_per_epoch=200, epochs=60, validation_data=validation_generator
tuner.search(train_gen_bf, epochs=N_EPOCH_SEARCH, validation_data=valid_gen_bf)

print(tuner.results_summary())

best_model = tuner.get_best_models(num_models=1)[0]
best_model.save('/DFS-L/DATA/pritchard/ankitesg/models/BFv12.h5')


