from cbrain.imports import *
from cbrain.data_generator import *
from cbrain.cam_constants import *
from cbrain.losses import *
from cbrain.utils import limit_mem
from cbrain.layers import *
from cbrain.data_generator import DataGenerator
import tensorflow as tf
from tensorflow import math as tfm
import tensorflow_probability as tfp
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

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Load coordinates (just pick any file from the climate model run)
coor = xr.open_dataset("/DFS-L/DATA/pritchard/tbeucler/SPCAM/sp8fbp_minus4k/sp8fbp_minus4k.cam2.h2.0000-01-01-00000.nc",\
                    decode_times=False)
lat = coor.lat; lon = coor.lon; lev = coor.lev;
coor.close();
path = '/export/nfs0home/ankitesg/CBrain_project/CBRAIN-CAM/cbrain/'
path_hyam = 'hyam_hybm.pkl'

hf = open(path+path_hyam,'rb')
hyam,hybm = pickle.load(hf)

scale_dict = load_pickle('/export/nfs0home/ankitesg/CBrain_project/CBRAIN-CAM/nn_config/scale_dicts/009_Wm2_scaling.pkl')
scale_dict['RH'] = 0.01*L_S/G, # Arbitrary 0.1 factor as specific humidity is generally below 2%

in_vars_RH = ['RH','TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars_RH = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']

TRAINFILE_RH = 'CI_RH_M4K_NORM_train_shuffle.nc'
NORMFILE_RH = 'CI_RH_M4K_NORM_norm.nc'
VALIDFILE_RH = 'CI_RH_M4K_NORM_valid.nc'
BASE_DIR = '/DFS-L/DATA/pritchard/ankitesg/'

train_gen_RH = DataGenerator(
    data_fn = f"{BASE_DIR}data/{TRAINFILE_RH}",
    input_vars = in_vars_RH,
    output_vars = out_vars_RH,
    norm_fn = f"{BASE_DIR}data/{NORMFILE_RH}",
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True,
    normalize_flag=True
)


in_vars = ['QBP','TfromNSV2','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']

TRAINFILE_TNS = 'CI_TNSV2_M4K_NORM_train_shuffle.nc'
NORMFILE_TNS = 'CI_TNSV2_M4K_NORM_norm.nc'
VALIDFILE_TNS = 'CI_TNSV2_M4K_NORM_valid.nc'

train_gen_TNS = DataGenerator(
    data_fn = f"{BASE_DIR}data/{TRAINFILE_TNS}",
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = f"{BASE_DIR}data/{NORMFILE_TNS}",
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True,
    normalize_flag=True
)



in_vars = ['QBP','TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']

## this won't be used just to show we can use it overall
TRAINFILE = 'CI_SP_M4K_train_shuffle.nc'
NORMFILE = 'CI_SP_M4K_NORM_norm.nc'
VALIDFILE = 'CI_SP_M4K_valid.nc'


inter_dim_size = 40


train_gen = DataGeneratorClimInv(
    data_fn = f"{BASE_DIR}data/{TRAINFILE}",
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = f"{BASE_DIR}data/{NORMFILE}",
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True,
    normalize_flag=True,
    lev=lev,
    hyam=hyam,hybm=hybm,
    inp_subRH=train_gen_RH.input_transform.sub, inp_divRH=train_gen_RH.input_transform.div,
    inp_subTNS=train_gen_TNS.input_transform.sub,inp_divTNS=train_gen_TNS.input_transform.div,
    rh_trans = True,t2tns_trans=False,
    lhflx_trans=False,
    scaling=False,
    interpolate=True,
    exp={"LHFLX":True}
)

valid_gen = DataGeneratorClimInv(
    data_fn = f"{BASE_DIR}data/{VALIDFILE}",
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = f"{BASE_DIR}data/{NORMFILE}",
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True,
    normalize_flag=True,
    lev=lev,
    hyam=hyam,hybm=hybm,
    inp_subRH=train_gen_RH.input_transform.sub, inp_divRH=train_gen_RH.input_transform.div,
    inp_subTNS=train_gen_TNS.input_transform.sub,inp_divTNS=train_gen_TNS.input_transform.div,
    rh_trans = True,t2tns_trans=False,
    lhflx_trans=False,
    scaling=False,
    interpolate=True,
        exp={"LHFLX":True}
)



inp = Input(shape=(173,))
offset = 64
inp_TNS = inp[:,offset:offset+2*inter_dim_size+4]
offset = offset+2*inter_dim_size+4
lev_tilde_before = inp[:,offset:offset+25]
offset = offset+25

densout = Dense(128, activation='linear')(inp_TNS)
densout = LeakyReLU(alpha=0.3)(densout)
for i in range (6):
    densout = Dense(128, activation='linear')(densout)
    densout = LeakyReLU(alpha=0.3)(densout)
denseout = Dense(2*inter_dim_size+4, activation='linear')(densout)
out = reverseInterpLayer(inter_dim_size)([denseout,lev_tilde_before])
model = tf.keras.models.Model(inp, out)


model.compile(tf.keras.optimizers.Adam(), loss=mse)
path_HDF5 = '/DFS-L/DATA/pritchard/ankitesg/models/'
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint(path_HDF5+'RH_Interp.hdf5',save_best_only=True, monitor='val_loss', mode='min')



with tf.device('/gpu:1'):
    Nep = 10
    model.fit_generator(train_gen, epochs=Nep, validation_data=valid_gen,\
                  callbacks=[earlyStopping, mcp_save])



