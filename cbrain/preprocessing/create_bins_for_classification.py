from cbrain.imports import *
from cbrain.data_generator import *
from cbrain.cam_constants import *
from cbrain.losses import *
from cbrain.utils import limit_mem
from cbrain.layers import *
from cbrain.data_generator import DataGenerator
import tensorflow as tf
#import tensorflow.math as tfm
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



fz = 15
lw = 4
siz = 1000
S0 = 320 # Representative mean solar insolation for normalization purposes
S0max = 1410.6442 # Max solar insolation for normalization purposes
SN = S0/100 # Representative target = mean insolation / 4
XNNA = 1.25 # Abscissa where architecture-constrained network will be placed
XTEXT = 0.25 # Text placement
YMIN = -1 # Representative value for conserving network
YTEXT = 0.3 # Text placement
bin_size = 1001



M4k_path = '/DFS-L/DATA/pritchard/tbeucler/SPCAM/sp8fbp_minus4k/sp8fbp_minus4k.cam2.h2.0001-01-0?-00000.nc'
P4K_path = '/DFS-L/DATA/pritchard/tbeucler/SPCAM/sp8fbp_4k/sp8fbp_4k.cam2.h2.0001-0?-15-00000.nc'
Ref_path = '/DFS-L/DATA/pritchard/tbeucler/SPCAM/fluxbypass_aqua/AndKua_aqua_SPCAM3.0_sp_fbp_f4.cam2.h1.0001-01-0?-00000.nc'
dP = load_pickle('/export/nfs0home/ankitesg/CBRAIN-CAM/nn_config/scale_dicts/009_Wm2_dP.pkl')
scale_dict = {}
scale_dict['PHQ'] = L_V*dP/G
scale_dict['TPHYSTND'] = C_P*dP/G
scale_dict['FSNT'] = 1
scale_dict['FSNS'] = 1
scale_dict['FLNT'] = 1
scale_dict['FLNS'] = 1

print("Loading Datasets")
data_m4k = xr.open_mfdataset(M4k_path,decode_times=False)
data_p4k = xr.open_mfdataset(P4K_path,decode_times=False)
data_ref = xr.open_mfdataset(Ref_path,decode_times=False)

output_dict = {} ## This will be pickled and loaded


data_dict = {'M4K':data_m4k, 'P4K':data_p4k, 'REF':data_ref}
lev_dep_vars = ['PHQ','TPHYSTND']
lev_non_dep_vars = ['FSNT', 'FSNS', 'FLNT', 'FLNS']
output_vars = [lev_dep_vars,lev_non_dep_vars]

PERC = np.linspace(0,100,bin_size).astype('int')

## dump Percentiles
perc = {}
print(f"Processing Percentiles")
for data_name in data_dict.keys():
    print(f"Processing Data {data_name}")
    perc[data_name] = {}
    data = data_dict[data_name]
    for var in lev_dep_vars:
        print(f"Processing Variable {var}")
        perc[data_name][var] = {}
        for ilev in range(30):
            print('ilev=',ilev,'        ',end='\r')
            perc[data_name][var][ilev] = np.percentile(a=scale_dict[var][ilev]*data[var][:,ilev,:,:].values.flatten(),q=PERC)
            
    for var in lev_non_dep_vars:
        print(f"Processing Variable {var}")
        perc[data_name][var] = np.percentile(a=scale_dict[var]*data[var][:,:,:].values.flatten(),q=PERC)

output_dict['Percentile'] = perc
        
## dump pdf of percentiles
pdf = {}
edg = {}
print(f"Processing PDF of Percentiles")
for data_name in data_dict.keys():
    print(f"Processing Data {data_name}")    
    pdf[data_name] = {}
    edg[data_name] = {}
    data = data_dict[data_name]
    for var in lev_dep_vars:
        print(f"Processing Variable {var}")
        pdf[data_name][var] = {}
        edg[data_name][var] = {}
        for ilev in range(30):
            print('ilev=',ilev,'        ',end='\r')
            pdf[data_name][var][ilev],edg[data_name][var][ilev] = \
            np.histogram(scale_dict[var][ilev]*data[var][:,ilev,:,:].values.flatten(),bins=perc[data_name][var][ilev])
            
    for var in lev_non_dep_vars:
        print(f"Processing Variable {var}")
        pdf[data_name][var],edg[data_name][var] = \
            np.histogram(scale_dict[var]*data[var][:,:,:].values.flatten(),bins=perc[data_name][var])

        
output_dict['PDF_PERCENTILE'] = pdf 
output_dict['BIN_EDGES_PERCENTILE'] = edg

out_path = '/export/nfs0home/ankitesg/data/percentile_data_bin_size_1000.pkl'
print("Saving Pickle File")
with open(out_path, 'wb') as handle:
    pickle.dump(output_dict, handle)
    
print(f"Saved file to {out_path}")
