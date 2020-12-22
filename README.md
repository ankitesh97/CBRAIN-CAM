# CBRAIN-CAM - a neural network for climate invariant parameterization


Hi, thanks for checking out this repository. It contains the code that was used for Rasp et al. 2018 and serves as the basis for ongoing work. In particular, check out the [`climate_invariant`](https://github.com/raspstephan/CBRAIN-CAM/tree/climate_invariant) branch for [Tom Beucler's](http://tbeucler.scripts.mit.edu/tbeucler/) work on physically consistent ML parameterizations.

If you are looking for the exact version of the code that corresponds to the PNAS paper, check out this release: https://github.com/raspstephan/CBRAIN-CAM/releases/tag/PNAS_final [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1402384.svg)](https://doi.org/10.5281/zenodo.1402384)

For a sample of the SPCAM data used, click here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2559313.svg)](https://doi.org/10.5281/zenodo.2559313)




## About

Set up the conda environment using the [env.yml](env.yml) file

### Papers

> T. Beucler, M. Pritchard, P. Gentine and S. Rasp, 2020.
> Towards Physically-consistent, Data-driven Models of Convection.
> https://arxiv.org/abs/2002.08525

> T. Beucler, M. Pritchard, S. Rasp, P. Gentine, J. Ott and P. Baldi, 2019.
> Enforcing Analytic Constraints in Neural-Networks Emulating Physical Systems.
> https://arxiv.org/abs/1909.00912

> S. Rasp, M. Pritchard and P. Gentine, 2018.
> Deep learning to represent sub-grid processes in climate models.
> PNAS. https://doi.org/10.1073/pnas.1810286115
 
> P. Gentine, M. Pritchard, S. Rasp, G. Reinaudi and G. Yacalis, 2018. 
> Could machine learning break the convection parameterization deadlock? 
> Geophysical Research Letters. http://doi.wiley.com/10.1029/2018GL078202
=======
>>>>>>> climate_invar_classification


## Repository description

The main components of the repository are:

- `cbrain`: Contains the cbrain module with all code to preprocess the raw data, run the neural network experiments and analyze the data.

The process of creating a model is as follows.

### Preprocessing

To preprocess the data you the major files are  

**preprocessing.py** and **convert_dataset.py**.
You can check out [this](notebooks/ankitesh-devlog/01_Preprocessing.ipynb) notebook for more information about it.

### Model Training

Once the data is processed we can train the model as a whole or in progression. Below is the architecture of the whole network

Inp -> RH Transformation -> LH Transformation -> T-TNS Transformation -> Split + Scaling -> Vertical Interpolation.

Check out [this](notebooks/ankitesh-devlog/02_Model.ipynb) notebook to know more about training the network.

### Model Diagnostics


Once the model is trained you can run model diagnostics to visualize the learnings.

Check out [this](notebooks/ankitesh-devlog/03_ModelDiagnostics.ipynb) notebook to know more about model diagnostics.
