#!/usr/bin/env python
# coding: utf-8

# In[35]:


import bilby
from bilby.core.prior import PriorDict, Uniform, Sine
from gwpopulation_pipe.utils import prior_conversion
import numpy as np
#from gwpopulation.experimental import jax
from gwpopulation.experimental.jax import NonCachingModel, JittedLikelihood
import gwpopulation as gwpop
from gwpopulation_pipe import data_simulation
import pandas as pd
import dill
import corner
import astropy.cosmology as cosmo
import matplotlib.pyplot as plt
import math

#gwpop.set_backend("jax")

#get_ipython().run_line_magic('matplotlib', 'inline')

# import warnings
# warnings.filterwarnings("once", "ChainedAssignmentError")
# warnings.filterwarnings("once", "SettingWithCopyWarning")
# warnings.filterwarnings("ignore", "FutureWarning")


# In[36]:


posteriors = pd.read_pickle('/storage/physics/phucgh/properdataoutput_50set2.pkl')

with open("/storage/physics/phucgh/gwtc-3-injections.pkl", "rb") as ff:
    injections = dill.load(ff)
    injections_df = pd.DataFrame(injections)


# In[37]:


#print(posteriors[0])
for i in range(0,len(posteriors)):
    post=posteriors[i]
    post["luminosity_distance"] = bilby.gw.conversion.redshift_to_luminosity_distance(post['redshift'])
    # post['cos_tilt_1'] = 0
    # post['cos_tilt_2'] = 0
    # for i in range(0,len(post)):
    #     costilt1=math.cos(post['tilt_1'][i])
    #     costilt2=math.cos(post['tilt_2'][i])
    #     post['cos_tilt_1'][i]=int(costilt1)
    #     post['cos_tilt_2'][i]=int(costilt2)
    zs = post["redshift"]
    post["mass_1"] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(post['chirp_mass'], post['mass_ratio'])[0]
    post["mass_1_detector"] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(post['chirp_mass'], post['mass_ratio'])[0] * (1 + zs)


# In[38]:


# In[39]:


model = NonCachingModel(
    [gwpop.models.mass.two_component_primary_mass_ratio,
     gwpop.models.spin.iid_spin,
     gwpop.models.redshift.PowerLawRedshift(cosmo_model="Planck15"),
    ],
)

vt = gwpop.vt.ResamplingVT(model=model, data=injections, n_events=len(posteriors))

likelihood = gwpop.hyperpe.HyperparameterLikelihood(
    posteriors=posteriors,
    hyper_prior=model,
    selection_function=vt,
)


likelihood = gwpop.hyperpe.HyperparameterLikelihood(
    posteriors=posteriors,
    hyper_prior=model,
    selection_function=vt,
    cupy=gwpop.backend == "cupy",
)


# In[40]:


priors = PriorDict()

# mass
priors["alpha"] = Uniform(minimum=-2, maximum=4, latex_label="$\\alpha$")
priors["beta"] = Uniform(minimum=-4, maximum=12, latex_label="$\\beta$")
priors["mmin"] = Uniform(minimum=10, maximum=20, latex_label="$m_{\\min}$")
priors["mmax"] = Uniform(minimum=70, maximum=80, latex_label="$m_{\\max}$")
priors["lam"] = Uniform(minimum=0, maximum=1, latex_label="$\\lambda_{m}$")
priors["mpp"] = Uniform(minimum=10, maximum=50, latex_label="$\\mu_{m}$")
priors["sigpp"] = Uniform(minimum=1, maximum=10, latex_label="$\\sigma_{m}$")
priors["gaussian_mass_maximum"] = 100

# Spin
priors["amax"] = 1
priors["alpha_chi"] = 2.57
#priors["alpha_chi"] = Uniform(minimum=1, maximum=6, latex_label="$\\alpha_{\\chi}$")
priors["beta_chi"] = 4.2
#priors["beta_chi"] = Uniform(minimum=1, maximum=6, latex_label="$\\beta_{\\chi}$")
priors["xi_spin"] = 0.68
#priors["xi_spin"] = Uniform(minimum=0, maximum=1, latex_label="$\\xi$")
priors["sigma_spin"] = 0.82
#priors["sigma_spin"] = Uniform(minimum=0.3, maximum=4, latex_label="$\\sigma$")

priors["lamb"] = 8.54
#priors["lamb"] = Uniform(minimum=-1, maximum=10, latex_label="$\\lambda_{z}$")

# cosmology
#priors["H0"] = Uniform(minimum=20, maximum=200, latex_label="$H_0$")
#priors["Om0"] = Uniform(minimum=0.1, maximum=0.9, latex_label="$\\Omega_{m}$")
#priors["w0"] = Uniform(minimum=-1.25, maximum=-0.75, latex_label="$w$")


# In[41]:


parameters = priors.sample()
likelihood.parameters.update(parameters)
likelihood.log_likelihood_ratio()
#get_ipython().run_line_magic('time', 'print(likelihood.log_likelihood_ratio())')
#jit_likelihood = JittedLikelihood(likelihood)
#jit_likelihood.parameters.update(parameters)
#%time print(jit_likelihood.log_likelihood_ratio())
#%time print(jit_likelihood.log_likelihood_ratio())



result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=1000,
    label="proper_run_50_events_set_2_run_2",
    outdir='proper_universe_model',
    sample="rwalk",
    npoints=1000,
    npool=32,
    dlogz=1,
    resume=True
)
# conversion_function=bilby.gw.conversion.generate_all_bbh_parameters

# In[ ]:




