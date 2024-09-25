#!/usr/bin/env python
# coding: utf-8

# In[4]:


from gwpopulation.experimental.jax import NonCachingModel
import gwpopulation as gwpop
from gwpopulation_pipe import data_simulation
from gwpopulation_pipe.utils import prior_conversion
import pandas as pd
from bilby.core.prior import PriorDict, Uniform, Sine
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np 
import bilby
import math
import csv
import json
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal

def UniverseSim(run_name, signif, batch):
    output=pd.read_csv(f'/storage/physics/phucgh/events/{batch}_{run_name}')
    
    outdir = f'SimLog_low_snr_{run_name}'
    label = f'SimLog_low_snr_{run_name}'
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    injection_parameters = dict(
    mass_1=output['mass_1'][0], 
    mass_2=output['mass_1'][0] * output['mass_ratio'][0], 
    a_1=output['a_1'][0], 
    a_2=output['a_2'][0], 
    cos_tilt_1=output['cos_tilt_1'][0], 
    cos_tilt_2=output['cos_tilt_2'][0],
    phi_12=0.5, 
    phi_jl=0.5, 
    luminosity_distance=float(bilby.gw.conversion.redshift_to_luminosity_distance(output['redshift'][0])), 
    theta_jn=0.4, 
    psi=0,
    phase=0.5, 
    geocent_time=1126259642.413, 
    ra=1.5, 
    dec=-1.5,
    fiducial=1)

    injection_parameters['chirp_mass'] = bilby.gw.conversion.component_masses_to_chirp_mass(injection_parameters['mass_1'],
                                                                                        injection_parameters['mass_2'])
    injection_parameters['mass_ratio'] = bilby.gw.conversion.component_masses_to_mass_ratio(injection_parameters['mass_1'],
                                                                                        injection_parameters['mass_2'])
    fiducial_parameters=injection_parameters.copy()

    waveform_arguments = dict(waveform_approximant='IMRPhenomXPHM',
                                  reference_frequency=50., minimum_frequency=20., catch_waveform_errors=True)
    duration = 4.
    sampling_frequency = 2048.
    
    waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_relative_binning,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)
    
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=injection_parameters['geocent_time'] - 3)
    injection = ifos.inject_signal(
        waveform_generator=waveform_generator,
        parameters=injection_parameters)

    H1_signal=ifos[0].get_detector_response(waveform_polarizations=waveform_generator.frequency_domain_strain(), parameters=injection_parameters)
    L1_signal=ifos[1].get_detector_response(waveform_polarizations=waveform_generator.frequency_domain_strain(), parameters=injection_parameters)
    SNR_H1=np.sqrt(ifos[0].optimal_snr_squared(H1_signal).real)
    SNR_L1=np.sqrt(ifos[1].optimal_snr_squared(L1_signal).real)
    SNR_network=np.sqrt(SNR_H1**2 +SNR_L1**2)

    if SNR_network>=5:
        print(f'event found for low SNR {batch} {filename}')
        prior = bilby.core.prior.PriorDict()
        prior['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=10, maximum=100, unit='$M_{\odot}$')
        prior['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1)
        prior['a_1'] = Uniform(name='a_1', minimum=0, maximum=0.99)
        prior['a_2'] = Uniform(name='a_2', minimum=0, maximum=0.99)
        prior['cos_tilt_1'] = Uniform(name='cos_tilt_1', minimum=-1.0, maximum=1.0)
        prior['cos_tilt_2'] = Uniform(name='cos_tilt_2', minimum=-1.0, maximum=1.0)
        for key in ['geocent_time', 'phi_12', 'phi_jl', 'psi', 'ra',
                    'dec','luminosity_distance', 'theta_jn', 'phase']:
            prior[key] = injection_parameters[key]
        for key in ['mass_1', 'mass_2', 'fiducial']:
            del fiducial_parameters[key]

        
        likelihood1 = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
        interferometers=ifos,
        fiducial_parameters=fiducial_parameters,
        waveform_generator=waveform_generator
        )

        result = bilby.run_sampler(
            likelihood=likelihood1,
            priors=prior,
            sampler="dynesty",
            sample='rslice',
            npoints=1000,
            npool=16,
            dlogz=signif,
            outdir=f'{batch}/result_lower_snr_{run_name}',
            label=f'result_lower_snr_{run_name}',
            conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            )
    else:
        print(f'insufficient evidence for a detection for {batch} {filename}')

batch='snr_checker_proper_run_1'
for i  in range(1, 500):
    filename=f'event_{i}'
    UniverseSim(filename, 0.1, batch)
print(f'Simulations finished for {batch}')


# In[ ]:




