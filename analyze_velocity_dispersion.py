import numpy as np
import h5py as h5
import os
from pathlib import Path
from astropy import units as u
import astropy.units.cds as cds
from astropy import constants as c
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize, differential_evolution
import parse_numerical_particles as pnp

cds.enable()

def read_velocity_dispersion(dir_data, file_name):
    '''
    Calculates the one and three dimensional velocity dispersions for the gravothermal simulations
    Parameters
    ------------
    file_name: string
            Name of the file being parsed. (Don't forget .h5 at the end) 
        
    dir_data: string
            Directory to store information for halo evolution.
            Either an absolute or a relative path may be provided
               
    Returns
    ------------
    v_1d, v_3d
            The one and three dimensional velocity dispersion respectively in each bin.
    '''
    with h5.File(os.path.join(dir_data.strip(), file_name), 'r') as file:
        params = pnp.get_all_params(dir_data, 'ini'+file_name)
        v_scale = (np.sqrt(4. * np.pi * c.G * (params['rho_s']* u.M_sun / u.pc**3.)) * (params['r_s'] * u.kpc)).to(u.km / u.s).value
        v_3d = np.sqrt(3. * file['p'][:][:] / file['rho'][:][:]) * v_scale
        v_1d = np.sqrt(file['p'][:][:] / file['rho'][:][:]) * v_scale
        return v_1d, v_3d

def bin_expected_partilces_prep(dir_N_body, file_N_body, dir_data):
    '''
    Creates functions describing the number of expected enclosed numerical as a function of parameters particles binned at N-body radii. To be used in bin_expected_partilces.
    Parameters
    ------------
    dir_N_body: string
                Directory to store information for halo evolution for the N_body simulations.
                Either an absolute or a relative path may be provided
            
    file_N_body: string
                Name of the file containing the N-body snapshots.

    dir_data: string
                Directory to store information for halo evolution.
                Either an absolute or a relative path may be provided
                   
    Returns
    ------------
    M_interp: array
                Array of functions describing the enclosed number of expected enclosed numerical particles binned at N-body radii as a function of parameters.
    '''
    N_body_data = pnp.parse_input_N_body(dir_N_body, file_N_body)
    r_N_body = N_body_data['ProfileEdges']

    M_interp = []

    for i in range(r_N_body.shape[0]):
        M_interp.append(pnp.sim_Number_particles_enclosed_prep2(dir_data, dir_N_body, file_N_body, r_N_body[i], 'thin_plate_spline'))
    return np.array(M_interp)

def bin_expected_partilces(params ,N_body_data, M_interp, m):
    '''
    Creates functions describing the number of expected enclosed numerical as a function of parameters particles binned at N-body radii. To be used in bin_expected_particles.
    Parameters
    ------------
    params: array
                    Array containing the parameters. C is in the first argument and alpha in the second argument.
                
    N_body_data: dict
                    dictionary containing the data from the N-body simulation. The keys are names of the data and the values are numpy arrays containing the data.
    
    M_interp: array
                    Array of functions describing the enclosed number of expected enclosed numerical particles binned at N-body radii as a function of parameters.

    m: float
                    The mass of the numerical particles.
                       
    Returns
    ------------
    Lambda: array
                    Array containing the number of expected enclosed numerical particles at each bin.

   '''
    r_N_body = N_body_data['ProfileEdges']

    Lambda = []
    
    for i in range(r_N_body.shape[0]-1):
        Lambda.append(pnp.sim_Number_particles_enclosed(params, M_interp[i+1], m) - pnp.sim_Number_particles_enclosed(params, M_interp[i], m))

    return np.array(Lambda).transpose()

def v_N_body_compatible(dir_data, file_name, N_body_data):
    '''
    Makes the velocity dispersion from the fluid simulations compatible with the one from the N-body simulations.
    Parameters
    ------------
    dir_data: string
                Directory to store information for halo evolution.
                Either an absolute or a relative path may be provided
                
    file_name: string
                Name of the file being parsed. (Don't forget .h5 at the end) 
    
    N_body_data: dict
                dictionary containing the data from the N-body simulation. The keys are names of the data and the values are numpy arrays containing the data.
                       
    Returns
    ------------
    v_tot: array
                Array of three dimensional velocity dispersions binned compatible with the N-body simulations.

    '''
    v_1d, v_3d = read_velocity_dispersion(dir_data, file_name)
    r_N_body = N_body_data['ProfileEdges']
    t_N_body = N_body_data['Time']

    with h5.File(os.path.join(dir_data.strip(), file_name), 'r') as file:
        params = pnp.get_all_params(dir_data, 'ini'+file_name)
        r_phys = (file['r'][:][:] * params['r_s'])
        t_phys = (file['t'][:] * (1. / np.sqrt(4. * np.pi * c.G * (params['rho_s']* u.M_sun / u.pc**3.))).to('Gyr')).value
        m_phys = np.empty(v_3d.shape)
        m_phys[:,0] = ((file['m'][:,0]) * (4. * np.pi * (params['rho_s']* u.M_sun / u.pc**3.) * (params['r_s'] * u.kpc)**3).to('Msun')).value
        m_phys[:,1:] = ((file['m'][:,1:] - file['m'][:,:-1])* (4. * np.pi * (params['rho_s']* u.M_sun / u.pc**3.) * (params['r_s'] * u.kpc)**3).to('Msun')).value

    v_tmp = np.empty((r_phys.shape[0], len(r_N_body) - 1)) #radial binning 
    for idx_t in range(r_phys.shape[0]):    
        for i in range(len(r_N_body) - 1):
            idx_r_right = np.min(np.where(r_N_body[i+1] <= r_phys[idx_t,:]))
            try: 
                idx_r_left = np.max(np.where(r_N_body[i] >= r_phys[idx_t,:]))
            except ValueError: #if there are no fluid values greater than the N-body radius
                idx_r_left = 0
            if (idx_r_right > idx_r_left):
                r_l = min(r_phys[idx_t,idx_r_left+1], r_N_body[i+1])
                r_r = max(r_phys[idx_t,idx_r_right-1], r_N_body[i])
                if r_l != r_phys[idx_t,idx_r_left]:
                    w_l = (r_l**3. - r_N_body[i]**3.) / (r_l**3. - r_phys[idx_t,idx_r_left]**3.)
                else: 
                    w_l = 0.
                if r_r != r_phys[idx_t,idx_r_right]:
                    w_r = (r_N_body[i+1]**3. - r_r**3.) / (r_phys[idx_t,idx_r_right]**3. - r_r**3.)
                else:
                    w_r = 0.
                v_tmp[idx_t, i] = (np.sum(v_3d[idx_t,idx_r_left+1:idx_r_right] * m_phys[idx_t,idx_r_left+1:idx_r_right]) + (v_3d[idx_t,idx_r_right] * m_phys[idx_t,idx_r_right]) * w_r \
                                + (v_3d[idx_t,idx_r_left] * m_phys[idx_t,idx_r_left]) * w_l) / (np.sum(m_phys[idx_t,idx_r_left+1:idx_r_right]) + w_l * m_phys[idx_t,idx_r_left] + w_r * m_phys[idx_t,idx_r_right])
            else:
                v_tmp[idx_t, i] = np.nan

    v_tot = np.empty(N_body_data['ProfileVelDisprDM'].shape) #temporal binning
    idx_t = np.empty(t_N_body[:-1].shape, dtype=np.int64)
    for i in range(len(t_N_body)):
        if t_phys[-1] < t_N_body[i]:
            v_tot[i,:] = np.nan
            continue
        try:
            if i == 0:
                t_tmp = t_N_body[1] / 2.
                idx_t[0] = np.argmin(np.abs(t_phys - t_tmp))
                v_tot[0,:] = np.mean(v_tmp[:idx_t[0],:], axis=0)
            elif i == len(t_N_body) - 1:
                v_tot[i,:] = np.mean(v_tmp[idx_t[i-1]:,:], axis=0)
            else:
                t_tmp = (t_N_body[i+1] + t_N_body[i]) / 2. 
                idx_t[i] = np.argmin(np.abs(t_phys - t_tmp))
                if v_tmp[idx_t[i-1]:idx_t[i],:].size == 0:
                    v_tot[i,:] = np.nan
                else:
                    v_tot[i,:] = np.mean(v_tmp[idx_t[i-1]:idx_t[i],:], axis=0)
        except ValueError:
            v_tot[i,:] = np.nan

    return v_tot


def combine_v_runs(dir_Gravothermal_data, N_body_data, RBF_kernel, epsilon = 1.):
    '''
    Combines the binned veloctiy dispersion from different fluid simulations to build an array of function, interpolating the velocity dispersion at each point in parameter space.
    Parameters
    ------------
    dir_Gravothermal_data: string
                Directory to store information for halo evolution.
                Either an absolute or a relative path may be provided
                            
    N_body_data: dict
                dictionary containing the data from the N-body simulation. The keys are names of the data and the values are numpy arrays containing the data.

    RBF_kernel: string
                Name of the RBF kernel used, only the in scipy implemented RBF kernels are valid arguments

    epsilon: float (default: 1.)
                Shape parameter for the RBF-kernel; check scipy documentary for further information
                           
    Returns
    ------------
    v_interp: array
                Array of functions having the velocity dispersion as a function of the parameters for each bin.
    '''
    folder_Gravothermal_data = Path(dir_Gravothermal_data)
    if not folder_Gravothermal_data.is_dir():
        raise ValueError('Folder argument is incorrect.')
    
    list_params = []
    list_v = []

    for file in folder_Gravothermal_data.iterdir(): #fills the list of enclosed mass with the enclosed mass at the N-body times with arrays containg the mass, as well as the parameters C and alpha
        if (file.is_file() and file.suffix == '.h5') and not file.name.startswith('ini'):
            list_params.append(pnp.get_params_from_name(str(file.name)))
            v = v_N_body_compatible(dir_Gravothermal_data, str(file.name), N_body_data)
            list_v.append(v)

    list_v = np.array(list_v)
    list_params = np.array(list_params)
    v_interp = []
    for j in range(list_v.shape[1]):
        v_interp_inner = []
        for i in range(list_v.shape[2]):
            where_nan = np.isnan(list_v[:,j,i])
            list_v_no_nan = list_v[:,j,i][~where_nan]
            list_params_no_nan = list_params[~where_nan]
            if list_params_no_nan.size >= 6:
                try:
                    v_interp_inner.append(RBFInterpolator(y=list_params_no_nan, d=list_v_no_nan, kernel=RBF_kernel, epsilon=epsilon))
                except np.linalg.LinAlgError:
                    v_interp_inner.append(np.nan)
            else:
                v_interp_inner.append(np.nan)
        v_interp.append(np.array(v_interp_inner))
        
    return np.array(v_interp)

def bin_v(idx_bin, v, N):
    '''
    bin different N-body velocity dispersions to reduce the impact of noise.
    Parameters
    ------------
    idx_bin: integer
                index of the up to the velocity dispersion should be binned. It must be positive
                                
    v: array
                Array containing the velocity dispersion at each element for each bin.
    
    N: array
                Number of enclosed numerical particles at each position.
                                   
    Returns
    ------------
    v_binned: array
                Array of the binned velocity dispersions.

    '''
    v_binned = np.empty((v[:,idx_bin-1:]).shape)
    mask_nan = ~np.isnan(v[:,:idx_bin])
    v_binned[:,0] = np.nansum(N[:,:idx_bin] * v[:,:idx_bin], axis=1) / np.sum(N[:,:idx_bin] * mask_nan, axis=1)
    v_binned[:,1:] = v[:,idx_bin:]
    return v_binned

def sum_squared_v(params ,v_N_body, N_body_data, v_interp, m_interp, m, idx_noise ,modes, L, R):
    '''
    Calculates the sum squared difference between the N-body and Gravothermal simulations.
    Parameters
    ------------
    params: array
                Array containing the parameters. C is in the first argument and alpha in the second argument.

    v_N_body: array 
                Array containg the N-body velocity dispersions

    N_body_data: dict
                dictionary containing the data from the N-body simulation. The keys are names of the data and the values are numpy arrays containing the data.

    v_interp: array
                Array of functions having the velocity dispersion as a function of the parameters for each bin.

    m_interp: array
                Array of functions describing the enclosed number of expected enclosed numerical particles binned at N-body radii as a function of parameters.

    m: float
                The mass of the numerical particles.

    idx_noise: integer (deprecated)
                index of the up to the velocity dispersion should be binned, due to noise. It must be positive.

    modes:  string
                if mode is 'ignore_noise', then the velocity dispersion up to a certain bon will be binned (deprcated).

    L: integer 
                index, where the summation starts, for the normal case

    R: integer 
                index, where the summation ends, for the normal case 

    Returns
    ------------
    sorted_list: array
                Array containing the sorted parameters
    '''
    Lambda = bin_expected_partilces(params, N_body_data, m_interp, m)
    N = N_body_data['ProfileNumDM']

    v_Gr = []
    for i in range(v_interp.shape[0]):
        v_Gr_inner = []
        for j in range(v_interp.shape[1]):
            try:
                v_Gr_inner.append(v_interp[i,j](np.atleast_2d(params))[0])
            except TypeError:
                v_Gr_inner.append(np.nan)
        v_Gr.append(v_Gr_inner)
    v_Gr = np.array(v_Gr)

    if modes == 'ignore_noise':
        v_Gr_binned = bin_v(idx_noise, v_Gr, Lambda)
        v_N_body_binned = bin_v(idx_noise, v_N_body, N)
        return np.nansum(np.abs((v_N_body_binned - v_Gr_binned)))
    else:
        return np.nansum(((v_N_body[:,L:R] - v_Gr[:,L:R]))**2.)

def optimize_params_v(dir_Gravothermal_data, dir_N_body, file_N_body, RBF_kernel, bounds, modes, idx_noise, m, epsilon = 1.):
    '''
    Finds the global minimum of the sum squared difference between the N-body and the gavothermal fluid velocity dispersions.
    Parameters
    ------------
    dir_Gravothermal_data: string 
                Directory to store information for halo evolution.
                Either an absolute or a relative path may be provided

    dir_N_body: string
                Directory to store information for halo evolution for the N_body simulations.
                Either an absolute or a relative path may be provided
            
    file_N_body: string
                Name of the file containing the N-body snapshots.

    RBF_kernel: string
                Name of the RBF kernel used, only the in scipy implemented RBF kernels are valid arguments

    bounds: array
                Array containg the bound sfor the minimization search. The first element takes boundaries ofr C and the second for alpha.

    modes:  string
                if mode is 'ignore_noise', then the velocity dispersion up to a certain bon will be binned (deprcated).

    idx_noise: integer (deprecated)
                index of the up to the velocity dispersion should be binned, due to noise. It must be positive.
    
    m: float
                The mass of the numerical particles.

    epsilon: float (default: 1.)
                Shape parameter for the RBF-kernel; check scipy documentary for further information
    Returns
    ------------
    result: OptimizeResult class from scipy.optimzie
                contains all information about the result
    '''
    N_body_data = pnp.parse_input_N_body(dir_N_body, file_N_body)
    v_N_body = np.sqrt(N_body_data['ProfileVelDisprDM'])
    v_interp = combine_v_runs(dir_Gravothermal_data, N_body_data, RBF_kernel, epsilon)
    m_interp = bin_expected_partilces_prep(dir_N_body, file_N_body, dir_Gravothermal_data)

    args =  (v_N_body ,N_body_data, v_interp, m_interp, m, idx_noise, modes)
    result = differential_evolution(sum_squared_v, bounds=bounds, args=args)
    return result
    
def optimize_params_v_local(dir_Gravothermal_data, dir_N_body, file_N_body, RBF_kernel, bounds, modes, idx_noise, m, IC, epsilon = 1.):
    '''
    Finds the global minimum of the sum squared difference between the N-body and the gavothermal fluid velocity dispersions.
    Parameters
    ------------
    dir_Gravothermal_data: string 
                Directory to store information for halo evolution.
                Either an absolute or a relative path may be provided

    dir_N_body: string
                Directory to store information for halo evolution for the N_body simulations.
                Either an absolute or a relative path may be provided
            
    file_N_body: string
                Name of the file containing the N-body snapshots.

    RBF_kernel: string
                Name of the RBF kernel used, only the in scipy implemented RBF kernels are valid arguments

    bounds: array
                Array containg the bound sfor the minimization search. The first element takes boundaries ofr C and the second for alpha.

    modes:  string
                if mode is 'ignore_noise', then the velocity dispersion up to a certain bon will be binned (deprcated).

    idx_noise: integer (deprecated)
                index of the up to the velocity dispersion should be binned, due to noise. It must be positive.
    
    m: float
                The mass of the numerical particles.

    IC: array
                Initial conditions in parameterspace for the local minimizer. The first initial condition is for C and the second for alpha

    epsilon: float (default: 1.)
                Shape parameter for the RBF-kernel; check scipy documentary for further information
    Returns
    ------------
    result: OptimizeResult class from scipy.optimzie
                contains all information about the result
    '''
    N_body_data = pnp.parse_input_N_body(dir_N_body, file_N_body)
    v_N_body = np.sqrt(N_body_data['ProfileVelDisprDM'])
    v_interp = combine_v_runs(dir_Gravothermal_data, N_body_data, RBF_kernel, epsilon)
    m_interp = bin_expected_partilces_prep(dir_N_body, file_N_body, dir_Gravothermal_data)  

    args = (v_N_body, N_body_data, v_interp, m_interp, m, idx_noise, modes)
    result = minimize(sum_squared_v, IC, args=args, bounds=bounds)
    if result.success:
        return result
    else:
        print('Warning no minimisation reached, with', IC)
        return result