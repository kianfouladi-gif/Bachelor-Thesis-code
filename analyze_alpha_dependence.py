import numpy as np
import h5py as h5
import os
from pathlib import Path
from astropy import units as u
from scipy.interpolate import CubicSpline
import parse_numerical_particles as pnp

def calculate_K_eff(dir_data, file_name):
    '''
    calculates the dimensionless effective heat conductivity everywhere in space
    Parameters
    ------------
    dir_data: string
                Directory to store information for halo evolution.
                Either an absolute or a relative path may be provided
                    
    file_name: string
                Name of the file being parsed. (Don't forget .h5 at the end) 
                                       
    Returns
    ------------
    K: array
                Array containing dimensionless effective heat conductivity

    '''
    with h5.File(os.path.join(dir_data.strip(), file_name), 'r') as file:
        params = pnp.get_all_params(dir_data, 'ini'+file_name)
        p = file['p'][:][:]
        v = np.sqrt(file['p'][:][:] / file['rho'][:][:])
        alpha = params['alpha']
        sigma_dim_less = ((params['sigma_m_with_units'] * u.cm**2. / u.g) * ((params['rho_s']* u.M_sun / u.pc**3.) * (params['r_s'] * u.kpc))).to(u.dimensionless_unscaled).value
        K_l = params['a'] * params['C'] * v * p * sigma_dim_less
        K_s = params['b'] * v / sigma_dim_less
        K = (K_s**(-alpha) + K_l**(-alpha))**(-1./alpha)
    return K

def dK_dalpha_fd(dir_data, file_name, file_name_deltaalpha, delta_alpha):
    '''
    calculates the finite difference derivative of the dimensionless effective heat conductivity w.r.t. alpha
    Parameters
    ------------
    dir_data: string
                Directory to store information for halo evolution.
                Either an absolute or a relative path may be provided
                    
    file_name: string
                Name of the file being parsed for alpha = alpha. (Don't forget .h5 at the end) 
    
    file_name_deltaalpha: string
                Name of the file being parsed for alpha = alpha + delta_alpha. (Don't forget .h5 at the end) 

    delta_alpha: float
                The value for delta_alpha used in the finite difference derivative 

    Returns
    ------------
    K: array
                Array containing dimensionless effective heat conductivity
    '''
    K = calculate_K_eff(dir_data, file_name)
    with h5.File(os.path.join(dir_data.strip(), file_name), 'r') as file:
        t = file['t'][:]

    K_deltaalpha = calculate_K_eff(dir_data, file_name_deltaalpha)
    with h5.File(os.path.join(dir_data.strip(), file_name_deltaalpha), 'r') as file:
        t_deltaalpha = file['t'][:]

    if t[-1] <= t_deltaalpha[-1]:
        K_deltaalpha_func = CubicSpline(t_deltaalpha, K_deltaalpha)
        K_deltaalpha = K_deltaalpha_func(t)
    else:
        K_func = CubicSpline(t, K)
        K = K_func(t_deltaalpha)
    
    dK_dalpha = (K_deltaalpha -  K) / delta_alpha
    
    return np.abs(np.mean(dK_dalpha))

def sort_params(list_params):
    '''
    Sorts parameters in the first argument
    Parameters
    ------------
    list_params: array 
                Array containing the unsorted parameters
    
    Returns
    ------------
    sorted_list: array
                Array containing the sorted parameters
    '''
    idx_sorted = np.argsort(list_params[:,0])
    list_params = list_params[idx_sorted]
    
    idx_l = 0
    idx_r = 0
    const_val = list_params[0,0]
    sorted_lists = []
    for i in range(list_params.shape[0]):
        if list_params[i,0] != const_val or i == (list_params.shape[0] - 1):
            idx_r = i
            if i == (list_params.shape[0] - 1):
                idx_r += 1
            idx_sorted = np.argsort(list_params[idx_l:idx_r,1])
            sorted_lists.append(list_params[idx_sorted + idx_l])
            const_val = list_params[i,0]
            idx_l = idx_r

    sorted_lists = np.array(sorted_lists)
    sorted_list = np.concatenate(sorted_lists)
    return sorted_list

def calculate_alpha_crit(dir_Gravothermal_data, name_halo):
    '''
    Calculates the dK/d alpha values for every C and all possible alphas for each halo.
    Parameters
    ------------
    dir_Gravothermal_data: string 
                Directory to store information for halo evolution.
                Either an absolute or a relative path may be provided

    name_halo: string 
                Name of the studied halo.
                
    Returns
    ------------
    sorted_list: array
                Array containing the sorted parameters
    '''
    folder_Gravothermal_data = Path(dir_Gravothermal_data)
    if not folder_Gravothermal_data.is_dir():
        raise ValueError('Folder argument is incorrect.')
    
    list_params = []

    for file in folder_Gravothermal_data.iterdir(): 
        if (file.is_file() and file.suffix == '.h5') and not file.name.startswith('ini'):
            list_params.append(pnp.get_params_from_name(str(file.name)))
    
    list_params = np.array(list_params)
    list_params = sort_params(list_params)
    list_dK_dalpha = []
    
    const_val = list_params[0,0]
    list_dK_dalpha_int = []
    for i in range(list_params.shape[0]-1):
        if list_params[i+1,0] == const_val:
            file = name_halo + '_C_' + str(list_params[i,0]) + '_alpha_' + str(list_params[i,1]) + '.h5'
            file_deltaalpha = name_halo + '_C_' + str(list_params[i+1,0]) + '_alpha_' + str(list_params[i+1,1]) + '.h5'
            deltaalpha = np.abs(list_params[i+1,1] - list_params[i,1])
            list_dK_dalpha_int.append(dK_dalpha_fd(dir_Gravothermal_data, file, file_deltaalpha, deltaalpha)) 
        else:
            const_val = list_params[i+1,0]
            list_dK_dalpha.append(list_dK_dalpha_int)
            list_dK_dalpha_int = []
        
        if (i == list_params.shape[0]-2):
            list_dK_dalpha.append(list_dK_dalpha_int)
        
    return np.array(list_dK_dalpha)