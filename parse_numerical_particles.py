import numpy as np
import h5py as h5
import os
from pathlib import Path
from astropy import units as u
import astropy.units.cds as cds
from astropy import constants as c
from scipy.interpolate import CubicSpline, interp1d, RBFInterpolator

cds.enable()

def get_params(dir_data, file_name = 'halo_ini.h5'):
        '''
        Gets relevant Halo IC parameters. Note the ini file needs to be in the same folder as the file containing the data.
        
        Parameters
        ------------
        file_name: string (dafault halo_ini.h5)
        Name of the file being parsed. (Don't forget .h5 at the end) 

        dir_data: string
            Directory to store information for halo evolution.
            Either an absolute or a relative path may be provided
       
        Returns
        ------------
        dictionary
            Dictionary containg the scale radius and density of the halo. r_s in kpc and rho_s in Msun / pc**3.
        '''
        with h5.File(os.path.join(dir_data.strip(), file_name), 'r') as file:
             params_dict = dict(file.attrs)
             return {'r_s': params_dict['r_s'] * u.kpc, 'rho_s': params_dict['rho_s'] * u.Msun / u.pc**3}

def GravothermalSIDM_M_enclosed(dir_data, file_name, radius, t_N_body, local=False): 
    '''
    Read the simulation data from a h5py file, compute properties used in the further analysis: enclosed mass
    at a given radius and the associated errors. Remark: the h5 files containing the initial data need to be named 'ini'+filename and stored in the same directory.

    Parameters
    ------------
    file_name: string  
            Name of the file being parsed. (Don't forget .h5 at the end) 

    dir_data: string
            Directory to store information for halo evolution.
            Either an absolute or a relative path may be provided

    radius: float (in kpc, do not use astropy units however, this holds for all functions, if not explicitly stated)
            Radius at which the enclosed mass is calculated
    
    t_N_body: float (in Gyr)
            Maximum time up to which the data is parsed and the quantities evaluated. Longer times are not required since the model is being compared to N-body simulations. 

    local: bool
            If true a linear interpolation of the enclosed mass is made, if not a cubic spline (global method) is used.
    Returns
    ------------
    GravothermalSIDM_data: dict
    dictionary containing the data from the simulation and calculated quantities for the analysis. (See code for details)
    '''
    params = get_params(dir_data, file_name= 'ini'+file_name)   

    with h5.File(os.path.join(dir_data.strip(), file_name), 'r') as file:
        M_0 = (4. * np.pi * params['rho_s'] * params['r_s']**3).to('Msun')
        t_0 = (1. / np.sqrt(4. * np.pi * c.G * params['rho_s'])).to('Gyr')

        t_phys = (file['t'][:] * t_0).value #transforming dimensionless to physical quantities, [t_phys] = Gyr, note astropy units were removed.
        r_phys = (file['r'][:][:] * params['r_s']).value #[r_phys] = kpc 
        M_phys = (file['m'][:][:] * M_0).value# [M_phys] = Msun

        M_enclosed, delta_m, nearest_r, error_r, t_tmp = [], [], [], [], []
        GravothermalSIDM_data = {}

        if not local:
            for i in range(len(t_phys)):  #Cubic spline
                if (t_phys[i-1] > t_N_body) and i >= 1: #calculate properties for analysis and plots, that there is at least always one data point with a time > t_N_body, in order to never extrapolate
                    continue                    
                
                t_tmp.append(t_phys[i])
                M_func = CubicSpline(r_phys[i][:], M_phys[i][:]) #check the neccesisty of cubic spline vs linear interpolation
                M_enclosed.append(M_func(radius))
                nearest_r.append(r_phys[i][np.argmin(np.abs(r_phys[i] - radius))])
                delta_m.append(M_func(radius) - M_phys[i][np.argmin(np.abs(r_phys[i] - radius))]) #compute 
                error_r.append((nearest_r[-1] - radius) / radius)         
        else:
             for i in range(len(t_phys)): #linear interpolation
                if (t_phys[i-1] > t_N_body) and i >= 1:  #calculate properties for analysis and plots
                    continue    
                
                t_tmp.append(t_phys[i])
                M_interp = np.interp(radius ,r_phys[i][:], M_phys[i][:]) #check the neccesisty of cubic spline vs linear interpolation
                M_enclosed.append(M_interp)
                nearest_r.append(r_phys[i][np.argmin(np.abs(r_phys[i] - radius))])
                delta_m.append(M_interp - M_phys[i][np.argmin(np.abs(r_phys[i] - radius))])
                error_r.append((nearest_r[-1] - radius) / radius)


        zipped_sorted = sorted(zip(t_tmp, M_enclosed, delta_m, nearest_r, error_r))
        GravothermalSIDM_data['t_phys'], GravothermalSIDM_data['M_enclosed'], GravothermalSIDM_data['delta_m'], GravothermalSIDM_data['nearest_r'], GravothermalSIDM_data['error_r'] = np.array(list(zip(*zipped_sorted)))
        return GravothermalSIDM_data
                 
def parse_input_N_body(dir_data ,file_name = 'results_profiles.txt'):
    '''
    Read the N-body simulation data from a structured text file. 

    Parameters
    ------------
    file_name: string (default: results_profiles.txt) 
        Name of the file beeing parsed. (Don't forget .txt at the end) 

    dir_data: string
            Directory to store information for halo evolution.
            Either an absolute or a relative path may be provided

    Returns
    ------------
    result: dict
            dictionary containing the data from the N-body simulation. The keys are names of the data and the values are numpy arrays containing the data.
    '''

    result = {}
    with open(os.path.join(dir_data.strip(), file_name), 'r') as fp:
        next(fp) #no data in fist line -> skip
        for line in fp:
            line = line.strip()
            pos0 = line.find("Key:")+5 #determine the position of the keys
            pos1 = line.find(",",pos0)
            key = line[pos0:pos1]
            
                
            pos0 = line.find("Shape:")+8 #determine the shape of the objects
            pos1 = line.find("),",pos0)
            shape = list(map(int, line[pos0:pos1].split(', ')))
                
            pos0 = line.find("DataType:")+10 #get the data type
            pos1 = line.find(",",pos0)
            datatype = line[pos0:pos1]
                
            pos = line.find("Data: ")+6 #read the data
            line2 = line[pos:].replace("["," ").replace("]"," ").replace(","," ").strip()
                
            if datatype in ["double", "float"]: #data is given in string, transform it to appropiate data type
                data = list(map(float, line2.split()))
            elif datatype in ["int", "unsigned int"]:
                data = list(map(int, line2.split()))
            elif datatype in ["long long unsigned int", "long unsigned int", "long long int", "long int"]:
                line2 = line2.replace("18446744073709551615","-1")
                data = list(map(int, line2.split()))
            elif datatype == "string":
                if shape[0]!=1 or len(shape)>1:
                    data = line2.split("  ")
                else:
                    data = line2
            elif datatype == "bool":
                data = list(map(int, line2.split()))
            else:
                print("datatype unkown,",datatype)
            data = np.array(data)
            if data.size != np.prod(shape):
                raise ValueError("Error with the shape of the data array in ", key, file_name)

            result[key] = data.reshape(shape)
    
    try: 
        result["Time"] *= 0.979 #change the time to physical time in Gyr.
    except KeyError:
        raise Exception("Some error with time. In", file_name)

    return result

def get_params_from_name(name):
    '''
    From a 'structured name' (i.e. the name of the h5 file containing the data), it reads of the paraneters C and alpha. (See workflow for details)

    Parameters
    ------------
    name: string  
            Name of the file. The files need to named in a specific way for this to work, see RunHaloEvolution.py for that

    Returns
    ------------
            Array: An array containing the values of C and alpha. C in the first and alpha in the second position.
    '''
    name_split = name.split('_')
    C = float(name_split[3])
    alpha = float(name_split[-1][:-3])
    return np.array([C, alpha])

def sim_Number_particles_enclosed_prep2(dir_Gravothermal_data, dir_data_N_body, file_name_N_body, radius, RBF_kernel, cubic_spline = False, epsilon = 1.):
    '''
        Calculates the expected number of simulation particles for parameters alpha and C.

    Parameters
    ------------
    dir_Gravothermal_data: string  
            Directory to store information for halo evolution in the Gravothermal SIDM case. An absolute path needs to provided.

    dir_data_N_body: string
            Directory to store information for halo evolution in the N-body simulation case. An absolute path needs to provided.

    file_name_N_body: string
            Name of the text file containing the data from the N-body simulation.  

    radius: float (in kpc)
            Radius to which the enclosed mass is computed. Remark: the radius
            needs to be included in the N-body simulation data.

    bounds: tuple
            Having the upper and lower bound of the Parameters of the fit.

    RBF_kernel: string
            Kernel used for the RBF-interpolator, see scipy documentation for options.

    cubic_spline: bool (default: False)
           If true a cubic spline for interpolation of the enclosed mass, a cubic spline interpolation is used, else a linear interpolation is used.

    epsilon: float (default 1.)
           Shape parameter for the RBF-kernel.
    
    Returns
    ------------
    Array
            Array containg the function M(alpha, C) in each time step.
    '''
    N_body_result = parse_input_N_body(dir_data_N_body, file_name_N_body) #function name is WIP

    folder_Gravothermal_data = Path(dir_Gravothermal_data)
    if not folder_Gravothermal_data.is_dir():
        raise ValueError('Folder argument is incorrect.')
    
    list_M_enclosed_Gravothermal = []
    list_params = []

    for file in folder_Gravothermal_data.iterdir(): #fills the list of enclosed mass with the enclosed mass at the N-body times with arrays containg the mass, as well as the parameters C and alpha
        if (file.is_file() and file.suffix == '.h5') and not file.name.startswith('ini'):
            Gravothermal_data = GravothermalSIDM_M_enclosed(dir_Gravothermal_data, str(file.name), radius, N_body_result['Time'][-1])
            if cubic_spline:
                M_enclosed_func = CubicSpline(Gravothermal_data['t_phys'], Gravothermal_data['M_enclosed'])
            else:
                M_enclosed_func = interp1d(Gravothermal_data['t_phys'], Gravothermal_data['M_enclosed'], kind='linear', bounds_error=False, fill_value='extrapolate') #the enclosed mass from the gravothermal fluid simulation and the N-body simulation do not neccercerally match.
            idx_t_N_body = np.max(np.where(np.abs(N_body_result['Time'] <= Gravothermal_data['t_phys'][-1])))
            list_M_enclosed_Gravothermal.append(M_enclosed_func(N_body_result['Time'][:(idx_t_N_body+1)]))#Keep the time t_N_body, thats why idx_t_N_body+1 
            list_params.append(get_params_from_name(str(file.name)))                                           
    
    list_interpolations = []
    for i in range(len(N_body_result['Time'])):
        list_interp_evals = []
        list_interp_params = []
        for j in range(len(list_M_enclosed_Gravothermal)):
            try:    
                list_interp_evals.append(list_M_enclosed_Gravothermal[j][i])
                list_interp_params.append(list_params[j])
            except IndexError:
                pass
        list_interpolations.append(RBFInterpolator(y=list_interp_params, d=list_interp_evals, kernel=RBF_kernel, epsilon=epsilon))

    return np.array(list_interpolations)

def sim_Number_particles_enclosed(params , M_interp, m):
    '''
    Calculates the expected number of simulation particles for parameters alpha and C.

    Parameters
    ------------
    params: array
            Array containing the parameters. C is in the first argument and alpha in the second argument.

    m: float
            Mass of the particle of the N-body simulation.

    M_interp: array
            Array conatining the function M(alpha, C) in each time step.
    
    Returns
    ------------
    Array
            Array containing the Number of expected N-body simulation particles for all N-body times, for parameters alpha and C.
    '''
    
    M_Gr = np.abs(np.array([M_interp[i](np.atleast_2d(params))[0] for i in range(len(M_interp))])) #interpolation can result in negative values, which is unphysical, hence the absolute value
    return M_Gr / m

def N_body_particles_enclosed(N_body_data, radius):
    '''
    Computes the enclosed mass at a given radius and the associated error from the N-body simulation data.

    Parameters
    ------------
    N_body_data: dict
            dictionary containing the data from the N-body simulation. The keys are names of the data and the values are numpy arrays containing the data. 
    
    radius: float (in kpc)
            Radius to which the enclosed mass is computed. Remark: the radius
            needs to be included in the N-body simulation data.
    
    Returns
    ------------
    Num: array
            Array containing the number of enclosed patricles at the given radius in Msun.

    '''
    idx_r = np.argmin(np.abs(N_body_data['ProfileEdges'] - radius))
    Num = (N_body_data['CentralNumDM'] + np.sum(N_body_data['ProfileNumDM'][:,:idx_r], axis=1))  
        
    return Num

def get_all_params(dir_data, file_name):
    '''
    Gets all Halo IC parameters. Note the ini file needs to be in the same folder as the file containing the data. 
    Parameters
    ------------
    file_name: string
            Name of the file being parsed. (Don't forget .h5 at the end) 
    
    dir_data: string
                Directory to store information for halo evolution.
                Either an absolute or a relative path may be provided
           
    Returns
    ------------
    dictionary
                Dictionary containing all initial Halo parameters.
    '''
    with h5.File(os.path.join(dir_data.strip(), file_name), 'r') as file:
        return dict(file.attrs)
