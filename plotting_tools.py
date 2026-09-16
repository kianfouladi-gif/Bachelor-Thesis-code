import parse_numerical_particles as pnp
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True

def genenerate_label(name):
    '''
    Generates a label for the plots below.

    Parameters
    ------------
    name: string  
            Name of the file. The files need to named in a specific way for this to work, see RunHaloEvolution.py for that, or see workflow for details.

    Returns
    ------------
            String: The label for the plots
    '''

    name_split = name.split('_')
    C = name_split[3]
    alpha = name_split[-1][:-3]

    return r'C = ' + C +r', $\alpha$ = ' + alpha

def plot_enclosed_particles2(dir_Gravothermal_data, dir_data_N_body, file_name_N_body, radius, m, figure_name, local = False):
    '''
    Plots the number enclosed N-body simulation particles at a given radius as a function of time, as well as the deviation from the N-body simulation. Used for plotting the halo best fit and universal fit

    Parameters
    ------------
    dir_Gravothermal_data: string  
            Directory to store information for halo evolution in the Gravothermal SIDM case. An absolute path needs to provided.

    dir_data_N_body: string
            Directory to store information for halo evolution in the N-body simulation case. An absolute path needs to provided.

    file_name_N_body: string
            Name of the text file containing the data from the N-body simulation.  

    radius: float (in kpc)
            Radius to which the enclosed mass is computed. 

    m: float
            Mass of the particle of the N-body simulation.

    figure_name: string
            Name of the figure beeing saved. (Do not forget the format)

    local: bool
            If true a linear interpolation of the enclosed mass is made, if not a cubic spline (global method) is used.

    Returns
    ------------
    A figure is created
    '''
    N_body_result = pnp.parse_input_N_body(dir_data_N_body, file_name_N_body)
    Num_N_body = pnp.N_body_particles_enclosed(N_body_result, radius)
    
    fig, ax = plt.subplots(1, 1, figsize = (6, 6), constrained_layout=True)
    colors = plt.cm.tab20(np.linspace(0., 1., 20)) #generate color for the plots
    colors_gravothermal = ['orange', 'indianred']

    ax.plot(N_body_result['Time'], Num_N_body, label = r'N-body', color = colors[0]) 

    folder_Gravothermal_data = Path(dir_Gravothermal_data)
    if not folder_Gravothermal_data.is_dir():
        raise ValueError('Folder argument is incorrect.')

    list_C = []
    list_alpha = []
    list_files = []
    for file in folder_Gravothermal_data.iterdir():
        list_C.append(pnp.get_params_from_name(str(file.name))[0])
        list_alpha.append(pnp.get_params_from_name(str(file.name))[1])  
        list_files.append(file)

    zipped_sorted = sorted(zip(list_C, list_alpha, list_files))
    list_C, list_alpha, list_files = list(zip(*zipped_sorted))

    color_idx = 0
    for idx, file in enumerate(list_files): 
        if (file.is_file() and file.suffix == '.h5') and not file.name.startswith('ini'):
            if pnp.get_params_from_name(str(file.name))[0] == 0.857 and pnp.get_params_from_name(str(file.name))[1] == 0.57:
                color_idx = 0                
            else:
                color_idx = 1
            Gravothermal_data = pnp.GravothermalSIDM_M_enclosed(dir_Gravothermal_data, str(file.name), radius, N_body_result['Time'][-1]) #plots for the Gravothermal fluid model
            ax.plot(Gravothermal_data['t_phys'], Gravothermal_data['M_enclosed'] / m, label = genenerate_label(str(file.name)), color = colors_gravothermal[color_idx]) #index shifted by 1, due to the coloring of the N-body simulations
            color_idx += 1
            
    ax.legend()
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$ [Gyr]', fontsize=12)
    ax.set_ylabel(r'$M \left(r = radius \, \mathrm{kpc} \right) / m_N$'.replace('radius', str(radius)), fontsize=12)

    plt.savefig(figure_name)
    plt.show()
    return

def plot_enclosed_pasarticles_all_halos_individual(list_gravothermal, name, list_m, list_file_N_body, list_data_N_body):
    '''
    Plots the number enclosed N-body simulation particles at a given radius as a function of time, as well as the deviation from the N-body simulation of the four halos individually.

    Parameters
    ------------
    list_gravothermal: string  
            list of Directory to store information for halo evolution in the Gravothermal SIDM case. An absolute path needs to provided.

    name: string
            Name of the figure beeing saved. (Do not forget the format)

    list_m: float
            list of the masses of the particles of the N-body simulation.

    list_file_name_N_body: string
            list of the name sof the text file containing the data from the N-body simulation.  
            
    list_dir_data_N_body: string
            list of the directories to store information for halo evolution in the N-body simulation case. An absolute path needs to provided.

    Returns
    ------------
    A figure is created
    '''
    data_N_body, data_N_body_0, data_N_body_2, data_N_body_3 = list_data_N_body
    file_N_body, file_N_body_0, file_N_body2, file_N_body3 = list_file_N_body
    m_1, m_0, m_2, m_3 = list_m
    N_body_data = pnp.parse_input_N_body(data_N_body, file_N_body)
    N_body_data_0 = pnp.parse_input_N_body(data_N_body_0, file_N_body_0)
    N_body_data_2 = pnp.parse_input_N_body(data_N_body_2, file_N_body2)
    N_body_data_3 = pnp.parse_input_N_body(data_N_body_3, file_N_body3)
    N_body_list = [N_body_data, N_body_data_0, N_body_data_2, N_body_data_3]
    m_list = [m_1, m_0, m_2, m_3]

    for i in range(len(m_list)):
        N_body_data_tmp = N_body_list[i]
        m_tmp = m_list[i]

        fig, ax = plt.subplots(1, 1)
        colors = plt.cm.tab20(np.linspace(0., 1., 20))

        if i == 2:
            radius = 0.01
        else:
            radius = 0.1

        Num_N_body = pnp.N_body_particles_enclosed(N_body_data_tmp, radius)
        ax.plot(N_body_data_tmp['Time'], Num_N_body, label = r'N-body', color = colors[0]) 

        folder_Gravothermal_data = Path(list_gravothermal[i])
        if not folder_Gravothermal_data.is_dir():
            raise ValueError('Folder argument is incorrect.')
        
        list_C = []
        list_alpha = []
        list_files = []
        for file in folder_Gravothermal_data.iterdir():
            list_C.append(pnp.get_params_from_name(str(file.name))[0])
            list_alpha.append(pnp.get_params_from_name(str(file.name))[1])  
            list_files.append(file)

        zipped_sorted = sorted(zip(list_C, list_alpha, list_files))
        list_C, list_alpha, list_files = list(zip(*zipped_sorted))
        
        for idx, file in enumerate(list_files): 
            if (file.is_file() and file.suffix == '.h5') and not file.name.startswith('ini'):
                Gravothermal_data = pnp.GravothermalSIDM_M_enclosed(list_gravothermal[i], str(file.name), radius, N_body_data_tmp['Time'][-1]) #plots for the Gravothermal fluid model
                ax.plot(Gravothermal_data['t_phys'], Gravothermal_data['M_enclosed'] / m_tmp, label = genenerate_label(str(file.name)), color = colors[idx+1]) #index shifted by 1, due to the coloring of the N-body simulations
    
        ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.set_xlabel(r'$t \, \mathrm{[Gyr]}$', fontsize=12)
        ax.set_ylabel(r'$M (r = radius \, \mathrm{kpc}) / m_N$'.replace('radius', str(radius)), fontsize=12)

        if i == 0:
            ax.legend()
        plt.savefig(name + 'inidivual_number.png'.replace('number', str(i)))
    plt.show()
    return

def plot_params(dir_N_body, file_N_body, C, alpha, C_err_plus, C_err_minus, alpha_err_plus, alpha_err_minus, inner_size,name):
    '''
    Plots the best fit values for both parameters as a function of radii.

    Parameters
    ------------
    dir_data_N_body: string
            Directory to store information for halo evolution in the N-body simulation case. An absolute path needs to provided.

    file_name_N_body: string
            Name of the text file containing the data from the N-body simulation.  

    C: array
            Array of best fit values for C.

    alpha: array
            Array of the best fit values for alpha.

    C_err_plus: array
            Array of the upper errors for alpha.

    C_err_minus: array
            Array of the lower errors for C.

    alpha_err_plus: array
            Array of the upper errors for alpha.

    alpha_err_minus: array
            Array of the lower errors for alpha.

    inner_size: float (in kpc)
        position of the inner dotted line

    name: string
        
            
    Returns
    ------------
    A figure is created    
    '''
    N_body_data = pnp.parse_input_N_body(dir_N_body, file_N_body)
    r = N_body_data['ProfileEdges']
    C_err = np.column_stack((C_err_minus, C_err_plus)).transpose()
    alpha_err = np.column_stack((alpha_err_minus, alpha_err_plus)).transpose()
    plt.errorbar(r, alpha, yerr=alpha_err, label=r'$\alpha$', fmt='o', capsize=5., markersize=2, color='red')
    plt.errorbar(r, C, yerr=C_err, label=r'$\mathrm{C}$', fmt='o', capsize=5., markersize=2, color='blue')
    plt.legend()
    plt.xscale('log')
    plt.xlabel(r'$r \, \mathrm{[kpc]}$', fontsize=12)
    plt.ylabel(r'$\mathrm{Parameters}$', fontsize=12)
    plt.vlines(inner_size, 0.2,  1.2,colors='gray', linestyles='dashed')
    plt.vlines(1., 0.2, 1.2, colors='gray', linestyles='dashed')
    plt.savefig(name + 'new.png')
    plt.show()


def plot_params_actual(dir_N_body, file_N_body, C, alpha, C_err_plus, C_err_minus, alpha_err_plus, alpha_err_minus, considered_idx, inner_size, mean_c, mean_alpha, name):
    '''
    Plots the best fit values for both parameters as a function of radii.

    Parameters
    ------------
    dir_data_N_body: string
            Directory to store information for halo evolution in the N-body simulation case. An absolute path needs to provided.

    file_name_N_body: string
            Name of the text file containing the data from the N-body simulation.  

    C: array
            Array of best fit values for C.

    alpha: array
            Array of the best fit values for alpha.

    C_err_plus: array
            Array of the upper errors for alpha.

    C_err_minus: array
            Array of the lower errors for C.

    alpha_err_plus: array
            Array of the upper errors for alpha.

    alpha_err_minus: array
            Array of the lower errors for alpha.

    considered_idx: array
            array of indices considered for the fit

    inner_size: float (in kpc)
            position of the inner dotted line
    
    mean_c: float
            mean value for C

    mean_alpha: float
            mean value for alpha

    name:
            Name of the figures beeing saved. (Do not forget the format)
            
    Returns
    ------------
    Two figures are created    
    '''
    N_body_data = pnp.parse_input_N_body(dir_N_body, file_N_body)
    r = N_body_data['ProfileEdges']
    C_err = np.column_stack((C_err_minus, C_err_plus)).transpose()
    alpha_err = np.column_stack((alpha_err_minus, alpha_err_plus)).transpose()

    C_final = C[considered_idx]
    C_err_final = C_err[:,considered_idx]
    alpha_final = alpha[considered_idx]
    alpha_err_final = alpha_err[:,considered_idx]
    r_final = r[considered_idx]

    fig, ax = plt.subplots(2, 1, figsize=(6, 6) ,constrained_layout=True)
    ax[0].errorbar(r_final, alpha_final, yerr=alpha_err_final, label=r'$\alpha$', fmt='o', capsize=5., markersize=2, color='red')
    ax[1].errorbar(r_final, C_final, yerr=C_err_final, label=r'$\mathrm{C}$', fmt='o', capsize=5., markersize=2, color='blue')
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[0].set_xlabel(r'$r \, \mathrm{[kpc]}$', fontsize=12)
    ax[0].set_ylabel(r'$\alpha$', fontsize=12)
    ax[1].set_xlabel(r'$r \, \mathrm{[kpc]}$', fontsize=12)
    ax[1].set_ylabel(r'$C$', fontsize=12)
    ax[1].hlines(mean_c, r_final[0], r_final[-1], linestyles='dotted', colors='blue')
    ax[0].hlines(mean_alpha, r_final[0], r_final[-1], linestyles='dotted', colors='red')
    plt.savefig(name + '_extended_new.png')
    plt.show()