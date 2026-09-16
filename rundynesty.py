import Analysis_Halo as AH
import numpy as np
from scipy.special import gammaln
import matplotlib as mpl
import matplotlib.pyplot as plt
import dynesty
from dynesty import plotting as dplt
from dynesty import utils as dyfunc
import argparse

mpl.rcParams["text.usetex"] = True

def prior_trafo(params, bounds):
    '''
    transforms the prior from the unit cube, as required by dynesty
    Parameters
    ------------
    params: array
                Array containing parameters C in the first argument and alpha in the second.

    bounds: array
                Array containing the bounds for each parameter, C in the first argument and alpha in the second
    Returns
    ------------
    float
    '''
    C, alpha = params
    C_lower, C_upper = bounds[0]
    alpha_lower, alpha_upper = bounds[1]
    C_scaled = C_lower + (C_upper - C_lower) * C
    alpha_scaled = alpha_lower + (alpha_upper - alpha_lower) * alpha
    return (C_scaled, alpha_scaled)

def log_prob(params, M_interp, m, N_body_data, radius, bounds):
    '''
    negative logarithmn of the likelihood 
    Parameters
    ------------
    params: array
                Array containing parameters C in the first argument and alpha in the second.
    
    M_interp: array
                Array containg the function M(alpha, C) in each time step.
    
    m: float
                The mass of the numerical particles.
    
    N_body_data: dict
                dictionary containing the data from the N-body simulation. The keys are names of the data and the values are numpy arrays containing the data.
    
    radius: float (in kpc)
            Radius to which the enclosed mass is computed. Remark: the radius
            needs to be included in the N-body simulation data.
        
    bounds: array
                Array containing the bounds for each parameter, C in the first argument and alpha in the second
    ------------
    float
                negative logarithmn of the likelihood at the parameters given in params
    '''
    N = AH.N_body_particles_enclosed(N_body_data, radius)
    Lambda = AH.sim_Number_particles_enclosed(params, M_interp, m)
    return np.sum(N * np.log(Lambda) - Lambda - gammaln(N+1))

def run_dynesty(Halo ,data, data_N_body, file_N_body, m, bounds, log_prob, prior_trafo, start_idx):
    '''
    performs the dynesty run creates cornerplots and writes all the relevant data into a csv file
    Parameters
    ------------
    Halo: string
                Name of the halo.

    data: string
                Directory to store information for halo evolution for the gravothermal fluid simulation.

    data_N_body: string
            Directory to store information for halo evolution for the N-body simulation.
        
    file_N_body: string
            Name of the file beeing parsed. (Don't forget .txt at the end) 

    m: float
            The mass of the numerical particles.
        
    bounds: array
                Array containing the bounds for each parameter, C in the first argument and alpha in the second+

    log_prob: function
                the logarithmic likelihood function
            
    prior_trafo: function
                the prior transformation function

    start_idx: integer (positive)
                index to start the dynamic nested sampling from

    Returns
    ------------
    mean_val, res_list: arrays
            Array of the mean values and results respectively
    '''
    N_body_data = AH.parse_input_N_body(data_N_body, file_N_body)
    mean_val, res_list = [], []
    with open('results' + Halo + '.csv', 'a') as file:    
        for i in range(start_idx, len(N_body_data['ProfileEdges'])):
            M_interp = AH.sim_Number_particles_enclosed_prep2(data, data_N_body, file_N_body, N_body_data['ProfileEdges'][i], 'thin_plate_spline')
            args = [M_interp, m, N_body_data, N_body_data['ProfileEdges'][i], bounds]
            args_prior = [bounds]
            dyn_sampler = dynesty.DynamicNestedSampler(log_prob, prior_trafo, 2, 2000, 'cubes', bootstrap=50, logl_args=args, ptform_args=args_prior)
            dyn_sampler.run_nested(dlogz_init=0.01)
            res = dyn_sampler.results
            res_samples = res.samples
            res_weights = res.importance_weights()
            mean, cov = dyfunc.mean_and_cov(res_samples, res_weights)
            percentille_C = dyfunc.quantile(res_samples[:,0], [0.16, 0.5, 0.84], res_weights)
            percentille_alpha = dyfunc.quantile(res_samples[:,1], [0.16, 0.5, 0.84], res_weights)
            percentille_C_3s = dyfunc.quantile(res_samples[:,0], [0.0013, 0.5, 0.9987], res_weights)
            percentille_alpha_3s = dyfunc.quantile(res_samples[:,1], [0.0013, 0.5, 0.9987], res_weights)
            q_C = np.diff(percentille_C)
            q_C_3s = np.diff(percentille_C_3s)
            q_alpha = np.diff(percentille_alpha)
            q_alpha_3s = np.diff(percentille_alpha_3s)
            file.write(str(mean[0]) + ',' + str(mean[1]) + ',' + str(q_C[0]) + ',' + str(q_C[1]) + ',' + str(q_C_3s[0]) + ',' + str(q_C_3s[1]) + ',' + str(q_alpha[0]) + ',' + str(q_alpha[1]) + ',' + str(q_alpha_3s[0]) + ',' + str(q_alpha_3s[1]) + ',' + str(percentille_C[1]) + ',' + str(percentille_alpha[1]) + '\n')
            mean_val.append(mean)
            res_list.append(res)
            fig, ax = dplt.cornerplot(res, color='black', quantiles=None, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.4f', labels=[r'$\mathrm{C}$', r'$\mathrm{\alpha}$'], use_math_text=True, truth_color='black', truths=[percentille_C[1], percentille_alpha[1]], quantiles_2d=[0.393, 0.864, 0.989], max_n_ticks=3, hist2d_kwargs={'contourf_kwargs':{'colors': ["white" ,"lightskyblue", "blue", "darkblue"]}}, smooth=0.01)
            plt.savefig(Halo + 'r_{k}.png'.format(k=i)) 
            plt.close(fig)
            fig_, ax_ = dplt.cornerplot(res, color='black', quantiles=None, title_quantiles=[0.0013, 0.5, 0.9987], show_titles=True, title_fmt='.4f', labels=[r'$\mathrm{C}$', r'$\mathrm{\alpha}$'], use_math_text=True, truth_color='black', truths=[percentille_C[1], percentille_alpha[1]], quantiles_2d=[0.393, 0.864, 0.989], max_n_ticks=3, hist2d_kwargs={'contourf_kwargs':{'colors': ["white" ,"lightskyblue", "blue", "darkblue"]}}, smooth=0.01)
            plt.savefig(Halo + 'r_{k}_3s.png'.format(k=i)) 
            plt.close(fig_)
    return mean_val, res_list

def violin_plotting(Halo, data, data_N_body, file_N_body, m, bounds, log_prob, prior_trafo, start_idx):
    '''same as above but creates violinplots of the chosen radii as well'''
    N_body_data = AH.parse_input_N_body(data_N_body, file_N_body)
    r = N_body_data['ProfileEdges']
    C_data, alpha_data = [], []
    C_means, alpha_means = [], []

    with open('results' + Halo + '_violin_.csv', 'a') as file:
        if Halo == 'Halo1_sig30':
            idx_considered = np.array([4, 5, 6, 7, 8, 15, 16])   
        elif  Halo == 'Halo1_sig80':
            idx_considered = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]) + 4
        elif Halo == 'Halo2_sig80':
            idx_considered = np.array([3, 4, 5, 6, 7, 8, 9, 10]) + 4
        elif Halo == 'Halo3_sig700':
            idx_considered = np.array([8, 9, 10, 11, 12, 13, 14, 18, 19, 20]) 
        
        for i in range(start_idx ,len(idx_considered)):
            idx = idx_considered[i]
            M_interp = AH.sim_Number_particles_enclosed_prep2(data, data_N_body, file_N_body, N_body_data['ProfileEdges'][idx], 'thin_plate_spline')#
            args = [M_interp, m, N_body_data, N_body_data['ProfileEdges'][idx], bounds]
            args_prior = [bounds]
            dyn_sampler = dynesty.DynamicNestedSampler(log_prob, prior_trafo, 2, 2000, 'cubes', bootstrap=50, logl_args=args, ptform_args=args_prior)
            dyn_sampler.run_nested(dlogz_init=0.01)
            res = dyn_sampler.results
            res_samples = res.samples
            res_weights = res.importance_weights()
            sample_equal = dyfunc.resample_equal(res_samples, res_weights)
            C_data.append(sample_equal[:,0])
            alpha_data.append(sample_equal[:,1])
            mean, cov = dyfunc.mean_and_cov(res_samples, res_weights)
            percentille_C = dyfunc.quantile(res_samples[:,0], [0.16, 0.5, 0.84], res_weights)
            percentille_alpha = dyfunc.quantile(res_samples[:,1], [0.16, 0.5, 0.84], res_weights)
            percentille_C_3s = dyfunc.quantile(res_samples[:,0], [0.0013, 0.5, 0.9987], res_weights)
            percentille_alpha_3s = dyfunc.quantile(res_samples[:,1], [0.0013, 0.5, 0.9987], res_weights)
            q_C = np.diff(percentille_C)
            q_C_3s = np.diff(percentille_C_3s)
            q_alpha = np.diff(percentille_alpha)
            q_alpha_3s = np.diff(percentille_alpha_3s)
            C_means.append(percentille_C[1])
            alpha_means.append(percentille_alpha[1])
            file.write(str(mean[0]) + ',' + str(mean[1]) + ',' + str(q_C[0]) + ',' + str(q_C[1]) + ',' + str(q_C_3s[0]) + ',' + str(q_C_3s[1]) + ',' + str(q_alpha[0]) + ',' + str(q_alpha[1]) + ',' + str(q_alpha_3s[0]) + ',' + str(q_alpha_3s[1]) + ',' + str(percentille_C[1]) + ',' + str(percentille_alpha[1]) + '\n')
            fig_1, ax_1 = dplt.cornerplot(res, color='black', quantiles=None, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.4f', labels=[r'$\mathrm{C}$', r'$\mathrm{\alpha}$'], use_math_text=True, truth_color='black', truths=[percentille_C[1], percentille_alpha[1]], quantiles_2d=[0.393, 0.864, 0.989], max_n_ticks=3, hist2d_kwargs={'contourf_kwargs':{'colors': ["white" ,"lightskyblue", "blue", "darkblue"]}}, smooth=0.01)
            plt.savefig(Halo + 'r_{k}_cornerviolin.png'.format(k=i)) 
            plt.close(fig_1)
            fig_2, ax_2 = dplt.cornerplot(res, color='black', quantiles=None, title_quantiles=[0.0013, 0.5, 0.9987], show_titles=True, title_fmt='.4f', labels=[r'$\mathrm{C}$', r'$\mathrm{\alpha}$'], use_math_text=True, truth_color='black', truths=[percentille_C[1], percentille_alpha[1]], quantiles_2d=[0.393, 0.864, 0.989], max_n_ticks=3, hist2d_kwargs={'contourf_kwargs':{'colors': ["white" ,"lightskyblue", "blue", "darkblue"]}}, smooth=0.01)
            plt.savefig(Halo + 'r_{k}_3s_cornerviolin.png'.format(k=i)) 
            plt.close(fig_2)

        alpha_means = np.array(alpha_means)
        C_means = np.array(C_means)
        alpha_tot = np.mean(alpha_means)
        C_tot = np.mean(C_means)
        positions = np.arange(1, len(r[idx_considered])+1)
        labels = np.array([str(r) for r in r[idx_considered]])
        tex_labels = [rf'$\mathrm{{{label}}}$' for label in labels]    
        
        fig, ax = plt.subplots(2, 1, constrained_layout=True)
        ax[0].set_xlabel(r'$\mathrm{r [kpc]}$', fontsize=12)
        ax[0].set_ylabel(r'$\mathrm{C}$', fontsize=12)
        ax[0].set_xticks(positions)
        ax[0].set_xticklabels(tex_labels)
        ax[0].hlines(C_tot, positions[0], positions[-1], linestyles='dotted', colors='dodgerblue')
        ax[1].set_xlabel(r'$\mathrm{r [kpc]}$', fontsize=12)
        ax[1].set_ylabel(r'$\mathrm{\alpha}$', fontsize=12)
        ax[1].set_xticks(positions)
        ax[1].set_xticklabels(tex_labels)
        ax[1].hlines(alpha_tot, positions[0], positions[-1], linestyles='dotted', colors='indianred')
        parts_1 = ax[0].violinplot(C_data, positions=positions, widths=0.5, showmedians=True)
        parts_2 = ax[1].violinplot(alpha_data, positions=positions, widths=0.5, showmedians=True)

        for pc in parts_1['bodies']:
            pc.set_facecolor('blue')
            pc.set_edgecolor('blue')
            pc.set_alpha(0.5)

        for pc in parts_2['bodies']:
            pc.set_facecolor('red')
            pc.set_edgecolor('red')
            pc.set_alpha(0.5)

        parts_2['cmins'].set_color('red')
        parts_2['cmaxes'].set_color('red')
        parts_2['cmedians'].set_color('red')
        parts_2['cbars'].set_color('red')

        parts_1['cmins'].set_color('blue')
        parts_1['cmaxes'].set_color('blue')
        parts_1['cmedians'].set_color('blue')
        parts_1['cbars'].set_color('blue')
        plt.savefig(Halo + '_violin.png')
        plt.close(fig)
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make a run with dynamic nested sampling.')
    parser.add_argument('--Halo', type=str, help='Halo beeing simulated.')
    parser.add_argument('--start', type=int, help='idx where sampler starts from')
    args = parser.parse_args()

    bounds = [(0.4, 1.2), (0.2, 1.2)]

    if args.Halo == 'Halo1_sig30':
        data = '/home/kian/Dokumente/Uni/Programmieren/BA/GravothermalSIDM/Data/sim_data/Halo1_sig30/plots/fit/'
        data_N_body = '/home/kian/Dokumente/Uni/Programmieren/BA/GravothermalSIDM/Data/sim_data/Halo1_sig30'
        file_N_body = 'results_halo_1_sig30.txt'
        m = (2.783 * 10**9.) / (1. * 10**7.)
        Halo_name = 'Halo1_sig30'
    elif args.Halo == 'Halo2_sig80':
        data = '/home/kian/Dokumente/Uni/Programmieren/BA/GravothermalSIDM/Data/sim_data/Halo2_sig80/plots/fit/'
        data_N_body = '/home/kian/Dokumente/Uni/Programmieren/BA/GravothermalSIDM/Data/sim_data/Halo2_sig80'
        file_N_body = 'results_halo_2_sig80.txt'
        m = (3.2378433 * 10**7.) / (2. * 10**6.)
        Halo_name = 'Halo2_sig80'
    elif args.Halo == 'Halo3_sig700':
        data = '/home/kian/Dokumente/Uni/Programmieren/BA/GravothermalSIDM/Data/sim_data/Halo3_sig700/plots/fit/'
        data_N_body = '/home/kian/Dokumente/Uni/Programmieren/BA/GravothermalSIDM/Data/sim_data/Halo3_sig700'
        file_N_body = 'results_halo_3_sig700.txt'
        m = (5.7733517 * 10**7.) / (2. * 10**6.)
        Halo_name = 'Halo3_sig700'
    elif args.Halo == 'Halo1_sig80':
        data = '/home/kian/Dokumente/Uni/Programmieren/BA/GravothermalSIDM/Data/sim_data/Halo1_sig80/plots/fit/'
        data_N_body = '/home/kian/Dokumente/Uni/Programmieren/BA/GravothermalSIDM/Data/sim_data/Halo1_sig80'
        file_N_body = 'results_halo_1_sig80.txt'
        m = (2.783 * 10**9.) / (5. * 10**7.)
        Halo_name = 'Halo1_sig80'
    else:
        raise ValueError('Halo not implemented.')
    
    violin_plotting(Halo_name, data, data_N_body, file_N_body, m, bounds, log_prob, prior_trafo, args.start)