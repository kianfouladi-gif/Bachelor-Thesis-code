import numpy as np
from functools import partial
from scipy.optimize import minimize
from scipy.special import gammaln
import parse_numerical_particles as pnp

def log_prior(params, bounds):
    '''
    logarithm of the prior
    Parameters
    ------------
    params: array
                Array containing parameters C in the first argument and alpha in the second.

    bounds: array
                Array containing the bounds for each parameter, C in the first argument and alpha in the second
    Returns
    ------------
    float
                0 if within the bounds - infinity if without
    '''
    C, alpha = params
    C_lower, C_upper = bounds[0]
    alpha_lower, alpha_upper = bounds[1]
    if C_lower <= C <= C_upper and alpha_lower <= alpha <= alpha_upper:
        return 0.0
    return -np.inf

def neg_log_prob(params, M_interp, m, N_body_data, radius, bounds):
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
    Returns
    ------------
    float
                negative logarithmn of the likelihood at the parameters given in params
    '''
    N = pnp.N_body_particles_enclosed(N_body_data, radius)
    Lambda = pnp.sim_Number_particles_enclosed(params, M_interp, m)
    return -np.sum(N * np.log(Lambda) - Lambda - gammaln(N+1)) + log_prior(params, bounds)

def find_local_maximum(neg_log_prob, boundaries, args, IC):
    '''
    Finds the local best fit with the method of fitting the number of enclosed (expected) numerical particles.
    ------------
    neg_log_prob: function 
                function calculating the negative of the liklihood as function of the parameters alpha and C; further arguments are given with args

    boundaries: array
                Array having the boundaries for the minimization.
                
    args: array
                Addiational arguments for neg_log_prob; to be properly defined in find_local_maxima

    IC: array
                Initial conditions in parameterspace for the local minimizer. The first initial condition is for C and the second for alpha

    Returns
    ------------
    result: result.x, result.success, result.fun
                the resut of the minimization, whethere the minimizer found a local extremum, the function value at the minimum 
    '''
    M_interp, m, N_body_data, radius = args
    neg_log_prob = partial(neg_log_prob, M_interp=M_interp, m=m, N_body_data=N_body_data, radius=radius, bounds=boundaries)
    result = minimize(neg_log_prob, x0=IC, bounds=boundaries)
    return result.x, result.success, result.fun

def find_local_maxima(dir, file ,neg_log_prob, boundaries, m, IC, RBF_kernel, epsilon = 1.):
    '''
    Finds the local best fit with the method of fitting the number of enclosed (expected) numerical particles for each radius part of the N-body snapshots.
    Parameters
    ------------
    neg_log_prob: function 
                function calculating the negative of the liklihood as function of the parameters alpha and C; further arguments are given with args

    boundaries: array
                Array having the boundaries for the minimization.
                
    m: float
                The mass of the numerical particles.

    IC: array
                Initial conditions in parameterspace for the local minimizer. The first initial condition is for C and the second for alpha

    RBF_kernel: string
                Name of the RBF kernel used, only the in scipy implemented RBF kernels are valid arguments

    epsilon: float (default: 1.)
                Shape parameter for the RBF-kernel; check scipy documentary for further information
    Returns
    ------------
    result: results_params, results_succ, results_fun
                array of the resut of the minimization, list of whethere the minimizer found a local extremum, list of the function value at the minimum 
    '''
    N_body_data = pnp.parse_input_N_body(dir, file)
    results = []
    for radius in N_body_data['ProfileEdges']:
        M_interp = pnp.sim_Number_particles_enclosed_prep2(dir, dir, file, radius, RBF_kernel, epsilon=epsilon)
        args = [M_interp, m, N_body_data, radius]
        results.append(find_local_maximum(neg_log_prob, boundaries, args, IC))
    results_params = np.array([res[0] for res in results])
    results_succ = [res[1] for res in results]
    results_fun = [res[2] for res in results]
    return results_params, results_succ, results_fun