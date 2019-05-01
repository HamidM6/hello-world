# apollo demand forecast platform
# model hyperparameter tuning functions
# @author: vikram govindan

import pydlm
from bayes_opt import BayesianOptimization
from scipy.optimize import minimize


def optimize_hyperparameters(
                                tuning_method,
                                cost_function, 
                                parameter_bounds, 
                                iterations,
                                training_wfa,
                                n_changepoints_default,
                                initial_values = None
                                ):
    
    if tuning_method == 'bayesian':
    
        optimizer = BayesianOptimization(
                                            f = cost_function,
                                            pbounds = parameter_bounds,
                                            random_state = 1,
                                        )
        
        optimizer.maximize(
                            n_iter = iterations,
                          )
        
        optimized_params = optimizer.max
    
    
    elif tuning_method == 'nelder-mead':
        
        optimized_params = minimize(
                                        fun = cost_function,
                                        method = 'Powell',              
                                        bounds = parameter_bounds,
                                        x0 = initial_values,
                                        options = {
                                                    'maxiter': iterations,
                                                    'disp': True
                                                  }
                                    )
    
    elif tuning_method == 'grid-search':
      wfa_dict = {}
      optimized_params = {}
      
      for num_changepoints in parameter_bounds['n_changepoints']:
        for yearly_seasonality in parameter_bounds['yearly_seasonality']:
          
          wfa_dict[(num_changepoints, yearly_seasonality)] = cost_function(num_changepoints, yearly_seasonality)
        
      max_wfa = max(wfa_dict.values())
      if max_wfa > training_wfa:
        optimized_params['n_changepoints'] = max(wfa_dict, key=wfa_dict.get)[0]
        optimized_params['yearly_seasonality'] = max(wfa_dict, key=wfa_dict.get)[1]
        
      else:
        optimized_params['n_changepoints'] = n_changepoints_default
        optimized_params['yearly_seasonality'] = 'auto'
      
    return optimized_params



def tune_dlm(dlm_model, tuning_iterations = 100):
    
    dlm_tuner = pydlm.modelTuner(method='gradient_descent', loss='mse')
    tuned_dlm = dlm_tuner.tune(dlm_model, maxit = tuning_iterations)
    
    return tuned_dlm
