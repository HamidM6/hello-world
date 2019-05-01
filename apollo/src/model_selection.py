"""
# apollo demand forecast platform
# best and winning model contest
# @author: vikram govindan
"""

from apollo.src import utils

def pick_winning_model(
                        fit_results_training,
                        model_list,
                        epsilon                       
                      ):
    """

    :param fit_results_training:
    :param model_list:
    :param epsilon:
    :return:
    """
    fit_results_training = _compute_constrained_abs_error(
                                                            fit_results_training,
                                                            model_list
                                                         )
    fit_results_training = _compute_constrained_wfa(
                                                        fit_results_training, 
                                                        model_list,
                                                        epsilon
                                                   )
    fit_results_training = _pick_highest_holdout_wfa(
                                                        fit_results_training,
                                                        model_list
                                                    )      
            
    return fit_results_training



def pass_winning_and_best_model(
                                 pre_processed_dict, 
                                 fit_results_training
                               ):
    """

    :param pre_processed_dict:
    :param fit_results_training:
    :return:
    """
    for fit_results_training_key, fit_results_training_value in fit_results_training.items():
            pre_processed_dict[fit_results_training_key]['winning_model'] = fit_results_training_value['winning_model']
            pre_processed_dict[fit_results_training_key]['best_model'] = fit_results_training_value['best_model']
    
    return pre_processed_dict



def _compute_constrained_wfa(
                                fit_results_training,
                                model_list,
                                epsilon
                            ):
     """

     :param fit_results_training:
     :param model_list:
     :param epsilon:
     :return:
     """
     for fit_results_training_key, fit_results_training_value in fit_results_training.items():
         r = src.utils.get_values(fit_results_training_value['forecast_response'])
         for model in (model_list + ['benchmark']):
             e = fit_results_training_value['abs_error_' + model]
             f = src.utils.get_values(fit_results_training_value[model + '_holdout'])
             fit_results_training_value[model + '_wfa'] = src.utils.wfa(
                                                                         abs_error = e,
                                                                         actual = r,
                                                                         forecast = f,
                                                                         epsilon = epsilon
                                                                      )
     return fit_results_training


def _compute_constrained_abs_error(
                                    fit_results_training,
                                    model_list
                                ):
     """

     :param fit_results_training:
     :param model_list:
     :return:
     """
     for fit_results_training_key, fit_results_training_value in fit_results_training.items():
         r = src.utils.get_values(fit_results_training_value['forecast_response'])
         for model in (model_list + ['benchmark']):
             f = src.utils.get_values(fit_results_training_value[model + '_holdout'])
             fit_results_training_value['abs_error_' + model] = abs(f - r)
            
     return fit_results_training
 
    
    
def _pick_highest_holdout_wfa(
                                fit_results_training,
                                model_list
                             ):
    """

    :param fit_results_training:
    :param model_list:
    :return:
    """
    for fit_results_training_key, fit_results_training_value in fit_results_training.items():    
    
        best_model_wfa = -1
        best_model = ''
        
        for model in model_list:
            
            model_wfa = fit_results_training_value[model + '_wfa']
            
            model_exists = _test_model_exist(
                                                fit_results_training_value[model + '_fitted_training'],
                                                fit_results_training_value[model + '_holdout']
                                            )
            
            if model_wfa > best_model_wfa and model_exists:
                
                best_model = model
                best_model_wfa = model_wfa
                fit_results_training_value['best_model'] = model
                fit_results_training_value['best_model_holdout'] = fit_results_training_value[model + '_holdout']
                fit_results_training_value['best_model_abs_error'] = fit_results_training_value['abs_error_' + model]
                fit_results_training_value['best_model_wfa'] = fit_results_training_value[model + '_wfa']    
                fit_results_training_value['best_model_fitted_training'] = fit_results_training_value[model + '_fitted_training']  
        
        if best_model == '':
            
            fit_results_training_value['best_model'] = ''
            fit_results_training_value['best_model_holdout'] = 0
            fit_results_training_value['best_model_abs_error'] = fit_results_training_value['forecast_response']
            fit_results_training_value['best_model_wfa'] = 0
            fit_results_training_value['model_better'] = 'No model'
            winning_model = 'benchmark'
        
        else:        
           
            if fit_results_training_value['best_model_wfa'] >= fit_results_training_value['benchmark_wfa'] and sum(fit_results_training_value['best_model_abs_error']) < sum(fit_results_training_value['abs_error_benchmark']):            
                winning_model = best_model
                fit_results_training_value['model_better'] = 'Y'
            else:
                winning_model = 'benchmark'
                fit_results_training_value['model_better'] = 'N'
            
        fit_results_training_value['winning_model'] = winning_model
        fit_results_training_value['winning_model_holdout'] = fit_results_training_value[winning_model + '_holdout']
        fit_results_training_value['winning_model_abs_error'] = fit_results_training_value['abs_error_' + winning_model]
        fit_results_training_value['winning_model_wfa'] = fit_results_training_value[winning_model + '_wfa']                
        fit_results_training_value['winning_model_fitted_training'] = fit_results_training_value[winning_model + '_fitted_training']
        
        for model in model_list:
            
            if model != 'ensemble':
                del fit_results_training_value[model + '_training_modelobj']
            del fit_results_training_value[model + '_holdout']
            del fit_results_training_value[model + '_fitted_training']
            del fit_results_training_value['abs_error_' + model] 
            del fit_results_training_value[model + '_wfa']
        
        del fit_results_training_value['training_response']
        del fit_results_training_value['forecast_response']
        
    return fit_results_training

def _test_model_exist(
                        training_fit,
                        holdout_fit
                     ):
    """

    :param training_fit:
    :param holdout_fit:
    :return:
    """
    model_exists = True
    
    if training_fit is None or holdout_fit is None:
        model_exists = False
        
    if model_exists == True:
        if type(training_fit) == float:
            if len([training_fit]) == 0:
                model_exists = False
        elif len(training_fit) == 0 :
            model_exists = False
        if model_exists == True:
            if type(holdout_fit) == float:
                if len([holdout_fit]) == 0:
                    model_exists = False
            elif len(holdout_fit) == 0:
                model_exists = False
    
    return model_exists