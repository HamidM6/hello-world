"""ensemble of high performance models"""

import apollo.src.utils

def fit_ensemble(
                     fit_results,
                     fit_type,
                     ensemble_model_list
                ):
    """

    :param fit_results:
    :param fit_type:
    :param ensemble_model_list:
    :return:
    """
    for fit_results_key, fit_results_value in fit_results.items():
        
        ensemble_exists = True
        
        # if ensemble didn't win in training, simply exit
        if fit_type == 'complete' and fit_results_value['best_model'] != 'ensemble':
            ensemble_exists = False
        
        if ensemble_exists == True:
        
            ensemble_forecast = 0 
            suffix = src.utils.prediction_suffix_by_fit_type(fit_type)
            ensemble_fitted = 0
    
            for model in ensemble_model_list:  
                
                    try:
        
                        ensemble_forecast = ensemble_forecast + fit_results_value[model + suffix].reset_index(drop = True)
                        ensemble_fitted = ensemble_fitted + fit_results_value[model + '_fitted_' + fit_type].reset_index(drop = True)
                        
                    except:
                        
                        if fit_type == 'complete':
                            if fit_results_value['winning_model'] == 'ensemble':
                                fit_results_value['winning_model_fitted_complete'] =  None 
                                fit_results_value['winning_model_forecast'] = None
                                fit_results_value['best_model_fitted_complete'] =  None 
                                fit_results_value['best_model_forecast'] = None
                            else:
                                fit_results_value['best_model_fitted_complete'] =  None 
                                fit_results_value['best_model_forecast'] = None
                        else:
                            fit_results_value['ensemble' + suffix] = None
                            fit_results_value['ensemble_fitted_' + fit_type] = None
                    
            try:
                
                fcst = ensemble_forecast / len(ensemble_model_list)
                fit = ensemble_fitted / len(ensemble_model_list)
                
                if fit_type == 'complete':
                    if fit_results_value['winning_model'] == 'ensemble':
                        fit_results_value['winning_model_forecast'] = fcst
                        fit_results_value['winning_model_fitted_complete'] = fit
                        fit_results_value['best_model_forecast'] = fcst
                        fit_results_value['best_model_fitted_complete'] = fit
                    else:
                        fit_results_value['best_model_forecast'] = fcst
                        fit_results_value['best_model_fitted_complete'] = fit
                else:
                    fit_results_value['ensemble' + suffix] = fcst
                    fit_results_value['ensemble_fitted_' + fit_type] = fit
                
            except:
                
                if fit_type == 'complete':
                    if fit_results_value['winning_model'] == 'ensemble':
                        fit_results_value['winning_model_fitted_complete'] =  None 
                        fit_results_value['winning_model_forecast'] = None
                        fit_results_value['best_model_fitted_complete'] =  None 
                        fit_results_value['best_model_forecast'] = None
                    else:
                        fit_results_value['best_model_fitted_complete'] =  None 
                        fit_results_value['best_model_forecast'] = None
                else:
                    fit_results_value['ensemble' + suffix] = None
                    fit_results_value['ensemble_fitted_' + fit_type] = None
                    
        if fit_type == 'complete':
            del fit_results_value['training_response']
            del fit_results_value['forecast_response']
            del fit_results_value['best_model']
            del fit_results_value['winning_model']                
                
    return fit_results