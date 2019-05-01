# apollo demand forecast platform
# model orchestration
# @author: vikram govindan

import statsmodels.api 

def fit_exponential_smoothing(
                                data,
                                model_time_series_required_length,
                                training_endog,
                                training_dates,
                                training_length,
                                holdout_end,
                                complete_endog,
                                complete_dates,
                                complete_length,
                                complete_end,
                                error_logger
                              ):
    
        
        model = 'exponential_smoothing'
        
        if data['training_length_in_years'].sample(1).values.item(0) < model_time_series_required_length.get(model, 0.5):
        
                try:                           
                    
                    exp_smoothing_training_model = statsmodels.api.tsa.SimpleExpSmoothing(
                                                                                            endog = training_endog
                                                                                         ).fit(optimized = True)
                    exp_smoothing_training_fittedvalues = exp_smoothing_training_model.fittedvalues     
                    exp_smoothing_holdout_prediction = exp_smoothing_training_model.predict(
                                                                                                start = training_length,
                                                                                                end =   holdout_end
                                                                                            )
                    
                    exp_smoothing_forecast_model = statsmodels.api.tsa.SimpleExpSmoothing(
                                                                                            endog = complete_endog
                                                                                         ).fit(optimized = True)
                    exp_smoothing_forecast = exp_smoothing_forecast_model.predict(
                                                                                    start = complete_length,
                                                                                    end =   complete_end
                                                                                )
                except Exception as e:
                    
                    exp_smoothing_training_model = None
                    exp_smoothing_training_fittedvalues = None
                    exp_smoothing_holdout_prediction = None
                    exp_smoothing_forecast_model = None
                    exp_smoothing_forecast = None
                    error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data.name) + ' with error ' + str(e))
        
        else:
            
            exp_smoothing_training_model = None
            exp_smoothing_training_fittedvalues = None
            exp_smoothing_holdout_prediction = None
            exp_smoothing_forecast_model = None
            exp_smoothing_forecast = None
        
        
        return exp_smoothing_training_model, exp_smoothing_training_fittedvalues, exp_smoothing_holdout_prediction, exp_smoothing_forecast_model, exp_smoothing_forecast