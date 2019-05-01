# apollo demand forecast platform
# model orchestration
# @author: vikram govindan

import pandas 

def fit_moving_average(
                        data,
                        model_time_series_required_length,
                        training_endog,
                        training_dates,
                        training_length,
                        holdout_end,
                        complete_endog,
                        complete_dates,
                        complete_length,
                        complete_end
                        error_logger
                      ):
    
        
        model = 'moving_average'
    
        if data['training_length_in_years'].sample(1).values.item(0) > model_time_series_required_length.get(model, 0.5):
                
                        try:
                            
                            moving_average_training_model = training_endog.rolling(window=3, min_periods=1)
                            
                            moving_average_training_fittedvalues = moving_average_training_model.mean().round()    
                            moving_average_holdout_prediction = pandas.Series([moving_average_training_fittedvalues.values[-1]] * (holdout_end - training_length + 1))
                            
                            moving_average_forecast_model = complete_endog.rolling(window=3, min_periods=1)
                            moving_average_forecast = moving_average_forecast_model.mean().round()
                            moving_average_forecast = pandas.Series([moving_average_forecast.values[-1]] * (complete_end - complete_length + 1))
                    
                        except Exception as e:
                            moving_average_training_model = None
                            moving_average_training_fittedvalues = None
                            moving_average_holdout_prediction = None
                            moving_average_forecast_model = None
                            moving_average_forecast = None    
                            error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data.name) + ' with error ' + str(e))
                
        else:
            
            moving_average_training_model = None
            moving_average_training_fittedvalues = None
            moving_average_holdout_prediction = None
            moving_average_forecast_model = None
            moving_average_forecast = None
        
        
        
        return moving_average_training_model, moving_average_training_fittedvalues, moving_average_holdout_prediction, moving_average_forecast_model, moving_average_forecast
