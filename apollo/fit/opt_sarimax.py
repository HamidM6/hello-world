# apollo demand forecast platform
# model orchestration
# @author: vikram govindan

import pandas
import pmdarima as pm

def fit_opt_sarimax(
                    data,
                    model_time_series_required_length,
                    training_exog_var,
                    training_endog,
                    training_dates,
                    training_length,
                    holdout_exog_var,
                    holdout_end,
                    complete_exog_var,
                    complete_endog,
                    complete_dates,
                    complete_length,
                    complete_end,
                    forecast_exog_var,
                    error_logger,
                    time_grain,
                    training_length_in_years,
                    time_series_class
               ):
    
        
        model = 'opt_sarimax'
    
        if training_length_in_years >= model_time_series_required_length.get('opt_sarimax', 0.5) and time_series_class == 'nominal':
                        

                    if time_grain == 'week':
                        seasonal_order = 52
                    elif time_grain == 'month':
                        seasonal_order = 12
                    
                    try:                            
                        
                        if time_grain == 'month':
                            
                            opt_sarimax_training_model = pm.auto_arima(
                                                                   training_endog.values,
                                                                   start_p=0, start_q=0,
                                                                   max_p=3, max_q=3, m=seasonal_order,
                                                                   start_P=0, seasonal=True,
                                                                   d=0, D=1, trace=True,
                                                                   error_action='ignore',  # don't want to know if an order does not work
                                                                   suppress_warnings=True,  # don't want convergence warnings
                                                                   stepwise=True
                                                                   )
                            opt_sarimax_training_fittedvalues = opt_sarimax_training_model.predict_in_sample()   

                            opt_sarimax_holdout_prediction = opt_sarimax_training_model.predict(holdout_end - training_length + 1)
                            
                            opt_sarimax_forecast_model = pm.auto_arima(
                                                                   complete_endog.values,
                                                                   start_p=0, start_q=0,
                                                                   max_p=3, max_q=3, m=seasonal_order,
                                                                   start_P=0, seasonal=True,
                                                                   d=0, D=1, trace=True,
                                                                   error_action='ignore',  # don't want to know if an order does not work
                                                                   suppress_warnings=True,  # don't want convergence warnings
                                                                   stepwise=True
                                                                   )
                            opt_sarimax_forecast = opt_sarimax_forecast_model.predict(complete_end - complete_length + 1)

                        elif time_grain == 'week':
                            
                            opt_sarimax_training_model = pm.auto_arima(
                                                                   training_endog.values,
                                                                   start_p=0, start_q=0,
                                                                   max_p=3, max_q=3, m=seasonal_order,
                                                                   start_P=0, seasonal=True,
                                                                   d=1, D=1, trace=True,
                                                                   error_action='ignore',  # don't want to know if an order does not work
                                                                   suppress_warnings=True,  # don't want convergence warnings
                                                                   stepwise=True
                                                                   )
                            
                            
                            opt_sarimax_training_fittedvalues = opt_sarimax_training_model.predict_in_sample(training_length)   
                            
                            opt_sarimax_holdout_prediction = opt_sarimax_training_model.predict(holdout_end - training_length + 1)

                            opt_sarimax_forecast_model = pm.auto_arima(
                                                                   complete_endog.values,
                                                                   start_p=0, start_q=0,
                                                                   max_p=3, max_q=3, m=seasonal_order,
                                                                   start_P=0, seasonal=True,
                                                                   d=1, D=1, trace=True,
                                                                   error_action='ignore',  # don't want to know if an order does not work
                                                                   suppress_warnings=True,  # don't want convergence warnings
                                                                   stepwise=True
                                                                   )
                            opt_sarimax_forecast = opt_sarimax_forecast_model.predict(complete_end - complete_length + 1)
 
                                    
                        else:
                            opt_sarimax_training_model = None
                            opt_sarimax_training_fittedvalues = None
                            opt_sarimax_holdout_prediction = None
                            opt_sarimax_forecast_model = None
                            opt_sarimax_forecast = None

                    except Exception as e:
                       error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data.name) + ' with error ' + str(e))
                       opt_sarimax_training_model = None
                       opt_sarimax_training_fittedvalues = None
                       opt_sarimax_holdout_prediction = None
                       opt_sarimax_forecast_model = None
                       opt_sarimax_forecast = None
        else:
            
            opt_sarimax_training_model = None
            opt_sarimax_training_fittedvalues = None
            opt_sarimax_holdout_prediction = None
            opt_sarimax_forecast_model = None
            opt_sarimax_forecast = None
        
        return opt_sarimax_training_model, pandas.Series(opt_sarimax_training_fittedvalues), pandas.Series(opt_sarimax_holdout_prediction), opt_sarimax_forecast_model, pandas.Series(opt_sarimax_forecast)
        
        
