"""Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model"""

import statsmodels.api

def fit_sarimax(
                    data_name,
                    model_time_series_required_length,
                    input_endog,
                    input_dates,
                    input_length,
                    forecast_length,
                    time_grain,
                    input_endog_shifted,
                    forecast_shifted_response,
                    error_logger,
                    training_length_in_years,
                    time_series_class,
                    holidays,
                    training_exog_var,
                    forecast_exog_var
                    ):
        """

        :param data_name:
        :param model_time_series_required_length:
        :param input_endog:
        :param input_dates:
        :param input_length:
        :param forecast_length:
        :param time_grain:
        :param input_endog_shifted:
        :param forecast_shifted_response:
        :param error_logger:
        :param training_length_in_years:
        :param time_series_class:
        :param holidays:
        :param training_exog_var:
        :param forecast_exog_var:
        :return:
        """
        if training_length_in_years >= model_time_series_required_length.get('sarimax', 0.5) and time_series_class == 'nominal':
                        
                    model = 'sarimax'

                    if time_grain == 'week':
                        seasonal_order = 52
                    elif time_grain == 'month':
                        seasonal_order = 12
                    
                    try:                            
                        
                        if time_grain == 'month':
                            
                            sarimax_model = statsmodels.api.tsa.statespace.SARIMAX(
                                                                                        endog = input_endog,
                                                                                        #exog = training_exog_var,
                                                                                        order = (1, 1, 1),
                                                                                        seasonal_order = (0,1,0,seasonal_order),
                                                                                        trend = 'n'
                                                                                   ).fit()
                        elif time_grain == 'week':
                            
                            sarimax_model = statsmodels.api.tsa.statespace.SARIMAX(
                                                                                        endog = input_endog,
                                                                                        exog = training_exog_var,
                                                                                        order = (0 , 0, 1),
                                                                                        seasonal_order = (0,0,0,seasonal_order),
                                                                                        trend = 'c'
                                                                                   ).fit()
                            
                            
                        sarimax_fittedvalues = sarimax_model.fittedvalues   
                        
                        if time_grain == 'month':
                            
                            sarimax_forecast = sarimax_model.predict(
                                                                        start = input_length,
                                                                        end =   input_length + forecast_length - 1,
                                                                        #exog = forecast_exog_var
                                                                     )
                        elif time_grain == 'week':
                            
                            sarimax_forecast = sarimax_model.predict(
                                                                        start = input_length,
                                                                        end =   input_length + forecast_length - 1,
                                                                        exog = forecast_exog_var
                                                                     )
                        
                    except Exception as e:
                        
                        if 'enforce_invertibility' in str(e):
                            
                            error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data_name) + ' with error ' + str(e))
                            
                            try:
                            
                                if time_grain == 'month':
                                
                                    sarimax_model = statsmodels.api.tsa.statespace.SARIMAX(
                                                                                                endog = input_endog,
                                                                                                #exog = training_exog_var,
                                                                                                seasonal_order=(0,1,0,seasonal_order),
                                                                                                trend = 'n'
                                                                                           ).fit()
                                
                                elif time_grain == 'week':
                                    
                                    sarimax_model = statsmodels.api.tsa.statespace.SARIMAX(
                                                                                                endog = input_endog,
                                                                                                exog = training_exog_var,
                                                                                                seasonal_order=(0,0,0,seasonal_order),
                                                                                                trend = 'c'
                                                                                           ).fit()
                                
                            
                                sarimax_fittedvalues = sarimax_model.fittedvalues
                                
                                if time_grain == 'month':
                                
                                    sarimax_forecast = sarimax_model.predict(
                                                                                start = input_length,
                                                                                end =   input_length + forecast_length - 1,
                                                                                #exog = forecast_exog_var
                                                                            )
                                
                                elif time_grain == 'week':
                                    
                                    sarimax_forecast = sarimax_model.predict(
                                                                                start = input_length,
                                                                                end =   input_length + forecast_length - 1,
                                                                                exog = forecast_exog_var
                                                                            )
                                
                            except Exception as e:       
                                 
                               sarimax_model = None
                               sarimax_fittedvalues = None
                               sarimax_forecast = None
                               error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data_name) + ' with error ' + str(e))
                                    
                        else:
                            sarimax_model = None
                            sarimax_fittedvalues = None
                            sarimax_forecast = None
                            error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data_name) + ' with error ' + str(e))  
        
        else:
            
            sarimax_model = None
            sarimax_fittedvalues = None
            sarimax_forecast = None
        
        return sarimax_model, sarimax_fittedvalues, sarimax_forecast