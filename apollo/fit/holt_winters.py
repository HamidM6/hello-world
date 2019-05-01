"""Holt-Winters exponential smoothing"""

from statsmodels.tsa.holtwinters import ExponentialSmoothing 

def fit_holt_winters(
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
        if training_length_in_years >= model_time_series_required_length.get('holt_winters', 0.5) and time_series_class == 'nominal':
                
                model = 'holt_winters'

                if time_grain == 'week':
                    seasonal_periods = 52
                elif time_grain == 'month':
                    seasonal_periods = 12
                
                try:                           
                    
                    holt_winters_model = ExponentialSmoothing(
                                                                endog = input_endog,
                                                                trend = 'add',
                                                                damped = True,
                                                                seasonal = 'add',
                                                                seasonal_periods = seasonal_periods
                                                              ).fit(
                                                                    optimized = True
                                                                   )

                    holt_winters_fittedvalues = holt_winters_model.fittedvalues.round().clip(lower=0)     

                    holt_winters_forecast = holt_winters_model.predict(
                                                                            start = input_length,
                                                                            end =   input_length + forecast_length - 1
                                                                        ).round().clip(lower=0)

                except Exception as e:
                    
                    if 'broadcast' in str(e):
                        try:
                            
                            holt_winters_model = ExponentialSmoothing(
                                                                        endog = input_endog,
                                                                        trend = None,
                                                                        damped = False,
                                                                        seasonal = 'add',
                                                                        seasonal_periods = seasonal_periods
                                                                      ).fit(
                                                                            optimized = True
                                                                           )
        
                            holt_winters_fittedvalues = holt_winters_model.fittedvalues     
        
                            holt_winters_forecast = holt_winters_model.predict(
                                                                                    start = input_length,
                                                                                    end =   input_length + forecast_length - 1
                                                                                )
                        except Exception as e:
                            
                            holt_winters_model = None
                            holt_winters_fittedvalues = None
                            holt_winters_forecast = None
                            error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data_name) + ' with error ' + str(e))

                    else:
                        
                        holt_winters_model = None
                        holt_winters_fittedvalues = None
                        holt_winters_forecast = None
                        error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data_name) + ' with error ' + str(e))

        else:
            
            holt_winters_model = None
            holt_winters_fittedvalues = None
            holt_winters_forecast = None
        
        
        return holt_winters_model, holt_winters_fittedvalues, holt_winters_forecast
