"""facebook's forecasting engine"""

from fbprophet import Prophet
import pandas

def fit_prophet(
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
        
        model = 'prophet'
    
        if training_length_in_years >= model_time_series_required_length.get(model, 0.5) and time_series_class == 'nominal':
            
            daily_seasonality = False
            weekly_seasonality = False
            changepoint_range  = 0.8
        
            if time_grain == 'week':
                freq = 'W-SAT'                

            elif time_grain == 'month':
                freq = 'M'
                holidays = None
                
            try:
                
                df = pandas.DataFrame({'ds':input_dates, 'y':input_endog})
                
                prophet_model = Prophet(
                
                                             holidays=holidays, 
                                             daily_seasonality=daily_seasonality, 
                                             weekly_seasonality=weekly_seasonality,
                                             changepoint_range  = changepoint_range 
                                             
                                         ).fit(df)
                
                future = prophet_model.make_future_dataframe(periods=forecast_length, freq=freq, include_history=True)
                
                prophet_model_predictions = prophet_model.predict(future)['yhat'].clip_lower(0).round()
                
                prophet_fittedvalues = prophet_model_predictions.head(input_length)                
                
                prophet_forecast = prophet_model_predictions.tail(forecast_length)
                
            except Exception as e:
                 
                 error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data_name) + ' with error ' + str(e))
                 prophet_model = None
                 prophet_fittedvalues = None
                 prophet_forecast = None 
        else:
            
            prophet_model = None
            prophet_fittedvalues = None
            prophet_forecast = None 
            
        return prophet_model, prophet_fittedvalues, prophet_forecast

