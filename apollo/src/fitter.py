from apollo.src.fit.holt_winters import fit_holt_winters
from apollo.src.fit.mean import fit_mean
from apollo.src.fit.zero import fit_zero
from apollo.src.fit.sarimax import fit_sarimax
from apollo.src.fit.opt_sarimax import fit_opt_sarimax
from apollo.src.fit.ewm import fit_ewm
from apollo.src.fit.prophet import fit_prophet
from apollo.src.fit.fft import fit_fft
#from src.fit.fit_bdlm import fit_bdlm
from apollo.src.fit.croston import fit_croston
from apollo.src.fit.last_year import fit_last_year

class fitter:    
    
    FIT_MODELS = {
                    'holt_winters':     fit_holt_winters,
                    'mean_model':       fit_mean,
                    'zero_model':       fit_zero,
                    'sarimax':          fit_sarimax,
                    'opt_sarimax':      fit_opt_sarimax,
                    'ewm_model':        fit_ewm,
                    'prophet':          fit_prophet,
                    'fft':              fit_fft,
                    'croston':          fit_croston,
                    'last_year':        fit_last_year
                 }
    
    
    def __init__(self, model):
         self.model = model
         
    
    def fit(
                                self,
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
        
        
        fit_model = self.FIT_MODELS[self.model]
        
        return fit_model (
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
                          )