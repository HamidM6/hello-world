"""Fast Fourier Transform"""

import numpy
import pandas

def fit_fft(
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
        model = 'fft'
    
        if training_length_in_years >= model_time_series_required_length.get(model, 0.5) and time_series_class == 'nominal':
                     
                try:                            
                        
                        if time_grain == 'week':
                            n_harmonics = 20
                        elif time_grain == 'month':
                            n_harmonics = 8
                                                                    # number of harmonics in model
                        t = numpy.arange(0, input_length)
                        linear_trend = numpy.polyfit(t, input_endog, 1)                  # find linear trend in x
                        training_endog_detrend = input_endog - linear_trend[0] * t       # detrended training signal
                        fft_model = numpy.fft.fft(training_endog_detrend)          # detrended training data in frequency domain (FT)
#                       f = fft.fftfreq(training_length)                                    # frequencies in Hertz
                        indexes = list(range(input_length))                            
                        # sort by amplitude
                        indexes.sort(
                                        key = lambda i: numpy.absolute(fft_model[i]) / input_length,
                                        reverse = True
                                    )
                        fft_terms_for_reconstruction = indexes[:1 + n_harmonics * 2]
                        ft_sample_frequencies = numpy.fft.fftfreq(
                                                                    n = input_length,
                                                                    d = 1
                                                                 ) 
                        fft_fittedvalues = reconstruct_signal(
                                                                 n_periods = input_length,
                                                                 fft_model = fft_model,
                                                                 ft_sample_frequencies = ft_sample_frequencies,
                                                                 fft_terms_for_reconstruction = fft_terms_for_reconstruction,
                                                                 linear_trend = linear_trend
                                                              )
                        fft_forecast = reconstruct_signal(
                                                                n_periods = input_length + forecast_length,
                                                                fft_model = fft_model,
                                                                ft_sample_frequencies = ft_sample_frequencies,
                                                                fft_terms_for_reconstruction = fft_terms_for_reconstruction,
                                                                linear_trend = linear_trend
                                                           )[-(forecast_length):]
                        
                        
                except Exception as e:     
                                 
                       fft_model = None
                       fft_fittedvalues = None
                       fft_forecast = None
                       error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data_name) + ' with error ' + str(e))
        
        else:
            
            fft_model = None
            fft_fittedvalues = None
            fft_forecast = None
        
        return fft_model, pandas.Series(fft_fittedvalues), pandas.Series(fft_forecast)



def reconstruct_signal(
                         n_periods,
                         fft_model,
                         ft_sample_frequencies,
                         fft_terms_for_reconstruction,
                         linear_trend
                      ):
    """

    :param n_periods:
    :param fft_model:
    :param ft_sample_frequencies:
    :param fft_terms_for_reconstruction:
    :param linear_trend:
    :return:
    """
    pi = numpy.pi
    t = numpy.arange(0, n_periods)
    restored_sig = numpy.zeros(t.size)
    for i in fft_terms_for_reconstruction:
        ampli = numpy.absolute(fft_model[i]) / n_periods   # amplitude
        phase = numpy.angle(
                             fft_model[i],
                             deg = False
                           )                       # phase in radians
        restored_sig += ampli * numpy.cos(2 * pi * ft_sample_frequencies[i] * t + phase)
    return restored_sig + linear_trend[0] * t
