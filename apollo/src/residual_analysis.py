"""
residual analysis
"""
from statsmodels.stats.diagnostic import acorr_ljungbox

# returns 1 if p-value of all lags are significant. needs more investigation!
def _compute_ljung_box(x, model):
    """

    :param x:
    :param model:
    :return:
    """
    training_residuals = x[x.data_split == 'Training']['abs_error_' + model].values
    
    ljung_box_test = acorr_ljungbox(x=training_residuals, lags=None, boxpierce=False)
    
    x['ljung_box_' + model] = 1 if all(ljung_box_test[1] > 0.05) else 0

    return x  