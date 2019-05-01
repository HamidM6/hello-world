"""
code performance functions
"""
import numpy
import matplotlib.pyplot as plt 

def _runtime_stats(
                    fit_results,
                    model_list,
                    runtime_stats_info_logger,
                    root
                    ):
    """

    :param fit_results:
    :param model_list:
    :param runtime_stats_info_logger:
    :param root:
    """
    graph_root = root + '\\log\\graphs\\'    
    #n_keys = len(fit_results)
    
    for model in model_list:
        
        models_runtimes_training = {}
        models_runtimes_training[model] = []
        #num_of_preds = 0
        #percent_prediction = 0
        
        for fit_results_key in fit_results:
    
            models_runtimes_training[model].append(fit_results[fit_results_key][model+'_runtime_training'])

        models_runtimes_training = [i for i in models_runtimes_training[model] if i > 0]
        
        if len(models_runtimes_training) > 0:
            
            plt.figure(figsize=(10,5))
            plt.plot(models_runtimes_training)
            plt.title(model)
            plt.savefig(graph_root + model + '_training.pdf')
            
            mean_runtime_training = str(numpy.round(numpy.mean(models_runtimes_training), 2))
            median_runtime_training = str(numpy.round(numpy.median(models_runtimes_training), 2))
            max_runtime_training = str(numpy.round(numpy.max(models_runtimes_training), 2))
            #percent_prediction = str(numpy.round(num_of_preds / n_keys, 2))
            runtime_stats_info_logger.info('average training run time for ' + model + ': ' + mean_runtime_training) 
            runtime_stats_info_logger.info('median training run time for ' + model + ': ' + median_runtime_training)    
            runtime_stats_info_logger.info('max training run time for ' + model + ': ' + max_runtime_training)  
            #runtime_stats_info_logger.info('% combinations run through ' + model + ': ' + percent_prediction) 
