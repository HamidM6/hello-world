"""
utility functions used across platform to help parallilize its processes.
"""

import pandas
import dask


def get_partitioned_list(x, parallel_partition_key):
    """

    :param x:
    :param parallel_partition_key:
    :return:
    """
    pre_processed_data_set = x.groupby(parallel_partition_key)
    pre_processed_data_set_list = list(pre_processed_data_set)
    
    return pre_processed_data_set_list

def combine_into_df(x):
    """

    :param x:
    :return:
    """
    result = pandas.DataFrame()
    for i in x:
        result = result.append(i)
    
    return result

def _apply_fun(gb, fun):
    """

    :param gb:
    :param fun:
    :return:
    """
    x = gb.apply(fun)
    
    return x

def parallel_apply_fun(pre_processed_data_set_list, fun, key):
    """

    :param pre_processed_data_set_list:
    :param fun:
    :param key:
    :return:
    """
    task_list = []
    for i in range(len(pre_processed_data_set_list)):
        gb = pre_processed_data_set_list[i][1].groupby(key)
        fit_task = dask.delayed(_apply_fun)(gb, fun)
        task_list.append(fit_task)
            
    fit_task = dask.delayed(combine_into_df)(task_list)            
    x  = fit_task.compute()
    
    return x