"""
utility functions used across platform
"""

import json
import os
import logging
import math
import pickle


def combine_into_dict(x):
    """

    :param x: list
    :return: dictionary
    """
    result = dict()
    for i in x:
        result.update(i)
    
    return result


def read_params_in_from_json(filename):
    """

    :param filename:
    :return:
    """
    root = os.path.dirname(os.getcwd())
    credential_file_path = root + '\\params\\' + filename
    json_string = open(credential_file_path).read()
    params = json.loads(json_string.replace('\n', ' ').replace('\t', ' '))
    return params


def setup_logger(logger_name, log_file, level=logging.INFO):
    """

    :param logger_name:
    :param log_file:
    :param level:
    """
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_in_file(file_location):
    """

    :param file_location:
    :return:
    """
    file_content = open(
                        file = file_location,
                        mode = 'r'
                        ).read()
    
    return file_content


def wfa(abs_error, actual, forecast, epsilon):
    """

    :param abs_error:
    :param actual:
    :param forecast:
    :param epsilon:
    :return:
    """
    if sum(forecast) == 0 and sum(actual) == 0 and sum(abs_error) == 0:
        wfa = 0
    else:
        wfa = max(0, 1 - 2*sum(abs_error)/(epsilon + sum(forecast) + sum(actual)))
    
    return wfa


def pickle_object(obj, output_filename):
    """

    :param obj:
    :param output_filename:
    """
    with open(output_filename, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        

def prediction_suffix_by_fit_type(fit_type):
    """

    :param fit_type:
    :return:
    """
    if fit_type == 'complete':
        return '_forecast'
    else:
        return '_holdout'
    
    
def chunk_dict(d, number_of_chunks):
    """

    :param d:
    :param number_of_chunks:
    :return:
    """
    chunk_size = math.ceil(len(d.items())/number_of_chunks)
    n = 1
    chunk = {}
    list_of_chunks = []
    for key, value in d.items():
        chunk[key] = value
        if (n % chunk_size) == 0:            
            list_of_chunks.append(chunk)
            n = 1
            chunk = {}        
        n += 1
    
    return list_of_chunks
    
    
    
    return chunk


def get_values(x):
    """

    :param x:
    :return:
    """
    try:
        if x is not None and len(x) > 0:
            x = x.values
        else:
            x = [0]
    except:
        x = [0]
        
    return x