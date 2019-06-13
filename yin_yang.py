import math 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

"""
Creates a dataset (data, labels) of required size, number of targets, and complexity*.

yin_yang_datagen():

attrs:

(1) n : The number of data points required to be generated
(2) num_target_classes : int; value between 2 and 3 (both included); default set to 2
(3) balanced : True; available only for binary target datasets*
(4) size_ratio : float between 0 and 1; controls the ratio of radius of circles TopSm to TopBig and/or BotSm to BotBig respectively. This controls complexity of the dataset.


* When working with small datasets (n < 4000), a size_ratio greater than 0.1 is recommended.

methods:

yin_yang_datagen() : 
Returns data : an (m x 2) matrix, where m is the desired number of training examples.
        labels : am (m x 1) array, where each instance corresponds to a row in the "data" matrix. 

"""


"""
parameters:

The parameters of the data generating mechanism are the radii and centre coordinates of the following 5 circles.

Main(x,y)    -> Coordinates of Main circle
TopSm(x,y)   -> Coordinates of TopSmall circle
TopBig(x,y)  -> Coordinates of TopBig circle
BotSm(x,y)   -> Coordinates of Bottom Small circle 
BotBig(x,y)  -> Coordinates of Bottom Big circle

"""

"""
yin_yang_generator utils:

Functions for use in labeling.
"""


def circle_contains(data, center, radius):
    """
    Accepts an m x 2 dimensional array; checks whether each point/row (1 x 2) coordinate vector is contained in a circle:
    center -> Tuple of form (x, y) denoting center of circle
    radius -> Positive number; radius of circle in question
    """
    x, y = center
    ed = np.sqrt((data[:,0] - x) ** 2 + (data[:,1] - y) ** 2)

    return (ed < radius).astype(int)


def params_dict_gen(size_ratio):
    """
    Controls the complexity of the dataset.
    """

    main_rad = 8
    # This is an arbitrary choice of main_rad; in theory any value may be assigned.
    big_rad = main_rad/2
    small_rad = size_ratio * big_rad

    params_dict = {'main' : ((0, 0), main_rad),
                   'topsm' : ((0, big_rad), small_rad),
                   'topbig' : ((0, big_rad), big_rad),
                   'botsm' : ((0, -big_rad), small_rad),
                   'botbig' : ((0, -big_rad), big_rad)} 


    return params_dict


def euc_dist_checker(data, params_dict):
    """
    Returns a dictionary of boolean values; testing for presence of point in the circles parameterized by params_dict.

    params_dict -> Dictionary of inner-circle parameters:
    params_dict = {'main' : [(x, y), radius],
                   'topsm' : [(x, y), radius],
                   'topbig' : [(x, y), radius],
                   'botsm' : [(x, y), radius],
                   'botbig' : [(x, y), radius]}  
  
    
    """
    cc_dict = {}
    for circle in params_dict.keys():
        c, r = params_dict[circle][0], params_dict[circle][1]
        cc_dict[circle] = circle_contains(data, c, r)

    return cc_dict



def quad_checker(data, params_dict):
    """
    Checks for the relative "quadrant" of a point of interest.

    Scaled to center;:
    (+, +) -> 3
    (+, -) -> -1
    (-, +) -> 1
    (-, -) -> -3  
    """
    main_center = params_dict['main'][0]
    
    test1 = np.sign(data - main_center).sum(axis = 1)
    test2 = np.sign((data - main_center)[:,1])
    return (test1 + test2).astype(int)


    #def data_label_matrix(data, params_dict):    
    #    circ_cont = pd.DataFrame(euc_dist_checker(data, params_dict))    
    #    quad_check = pd.DataFrame(quad_checker(data, params_dict))
    #    return pd.merge(circ_cont, quad_check, left_index = True, right_index = True).\
    #            rename(columns = {0 : 'quad'})



def yin_yang_datagen(n = 1000, random_seed = 19,  size_ratio = 0.25, num_target_classes = 2, balanced = 1):

    np.random.seed(seed = random_seed)

    # Exception Handling:

    if 0 < size_ratio <= 1:
        pass
    else:
        raise Exception('size_ratio must lie between 0 and 1!')
    
    # NOTE: Future iterations will be modified to allow multiclass labeling (4 to 16 classes)
    if num_target_classes in np.arange(2, 4, 1):
        pass 
    else:
        raise Exception('The number of target classes must lie between 2 and 16!') 

    if balanced in [0, False, 1, True]:
        pass
    else:
        raise Exception('balanced is a Boolean (0/1) input!')

    params_dict = params_dict_gen(size_ratio = size_ratio) 

    center, scale = params_dict['main'][0], params_dict['main'][1]

    if num_target_classes in [2, 3]:

        safety_cushion = 1.5
        # safety_cushion is the amount by which we will scale the requested dataset size. This is because we can whittle away at this larger dataset to get to two classes (if num_target_classes = 2)

        data = np.random.uniform(np.array(center) - (scale, scale), 
                                 np.array(center) + (scale, scale),
                                 size = (int(n * safety_cushion), 2))

        cc_dict = pd.DataFrame(euc_dist_checker(data, params_dict))
        quad_check = pd.DataFrame(quad_checker(data, params_dict))
        
        df_dat = pd.merge(quad_check, cc_dict, left_index = True, right_index = True).\
                rename(columns = {0 : 'quad'})
        
        df_dat['cc_str'] = df_dat[['quad', 'main', 'topsm', 'topbig', 'botsm', 'botbig']].apply(lambda x : x.astype(str).sum(), axis = 1)

        
        ## Label Dictionary
        pos_vals = ['311100', '310000', '-110001', '-110000', '-310001', '111100']
        neg_vals = ['310100', '-110011', '-310011', '-310000', '110100', '110000']
        neu_vals = ['-300000', '-100000', '100000', '300000']

        def class_mapper(x):
            if x in pos_vals:
                return 1
            elif x in neg_vals:
                return -1
            elif x in neu_vals:
                return 0
        

        if (num_target_classes == 2) & (balanced in [1, True]):

            df_dat['Class'] = df_dat['cc_str'].apply(lambda x: class_mapper(x))
            dataset = pd.DataFrame(data = data, index = df_dat['Class'].values)

            dataset = dataset[dataset.index.isin([-1, 1])].sample(n)
            return dataset.values, dataset.index

        if (num_target_classes == 2) & (balanced in [0, False]):
            pos_vals = ['311100', '111100']
            neg_vals = ['310100', '-310000', '110100', '110000', '310000', '-110001', '-110000', '-310001', '-110011', '-310011' ]
            neu_vals = ['-300000', '-100000', '100000', '300000']

            df_dat['Class'] = df_dat['cc_str'].apply(lambda x: class_mapper(x))
            dataset = pd.DataFrame(data = data, index = df_dat['Class'].values)

            dataset = dataset[dataset.index.isin([-1, 1])].sample(n)
            return dataset.values, dataset.index


        if num_target_classes == 3:
            df_dat['Class'] = df_dat['cc_str'].apply(lambda x: class_mapper(x))
            dataset = pd.DataFrame(data = data, index = df_dat['Class'].values)
            
            dataset = dataset.sample(n)
            return dataset.values, dataset.index

    else:
        "We'll get there!"               

