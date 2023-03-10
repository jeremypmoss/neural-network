# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:17:25 2023

@author: JeremyMoss
"""

import sys; sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')
import matplotlib.pyplot as plt
import quasar_functions as qf
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import cm

#%% Load training/validation dataset
sdss16 = qf.loaddata('sdss16qso',
                    dropna = False,
                    colours = False,
                    impute_method = None)

steves = qf.loaddata('sdssmags',
                   dropna = False,
                   colours = False,
                   impute_method = None)

skymap = qf.loaddata('skymapper_wise',
                    dropna = False,
                    colours = False,
                    impute_method = None)

galex = qf.loaddata('galexqso',
                    dropna = False,
                    colours = False,
                    impute_method = None)

oldfit = qf.loaddata('old_fitted',
                    dropna = False,
                    colours = False,
                    impute_method = None)

xlsoptn = qf.loaddata('xlsoptn',
                    dropna = False,
                    colours = False,
                    impute_method = None)

sdss12 = qf.loaddata('sdss12_petrosian',
                    dropna = False,
                    colours = False,
                    impute_method = None)

def plot_annotations(dataset, band):
    label = r'''{0}
    n = {1}
    $\mu={2:.2f}$
    $\sigma={3:.2f}$'''.format(dataset[1],
                            sum(~np.isnan(dataset[0][band])),
                            dataset[0][band].mean(),
                            dataset[0][band].std())

    return label
#%% Plot the distributions

plot_params = {'density': False,
                 'histtype': 'step',
                 'bins': 200,
                 'linewidth': 2,
                 'alpha': 1}

tick_params = {'rotation': 0,
                 'fontsize': 'small'}

scale = 'linear'
figsize = (18, 10)
data_list = [galex, skymap]
colour_list = cm.hsv(np.linspace(0, 0.8, len(data_list)))
bands = ['u'
          , 'g', 'r', 'i', 'zmag', 'W1', 'W2'
         ]

for band in bands:
    fig, ax = plt.subplots(figsize = figsize)
    text_y = 0.7
    for dataset, colour in zip(data_list,
                                       colour_list,
                                       # np.linspace(0.3, 2/len(data_list), len(data_list)) # for even spacing of text
                                       ):
        ax.hist(dataset[0][band], **plot_params, color = colour)
        print('Histogram for {0} {1}'.format(band, dataset[1]))
        ax.text(0.15, text_y, plot_annotations(dataset, band),
                            transform = fig.transFigure,
                            color = colour, size = 'x-small')
        text_y = text_y - 0.1
    ax.set_yscale(scale)
    ax.grid(which = 'both', axis = 'both')
    ax.set_xticklabels(ax.get_xticks(), **tick_params)
    ax.set_yticklabels(ax.get_yticks(), **tick_params)
    ax.set_xlabel('{0} [mag]'.format(band))
    ax.set_ylabel('Count')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('../../Toy paper/plots/mag_dist_{0}'.format(band))
