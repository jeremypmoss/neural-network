# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:06:20 2023

@author: JeremyMoss
"""

import sys; sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')
import quasar_functions as qf

#%% This is fine, since redshift == True:
sdss_data = qf.loaddata('sdssmags',
                        dropna = False,
                        colours = True,
                        impute_method = 'max')

#%% This fails, since redshift == False:
skymapper_data = qf.loaddata('skymapper',
                             dropna = False,
                             colours = True,
                             impute_method = 'max')
