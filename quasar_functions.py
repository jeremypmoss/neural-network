'''
A collection of custom functions written for my MSc and PhD work.
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm
# import statsmodels.formula.api as smf
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import spearmanr, chisquare
import os
from tensorflow import keras
import tensorflow as tf
# import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
# import tensorflow_docs.modeling
from tensorflow.keras import layers
from scipy.stats import gaussian_kde, kurtosis, kurtosistest
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
import missingno as msno # https://github.com/ResidentMario/missingno
from sklearn.impute import SimpleImputer
from astropy.coordinates import SkyCoord

# from imutils import path
font = {'weight' : 'normal',
        'size'   : 15}

plt.rc('font', **font)
###############################################################################
def loaddata(name, colours = False, impute_method = None, cols = None,
             dropna = True, number_of_rows = 'all'):
    '''Load data from a text file on disk.

    Parameters
    ----------
    name: The name of the dataset (eg: galex, sdss12, vimos, ugriz, sdss, mgii,
                             largess).

    colours: Whether to compute the colours of the magnitudes selected.
            Default: False. If the features selected are not magnitudes
            (e.g. spectral index or turnover frequency, this should be False).

    impute_max: Whether to impute missing values with the maximum from the
                column (boolean; default = False.

    cols: The columns to select from the file; if None, read all columns.
    (default = None).

    dropna: Whether to drop NaNs (boolean; default = True).

    Returns
    -------
    df; pandas.dataframe containing data, columns read from `cols`.

    datasetname; a string used for naming plots or files.

    magnames; a list of the magnitude names used in the survey.

    mf; a dataframe consisting of the data in the mags columns of the dataset
    '''
    path = r'../../data_files'
#%% Skymapper
    if name.lower() == 'skymapper': # 999999
        redshift = False
        # fields = [_RAJ2000,_DEJ2000,ObjectId,RAICRS,DEICRS,e_RAICRS,e_DEICRS,SMSS,EpMean,flags,ClassStar,RadrPetro,a,b,PA,uPSF,uPetro,vPSF,vPetro,gPSF,gPetro,rPSF,rPetro,iPSF,iPetro,zPSF,zPetro,(u-v)PSF,(u-g)PSF,(g-r)PSF,(g-i)PSF,(i-z)PSF]
        magnames = ['u','g','r',
                    'i','zmag'
                    ]
        df = pd.read_csv(path + '/Skymapper/vizier_II_358_smss_20230207_2.csv',
                         sep = ',', index_col = False,
                         usecols = ['uPSF','gPSF','rPSF','iPSF','zPSF'])
        df.rename(columns={'uPSF':'u',
                           'gPSF':'g',
                           'rPSF':'r',
                           'iPSF':'i',
                           'zPSF':'zmag',}, inplace = True)
        datasetname = 'Skymapper'

#%% Processed MQ x GLEAM with radio spectral fits
    if name.lower() == 'mq_processed': # 9276
        redshift = True
        # fields = [_RAJ2000_1,_DEJ2000_1,GLEAM,RAJ2000_1,DEJ2000_1,Fpwide,Fintwide,eabsFpct,efitFpct,Fp076,Fint076,Fp084,Fint084,Fp092,Fint092,Fp099,Fint099,Fp107,Fint107,Fp115,Fint115,Fp122,Fint122,Fp130,Fint130,Fp143,Fint143,Fp151,Fint151,Fp158,Fint158,Fp166,Fint166,Fp174,Fint174,Fp181,Fint181,Fp189,Fint189,Fp197,Fint197,Fp204,Fint204,Fp212,Fint212,Fp220,Fint220,Fp227,Fint227,alpha,Fintfit200,_RAJ2000_2,_DEJ2000_2,recno,RAJ2000_2,DEJ2000_2,Name,Type,Rmag,Bmag,Comment,R,B,z,Qpct,XName,RName,Separation]
        magnames = [
            # 'my alpha',
                    'alpha_thin','alpha_thick','log nu_TO','log F_p'
                    ]
        df = pd.read_csv(path + '/x-matches/milliquas_x_gleam_fits_final.csv',
                         sep = ',', index_col = False,
                         usecols = cols)
        df.rename(columns={'z':'redshift'}, inplace=True)
        datasetname = 'GLEAM x Milliquas spectral fits'

#%% Spectroscopically-confirmed QSOs from milliquas
    if name.lower() == 'radio_z_mq': # 45318
        redshift = True
        # fields = [_RAJ2000_1,_DEJ2000_1,GLEAM,RAJ2000_1,DEJ2000_1,Fpwide,Fintwide,eabsFpct,efitFpct,Fp076,Fint076,Fp084,Fint084,Fp092,Fint092,Fp099,Fint099,Fp107,Fint107,Fp115,Fint115,Fp122,Fint122,Fp130,Fint130,Fp143,Fint143,Fp151,Fint151,Fp158,Fint158,Fp166,Fint166,Fp174,Fint174,Fp181,Fint181,Fp189,Fint189,Fp197,Fint197,Fp204,Fint204,Fp212,Fint212,Fp220,Fint220,Fp227,Fint227,alpha,Fintfit200,_RAJ2000_2,_DEJ2000_2,recno,RAJ2000_2,DEJ2000_2,Name,Type,Rmag,Bmag,Comment,R,B,z,Qpct,XName,RName,Separation]
        magnames = ['Rmag','Bmag']
        # cols = ['u_mag','g_mag','r_mag','i_mag','z_mag',
        #           'I_mag','J_mag','H_mag','K_mag','W1_mag','SPIT_5_mag','W2_mag',
        #           'SPIT_8_mag','W3_mag','W4_mag','NUV_mag','FUV_mag', 'redshift']
        df = pd.read_csv(path + '/radio_z_mq.csv',
                         sep = ',', index_col = False,
                         usecols = cols)
        df.rename(columns={'z':'redshift'}, inplace=True)
        datasetname = 'GLEAM x Milliquas'

#%% GLEAM x Milliquas
# First 50,000 GALEX QSOs with SDSS and WISE
    if name.lower() == 'gleam_x_milliquas': # 3513
        redshift = True
        # fields = [_RAJ2000_1,_DEJ2000_1,GLEAM,RAJ2000_1,DEJ2000_1,Fpwide,Fintwide,eabsFpct,efitFpct,Fp076,Fint076,Fp084,Fint084,Fp092,Fint092,Fp099,Fint099,Fp107,Fint107,Fp115,Fint115,Fp122,Fint122,Fp130,Fint130,Fp143,Fint143,Fp151,Fint151,Fp158,Fint158,Fp166,Fint166,Fp174,Fint174,Fp181,Fint181,Fp189,Fint189,Fp197,Fint197,Fp204,Fint204,Fp212,Fint212,Fp220,Fint220,Fp227,Fint227,alpha,Fintfit200,_RAJ2000_2,_DEJ2000_2,recno,RAJ2000_2,DEJ2000_2,Name,Type,Rmag,Bmag,Comment,R,B,z,Qpct,XName,RName,Separation]
        magnames = ['Rmag','Bmag','Fp076','Fp084','Fp092',
                  'Fp099','Fp107','Fp115',
                  'Fp122','Fp130','Fp143','Fp151',
                  'Fp158','Fp166','Fp174',
                  'Fp181','Fp189','Fp197','Fp204',
                  'Fp212','Fp220','Fp227',
                  # 'alpha'
                  ]
        # cols = ['u_mag','g_mag','r_mag','i_mag','z_mag',
        #           'I_mag','J_mag','H_mag','K_mag','W1_mag','SPIT_5_mag','W2_mag',
        #           'SPIT_8_mag','W3_mag','W4_mag','NUV_mag','FUV_mag', 'redshift']
        df = pd.read_csv(path + '/x-matches/gleam_x_milliquas.csv',
                         sep = ',', index_col = False,
                         usecols = cols)
        df.rename(columns={'z':'redshift'}, inplace=True)
        datasetname = 'GLEAM x Milliquas'

#%% GALEX QSO
# First 50,000 GALEX QSOs with SDSS and WISE
    if name.lower() == 'galexqso': # 33577
        redshift = False
        fields = ['no','NED','redshift','ez','type','class','no_radio','radio_top',
                  'no_UV','uv_bottom','u_mag','g_mag','r_mag','i_mag','z_mag',
                  'I_mag','J_mag','H_mag','K_mag','W1_mag','SPIT_5_mag','W2_mag',
                  'SPIT_8_mag','W3_mag','W4_mag','NUV_mag','FUV_mag']
        magnames = [
                    'u_mag','g_mag','r_mag','i_mag','z_mag', # SDSS
                    'I_mag',
                    # 'J_mag','H_mag','K_mag',         # 2MASS
                    'W1_mag','W2_mag','W3_mag','W4_mag',     # WISE
                    'NUV_mag','FUV_mag'                      # GALEX
                    ]
        # cols = ['u_mag','g_mag','r_mag','i_mag','z_mag',
        #           'I_mag','J_mag','H_mag','K_mag','W1_mag','SPIT_5_mag','W2_mag',
        #           'SPIT_8_mag','W3_mag','W4_mag','NUV_mag','FUV_mag', 'redshift']
        df = pd.read_csv(path + '/GALEX/QSOs_1st_50k-mags_GALEX-fixed.dat.dat',
                         sep = ' ', index_col = False, names = fields)
        datasetname = 'SDSS12 x GALEX x WISE x 2MASS'

#%% GALEX first 10 million
    if name.lower() == 'galex10k': # 999999
        redshift = False
        magnames = [
            # 'FUV', Very few measurements
            'NUV']
        df = pd.read_csv(path + '/GALEX/vizier_II_312_ais_20220606_10million.csv',
                         sep = ',', index_col = False,
                         usecols = cols)
        datasetname = 'GALEX'
        # if redshift:
        #     df.rename(columns={'zsp':'redshift'}, inplace=True)

#%% SDSS16QSO
    if name.lower() == 'sdss16qso': # 750414
       redshift = True
       fields = ['_RAJ2000','_DEJ2000','recno','SDSS','RAJ2000','DEJ2000',
                 'Plate','MJD','Fiber','Class','QSO','z','r_z','umag','gmag',
                 'rmag','imag','zmag','e_umag','e_gmag','e_rmag','e_imag',
                 'e_zmag','Extu','Extg','Extr','Exti','Extz','FFUV','FNUV',
                 'FY','FJ','FH','FK','FW1','fracW1','FW2','fracW2','2RXS',
                 'Gaia','Sp','Simbad']
       magnames = ['umag','gmag','rmag','imag','zmag','FFUV','FNUV',
                 # 'FY','FJ','FH','FK',
                 'FW1', 'FW2']
       df = pd.read_csv(path + '/SDSS/vizier_VII_289_dr16q_20210423.csv',
                        sep = ',', names = fields, index_col = False,
                        header = 0,
                        usecols = cols)
       df.rename(columns={'z':'redshift'}, inplace=True)
       datasetname = 'SDSS quasar catalog, sixteenth data release (DR16Q)'

#%% SDSS9 QSO
    if name.lower() == 'sdss9qso': # 230096
       redshift = True
       fields = ['_RAJ2000','_DEJ2000','mode','q_mode','cl','SDSS9','m_SDSS9',
                 'Im','objID','RA_ICRS','DE_ICRS','ObsDate','Q','umag','e_umag',
                 'gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag','zsp']
       magnames = ['umag','gmag','rmag', 'imag', 'zmag']
       # cols = ('umag', 'gmag', 'rmag', 'imag', 'zmag', 'zsp')
       df = pd.read_csv(path + '/SDSS/vizier_V_139_sdss9_20201125.csv',
                        sep = ',', header = 0, index_col = False,
                        usecols = cols)
       df = df.replace(-9999, np.nan)
       df.rename(columns={'zsp':'redshift'}, inplace=True)
       # df = df.drop(['q_mode', 'm_SDSS9', 'objID'], axis = 1)
       datasetname = 'SDSS Catalog, Data Release 9 QSOs'
#%% SDSS12
    if name.lower() == 'sdss12': # 43102
        redshift = True
        magnames = ['umag','gmag','rmag','imag','zmag']
        df = pd.read_csv(path + '/SDSS/vizier_V_147_sdss12_20200818.csv',
                         sep = ',', index_col = False,
                         usecols = cols)
        datasetname = 'SDSS DR12'
        df = df.drop(columns = ['_RAJ2000', '_DEJ2000'])
        df = df[df['zph'] > 0]
        df = df[df['<zph>'] > 0]
        if redshift:
            df.rename(columns={'zsp':'redshift'}, inplace=True)

#%% SDSS12 QSO
    if name.lower() == 'sdss12qso': # 439273
        redshift = True
        #   QSOs with dz < 0.01 from SDSS DR12
        magnames = ['umag', 'gmag', 'rmag', 'imag', 'zmag']
        df = pd.read_csv(path + '/SDSS/sdss12qso.csv',
                         sep = ',',
                         # header = 0,
                         index_col = False,
                         usecols = cols)
        # df = df.drop(columns = ['_RAJ2000' '_DEJ2000'])
        # df = df[df['zph']!=-9999]
        df.rename(columns={'zsp':'redshift'}, inplace=True)
        # df = df.drop(['q_mode', 'm_SDSS12'], axis = 1)
        datasetname = 'SDSS DR12 QSOs'

#%% SDSS12 Spectroscopic sources
    if name.lower() == 'sdss12spec': # 8857
        redshift = True
        magnames = ['umag', 'gmag', 'rmag', 'imag', 'zmag']
        df = pd.read_csv(path + '/SDSS/sdss12_spec_sources_DR16_quasars.csv',
                         sep = ',',
                         header = 0,
                         index_col = False,
                         usecols = cols)
        # df = df[df['zph']!=-9999]
        df.rename(columns={'zsp':'redshift'}, inplace=True)
        # df = df.drop(['q_mode', 'm_SDSS12'], axis = 1)
        datasetname = 'Spectroscopic sources from SDSS DR12 also in DR16Q'

#%% SDSS12 filtered for good zsp and with PSF and Petrosian mags
    if name.lower() == 'sdss12_petrosian': # 3118641
        magnames = ["u'mag","upmag","uPmag","g'mag","gpmag","gPmag",
                    "r'mag","rpmag","rPmag","i'mag","ipmag","iPmag",
                    "z'mag","zpmag","zPmag"]
        df = pd.read_csv(path + '/SDSS/vizier_V_147_sdss12_Pet_PSF.csv',
                         sep = ',',
                         header = 0,
                         index_col = False,
                         usecols = cols,
                         # nrows = number_of_rows
                         )
        # df = df[df['zph']!=-9999]
        df.rename(columns={'zsp':'redshift'}, inplace=True)
        # df = df.drop(['q_mode', 'm_SDSS12'], axis = 1)
        datasetname = 'SDSS12 with Petrosian and PSF mags'

#%% sdssmags
    if name.lower() == 'sdssmags': # 26301
        '''
        I think this data is from SDSS, WISE and GALEX, but I'm not sure
        what processing has been done on it. Steve gave me the data, but
        he doesn't remember. He refers to it in an email on 26/11/2020:
          | I've attached my python code and the data file it used (this is produced by an awk script which pulls the relevent
          | fields from the master file which I've sent before).
        '''
        redshift = True
        fields = ['redshift','u','g','r','i','zmag',
                  # 'W1','W2','NUV','FUV'
                  ]
        magnames = ['u','g','r','i','zmag',
                    # 'W1','W2','NUV','FUV'
                    ]
        df = pd.read_csv(path + '/SDSS/SDSS-mags.dat',
                         sep = ' ', names = fields, index_col = False,
                         usecols = cols)
        datasetname = 'Steve\'s SDSS data'

#%% GLEAM x SDSS16
    if name.lower() == 'gleam_x_sdss16': # 420
        redshift = True
#         fields = ['_RAJ2000_1','_DEJ2000_1','GLEAM','RAJ2000','DEJ2000','Fpwide',
#                   'Fintwide','eabsFpct','efitFpct','Fp076','Fint076','Fp084',
#                   'Fint084','Fp092','Fint092','Fp099','Fint099','Fp107','Fint107',
#                   'Fp115','Fint115','Fp122','Fint122','Fp130','Fint130','Fp143',
#                   'Fint143','Fp151','Fint151','Fp158','Fint158','Fp166','Fint166',
#                   'Fp174','Fint174','Fp181','Fint181','Fp189','Fint189','Fp197',
#                   'Fint197','Fp204','Fint204','Fp212','Fint212','Fp220','Fint220',
#                   'Fp227','Fint227','alpha','Fintfit200','_RAJ2000_2','_DEJ2000_2',
#                   'objID','RA_ICRS','DE_ICRS','mode','class','clean','e_RA_ICRS',
#                   'e_DE_ICRS','umag','gmag','rmag','imag','zmag','e_umag','e_gmag',
#                   'e_rmag','e_imag','e_zmag','zsp','e_zsp','f_zsp','zph','e_zph',
#                   '<zph>','Q','SDSS16','Sp-ID','MJD','Separation'
# ]
        magnames = ['Fp076','Fp084','Fp092','Fp099','Fp107',
                          'Fp115','Fint115','Fp122','Fint122','Fp130','Fint130','Fp143',
                          'Fp151','Fp158','Fp166',
                          'Fp174','Fp181','Fp189','Fp197',
                          'Fp204','Fp212','Fp220',
                          'Fp227','alpha','umag','gmag','rmag','imag','zmag']
        df = pd.read_csv(path + '/x-matches/gleam_x_sdss16_r10arcsec.csv',
                         sep = ',',
                         # names = fields,
                         index_col = False,
                         usecols = cols)
        if redshift:
            df.rename(columns={'zsp':'redshift'}, inplace=True)
        datasetname = 'GLEAM x SDSS 16, r=10"'

#%% GLEAM x SDSS12
    if name.lower() == 'gleam_x_sdss12': # 2028
        redshift = True
#         fields = ['_RAJ2000_1','_DEJ2000_1','GLEAM','RAJ2000','DEJ2000','Fpwide',
#                   'Fintwide','eabsFpct','efitFpct','Fp076','Fint076','Fp084',
#                   'Fint084','Fp092','Fint092','Fp099','Fint099','Fp107','Fint107',
#                   'Fp115','Fint115','Fp122','Fint122','Fp130','Fint130','Fp143',
#                   'Fint143','Fp151','Fint151','Fp158','Fint158','Fp166','Fint166',
#                   'Fp174','Fint174','Fp181','Fint181','Fp189','Fint189','Fp197',
#                   'Fint197','Fp204','Fint204','Fp212','Fint212','Fp220','Fint220',
#                   'Fp227','Fint227','alpha','Fintfit200','_RAJ2000_2','_DEJ2000_2',
#                   'objID','RA_ICRS','DE_ICRS','mode','class','clean','e_RA_ICRS',
#                   'e_DE_ICRS','umag','gmag','rmag','imag','zmag','e_umag','e_gmag',
#                   'e_rmag','e_imag','e_zmag','zsp','e_zsp','f_zsp','zph','e_zph',
#                   '<zph>','Q','SDSS16','Sp-ID','MJD','Separation'
# ]
        magnames = ['Fp076','Fp084','Fp092','Fp099','Fp107',
                          'Fp115','Fp122','Fp130','Fp143',
                          'Fp151','Fp158','Fp166',
                          'Fp174','Fp181','Fp189','Fp197',
                          'Fp204','Fp212','Fp220',
                          'Fp227','alpha','umag','gmag','rmag','imag','zmag']
        df = pd.read_csv(path + '/x-matches/gleam_x_sdss12.csv',
                         sep = ',',
                         # names = fields,
                         index_col = False,
                         usecols = cols)
        df = df.drop(['_RAJ2000_1'], axis=1)
        df = df.drop(['_DEJ2000_1'], axis=1)
        if redshift:
            df.rename(columns={'zsp':'redshift'}, inplace=True)
        datasetname = 'GLEAM x SDSS 12, r=10"'
#%% GLEAM x SIMBAD
    if name.lower() == 'gleam_x_simbad': # 33012
        redshift = True
        # fields = ["angDist","X_RAJ2000","X_DEJ2000","GLEAM","RAJ2000","DEJ2000",
        #           "Fpwide","Fintwide","eabsFpct","efitFpct","Fp076","Fint076",
        #           "Fp084","Fint084","Fp092","Fint092","Fp099","Fint099","Fp107",
        #           "Fint107","Fp115","Fint115","Fp122","Fint122","Fp130","Fint130",
        #           "Fp143","Fint143","Fp151","Fint151","Fp158","Fint158","Fp166",
        #           "Fint166","Fp174","Fint174","Fp181","Fint181","Fp189","Fint189",
        #           "Fp197","Fint197","Fp204","Fint204","Fp212","Fint212","Fp220",
        #           "Fint220","Fp227","Fint227","alpha","Fintfit200","main_id","ra",
        #           "dec","coo_err_maj","coo_err_min","coo_err_angle","nbref",
        #           "ra_sexa","dec_sexa","coo_qual","coo_bibcode","main_type",
        #           "other_types","radvel","radvel_err","redshift","redshift_err",
        #           "sp_type","morph_type","plx","plx_err","pmra","pmdec","pm_err_maj",
        #           "pm_err_min","pm_err_pa","size_maj","size_min","size_angle",
        #           "B","V","R","J","H","K","u","g","r","i","z"]

        magnames = ["Fp076","Fint076",
                  "Fp084","Fp092","Fp099","Fp107",
                  "Fp115","Fp122","Fp130",
                  "Fp143","Fp151","Fp158","Fp166",
                  "Fp174","Fp181","Fp189",
                  "Fp197","Fp204","Fp212","Fp220",
                  "Fp227","B","V","R","J","H","K","u","g","r","i","z"]
        df = pd.read_csv(path + '/x-matches/gleamxSIMBAD.csv',
                         sep = ',', index_col = False,
                         header = 0
                          # names = fields
                          )
        datasetname = 'GLEAM x SIMBAD'
#%% SDSS12 x UKIDSS
    if name.lower() == 'sdss12_x_ukidss': # 2417
        redshift = True
        fields = ['_RAJ2000_1','_DEJ2000_1','RA_ICRS','DE_ICRS','mode','q_mode',
                  'class','SDSS12','m_SDSS12','ObsDate','Q','umag','e_umag','gmag',
                  'e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag','zsp','spCl',
                  'zph','e_zph','<zph>','_RAJ2000_2','_DEJ2000_2','UDXS','m','RAJ2000',
                  'DEJ2000','Jmag','e_Jmag','Kmag','e_Kmag','Jdiam','Jell','Jflags',
                  'Kdiam','Kell','Kflags','Separation'
                  ]
        magnames = ['umag','gmag','rmag','imag','zmag', 'Jmag', 'Kmag']

        df = pd.read_csv(path + '/x-matches/sdss12_x_ukidss.csv',
                         sep = ',', index_col = False,
                         header = 0,
                         names = fields
                         )
        if redshift:
            df.rename(columns={'zsp':'redshift'}, inplace=True)
        datasetname = 'SDSS DR12 cross-match UKIDSS'
#%% GALEX all-Sky Survey
    if name.lower() == 'galexais': #
        redshift = False
        fields = ['recno','RAJ2000','DEJ2000','FUV','e_FUV','NUV','e_NUV','objid',
                  'tile','img','sv','r.fov','b','E(B-V)','FUV.b','e_FUV.b','NUV.b',
                  'e_NUV.b','FUV.a','e_FUV.a','NUV.a','e_NUV.a','FUV.4','e_FUV.4',
                  'NUV.4','e_NUV.4','FUV.6','e_FUV.6','NUV.6','e_NUV.6','Fafl',
                  'Nafl','Fexf','Nexf','Fflux','e_Fflux','Nflux','e_Nflux','FXpos',
                  'FYpos','NXpos','NYpos','Fima','Nima','Fr','Nr','phID','fRAdeg',
                  'fDEdeg'
                  ]
        magnames = [
            # 'FUV', Very few measurements
            'NUV']

        df = pd.read_csv(path + '/GALEX/GALEX_II_312_ais.csv',
                         sep = ',', index_col = False,
                         header = 0,
                         names = fields
                         )
        datasetname = 'GALEX AIS (All-sky Imaging Survey)'
#%% GALEX x SDSS12
    if name.lower() == 'galex_x_sdss12': # 73906
        redshift = True
        fields = ['recno','RAJ2000','DEJ2000','FUV','e_FUV','NUV','e_NUV','objid',
                  'tile','img','sv','r.fov','b','E(B-V)','FUV.b','e_FUV.b','NUV.b',
                  'e_NUV.b','FUV.a','e_FUV.a','NUV.a','e_NUV.a','FUV.4','e_FUV.4',
                  'NUV.4','e_NUV.4','FUV.6','e_FUV.6','NUV.6','e_NUV.6','Fafl',
                  'Nafl','Fexf','Nexf','Fflux','e_Fflux','Nflux','e_Nflux','FXpos',
                  'FYpos','NXpos','NYpos','Fima','Nima','Fr','Nr','phID','fRAdeg',
                  'fDEdeg','_RAJ2000','_DEJ2000','RA_ICRS','DE_ICRS','mode','q_mode',
                  'class','SDSS12','m_SDSS12','ObsDate','Q','umag','e_umag','gmag',
                  'e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag','zsp','spCl',
                  'zph','e_zph','<zph>','Separation'
                  ]
        magnames = [
            # 'FUV', Very few measurements
            'NUV','umag','gmag','rmag','imag','zmag']

        df = pd.read_csv(path + '/x-matches/galex_x_sdss12.csv',
                          sep = ',', index_col = False,
                          header = 0,
                          names = fields
                          )
        if redshift:
            df.rename(columns={'zsp':'redshift'}, inplace=True)
        datasetname = 'GALEX All-sky Survey X SDSS12'

#%% GALEX x SDSS12 x GLEAM
    if name.lower() == 'galex_x_sdss12_x_gleam': # 565
        redshift = True
        fields = ['recno','RAJ2000_1','DEJ2000_1','FUV','e_FUV',
                  'NUV','e_NUV','objid','tile','img','sv','r.fov','b',
                  'E(B-V)','FUV.b','e_FUV.b','NUV.b','e_NUV.b','FUV.a',
                  'e_FUV.a','NUV.a','e_NUV.a','FUV.4','e_FUV.4','NUV.4',
                  'e_NUV.4','FUV.6','e_FUV.6','NUV.6','e_NUV.6','Fafl',
                  'Nafl','Fexf','Nexf','Fflux','e_Fflux','Nflux','e_Nflux',
                  'FXpos','FYpos','NXpos','NYpos','Fima','Nima','Fr','Nr',
                  'phID','fRAdeg','fDEdeg','_RAJ2000_1','_DEJ2000_1',
                  'RA_ICRS','DE_ICRS','mode','q_mode','class','SDSS12',
                  'm_SDSS12','ObsDate','Q','umag','e_umag','gmag','e_gmag',
                  'rmag','e_rmag','imag','e_imag','zmag','e_zmag','zsp',
                  'spCl','zph','e_zph','<zph>','Separation_1','_RAJ2000_2',
                  '_DEJ2000_2','GLEAM','RAJ2000_2','DEJ2000_2','Fpwide',
                  'Fintwide','eabsFpct','efitFpct','Fp076','Fint076','Fp084',
                  'Fint084','Fp092','Fint092','Fp099','Fint099','Fp107',
                  'Fint107','Fp115','Fint115','Fp122','Fint122','Fp130',
                  'Fint130','Fp143','Fint143','Fp151','Fint151','Fp158',
                  'Fint158','Fp166','Fint166','Fp174','Fint174','Fp181',
                  'Fint181','Fp189','Fint189','Fp197','Fint197','Fp204',
                  'Fint204','Fp212','Fint212','Fp220','Fint220','Fp227',
                  'Fint227','alpha','Fintfit200','Separation']

        magnames = ['Fp076','Fp084','Fp092','Fp099','Fp107', 'Fp115',
                    'Fp122','Fp130','Fp143','Fp151','Fp158','Fp166','Fp174',
                    'Fp181','Fp189','Fp197','Fp204','Fp212','Fp220','Fp227',
                    'alpha','FUV','NUV','umag','gmag','rmag','imag','zmag']

        df = pd.read_csv(path + '/x-matches/galex_x_sdss12_x_gleam.csv',
                          sep = ',', index_col = False,
                          header = 0,
                          names = fields
                          )
        if redshift:
            df.rename(columns={'zsp':'redshift'}, inplace=True)
        datasetname = 'GALEX All-sky Survey X SDSS12 x GLEAM'
#%% Milliquas x GLEAM
    if name.lower() == 'milli_x_gleam': # 6834
        redshift = True
        fields = ['angDist','_RAJ2000_milli','_DEJ2000_milli','recno','RAJ2000_milli',
                  'DEJ2000_milli','Name',
                  'Type','Rmag','Bmag','Comment','R','B','z','Qpct','XName','RName',
                  '_RAJ2000','_DEJ2000','GLEAM','RAJ2000','DEJ2000','Fpwide','Fintwide',
                  'eabsFpct','efitFpct','Fp076','Fint076','Fp084','Fint084','Fp092',
                  'Fint092','Fp099','Fint099','Fp107','Fint107','Fp115','Fint115',
                  'Fp122','Fint122','Fp130','Fint130','Fp143','Fint143','Fp151',
                  'Fint151','Fp158','Fint158','Fp166','Fint166','Fp174','Fint174',
                  'Fp181','Fint181','Fp189','Fint189','Fp197','Fint197','Fp204',
                  'Fint204','Fp212','Fint212','Fp220','Fint220','Fp227','Fint227',
                  'alpha','Fintfit200'
                  ]
        magnames = ['Rmag','Bmag','Fp076','Fp084','Fp092',
                  'Fp099','Fp107','Fp115',
                  'Fp122','Fp130','Fp143','Fp151',
                  'Fp158','Fp166','Fp174',
                  'Fp181','Fp189','Fp197','Fp204',
                  'Fp212','Fp220','Fp227',
                  # 'alpha'
                  ]

        df = pd.read_csv(path + '/x-matches/milliquas_x_gleam.csv',
                         sep = ',', index_col = False,
                         header = 0,
                         names = fields
                         )
        df = df.drop(['_RAJ2000_milli','_DEJ2000_milli','RAJ2000_milli',
                      'DEJ2000_milli','Name', 'XName', 'RName','_RAJ2000',
                      '_DEJ2000','GLEAM'], axis = 1)
        df = df[df['z'] < 5.84]

        if redshift:
            df.rename(columns={'z':'redshift'}, inplace=True)
        datasetname = 'Milliquas GLEAM cross-match'
#%% SDSS16
    if name.lower() == 'sdss16': # 28000
        redshift = True
        fields = ['_RAJ2000','_DEJ2000','objID','RA_ICRS','DE_ICRS','mode','class',
                  'clean','e_RA_ICRS','e_DE_ICRS','umag','gmag','rmag','imag','zmag',
                  'e_umag','e_gmag','e_rmag','e_imag','e_zmag','zsp','e_zsp','f_zsp',
                  'zph','e_zph','<zph>','Q','SDSS16','Sp-ID','MJD'
                  ]
        magnames = ['umag','gmag','rmag','imag','zmag']

        df = pd.read_csv(path + '/SDSS/vizier_V_154_sdss16_20220307.csv',
                         sep = ',', index_col = False,
                         header = 0,
                         names = fields
                         )
        df = df.drop(['SDSS16', '_RAJ2000', '_DEJ2000', 'Sp-ID'], axis = 1)

        if redshift:
            df.rename(columns={'zsp':'redshift'}, inplace=True)
        datasetname = 'SDSS 16'
#%% New Fitted
    if name.lower() == 'new_fitted': # 43699
        redshift = True
        fields = ['idx', 'z_spec','flag_TO','SI','S_400','S_1p4','S_5','S_8p7'
                  ]
        magnames = ['SI','S_400','S_1p4','S_5','S_8p7']

        df = pd.read_csv(path + '/out.csv',
                         sep = ',', index_col = False,
                         header = 0,
                         names = fields
                         )
        if redshift:
            df.rename(columns={'z_spec':'redshift'}, inplace=True)
        datasetname = 'The Million Quasars catalogue'
#%% Old fitted
    if name.lower() == 'old_fitted': # 43699
        sample = 'PdEAK'
        redshift = True
        fields = ['no','NED','zspec','z_NED','class','type',' Qpct','no_radio','fit',
                  'X2','X1','X0','double_dash_1','root_1','TO','flag_TO','TO_flux','SI',
                  'nu_peak','S_peak','thick','thin','S_70','S_150','S_400','S_700','S_1',
                  'S_1p4','S2p7','S_5','S_8p7','S_15','S_20','u','g','r','i','z','W1',
                  'W2','NUV','FUV'
                  ]
        if sample == 'PEAK':
            magnames = ['SI', 'nu_peak','S_peak','thick','thin']

        else:
            magnames = ['TO_flux','SI',
                  'nu_peak','S_peak','thick','thin','S_70','S_150','S_400','S_700','S_1',
                  'S_1p4','S2p7','S_5','S_8p7','S_15','S_20','u','g','r','i','z','W1',
                  'W2','NUV','FUV']

        df = pd.read_csv(path + '/new_fitted.dat',
                         sep = ' ', index_col = False,
                         header = None,
                         names = fields
                         )
        if redshift:
            df.rename(columns={'zspec':'redshift'}, inplace=True)
        datasetname = 'The Million Quasars (Milliquas) catalogue'
#%% XLSOptn
    if name.lower() == 'xlsoptn': # 31585
        redshift = True
        fields = ['_RAJ2000','_DEJ2000','Xcatname','RACtpdeg','DECtpdeg','zspec',
                  'uSDSSmag','gSDSSmag','rSDSSmag','iSDSSmag','zSDSSmag','uCFHTmag',
                  'gCFHTmag','rCFHTmag','iCFHTmag','yCFHTmag','zCFHTmag','zVISTAmag',
                  'YVISTAmag','JVISTAmag','HVISTAmag','KVISTAmag','JUKIDSSmag',
                  'HUKIDSSmag','KUKIDSSmag','KWIRcammag','IRAC3.6mag','IRAC4.5mag',
                  'GALEXFUVmag','GALEXNUVmag','WISE1mag','WISE2mag','WISE3mag',
                  'WISE4mag','recno'
                  ]
        magnames = ['uSDSSmag','gSDSSmag','rSDSSmag','iSDSSmag','zSDSSmag','uCFHTmag',
                  'gCFHTmag','rCFHTmag','iCFHTmag','yCFHTmag','zCFHTmag','zVISTAmag',
                  'YVISTAmag','JVISTAmag','HVISTAmag','KVISTAmag','JUKIDSSmag',
                  'HUKIDSSmag','KUKIDSSmag','KWIRcammag','IRAC3.6mag','IRAC4.5mag',
                  'GALEXFUVmag','GALEXNUVmag','WISE1mag','WISE2mag','WISE3mag']

        df = pd.read_csv(path + '/vizier_IX_52_3xlsoptn_20220120.csv',
                         sep = ',', index_col = False,
                         header = 0,
                          usecols = cols
                         )
        if redshift:
            df.rename(columns={'zspec':'redshift'}, inplace=True)
        datasetname = 'Spectrophotometric catalog of galaxies'
#%% XXGal
    if name.lower() == 'xxgal': # 24336
        redshift = True
        fields = ['_RAJ2000','_DEJ2000','Index','RAJ2000','DEJ2000','z','r_z','f_z',
                  'q_z','DRr200-1','DRr200-2','DRr200-3','DRr200-4','DRr200-5','Dv-1',
                  'Dv-2','Dv-3','Dv-4','Dv-5','XLSSC3r200','XLSSC3r200u','DRr200u',
                  'uMag','gMag','rMag','iMag','yMag','zMag','bMass','Mass','BMass',
                  'ComplSM'

                  ]
        magnames = ['uMag','gMag','rMag','iMag','yMag','zMag']

        df = pd.read_csv(path + '/vizier_IX_52_xxlngal_20220120.csv',
                         sep = ',', index_col = False,
                         header = 0,
                          usecols = cols
                         )
        if redshift:
            df.rename(columns={'z':'redshift'}, inplace=True)
        datasetname = 'Spectrophotometric catalog of galaxies'
#%% Milliquas
    if name.lower() == 'milliquas': # 31561
        redshift = True
        fields = ['recno','RAJ2000','DEJ2000','Name','Type','Rmag','Bmag','Comment',
                  'R','B','z','Qpct','XName','RName'
                  ]
        magnames = ['Rmag','Bmag']

        df = pd.read_csv(path + '/Milliquas/vizier_VII_290_catalog_20211213_fixed.csv',
                         sep = ',', index_col = False,
                         header = 0,
                          usecols = cols
                         )
        df['z'] = pd.to_numeric(df['z'], errors = 'coerce')
        df['Rmag'] = pd.to_numeric(df['Rmag'], errors = 'coerce')
        df['Bmag'] = pd.to_numeric(df['Bmag'], errors = 'coerce')
        df = df[df['z'] < 5.84]
        df = df[df['Type'].str.contains("QR")] # quasar + radio association

        if redshift:
            df.rename(columns={'z':'redshift'}, inplace=True)
        datasetname = 'The Million Quasars (Milliquas) catalogue, version 7.2'
#%% FIRST-NVSS
    if name.lower() == 'first_nvss': # 20648
        redshift = True
        fields = ['_RAJ2000','_DEJ2000','recno','UNIQ_ID','RAJ2000','DEJ2000',
                  'Fiflux','Nflux','Wflux','Gflux','SNgmag','SNimag','MJD',
                  'Plate','Fiber','z','Dist','WDist','GB6Dist','SNDist',
                  'SBDist','Plate1','MJD1','Ori'
                  ]
        magnames = ['Fiflux','Nflux','Wflux','Gflux','SNgmag','SNimag']

        df = pd.read_csv(path + '/vizier_J_ApJ_699_L43_catalog_20220117.csv',
                         sep = ',', index_col = False,
                         header = 0,
                          usecols = cols
                         )
        if redshift:
            df.rename(columns={'z':'redshift'}, inplace=True)
        datasetname = 'FIRST-NVSS-SDSS AGN sample catalog'
#%% GLEAM x NVSS, radius 5"
    if name.lower() == 'gxn': # 69745
        redshift = False
        fields = ['angDist','_RAJ2000','_DEJ2000','NVSS','RAJ2000','DEJ2000',
                  'e_RAJ2000','e_DEJ2000','S1.4','e_S1.4','l_MajAxis','MajAxis',
                  'l_MinAxis','MinAxis','f_resFlux','_RAJ2000','_DEJ2000','GLEAM',
                  'RAJ2000','DEJ2000','Fpwide','Fintwide','eabsFpct','efitFpct',
                  'Fp076','Fint076','Fp084','Fint084','Fp092','Fint092','Fp099',
                  'Fint099','Fp107','Fint107','Fp115','Fint115','Fp122','Fint122',
                  'Fp130','Fint130','Fp143','Fint143','Fp151','Fint151','Fp158',
                  'Fint158','Fp166','Fint166','Fp174','Fint174','Fp181','Fint181',
                  'Fp189','Fint189','Fp197','Fint197','Fp204','Fint204','Fp212',
                  'Fint212','Fp220','Fint220','Fp227','Fint227','alpha','Fintfit200']
        magnames = [mag for mag in fields if mag.startswith('Fp') and not mag.startswith("Fpw")]

        df = pd.read_csv(path + '/x-matches/gleam_x_nvss_5_arcsec.csv',
                         sep = ',', index_col = False,
                         header = 0,
                          usecols = cols
                         )
        if redshift:
            df.rename(columns={'zsp':'redshift'}, inplace=True)
        datasetname = 'GLEAM x NVSS, radius 5"'
#%% GLEAM x NVSS x SDSS12 radius 5"
    if name.lower() == 'gxnxs': # 28424
        redshift = True
        fields = ['angDist','angDist','_RAJ2000','_DEJ2000','NVSS','RAJ2000',
                  'DEJ2000','e_RAJ2000','e_DEJ2000','S1.4','e_S1.4','l_MajAxis',
                  'MajAxis','l_MinAxis','MinAxis','f_resFlux','_RAJ2000',
                  '_DEJ2000','GLEAM','RAJ2000','DEJ2000','Fpwide','Fintwide',
                  'eabsFpct','efitFpct','Fp076','Fint076','Fp084','Fint084',
                  'Fp092','Fint092','Fp099','Fint099','Fp107','Fint107','Fp115',
                  'Fint115','Fp122','Fint122','Fp130','Fint130','Fp143',
                  'Fint143','Fp151','Fint151','Fp158','Fint158','Fp166',
                  'Fint166','Fp174','Fint174','Fp181','Fint181','Fp189',
                  'Fint189','Fp197','Fint197','Fp204','Fint204','Fp212',
                  'Fint212','Fp220','Fint220','Fp227','Fint227','alpha',
                  'Fintfit200','RAdeg','DEdeg','errHalfMaj','errHalfMin',
                  'errPosAng','objID','mode','q_mode','class','SDSS12',
                  'm_SDSS12','flags','ObsDate','Q','umag','e_umag','gmag',
                  'e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag','zsp',
                  'e_zsp','f_zsp','zph','e_zph','avg_zph','pmRA','e_pmRA','pmDE',
                  'e_pmDE','SpObjID','spType','spCl','subClass']
        magnames1 = [mag for mag in fields if mag.startswith('Fp') and not mag.startswith("Fpw")]
        magnames2 = [mag for mag in fields if (mag.endswith('mag') and not mag.startswith("e_"))]
        magnames = magnames1 + magnames2
        df = pd.read_csv(path + '/x-matches/gleam_x_nvss_x_sdss12_5_arcsec.csv',
                         sep = ',', index_col = False,
                         header = 0,
                         usecols = cols)
        df.rename(columns={'zsp':'redshift'}, inplace=True)
        datasetname = 'GLEAM x NVSS x SDSS12 radius 5"'
#%% eMERLIN
    if name.lower() == 'emerlin': # 395
        redshift = False
        fields = ['IslId', 'RAdeg', 'e_RAdeg', 'DEdeg', 'e_DEdeg', 'FluxT',
                  'e_FluxT', 'FluxP', 'e_FluxP', 'Maj', 'e_Maj', 'Min', 'e_Min',
                  'PA', 'e_PA', 'DCMaj', 'e_DCMaj', 'DCMin', 'e_DCMin', 'DCPA',
                  'e_DCPA', 'FluxTIsl', 'e_FluxTIsl', 'Islrms', 'Islmean',
                  'ResIslrms', 'ResIslmean', 'SCode', 'ScaleFlag', 'ResFlag',
                  'SMorphFlag']
        magnames = ['FluxT','FluxP']
        df = pd.read_csv(path + '/emerlin_vla_subaru/vizier_J_MNRAS_495_1706_emerlin_20220112.csv',
                         sep = ',', index_col = False,
                         header = 0,
                         usecols = cols)
        datasetname = 'SuperCLASS eMERLIN survey'
#%% VLA
    if name.lower() == 'vla': # 887
        redshift = False
        fields = ['GausId','IslId','SourceId','WaveId','RAJ2000','DEJ2000',
                  'FluxT','FluxP','Maj','Min','PA','DCMaj','DCMin','DCPA',
                  'FluxTIsl','Islrms','Islmean','ResIslrms','ResIslmean',
                  'SCode','ResFlag','SMorphFlag','e1unc','e2unc','e1cal',
                  'e2cal','Valid','RadIm3Shape','e1calCorr','e2calCorr']
        magnames = ['FluxT','FluxP']
        df = pd.read_csv(path + '/emerlin_vla_subaru/vizier_J_MNRAS_495_1706_vla_20220112.csv',
                         sep = ',', index_col = False,
                         header = 0,
                         usecols = cols)
        datasetname = 'SuperCLASS VLA survey'
#%% Subaru
    if name.lower() == 'subaru': # 376380
        redshift = False
        fields = ['RAJ2000','DEJ2000','Bmag','Vmag','rmag','imag','zmag',
                      'ymag','[3.6]','[4.5]','Id','za','chiza','Nfilt','e1',
                      'e2','Rad','RadRatio','BulgeA','DiscA','BulgeIndex',
                      'DiscIndex','BulgeFlux','DiscFlux','FluxRatio','snr',
                      'SourceId']
        magnames = ['Bmag','Vmag','rmag','imag','zmag','ymag','[3.6]','[4.5]']
        df = pd.read_csv(path + '/emerlin_vla_subaru/vizier_J_MNRAS_495_1706_subaru_20220112.csv',
                             sep = ',', index_col = False,
                             header = 0,
                             usecols = cols)
        df.rename(columns={'za':'redshift'}, inplace=True)
        datasetname = 'SuperCLASS Subaru survey'
#%% GLEAM
    if name.lower() == 'gleam': # 307455
        redshift = False
        fields = ['_RAJ2000','_DEJ2000','GLEAM','RAJ2000','DEJ2000','Fpwide',
                  'Fintwide','eabsFpct','efitFpct','Fp076','Fint076','Fp084',
                  'Fint084','Fp092','Fint092','Fp099','Fint099','Fp107','Fint107',
                  'Fp115','Fint115','Fp122','Fint122','Fp130','Fint130','Fp143',
                  'Fint143','Fp151','Fint151','Fp158','Fint158','Fp166','Fint166',
                  'Fp174','Fint174','Fp181','Fint181','Fp189','Fint189','Fp197',
                  'Fint197','Fp204','Fint204','Fp212','Fint212','Fp220','Fint220',
                  'Fp227','Fint227','alpha','Fintfit200']
        magnames = ['Fp076','Fp084','Fp092','Fp099','Fp107', 'Fp115',
                    'Fp122','Fp130','Fp143','Fp151','Fp158','Fp166','Fp174',
                    'Fp181','Fp189','Fp197','Fp204','Fp212','Fp220','Fp227', 'alpha']
        df = pd.read_csv(path + '/GLEAM/vizier_VIII_100_gleamegc_20210608.csv',
                         sep = ',', index_col = False,
                         header = 0,
                         usecols = cols)
        df.rename(columns={'z':'redshift'}, inplace=True)
        datasetname = 'GaLactic and Extragalactic All-sky MWA survey'

#%% Markarian
    if name.lower() == 'markarian': # 1469
        redshift = False
        fields = ['name','ra','dec','vmag','major_axis','minor_axis','redshift',
                  'bv_color','ub_color','class']
        magnames = ['vmag','bv_color','ub_color']
        df = pd.read_csv(path + '/Markarian/markarian.csv',
                         sep = ',', header = 0, index_col = False,
                         usecols = cols)
        datasetname = 'First Byurakan Survey (Markarian) Catalog of UV-Excess Galaxies'
#%% LBQS
    if name.lower() == 'lbqs': # 1055
        redshift = True
        fields = ['name','ra','dec','bjmag','redshift','field_info']
        magnames = ['bjmag']
        df = pd.read_csv(path + '/lbqs.csv',
                         sep = ',', header = 0, index_col = False,
                         usecols = cols)
        datasetname = 'Large Bright Quasar Survey'
#%% LQAC
    if name.lower() == 'lqac': # 20000
        redshift = True
        fields = ['name','ra','dec','vmag','rmag','kmag','flux_20_cm','redshift','abs_imag']
        magnames = ['vmag','rmag','kmag']
        df = pd.read_csv(path + '/lqac.csv',
                         sep = ',', header = 0, index_col = False,
                         usecols = cols)
        datasetname = 'Large Quasar Astrometric Catalog, 3rd Release'
#%% allWISE AGN
    if name.lower() == 'allwiseagn': # 10000
        redshift = True
        fields = ['name','ra','dec','w1w2_color','w2w3_color','w1_mag','gmag','redshift']
        magnames = ['w1w2_color','w2w3_color','w1_mag','gmag']
        df = pd.read_csv(path + '/WISE/allwiseagn.csv',
                         sep = ',', header = 0, index_col = False,
                         usecols = cols)
        datasetname = 'AllWISE Catalog of Mid-IR AGNs'
#%% Crampton
    if name.lower() == 'crampton': # 777
        redshift = False
        fields = ['_RAJ2000','_DEJ2000','recno','Rem','Name','RA1950','DE1950',
                  'Bmag','n_Bmag','redshift','u_z','Nature','_RA.icrs','_DE.icrs']
        magnames = ['Bmag']
        cols = ['_RAJ2000','_DEJ2000','recno','Bmag','z','_RA.icrs','_DE.icrs']
        df = pd.read_csv(path + '/vizier_VII_143_catalog_20201104.csv',
                         sep = ',', header = 0, index_col = False,
                         usecols = cols)
        df.rename(columns={'z':'redshift'}, inplace=True)
        datasetname = 'Quasar Candidates (Crampton+ 1985-1990)'
#%% Rood
    if name.lower() == 'rood': # 3981
        redshift = False
        fields = ['_RAJ2000','_DEJ2000','recno','Seq','UGC','OtherName','RA1950',
                  'DE1950','Pmag','zFlag','r_e_HVel','HVel','e_HVel','u_e_HVel',
                  '_RA.icrs','_DE.icrs']
        magnames = ['Pmag']
        df = pd.read_csv(path + '/vizier_VII_36_catalog_20201104.csv',
                         sep = ',', header = 0, index_col = False,
                         usecols = cols)
        df.rename(columns={'HVel':'redshift', 'e_HVel':'e_z', 'u_e_HVel':'uncertain z'},
                  inplace=True)
        datasetname = 'Galaxy Redshifts (Rood 1980)'
 #%% DEEP2
    if name.lower() == 'deep2': # 52989
        redshift = True
        magnames = ['Bmag','Rmag','Imag', 'EBV']
        # cols = ['Bmag','Rmag','Imag','RG','BMag', 'z']
        df = pd.read_csv(path + '/DEEP2/vizier_III_268_deep2all_20220727_color_extinction.csv',
                         # sep = ',', header = 0, index_col = False, usecols = cols
                         )
        df.rename(columns={'z':'redshift'}, inplace=True)
        datasetname = 'DEEP2 Redshift Survey, DR4'

        if redshift:
            df = df[df['redshift'] > 0]
            df.insert(len(df.columns)-1, 'redshift', df.pop('redshift')) # move redshift column to the end

        deep2_agn = df[df['Cl'] == 'A']
        deep2_gal = df[df['Cl'] == 'G']
        deep2_stars = df[df['Cl'] == 'S']

#%% PanSTARRS
    if name.lower() == 'panstarrs': # 999999
        redshift = False
        fields = ['_RAJ2000','_DEJ2000','RAJ2000','DEJ2000','objID','f_objID',
                  'Qual','e_RAJ2000','e_DEJ2000','Epoch','Ns','Nd','gmag',
                  'e_gmag','gKmag','e_gKmag','gFlags','rmag','e_rmag','rKmag',
                  'e_rKmag','rFlags','imag','e_imag','iKmag','e_iKmag','iFlags',
                  'zmag','e_zmag','zKmag','e_zKmag','zFlags','ymag','e_ymag',
                  'yKmag','e_yKmag','yFlags']
        magnames = ['gmag','gKmag','rmag','rKmag','imag','iKmag','zmag','ymag','yKmag']
        df = pd.read_csv(path + '/vizier_II_349_ps1_20201102_10^6rows.csv',
                         sep = ',', header = 0, index_col = False, usecols = cols)
        datasetname = 'Pan-STARRS release 1 (PS1) Survey - DR1'

#%% VIMOS
    if name.lower() == 'vimos': # 8981
        redshift = False
        print("This dataset contains infs or NaNs; neural network may fail.")
        fields = ['ID','redshift','q_z','phf','UEmag','B','V','R','I','J',
                 'K','VVDS','FITS','RAJ2000','DEJ2000']
        magnames = ['B', 'V', 'R', 'I', 'J', 'K']
        cols = ['B', 'V', 'R', 'I', 'J', 'K', 'redshift']
        df = pd.read_csv(path + '/vizier_III_250_vvds_dp_20200908.csv', skiprows = [0],
                         sep = ',', names = fields, index_col = False, usecols = cols)
        datasetname = 'VIMOS VLT deep survey (VVDS-DEEP)'
#%% ugriz
    if name.lower() == 'ugriz': # 33643
        redshift = False
        print("This dataset contains infs or NaNs; neural network may fail.")
        #   ugriz data from Steve for start of PhD
        fields = ['ignore', 'name', 'redshift','delta_z','NED_class',
                  'SDSS_class','no_radio','radio_max','no_UV', 'UV_min',
                   'u', 'g', 'r', 'i', 'z_mag', 'I', 'J', 'H', 'K', 'W1', 'SPIT_5',
                   'W2', 'SPIT_8', 'W3', 'W4', 'NUV', 'FUV']
        magnames = ['u', 'g', 'r', 'i', 'z_mag', 'I', 'J', 'H', 'K', 'W1', 'SPIT_5',
                   'W2', 'SPIT_8', 'W3', 'W4', 'NUV', 'FUV']
        cols = ['u', 'g', 'r', 'i', 'z_mag', 'I', 'J', 'H', 'K', 'W1', 'SPIT_5',
                   'W2', 'SPIT_8', 'W3', 'W4', 'NUV', 'FUV', 'redshift']
        df = pd.read_csv(path + '//optical_data//QSOs_1st_50k.dat-mags.dat',
                         sep = ' ', names = fields, index_col = False, usecols = cols)
        # df = df.drop(['ignore'], axis = 1)
        datasetname = 'ugriz'
#%% SDSS
    if name.lower() == 'nedfirst50k': # 49738
        redshift = False
        #   SDSS data
        fields = ['ignore', 'NED','redshift','ez','type','class','no_radio','no_UV',
                  'U','B','V','R','I','J','H','K','W1','W2','W3','W4',
                  'SPIT_5','SPIT_8','NUV','FUV']
        magnames = ['U','B','V','R','I','J','H','K','W1','W2','W3','W4',
                  'SPIT_5','SPIT_8','NUV','FUV']
        cols = ['U','B','V','R','I','J','H','K','W1','W2','W3','W4',
                  'SPIT_5','SPIT_8','NUV','FUV', 'redshift']
        df = pd.read_csv(path + '/optical_data/QSOs-NED_1st_50K.dat-mags.dat',
                         sep = ' ', names = fields, index_col = False, usecols = cols)
        # df = df.drop_duplicates(['NED'], keep='first')
        # df = df.drop(['ignore'], axis = 1)
        datasetname = 'SDSS'

#%% MgII
    if name.lower() == 'mgii': # 29008
        redshift = True
        fields = ['no','NED','redshift','type','S_21','freq_21',
                       'SI_flag','U','B','V','R','I','J','H','K','W1','W2','W3',
                       'W4','L_UV','Q','flag_uv','n_abs','z_a','dz_a','EW_2803',
                       'dEW_2803','EW_2796','dEW_2796','SPIT_5','SPIT_8','NUV','FUV']
        magnames = ['U','B','V','R','I','J','H','K','W1','W2','W3',
                       'W4', 'SPIT_5','SPIT_8','NUV','FUV',
                        'chord'
                       ]
        # cols = ['U','B','V','R','I','J','H','K','W1','W2','W3',
        #                'W4', 'SPIT_5','SPIT_8','NUV','FUV','redshift']
        df = pd.read_csv(path + '/optical_data/MgII_mags+SPITZER+UV_0.050.dat',
                         sep = ' ', names = fields, index_col = False, usecols = cols)
        # df = df.drop_duplicates(['NED'], keep='first')
        # df = df.drop(['SI_flag'], axis = 1)
        df['chord'] = (df['I']-df['W2']) / (df['W3']-df['U'])

        datasetname = 'MgII'
#%% LARGESS
    if name.lower() == 'largess': # 10944
        redshift = True
        fields = ['no','NED','redshift','q_z','zsource','EWOIII','OIII_SN',
                          'VClass','BClass','SI','TO','flag_TO','L_21',
                          'P_21','L_UV','Q','flag_uv','U','B','V','R','I','J',
                          'H','K','W1','W2','W3','W4','SPIT_5','SPIT_8','NUV','FUV']
        magnames = ['U','B','V','R','I','J',
                          'H','K','W1','W2','W3','W4','SPIT_5','SPIT_8','NUV','FUV']
        cols = ['U','B','V','R','I','J','H','K','W1','W2','W3','W4',
                'SPIT_5','SPIT_8','NUV','FUV', 'redshift']
        df = pd.read_csv(path + '/optical_data/LARGESS_mag_0.050.dat', sep = ' ',
                         names = fields, index_col = False, usecols = cols)
        # df = df.drop_duplicates(['NED'], keep='first')
        # df = df.drop(['VClass', 'BClass'], axis = 1)
        datasetname = 'LARGESS'
#%% ICRF
    if name.lower() == 'icrf': # 1493
        redshift = True
        fields = ['no','NED','redshift','SI','TO','flag_TO','type','L_21','P_21','L_UV',
                  'Q','flag_uv','U','B','V','R','I','J','H','K','W1','W2','W3',
                  'W4','SPIT_5','SPIT_8','NUV','FUV']
        magnames = ['U','B','V','R','I','J','H','K','W1','W2','W3',
                  'W4','SPIT_5','SPIT_8','NUV','FUV']
        cols = ['U','B','V','R','I','J','H','K','W1','W2','W3',
                  'W4','SPIT_5','SPIT_8','NUV','FUV', 'redshift']
        df = pd.read_csv(path + '/optical_data/Hunstead_mags+SPITZER_0.050.dat',
                         sep = ' ', names = fields, index_col = False, usecols = cols)
        # df = df.drop_duplicates(['NED'], keep='first')

        datasetname = 'ICRF2'

#%% Test dataframe
    if name.lower() == 'test':
        redshift = True
        df = pd.read_csv(path + '/test_dataset.csv',
                         sep = ',', index_col = False, header = 0)

        datasetname = 'Test dataset'
        colours = False
        print('Colours cannot be computed for the test frame')
        magnames = df.columns[3:-1]

#%% Test dataframe 1
    if name.lower() == 'test1':
        redshift = True
        df = pd.read_csv(path + '/test_dataset1.csv',
                         sep = ',', index_col = False, header = 0)

        datasetname = 'Test dataset 1'
        magnames = df.columns[3:-1]

#%% Test dataframe 2
    if name.lower() == 'test2':
        redshift = True
        df = pd.read_csv(path + '/test_dataset2.csv',
                         sep = ',', index_col = False, header = 0)

        datasetname = 'Test dataset 2'
        magnames = df.columns[3:-1]

#%% Tidy the data
    if redshift:
        df = df[df['redshift'] > 0]
        df.insert(len(df.columns)-1, 'redshift', df.pop('redshift')) # move redshift column to the end

    mgf = df[magnames]
    df = df.where(df != -999, np.nan)
    mgf = mgf.where(mgf != -999, np.nan)

    print('{2} sources loaded from {0} with the following bands:\n{1}\n'.format(datasetname, magnames, df.shape[0]))

    contains_z = ['z', 'zsp', 'redshift', 'z_sp']
    for synonym in contains_z:
        if(synonym in df.columns):
            redshift = True

    # Inspect structure of missing data; requires dropna = False in qf.loaddata()
    if dropna:
        df = df.dropna(axis = 0, how = 'any')
        mgf = mgf.dropna(axis = 0, how = 'any')
        print('NaNs have been dropped from the original data.')
    else: # If selected to keep NaNs in dataset, show structure of missing data
        msno.matrix(df[magnames],
                    sort = 'descending',
                color = (0, .6, .8),
                fontsize = 10,
                figsize = (10, 6))

    if colours:
        # Compute colours using the magnitudes columns and add to the dataset
        colours = compute_colours(mgf); colours = colours.iloc[:, len(magnames):] # compute colours and add colours to magnitudes dataframe
        allcolumns = magnames + ['redshift']              # add redshift to the list of magnitude names
        df = df[allcolumns]                     # dataset is now magnitudes and redshift column
        df = pd.concat([df, colours], axis = 1) # dataset is now magnitudes and colours
        print('Colours have been computed and added to the dataset.')

    if impute_method == 'max':
        df = df.fillna(df.max()) # using max() assumes missing data are due to detection limit
        mgf = mgf.fillna(mgf.max())
        print('Missing values have been imputed with the maximum for each column.')
    elif impute_method == 'mean':
        impute_mean = SimpleImputer(missing_values = np.nan,
                                    strategy = 'mean')
        # impute_mean.fit(df)
        impute_mean.fit(mgf)
        # impute_mean.transform(df)
        mgf = impute_mean.transform(mgf) # converts to np.array
        mgf = pd.DataFrame(mgf, columns = magnames) # back to DataFrame
        print('Missing values have been imputed with the mean for each column.')

    # if ['RAJ2000', 'DEJ2000'] in df.columns:
    #     coords = df[['RAJ2000', 'DEJ2000']].to_numpy()
    #     coords = SkyCoord(coords, unit = 'deg')

    return df, datasetname, magnames, mgf

###############################################################################
#%% Create test dataframe
def make_test(n_galaxies=10, n_mags=5, seed=0, file_name = 'test_dataset.csv'):
    '''
    Generates a test dataset with the following structure:

        Name      | RAJ2000 | DEJ2000 | mag(n)  | redshift
        -------------------------------------------------
        Galaxy (n)| (random)| (random)| (random)| (random)

    n_galaxies: number of rows of galaxies, each numbered sequentially from 1
    n_mags    : number of columns of magnitudes, with a random element
                in each mag(n) column == np.nan
    seed      : customizable so you can generate different datasets

    NOTE: n_mags must be <= n_galaxies
    '''
    if seed:
        seed = seed
    else:
        seed = np.random.seed()
    np.random.seed(seed)
    data = np.random.uniform(10, 20, (n_galaxies,n_mags))
    try:
        data[np.diag_indices(n_mags)] = np.nan
    except IndexError:
        print('Cannot generate dataset: n_galaxies ({0}) must be >= n_mags ({1})'.format(n_galaxies, n_mags))
    np.random.shuffle(data)

    magnames = [f'mag{i}' for i in range(1, n_mags + 1)]

    df = pd.DataFrame(data, columns=magnames)
    df.insert(0, 'Name', [f'Galaxy {i}' for i in range(1, n_galaxies + 1)])

    # Generate redshift, RA and dec
    df['redshift'] = np.random.uniform(0.01, 5, n_galaxies) # generate redshift col
    df['RAJ2000'] = np.random.uniform(8, 8.1, n_galaxies)   # generate RA col
    df['DEJ2000'] = np.random.uniform(5, 5.1, n_galaxies)   # generate dec col

    # Move RA and dec to positions 1 and 2
    df.insert(1, 'RAJ2000', df.pop('RAJ2000'))
    df.insert(2, 'DEJ2000', df.pop('DEJ2000'))

    # Save as file
    path = 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/data_files/'
    df.to_csv(path + file_name, index = False)

#%% The rest

def grid_search_model(model, X_train, y_train, y_test, y_pred):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error

    parameters = {'loss'         : ['absolute_error', 'squared_error', 'huber', 'quantile'], # was squared error first
                  'optimizer'    : ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
                  'epochs'       : [10],
                  'batch_size'   : [5, 10, 50]}
                  # what else can I try in here?

    grid = GridSearchCV(estimator = model,
                        param_grid = parameters,
                        scoring = 'accuracy',
                        n_jobs = None, # not -1
                        refit = 'boolean',
                        verbose = 0)
    grid_result = grid.fit(X_train, y_train)

    mse_krr = mean_squared_error(y_test, y_pred)
    print(mse_krr)
    print(grid.best_params_)
    print(grid.best_estimator_)
    return (mse_krr, grid.best_params_, grid.best_estimator_)

###############################################################################

def print_two_lists(x, y):
    border = '-'*25
    separator = '\t\t|\t'
    results_list = zip(x, y)
    print('List 1' + separator + 'List 2')
    print(border)
    for mean, dev, *_ in results_list:
        print(f"{mean:7f}\t|\t{dev:7f}")
    print(border)
    print('Average of list1 = {avg_list1}\nAverage of list2 = {avg_list2}'.format(
        avg_list1=np.mean(x),
        avg_list2=np.mean(y)
    ))

###############################################################################

def norm(x):
    '''
    Still trying to get this one working
    '''
    stats = x.describe()  # overall statistics
    columns = x.columns
    stats = stats[[columns]]
    stats = stats.transpose()

    return (x - stats['mean']) / stats['std']

###############################################################################

def compute_colours(magnitudes):
    '''Computes colours (or any other combination) of magnitudes.
    '''
    comb = [(x, magnitudes.columns[(i+1) % len(magnitudes.columns)])
            for i, x in enumerate(magnitudes.columns)]
    for x, y in comb:
      magnitudes[f'{x} - {y}'] = magnitudes[x] - magnitudes[y]

    return magnitudes

###############################################################################

def plot_z(x_var, y_var, datasetname, ax = None):
    '''Displays a plot of x versus y, colour-coded with a colourmap.'''
    title = 'Redshift predictions compared to spectroscopic redshift\nfor %s'%datasetname
    xy = np.vstack([x_var,y_var])
    z = gaussian_kde(xy)(xy)
    if ax is None:
        plt.scatter(x_var, y_var,
                    s = 10,
                    alpha = 0.3,
                    marker = '.',
                    # edgecolor = '',
                    c = z)
        plt.colorbar()
        plt.grid()
        plt.plot(x_var, x_var, 'r-.')
        # plt.title(title)
        plt.xlabel(r'$z_{spec}$')
        plt.ylabel('Predicted z')
    else:
        ax.scatter(x_var, y_var,
                    s = 10,
                    alpha = 0.3,
                    marker = '.',
                    # edgecolor = '',
                    c = z)
        # fig.colorbar()
        ax.grid()
        ax.plot(x_var, x_var, 'r-.')
        ax.set_title(title)
        ax.set_xlabel(r'$z_{spec}$')
        ax.set_ylabel('Predicted z')
        # ax.set_ylim(0, 4)
    plt.show()

###############################################################################

def plot_z_sets(set1, set2, datasetname, ax = None):
    # valid_set['Delta z'] = y - x
    # x, y = valid_set['z'], valid_set['Delta z']

    fig, ax = plt.subplots(figsize = (12, 9))
    ax.hist(set1,
            bins = 100,
            edgecolor = 'red',
            color = 'pink',
            label = 'Training set')
    ax.hist(set2,
            bins = 100,
            edgecolor = 'blue',
            color = 'lightblue',
            label = 'Validation set',
            alpha = 0.5)
    ax.grid()
    ax.set_xlabel('Redshift')
    ax.set_yscale('log')
    ax.set_ylabel('Count')
    ax.set_title('Redshift distributions of training and validation sets for\n{0}'.format(datasetname))
    ax.legend(loc = 'best')

###############################################################################

def plot_deltaz(x_var, y_var, datasetname, ax = None):
    # valid_set['Delta z'] = y - x
    # x, y = valid_set['z'], valid_set['Delta z']

    # theta = np.polyfit(x_var, y_var, 2)
    # print("Curve parameters: {0}".format(theta))
    # y_line = theta[2] + theta[1] * pow(x_var, 1) + theta[0] * pow(x_var, 2)
    # plt.plot(x_var, y_line, 'r')

    xy = np.vstack([x_var,y_var])
    z = gaussian_kde(xy)(xy)
    plt.scatter(x_var, y_var,
                s = 50, alpha = 0.5, marker = '.', c = z)
    plt.title('Deviation of redshift predictions\nfrom spectroscopic redshift for\n%s'%datasetname)
    plt.xlabel(r'$z_{spec}$')
    plt.ylabel(r'$\Delta z$')
    plt.colorbar()
    plt.grid(which = 'both', axis = 'both')
    # plt.xlim([0, 1.5])
    plt.show()

###############################################################################

def plot_delta_z_hist(values, datasetname, model, ax = None):
    bins = 500
    title = r'Distribution of $\Delta z$ for %s'%datasetname
    if type(values) == np.ndarray:
        values = pd.DataFrame(values)
    stats = values.describe().transpose()
    mean, std = stats['mean'], stats['std']
    # mean, std = values.describe()['mean'], values.describe()[2]
    values.hist(label = ' mean = %.3f\n std dev = %.3f'%
                                      (mean, std),
                                      bins = bins)
    # x_norm = np.linspace(min(values), max(values), 100)
    # y_norm = stats.norm.pdf(x_norm, mean, std)
    # np.histogram(y_norm, bins = 100)
    zero_line_colour, zero_line_style = 'cyan', 'dashed'
    mean_std_colour, mean_style, std_style = 'red', 'solid', 'dotted'
    legend_pos = 'lower left'
    if ax is None:
        # plt.title(title)
        # plt.xlim(-1.5, 1.5)
        plt.xlabel(r'$\Delta z$')
        plt.ylabel('Count')
        plt.grid()
        ax.axvline(x = 0,
                   color = zero_line_colour,
                   linestyle = zero_line_style)
        ax.axvline(x = mean,
                   color = mean_std_colour,
                   linestyle = mean_style,
                   label = r'mean $\pm$ std dev')
        for std in [mean + std, mean - std]:
            ax.axvline(x = std,
                        color = mean_std_colour,
                        linestyle = std_style)

        plt.legend(loc = legend_pos)
    else:
        # ax.set_title(title)

        # plt.xlim(-1.5, 1.5)
        ax.set_xlabel(r'$\Delta z$')
        ax.set_ylabel('Count')
        ax.grid('both')
        ax.axvline(x = 0,
                   color = zero_line_colour,
                   linestyle = zero_line_style)
        ax.axvline(x = mean,
                   color = mean_std_colour,
                   linestyle = mean_style,
                   label = r'mean $\pm$ std dev')
        for std in [mean + std, mean - std]:
            ax.axvline(x = std,
                        color = mean_std_colour,
                        linestyle = std_style)
        ax.legend(loc = legend_pos)
        # ax.set_yscale('log')
    # if model != None: # print the model information onto the plot
    #     stringlist = []
    #     model.summary(print_fn=lambda x: stringlist.append(x))
    #     short_model_summary = "\n".join(stringlist)
    #     # print(short_model_summary)
    #     at = AnchoredText(
    #     short_model_summary, prop=dict(size=7, alpha = 0.6), frameon=False, loc='upper left')
    #     at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    #     ax.add_artist(at)
        # plt.annotate('Text', (1,1))

    plt.show()

###############################################################################

def kurt_result(data):
    kur = kurtosis(data, fisher=True)
    z_score, p_value = kurtosistest(data)
    print('''For delta z:
Kurtosis: {0:.4f}
z-score: {1:.4f}
p-value: {2:.2g}'''.format(kur, z_score, p_value))

###############################################################################

def build_nn_model(n,
               hyperparameters = [100, 'relu', 100, 'relu', 100, 'relu'],
               loss = 'mae',
               metrics = ['mae'],
               opt = 'Nadam'):
    ''' Define the prediction model. The NN takes magnitudes as input features
    and outputs the redshift. n should be len(train_set.keys())'''
    model = keras.Sequential([
    keras.layers.Dense(hyperparameters[0], activation=hyperparameters[1], # number of outputs to next layer
                           input_shape=[n]),  # number of features
    keras.layers.Dense(hyperparameters[2], activation=hyperparameters[3]),
    keras.layers.Dense(hyperparameters[4], activation=hyperparameters[5]),

    keras.layers.Dense(1) # 1 output (redshift)
    ])

    model.compile(loss=loss, optimizer = opt, metrics = metrics)
    print(model.summary())
    return model

###############################################################################

def plot_z_boxplot(dataset, x, y, datasetname, outliers, ax = None):
    columns = [y, x]
    df = pd.DataFrame(data = dataset,
                columns = columns)
    if ax is None:
        sns.boxplot(data = pd.melt(df),
            x = None, y = None,
            linewidth = 0.5,
            flierprops = dict(markerfacecolor = '0.1', markersize = 0.2),
            showfliers = outliers
            )
        # valid_dataset.boxplot(column = ['Predicted z', 'z'])
        plt.title('Distribution of statistical parameters\nfor %s'%datasetname)
        plt.tight_layout()
    else:
        ax.boxplot(data = pd.melt(df),
            x = None, y = None,
            linewidth = 0.5,
            flierprops = dict(markerfacecolor = '0.1', markersize = 0.2),
            showfliers = outliers
            )
        # valid_dataset.boxplot(column = ['Predicted z', 'z'])
        ax.set_title('Distribution of statistical parameters\nfor %s'%datasetname)
        ax.tight_layout()
    plt.show()

###############################################################################

def findCombos(x, xname, bands, names, operation, dataset, threshold, n, calc_tau = False):
    '''
    Find and plot all pair-wise combinations of magnitudes

    Parameters
    ----------
    x;

    xname;

    bands; list of np.arrays; a list of the wavelength bands used

    names; list of str; a list of the names of the wavelength bands

    operation; str; '-', '+', '*' or '/'

    dataset;

    threshold; a float for the minimum value of r^2

    n; a float to represent the minimum value of r^2*n

    calc_tau; Boolean; default = False

    Returns
    -------

    '''
    count = 0
    rVals = []
    combinations = []
    combinationNames = []
    bestComboNames = []
    bestCombosrVals = []
    ranges = []
    bestResults = []
    bestTau = []
    bestr2n = []
    for name1, array1 in zip(names, bands):
        for name2, array2 in zip(names, bands):
            if name1 == name2:
                continue
            else:
                if operation == '-':
                    array = array1 - array2
                    count +=1
                    name = '(' + name1 + '-' + name2 + ')'
                    filename = dataset + '_against_' + xname + '_' + 'sub'
                elif operation == '+':
                    array = array1 + array2
                    count +=1
                    name = '(' + name1 + '+' + name2 + ')'
                    filename = dataset + '_against_' + xname + '_' + 'add'
                elif operation == '*':
                    array = array1 * array2
                    count +=1
                    name = '(' + name1 + '*' + name2 + ')'
                    filename = dataset + '_against_' + xname + '_' + 'mult'
                elif operation == '/':
                    array = array1 / array2
                    count +=1
                    name = '(' + name1 + '/' + name2 + ')'
                    filename = dataset + '_against_' + xname + '_' + 'div'
                if calc_tau:
                    filename = filename + '_' + str(threshold) + '_tau.txt'
                else:
                    filename = filename + '_' + str(threshold) + '.txt'
                results, resids, params, fit, equation, tau = computeLinearStats(x,
                                                                            xname,
                                                                            array,
                                                                            name,
                                                                            calc_tau)
                rValue = params[2]
                combinations.append(array)
                combinationNames.append(name)
                rVals.append(rValue)
                residRange = max(resids) - min(resids)
                ranges.append(residRange)
                r2n = (rValue**2)*results.nobs

                #   Save the combinations with |r-val| > threshold
                if abs(rValue) > threshold and r2n > n:
                    bestComboNames.append(name)
                    bestCombosrVals.append(rValue)
                    bestResults.append(results)
                    bestTau.append(tau[0])
                    bestr2n.append(r2n)
    with open(filename, 'w') as outfile:
        saveBest(bestComboNames, bestCombosrVals, bestResults,
                     bestTau, threshold, n, bestr2n, outfile, calc_tau)

    print("\n%d combinations saved to %s"%(len(bestComboNames), filename))

#    print('Highest r-value of %g occurred for %s'%(max(rVals), combinationNames[rVals.index(max(rVals))]))
#    print('Lowest r-value of %g occurred for %s'%(min(rVals), combinationNames[rVals.index(min(rVals))]))
#    print('Lowest residual range of %g for %s'%(min(ranges), combinationNames[ranges.index(min(ranges))]))
#    print('Highest r^2n of %g for %s'%(max(bestr2n), combinationNames[bestr2n.index(max(bestr2n))]))
#    makeTable(filename[:-4])
    return combinations, combinationNames, rVals, bestResults, bestr2n

###############################################################################

def plotSelected(x, xname, arraysToPlot, arrayNames, dataset, residuals = False,
                 calc_tau = False):
    '''
    Plot magnitude-log(z) graphs for selected arrays.
    Save statistical parameter results for each in stats_results_selected.txt
    '''
    colors = iter(cm.tab20b(np.linspace(0.1,1,len(arraysToPlot)*2)))
    for array, name in zip(arraysToPlot, arrayNames):
        results, resids, params, fit, equation, tau = computeLinearStats(x,
                                                                xname,
                                                                array,
                                                                name,
                                                                calc_tau)
        colour = next(colors)
        graph(x, xname, array, name, dataset, colour, residuals, calc_tau)
        if residuals:
            residuals = True
    return results, resids, params, fit, equation, tau


###############################################################################

def computeLinearStats(x, xname, y, yName, calc_tau = False):
    '''
    Takes as an argument two numpy arrays, one for x and one y, and a string for the
    name of the y-variable, and a boolean for whether to calculate tau.
    Uses Ordinary Least Squares to compute the statistical parameters for the
    array against x, and determines the equation for the line of best fit.
    Returns the results summary, residuals, statistical parameters in a list,
    the best fit equation, and Kendall's tau.
    '''

    #   Mask NaN values in both axes
    mask = ~np.isnan(y) & ~np.isnan(x)
    #   Compute model parameters
    model = sm.OLS(y[mask], sm.add_constant(x[mask]), missing= 'drop')
    results = model.fit()
    residuals = results.resid
    if calc_tau:
        tau = stats.kendalltau(x, y, nan_policy= 'omit')
    else:
        tau = [1, 1]    #   Use this to exclude computation of tau
#

    #   Compute fit parameters
    params = stats.linregress(x[mask], y[mask])
    fit = params[0]*x + params[1]
    fitEquation = '$%s=(%.4g \pm %.4g) \\times$ %s $+%.4g$'%(yName,
                    params[0],  #   slope
                    params[4],  #   stderr in slope
                    xname,
                    params[1])  #   y-intercept
    return results, residuals, params, fit, fitEquation, tau
###############################################################################

def makeArray(dataframe, band):
    '''
    Takes as an argument a pandas dataframe and converts it to a numpy array.
    Replaces any invalid values (defined as value = -999) with NaN and returns
    the resultant array.
    '''
    array = np.array(dataframe[band])
    array[array==-999]=np.nan
    return array
###############################################################################

def graph(x, xname, y, yName, dataset, colour, residuals = False, calc_tau = False):
    '''
    Takes as an argument two arrays, one for x and one y, a string for the
    name of the y-variable, and two colours (which could be strings, values
    or iterators).
    Plots a scatter graph of the array against log(z), along with the best fit
    line, its equation and its regression coefficient, as well as Kendall's tau
    and the total number of points plotted.
    '''

    arrayresults, arrayresids, arrayparams, arrayfit, arrayequation, arraytau = computeLinearStats(x,
                                                                    xname,
                                                                    y,
                                                                    yName,
                                                                    calc_tau)
    count = np.count_nonzero(~np.logical_or(np.isnan(x), np.isnan(y)))
#    arrayequation = 'r\'' + yName +arrayequation[arrayequation.index('='):]
    plt.scatter(x, y,
                label = arrayequation,
                marker = '*',
                s = 30,
                alpha = 1,
                c = colour)
    if calc_tau:                #if calc_tau is set to True, display the value
                                #in the legend along with equation, r and n
        plt.plot(x, arrayfit,
                 label = r'''r=%g, $\tau$=%g, n=%d'''%(arrayparams[2], arraytau[0], count),
                 c = colour)
    else:                       #otherwise just display equation, r and n
        plt.plot(x, arrayfit,
                 label = r'''$r=%g$, $n=%d$, $r^2n=%.2f$'''%(arrayparams[2],
                                count,
                                arrayresults.nobs*arrayparams[2]**2),
                                c = 'r', lw = 5, alpha = 0.8)
    plt.plot(x, arrayfit+np.nanstd(y), 'r:', lw = 5, alpha = 0.8, label = '_nolegend_')
    plt.plot(x, arrayfit-np.nanstd(y), 'r:', lw = 5, alpha = 0.8, label = '_nolegend_')
#    plt.scatter(x, 0.28*x**2+2.48*x+14.12, s=5, c ='k', label = None)
    legendfont = 25
    labelfont = 25
    plt.xlabel(r'%s'%xname, fontsize = labelfont)
    plt.ylabel('%s [mag]'%yName, fontsize = labelfont)
    plt.legend(fontsize = legendfont)
    plt.xticks(fontsize = labelfont)
    plt.yticks(fontsize = labelfont)
    plt.xscale='linear'
    plt.grid(True, which = 'both')
    if residuals:
        plotResids(x, y, yName, dataset, colour)
    fig = plt.gcf()
    width, height = 15,35   #   inches
    fig.set_size_inches(width, height, forward = True)
#    plt.xlim(-0.5)
#    plt.ylim(-.9)
    plt.show()
    return arrayresults, arrayresids, arrayparams, arrayfit, arrayequation, arraytau

###############################################################################

def plotResids(x, xname, y, yName, dataset, colour, axis):
    '''
    Takes as an argument two arrays, one for x and one y, and a string for the
    name of the y-variable.
    Displays a scatter plot of the x and y variables, and a scatter plot of
    the residuals of the y-variable.
    '''

    # fit model
    model = sm.OLS(y, sm.add_constant(x), missing='drop').fit()
    upper = max(model.resid)
    lower = min(model.resid)
    plt.scatter(model.model.exog[:,1], model.resid, s = 10,
                marker = 'o',
                alpha = .5,
                label = '''Range = %g
Average = %.2e'''%(upper - lower, abs(sum(model.resid)/len(model.resid))))

    print('Range of residuals for %s: %g'%(yName, upper - lower))
    print('Mean of residuals for %s : %E'%(yName, sum(model.resid)/len(model.resid)))

#    plt.hlines(y = upper, xmin = min(x), xmax = max(x), alpha = 0.8, lw = 5,
#               color = colour, linestyle = 'dashed')
#    plt.hlines(y = lower, xmin = min(x), xmax = max(x), alpha = 0.8, lw = 5,
#               color = colour, linestyle = 'dashed')
#    plt.vlines(label = 'Range of residuals for %s: %g'%(yName, (upper - lower)),
#               x = min(x), ymin = lower, ymax = upper,
#               color = colour, linestyle = 'solid',
#               arrowprops = {'arrowstyle': '<->'}
#               )
#    plt.annotate('Range of residuals: %g'%(max(model.resid) - min(model.resid)),
#                 xy = (min(x), max(model.resid)),
#                 xytext=(min(x), max(model.resid))
#                )
#    plt.annotate("",
#            xy=(min(x), lower), xycoords='data',
#            xytext=(min(x), upper), textcoords='data',
#            arrowprops=dict(arrowstyle="<|-|>",
#                            connectionstyle="arc3", color=colour, lw=5, alpha = 0.8),
#            )

#    plt.text(1.05*min(x), .5*upper, 'Range: %g'%(upper - lower),
#         rotation = 90, fontsize = 9)
    legendfont = 25
    labelfont = 25
    plt.xlabel('%s'%xname, fontsize = labelfont)
    plt.ylabel('%s residuals'%yName, fontsize = labelfont)
#    plt.legend(fontsize = legendfont)
    plt.xticks(fontsize = labelfont)
    plt.yticks(fontsize = labelfont)
    plt.grid(True, which = 'both')
    plt.show()

###############################################################################

def ccdiagram(x, xName, y, yName):

    labelfont = 20
    plt.xlabel(xName, fontsize = labelfont)
    plt.ylabel(yName, fontsize = labelfont)
    count = np.count_nonzero(~np.logical_or(np.isnan(x), np.isnan(y)))
    plt.scatter(x, y, s = 10, alpha = 0.3)
    plt.xticks(fontsize = labelfont)
    plt.yticks(fontsize = labelfont)
    plt.grid(True, which = 'both')

###############################################################################

def saveStats(xname, arrayname, arrayresults, arrayparams,
              arrayfit, arrayequation, arraytau, filename):
    '''
     Writes the statistical parameters and best fit equation to a text file.
    '''
    r2n = arrayresults.nobs*arrayparams[2]**2
    lineTitle = "Results summary for %s\n" % arrayname
    filename.write(lineTitle)
    filename.write('''Straight line fit:
%s=(%.4g +/- %.4g)%s+%.4g
r=%g
r^2n=%g
Kendall's tau=%g\n'''%(arrayname,
                    arrayparams[0],  #   slope
                    arrayparams[4],  #   stderr in slope
                    xname,
                    arrayparams[1],  #   y-intercept
                    arrayparams[2],  #   r-value
                    r2n,             #   r^2n
                    arraytau[0]))    #    Kendall's tau
    filename.write(str(arrayresults.summary()))
    filename.write('\n')
    filename.write(("="*len(lineTitle)*3 + '\n') + '\n')

###############################################################################

def marginal(x, xname, y, yName, colour1, colour2, calc_tau):
    '''
    Takes as an argument two arrays, one for x and one y, a string for the
    name of the y-variable, and two colours (which could be strings, values
    or iterators).
    Plots a scatter plot of the x and y variables, along with the marginal
    PDFs of each variable.
    '''
    nullfmt = NullFormatter()         # no labels
    arrayresults,arrayresids,arrayparams,arrayfit,arrayequation,arraytau = computeLinearStats(x,
                                                                    xname,
                                                                    y,
                                                                    yName,
                                                                    calc_tau)

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_plot = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axPlot = plt.axes(rect_plot)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y, s = 0.5, alpha = 0.5,
#                      label = arrayequation,
                      c = colour1)
    axPlot.plot(x, arrayfit,
             label = '%s\nr=%g'%(arrayequation, arrayparams[2]),
             alpha = 0.3,
             c = colour2)

    # now determine nice limits:
    y_lower = min(y[~np.isnan(y)])
    y_upper = max(y[~np.isnan(y)])

    axScatter.set_xlim(min(x), max(x))
    axScatter.set_ylim(y_lower, y_upper)
    axScatter.legend()

    bins = 1000
    axHistx.hist(x[~np.isnan(x)], bins=bins)
    axHisty.hist(y[~np.isnan(y)], bins=bins, orientation='horizontal', label = 'y')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    plt.show()
###############################################################################

def saveBest(bestNames, bestrVals, bestResults, bestTau, threshold, n, bestr2n,
                                                         filename, calc_tau):
    '''
    Saves a list of the highest r-value results for a combination of magnitudes.
    '''
    try:
        if calc_tau:
            filename.write('''%d combinations with |r-value| > %g and r^2.n > %d
Combo\t&\t$|r|>%g$\t&\t$n$\t&\t$r^2n>%d\t&\ttau
=============================================\n'''%(len(bestNames), threshold, n, threshold, n))
            for bestname, bestvalue, bestResult, bestTau in zip(bestNames, bestrVals, bestResults, bestTau):
                    filename.write('%s\t&\t%.6f\t&\t%d\t&\t%g\t&\t%g\n'%(bestname,
                                                                         bestvalue,
                                                                         bestResult.nobs,
                                                                         (bestvalue**2)*bestResult.nobs,
                                                                         bestTau))
        else:
            filename.write('''%d combinations with |r-value| > %g and r^2.n > %d
Combo\t&\t$|r|>%g$\t&\t$n$\t&\t$r^2n > %d$
=============================================\n'''%(len(bestNames), threshold, n, threshold, n))
            for bestname, bestvalue, bestResult, bestTau in zip(bestNames, bestrVals, bestResults, bestTau):
                filename.write('%s\t&\t%.6f\t&\t%d\t&\t%g\n'%(bestname, bestvalue, bestResult.nobs, (float(bestvalue)**2)*int(bestResult.nobs)))
        filename.write('\nHighest r-value of %g occurred for %s\n'%(max(bestrVals), bestNames[bestrVals.index(max(bestrVals))]))
        filename.write('Lowest r-value of %g occurred for %s\n'%(min(bestrVals), bestNames[bestrVals.index(min(bestrVals))]))
        filename.write('Highest r^2n of %g for %s'%(max(bestr2n), bestNames[bestr2n.index(max(bestr2n))]))
        print('Highest r-value of %g occurred for %s'%(max(bestrVals), bestNames[bestrVals.index(max(bestrVals))]))
        print('Lowest r-value of %g occurred for %s'%(min(bestrVals), bestNames[bestrVals.index(min(bestrVals))]))
    except ValueError:
        print('\nThere were no r-values above %g\n'%threshold)
###############################################################################

def makeTable(file):
    '''
    Reformats a list produced by saveBest() to LaTex format for inclusion in
    final report.
    '''
    filename = file+'.txt'
    with open(filename, 'r') as infile:
        new_filename = 'Tables/' + file+'_table.txt'
        with open(new_filename, 'w') as outfile:
            lines = infile.readlines()
            outfile.write(r'''\begin{center}
\addtolength{\tabcolsep}{-0.3pt}
\sisetup{group-digits=integer}
\begin{tabular}{
    @{}
    l
    S[table-format=+1.6]
    S[table-format=5]
    S[table-format=4.4]
    @{}
}
\hline
Combo	&	{$|r|>0.5$}	&	{$n$}	&	{$r^2n > 1000$}\\
\hline''' + '\n')
            for line in lines[3:-3]:
                line = line.replace('\n', ' '+r'\\')
                outfile.write(line)
                outfile.write('\n')
            outfile.write(r'\hline' + '\n')
            outfile.write(r'\end{tabular}' + '\n')
            outfile.write(r'\end{center}' + '\n')
            for line in lines[-3:]:
                line = line.replace('\n', ' '+r'\\')
                outfile.write(line)
                outfile.write('\n')
            outfile.write(r'''\\
\captionof{table}[]{}
\label{}
\end{center}
\vfill\null



	\end{tabular}
\vfill\null
\columnbreak
\begin{tabular}{
	@{}
	l
	S[table-format=+1.6]
	S[table-format=5]
	S[table-format=4]
	@{}
}
\hline
Combo	&	{$|r|>0.5$}	&	{$n$}	&	{$r^2n$}\\
\hline
& {cont ...} 	&			&			\\''')
###############################################################################

#def plotRedshiftPredictions0(x1, y1, x1name, y1name, x2, y2,  x2name, y2name, equation, combination):
#    gridsize = (1,2)
#    labelfont = 20
#    titlefont = 20
#    plt.subplot2grid(gridsize, (0,0))
#    plt.title("Redshift prediction for\n%s"%combination, fontsize = titlefont)
#    plt.scatter(x1, y1, s = 10, alpha = 0.5)
#    plt.xlabel(x1name, fontsize = labelfont)
#    plt.ylabel(y1name, fontsize = labelfont)
#    plt.subplot2grid(gridsize, (0,1))
#    plt.title("Percentage difference between\nspectroscopic and predicted redshifts for\n%s"%equation, fontsize = titlefont)
#    plt.scatter(x2, y2, s = 10, alpha = 0.5)
#    plt.xlabel(x2name, fontsize = labelfont)
#    plt.ylabel(y2name, fontsize = labelfont)
###############################################################################

def zPredictions(x1, y1, x1name, y1name, x2, y2, x2name, y2name, equation, combination, save=False):
    labelfont = 25
    titlefont = 25
    fig1 = plt.figure()

    ax1 = fig1.add_subplot(111)
    ax1.scatter(x1, y1, alpha = 0.5, s = 30,
#                c= 'b', marker = 'o'
                )
    ax1.plot(x1, x1, c = 'g', ls = '-', lw = 5, alpha = 0.8)
    ax1.plot(x1, x1+np.nanstd(y1), c = 'green',  ls = ':', lw = 5, alpha = 0.8)
    ax1.plot(x1, x1-np.nanstd(y1), c = 'green',  ls = ':', lw = 5, alpha = 0.8)
    ax1.set_xlabel(x1name, fontsize = labelfont)
    ax1.set_ylabel(y1name, fontsize = labelfont)
    ax1.grid(True, linestyle = '-', which = 'both')
#    ax1.set_title("Redshift prediction for\n%s"%combination, fontsize = titlefont)
    ax1.set_xlim(min(x1), max(x1))
    ax1.set_yscale('linear')
    ax1.set_xscale('linear')
    ax1.tick_params(labelsize=labelfont)

#    ax2 = fig1.add_subplot(122)
#    ax2.set_title("Percentage difference between\nspectroscopic and predicted redshifts for\n%s"%combination, fontsize = titlefont)
#    ax2.scatter(x2, y2, alpha = 0.5, s = 30)
#    ax2.grid(True, linestyle = '-')
#    ax2.set_xlabel(x2name, fontsize = labelfont)
#    ax2.set_ylabel(y2name, fontsize = labelfont)
#    ax1.set_xlim(-0.1, max(x1))
#    ax1.set_ylim(-10, 40)
    fig = plt.gcf()
    width, height = 15,35   #   inches
    fig.set_size_inches(width, height, forward = True)
    count = np.count_nonzero((~np.isnan(y1)))
    plt.annotate('$n = %d$, $\sigma=%.4f$'%(count, np.nanstd(y1)), fontsize = 25, color = 'r', xy=(0.05, 0.01),
                 xycoords='axes fraction')
    plt.show()
    return np.nanstd(y1)
 ###############################################################################

def binning(col, cut_points, labels = None):
    #   Define min and max values:
    minval = col.min()
    maxval = col.max()
    #   Create list by adding min and max to cut_points
    break_points = [minval] + cut_points + [maxval]

    #   If no labels provided, use default labels 0 ... (n-1)
    if not labels:
        labels = range(len(cut_points)+1)

    #   Binning using cut function of pandas
    colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True, duplicates = 'drop')
    return colBin
###############################################################################

def compute_spearman(x, y):
    scorr, sp_value = spearmanr(x, y, nan_policy='omit')
    chisq, chip = chisquare(x, y)
    print('Spearman correlation = %g, p-value = %g'%(scorr, sp_value))
    print('chi squared = %g, chi p-value = %g'%(chisq, chip))
    return scorr, sp_value, chisq, chip
###############################################################################

def subgraph(x, xname, y, yname, dataset, colour, ax, residuals = False, calc_tau = False):
    '''
    Takes as an argument two arrays, one for x and one y, a string for the
    name of the y-variable, and two colours (which could be strings, values
    or iterators).
    Plots a scatter graph of the array against log(z), along with the best fit
    line, its equation and its regression coefficient, as well as Kendall's tau
    and the total number of points plotted.
    '''

    arrayresults, arrayresids, arrayparams, arrayfit, arrayequation, arraytau = computeLinearStats(x,
                                                                    xname,
                                                                    y,
                                                                    yname,
                                                                    calc_tau)
    count = np.count_nonzero(~np.logical_or(np.isnan(x), np.isnan(y)))
#    arrayequation = 'r\'' + yName +arrayequation[arrayequation.index('='):]
    ax.scatter(x, y,
#                label = arrayequation,
                s = 10,
                alpha = 0.5,
                c = colour)
    if calc_tau:                #if calc_tau is set to True, display the value
                                #in the legend along with equation, r and n
        ax.plot(x, arrayfit,
                 label = r'''$%s$,
r=%g
$\tau$=%g
n=%d'''%(arrayequation, arrayparams[2], arraytau[0], count),
                 c = colour)
    else:                       #otherwise just display equation, r and n
        ax.plot(x, arrayfit,
#                marker='.', s = 1,
                 label = r'''$r=%g$
$n=%d$
$r^2n=%.2f$'''%(arrayparams[2], count, arrayresults.nobs*arrayparams[2]**2),
                 c = 'r', lw = 5, alpha = 0.8)
    ax.plot(x, arrayfit+np.nanstd(y), 'r:', lw = 5, alpha = 0.8, label = '_nolegend_')
    ax.plot(x, arrayfit-np.nanstd(y), 'r:', lw = 5, alpha = 0.8, label = '_nolegend_')
    legendfont = 20
    labelfont = 25
    ax.tick_params(axis = 'both', labelsize=labelfont)
    ax.set_xlabel(r'%s'%xname, fontsize = labelfont)
    ax.set_ylabel('%s'%yname, fontsize = 20)
    ax.legend(fontsize = legendfont, loc = 'best')
#    ax.set_xticks(xticks)
#    ax.set_xticklabels(fontsize = labelfont)
#    ax.set_yticks(yticks)
#    ax.set_yticklabels(fontsize = labelfont)
#    ax.set_ylim(top = 11)
    ax.grid(True, which = 'both')
    if residuals:
        plotResids(x, y, yname, dataset, colour, ax)
    fig = plt.gcf()
    width, height = 15,15   #   inches
    fig.set_size_inches(width, height, forward = True)
    plt.show()
    return arrayresults, arrayresids, arrayparams, arrayfit, arrayequation, arraytau
###############################################################################

def redshiftEvolution(x, xname, y, yname):
    sources['z bins']=pd.cut(sources['z'], [0,1,2,3, max(z)],
                   labels = ['z < 1', '1 < z < 2', '2 < z < 3', 'z > 3'])
    fig = plt.figure()
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    gridsize = 1,4

    ax1 = plt.subplot2grid(gridsize, (0,0))
    ax1.scatter(x, y, s = 10, alpha = 0.5)
    ax1.set_ylabel('yname', fontsize=20)
    ax1.set_xlabel('xname', fontsize=20)
    ax1.grid()

    ax2 = plt.subplot2grid(gridsize, (0,1), sharey=ax1, sharex = ax1)
    ax2.scatter(x, y, s = 10, alpha = 0.5)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlabel('xname', fontsize=20)
    ax2.grid()

    ax3 = plt.subplot2grid(gridsize, (0,2), sharey=ax1, sharex = ax1)
    ax3.scatter(x, y, s = 10, alpha = 0.5)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.set_xlabel('xname', fontsize=20)
    ax3.grid()

    ax4 = plt.subplot2grid(gridsize, (0,3), sharey=ax1, sharex = ax1)
    ax4.scatter(x, y, s = 10, alpha = 0.5)
    plt.setp(ax4.get_yticklabels(), visible=False)
    ax4.set_xlabel('xname', fontsize=20)
    ax4.grid()

###############################################################################

def fixDat(file):
    '''
    Removes extra spaces in the data files. Replaces original file with new
    and renames original to "...._original.dat".
    '''

    import re
    with open(file+'.dat', 'r') as infile:
        with open(file+'_fixed.dat', 'w') as outfile:
            lines = infile.readlines()
            for line in lines:
                fixed = re.sub("\s\s+" , " ", line)
                outfile.write(fixed)

    os.rename(file+'.dat', file+'_original.dat')
    os.rename(file+'_fixed.dat', file+'.dat')

###############################################################################
