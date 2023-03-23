import pandas as pd
import numpy as np
import missingno as msno  # https://github.com/ResidentMario/missingno
from sklearn.impute import SimpleImputer


class DataLoader:
    '''
    Example syntax:

    # import sys; sys.path.insert(0, 'D:/Dropbox/Jim/Astro_at_VUW/PhD_stuff/code')
    from DataLoader import DataLoader
    dl = DataLoader(dropna = False,
                                colours = False,
                                impute_method = 'max')
    dataset, datasetname, magnames, mags = data_loader.load_data('test')
    '''

    def __init__(self, colours=False,
                 impute_method=None,
                 cols=None,
                 dropna=True,
                 number_of_rows='all'):

        self.colours = colours
        self.impute_method = impute_method
        self.cols = cols
        self.dropna = dropna
        self.number_of_rows = number_of_rows

    def compute_colours(self, magnitudes):
        '''Computes colours (or any other combination) of magnitudes.
        '''
        comb = [(x, magnitudes.columns[(i+1) % len(magnitudes.columns)])
                for i, x in enumerate(magnitudes.columns)]
        for x, y in comb:
            magnitudes[f'{x} - {y}'] = magnitudes[x] - magnitudes[y]

        return magnitudes

    def load_data(self, name):
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
        self.name = name
        path = r'../../data_files'

        # %%% Clarke's 2.1 million SDSS Quasar Catalogue
        if self.name.lower() == 'clarke':  # 2.1 million
            redshift = False
            # fields = [,'objid','ra','dec','psf_u','psf_g','psf_r','psf_i','psf_z',
#             'w1','w2','w3','w4','resolvedr','class_pred','class_prob_galaxy',
#             'class_prob_quasar','class_prob_star']
            magnames = ['u', 'g', 'r', 'i', 'zmag',
                        'W1', 'W2'
                        # , 'w3', 'w4'
                        ]
            df = pd.read_csv(path + '/SDSS/SDSS-ML-quasars.csv',
                             sep=',', index_col=False)
            df.rename(columns={'psf_u': 'u',
                               'psf_g': 'g',
                               'psf_r': 'r',
                               'psf_i': 'i',
                               'psf_z': 'zmag',
                               'w1': 'W1',
                               'w2': 'W2'
                               }, inplace=True)
            datasetname = 'Clarke Catalogue'
# %%% MilliQuas x GLEAM SED fits
        if self.name.lower() == 'milli_x_gleam_fits':  # 999999
            redshift = True
            # fields = [_RAJ2000,_DEJ2000,ObjectId,RAICRS,DEICRS,e_RAICRS,e_DEICRS,SMSS,EpMean,flags,ClassStar,RadrPetro,a,b,PA,uPSF,uPetro,vPSF,vPetro,gPSF,gPetro,rPSF,rPetro,iPSF,iPetro,zPSF,zPetro,(u-v)PSF,(u-g)PSF,(g-r)PSF,(g-i)PSF,(i-z)PSF]
            magnames = ['Rmag', 'Bmag',
                        'Fp076', 'Fp084', 'Fp092', 'Fp099', 'Fp107', 'Fp115', 'Fp122',
                        'Fp130', 'Fp143', 'Fp151', 'Fp158', 'Fp166', 'Fp174', 'Fp181',
                        'Fp189', 'Fp197', 'Fp204', 'Fp212', 'Fp220', 'Fp227',
                        'alpha', 'my alpha', 'alpha_thin', 'alpha_thick', 'log nu_TO', 'log F_p'
                        ]
            df = pd.read_csv(path + '/x-matches/milliquas_x_gleam_fits_final.csv',
                             sep=',', index_col=False)
            if redshift:
                df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'MilliQuas x GLEAM SED fits'
    # %%% Skymapper x WISE
        if self.name.lower() == 'skymapper_wise':  # 999999
            redshift = False
            # fields = [_RAJ2000,_DEJ2000,ObjectId,RAICRS,DEICRS,e_RAICRS,e_DEICRS,SMSS,EpMean,flags,ClassStar,RadrPetro,a,b,PA,uPSF,uPetro,vPSF,vPetro,gPSF,gPetro,rPSF,rPetro,iPSF,iPetro,zPSF,zPetro,(u-v)PSF,(u-g)PSF,(g-r)PSF,(g-i)PSF,(i-z)PSF]
            magnames = ['u', 'g', 'r', 'i', 'zmag', 'W1', 'W2'
                        ]
            df = pd.read_csv(path + '/x-matches/skymapper_wise.csv',
                             sep=',', index_col=False,
                             usecols=['uPetro', 'gPetro', 'rPetro', 'iPetro', 'zPetro',
                                      'W1mag', 'W2mag'
                                      ])
            df.rename(columns={'uPetro': 'u',
                               'gPetro': 'g',
                               'rPetro': 'r',
                               'iPetro': 'i',
                               'zPetro': 'zmag',
                               'W1mag': 'W1',
                               'W2mag': 'W2'
                               }, inplace=True)
            datasetname = 'Skymapper'
    # %%% Skymapper
        if self.name.lower() == 'skymapper':  # 999999
            redshift = False
            # fields = [_RAJ2000,_DEJ2000,ObjectId,RAICRS,DEICRS,e_RAICRS,e_DEICRS,SMSS,EpMean,flags,ClassStar,RadrPetro,a,b,PA,uPSF,uPetro,vPSF,vPetro,gPSF,gPetro,rPSF,rPetro,iPSF,iPetro,zPSF,zPetro,(u-v)PSF,(u-g)PSF,(g-r)PSF,(g-i)PSF,(i-z)PSF]
            magnames = ['u', 'g', 'r',
                        'i', 'zmag'
                        ]
            df = pd.read_csv(path + '/Skymapper/vizier_II_358_smss_20230207_2.csv',
                             sep=',', index_col=False,
                             usecols=['uPetro', 'gPetro', 'rPetro', 'iPetro', 'zPetro'])
            df.rename(columns={'uPetro': 'u',
                               'gPetro': 'g',
                               'rPetro': 'r',
                               'iPetro': 'i',
                               'zPetro': 'zmag', }, inplace=True)
            datasetname = 'Skymapper'

    # %%% Processed MQ x GLEAM with radio spectral fits
        if self.name.lower() == 'mq_processed':  # 9276
            redshift = True
            # fields = [_RAJ2000_1,_DEJ2000_1,GLEAM,RAJ2000_1,DEJ2000_1,Fpwide,Fintwide,eabsFpct,efitFpct,Fp076,Fint076,Fp084,Fint084,Fp092,Fint092,Fp099,Fint099,Fp107,Fint107,Fp115,Fint115,Fp122,Fint122,Fp130,Fint130,Fp143,Fint143,Fp151,Fint151,Fp158,Fint158,Fp166,Fint166,Fp174,Fint174,Fp181,Fint181,Fp189,Fint189,Fp197,Fint197,Fp204,Fint204,Fp212,Fint212,Fp220,Fint220,Fp227,Fint227,alpha,Fintfit200,_RAJ2000_2,_DEJ2000_2,recno,RAJ2000_2,DEJ2000_2,Name,Type,Rmag,Bmag,Comment,R,B,z,Qpct,XName,RName,Separation]
            magnames = [
                # 'my alpha',
                'alpha_thin', 'alpha_thick', 'log nu_TO', 'log F_p'
            ]
            df = pd.read_csv(path + '/x-matches/milliquas_x_gleam_fits_final.csv',
                             sep=',', index_col=False,
                             usecols=self.cols)
            df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'GLEAM x Milliquas spectral fits'

    # %%% Spectroscopically-confirmed QSOs from milliquas
        if self.name.lower() == 'radio_z_mq':  # 45318
            redshift = True
            # fields = [_RAJ2000_1,_DEJ2000_1,GLEAM,RAJ2000_1,DEJ2000_1,Fpwide,Fintwide,eabsFpct,efitFpct,Fp076,Fint076,Fp084,Fint084,Fp092,Fint092,Fp099,Fint099,Fp107,Fint107,Fp115,Fint115,Fp122,Fint122,Fp130,Fint130,Fp143,Fint143,Fp151,Fint151,Fp158,Fint158,Fp166,Fint166,Fp174,Fint174,Fp181,Fint181,Fp189,Fint189,Fp197,Fint197,Fp204,Fint204,Fp212,Fint212,Fp220,Fint220,Fp227,Fint227,alpha,Fintfit200,_RAJ2000_2,_DEJ2000_2,recno,RAJ2000_2,DEJ2000_2,Name,Type,Rmag,Bmag,Comment,R,B,z,Qpct,XName,RName,Separation]
            magnames = ['Rmag', 'Bmag']
            # cols = ['u_mag','g_mag','r_mag','i_mag','z_mag',
            #           'I_mag','J_mag','H_mag','K_mag','W1_mag','SPIT_5_mag','W2_mag',
            #           'SPIT_8_mag','W3_mag','W4_mag','NUV_mag','FUV_mag', 'redshift']
            df = pd.read_csv(path + '/radio_z_mq.csv',
                             sep=',', index_col=False,
                             usecols=self.cols)
            df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'GLEAM x Milliquas'

    # %%% GLEAM x Milliquas
    # First 50,000 GALEX QSOs with SDSS and WISE
        if self.name.lower() == 'gleam_x_milliquas':  # 3513
            redshift = True
            # fields = [_RAJ2000_1,_DEJ2000_1,GLEAM,RAJ2000_1,DEJ2000_1,Fpwide,Fintwide,eabsFpct,efitFpct,Fp076,Fint076,Fp084,Fint084,Fp092,Fint092,Fp099,Fint099,Fp107,Fint107,Fp115,Fint115,Fp122,Fint122,Fp130,Fint130,Fp143,Fint143,Fp151,Fint151,Fp158,Fint158,Fp166,Fint166,Fp174,Fint174,Fp181,Fint181,Fp189,Fint189,Fp197,Fint197,Fp204,Fint204,Fp212,Fint212,Fp220,Fint220,Fp227,Fint227,alpha,Fintfit200,_RAJ2000_2,_DEJ2000_2,recno,RAJ2000_2,DEJ2000_2,Name,Type,Rmag,Bmag,Comment,R,B,z,Qpct,XName,RName,Separation]
            magnames = ['Rmag', 'Bmag', 'Fp076', 'Fp084', 'Fp092',
                        'Fp099', 'Fp107', 'Fp115',
                        'Fp122', 'Fp130', 'Fp143', 'Fp151',
                        'Fp158', 'Fp166', 'Fp174',
                        'Fp181', 'Fp189', 'Fp197', 'Fp204',
                        'Fp212', 'Fp220', 'Fp227',
                        # 'alpha'
                        ]
            # cols = ['u_mag','g_mag','r_mag','i_mag','z_mag',
            #           'I_mag','J_mag','H_mag','K_mag','W1_mag','SPIT_5_mag','W2_mag',
            #           'SPIT_8_mag','W3_mag','W4_mag','NUV_mag','FUV_mag', 'redshift']
            df = pd.read_csv(path + '/x-matches/gleam_x_milliquas.csv',
                             sep=',', index_col=False,
                             usecols=self.cols)
            df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'GLEAM x Milliquas'

    # %%% GALEX QSO
    # First 50,000 GALEX QSOs with SDSS and WISE
    # I'm pretty sure this is the same dataset as 'sdssmags', but sdssmags has z
    # I think this is the one I made when I tried to replicate Steve's results.
        if self.name.lower() == 'galexqso':  # 33577
            redshift = False
            fields = ['no', 'NED', 'redshift', 'ez', 'type', 'class', 'no_radio', 'radio_top',
                      'no_UV', 'uv_bottom', 'u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag',
                      'I_mag', 'J_mag', 'H_mag', 'K_mag', 'W1_mag', 'SPIT_5_mag', 'W2_mag',
                      'SPIT_8_mag', 'W3_mag', 'W4_mag', 'NUV_mag', 'FUV_mag']
            magnames = [
                'u', 'g', 'r', 'i', 'zmag',  # SDSS
                # 'I_mag',
                # 'J_mag','H_mag','K_mag',         # 2MASS
                'W1', 'W2',
                # 'W3_mag','W4_mag',     # WISE
                # 'NUV_mag','FUV_mag'                      # GALEX
            ]
            # cols = ['u_mag','g_mag','r_mag','i_mag','z_mag',
            #           'I_mag','J_mag','H_mag','K_mag','W1_mag','SPIT_5_mag','W2_mag',
            #           'SPIT_8_mag','W3_mag','W4_mag','NUV_mag','FUV_mag', 'redshift']
            df = pd.read_csv(path + '/GALEX/QSOs_1st_50k-mags_GALEX-fixed.dat.dat',
                             sep=' ', index_col=False, names=fields)
            df.rename(columns={'u_mag': 'u',
                               'g_mag': 'g',
                               'r_mag': 'r',
                               'i_mag': 'i',
                               'z_mag': 'zmag',
                               'W1_mag': 'W1',
                               'W2_mag': 'W2'
                               }, inplace=True)
            datasetname = 'First 50,000 GALEX QSOs with SDSS and WISE'

    # %%% sdssmags
    # I think this is the same as 'galexqso'.
        if self.name.lower() == 'sdssmags':  # 26301
            '''
            I think this data is from SDSS, WISE and GALEX, but I'm not sure
            what processing has been done on it. Steve gave me the data, but
            he doesn't remember. He refers to it in an email on 26/11/2020:
              | I've attached my python code and the data file it used
              | (this is produced by an awk script which pulls the relevent
              | fields from the master file which I've sent before).
            '''
            redshift = True
            fields = ['redshift', 'u', 'g', 'r', 'i', 'zmag', 'W1', 'W2'
                      # 'NUV','FUV'
                      ]
            magnames = ['u', 'g', 'r', 'i', 'zmag', 'W1', 'W2'
                        # 'NUV','FUV'
                        ]
            df = pd.read_csv(path + '/SDSS/SDSS-mags.dat',
                             sep=' ', names=fields, index_col=False,
                             usecols=self.cols)
            datasetname = r'SDSS $\times$ WISE'
            datasetname = "Steve's SDSS data"

    # %%% GALEX first 10 million
        if self.name.lower() == 'galex10k':  # 999999
            redshift = False
            magnames = [
                # 'FUV', Very few measurements
                'NUV']
            df = pd.read_csv(path + '/GALEX/vizier_II_312_ais_20220606_10million.csv',
                             sep=',', index_col=False,
                             usecols=self.cols)
            datasetname = 'GALEX'
            # if redshift:
            #     df.rename(columns={'zsp':'redshift'}, inplace=True)

    # %%% SDSS16QSO with WISE mags
        if self.name.lower() == 'sdss16qso':  # 750404
            redshift = True
            fields = ['SDSS', 'RAJ2000', 'DEJ2000', 'Plate', 'MJD', 'Fiber', 'Class', 'QSO',
                      'z', 'r_z', 'umag', 'gmag', 'rmag', 'imag', 'zmag',
                      'Extu', 'Extg', 'Extr', 'Exti', 'Extz',
                      'FFUV', 'FNUV', 'FY', 'FJ', 'FH', 'FK', 'FW1',
                      'W1mag', 'fracW1', 'FW2', 'W2mag', 'fracW2',
                      'Jmag', 'Hmag', 'Kmag', '2RXS', 'Sp', 'Simbad', 'recno'
                      ]
            magnames = ['g', 'r', 'i', 'zmag',
                        'W1', 'W2'
                        ]
            df = pd.read_csv(path + '/SDSS/vizier_VII_289_dr16q_20230217_WISE.csv',
                             sep=',', names=fields, index_col=False,
                             header=0,
                             usecols=self.cols)
            df.rename(columns={'umag': 'u',
                               'gmag': 'g',
                               'rmag': 'r',
                               'imag': 'i',
                               'zmag': 'zmag',
                               'W1mag': 'W1',
                               'W2mag': 'W2'}, inplace=True)
            df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'SDSS DR16Q'

    # %%% SDSS16QSO
        if self.name.lower() == 'sdss16qso*':  # 750414
            redshift = True
            fields = ['_RAJ2000', '_DEJ2000', 'recno', 'SDSS', 'RAJ2000', 'DEJ2000',
                      'Plate', 'MJD', 'Fiber', 'Class', 'QSO', 'z', 'r_z', 'umag', 'gmag',
                      'rmag', 'imag', 'zmag', 'e_umag', 'e_gmag', 'e_rmag', 'e_imag',
                      'e_zmag', 'Extu', 'Extg', 'Extr', 'Exti', 'Extz', 'FFUV', 'FNUV',
                      'FY', 'FJ', 'FH', 'FK', 'FW1', 'fracW1', 'FW2', 'fracW2', '2RXS',
                      'Gaia', 'Sp', 'Simbad']
            magnames = ['umag', 'gmag', 'rmag', 'imag', 'zmag', 'FFUV', 'FNUV',
                        # 'FY','FJ','FH','FK',
                        'FW1', 'FW2']
            df = pd.read_csv(path + '/SDSS/vizier_VII_289_dr16q_20210423.csv',
                             sep=',', names=fields, index_col=False,
                             header=0,
                             usecols=self.cols)
            df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'SDSS quasar catalog, sixteenth data release (DR16Q)'

    # %%% SDSS9 QSO
        if self.name.lower() == 'sdss9qso':  # 230096
            redshift = True
            fields = ['_RAJ2000', '_DEJ2000', 'mode', 'q_mode', 'cl', 'SDSS9', 'm_SDSS9',
                      'Im', 'objID', 'RA_ICRS', 'DE_ICRS', 'ObsDate', 'Q', 'umag', 'e_umag',
                      'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 'zsp']
            magnames = ['umag', 'gmag', 'rmag', 'imag', 'zmag']
            # cols = ('umag', 'gmag', 'rmag', 'imag', 'zmag', 'zsp')
            df = pd.read_csv(path + '/SDSS/vizier_V_139_sdss9_20201125.csv',
                             sep=',', header=0, index_col=False,
                             usecols=self.cols)
            df = df.replace(-9999, np.nan)
            df.rename(columns={'zsp': 'redshift'}, inplace=True)
            # df = df.drop(['q_mode', 'm_SDSS9', 'objID'], axis = 1)
            datasetname = 'SDSS Catalog, Data Release 9 QSOs'
    # %%% SDSS12
        if self.name.lower() == 'sdss12':  # 43102
            redshift = True
            magnames = ['umag', 'gmag', 'rmag', 'imag', 'zmag']
            df = pd.read_csv(path + '/SDSS/vizier_V_147_sdss12_20200818.csv',
                             sep=',', index_col=False,
                             usecols=self.cols)
            datasetname = 'SDSS DR12'
            df = df.drop(columns=['_RAJ2000', '_DEJ2000'])
            df = df[df['zph'] > 0]
            df = df[df['<zph>'] > 0]
            if redshift:
                df.rename(columns={'zsp': 'redshift'}, inplace=True)

    # %%% SDSS12 QSO
        if self.name.lower() == 'sdss12qso':  # 439273
            redshift = True
            #   QSOs with dz < 0.01 from SDSS DR12
            magnames = ['umag', 'gmag', 'rmag', 'imag', 'zmag']
            df = pd.read_csv(path + '/SDSS/sdss12qso.csv',
                             sep=',',
                             # header = 0,
                             index_col=False,
                             usecols=self.cols)
            # df = df.drop(columns = ['_RAJ2000' '_DEJ2000'])
            # df = df[df['zph']!=-9999]
            df.rename(columns={'zsp': 'redshift'}, inplace=True)
            # df = df.drop(['q_mode', 'm_SDSS12'], axis = 1)
            datasetname = 'SDSS DR12 QSOs'

    # %%% SDSS12 Spectroscopic sources
        if self.name.lower() == 'sdss12spec':  # 8857
            redshift = True
            magnames = ['umag', 'gmag', 'rmag', 'imag', 'zmag']
            df = pd.read_csv(path + '/SDSS/sdss12_spec_sources_DR16_quasars.csv',
                             sep=',',
                             header=0,
                             index_col=False,
                             usecols=self.cols)
            # df = df[df['zph']!=-9999]
            df.rename(columns={'zsp': 'redshift'}, inplace=True)
            # df = df.drop(['q_mode', 'm_SDSS12'], axis = 1)
            datasetname = 'Spectroscopic sources from SDSS DR12 also in DR16Q'

    # %%% SDSS12 filtered for good zsp and with PSF and Petrosian mags
        if self.name.lower() == 'sdss12_petrosian':  # 3118641
            redshift = True
            magnames = [
                # "u'mag","g'mag","r'mag","i'mag","z'mag",
                # "upmag","gpmag","rpmag","ipmag","zpmag",
                "u", "g", "r", "i", "zmag",

            ]
            df = pd.read_csv(path + '/SDSS/vizier_V_147_sdss12_Pet_PSF.csv',
                             sep=',',
                             header=0,
                             index_col=False,
                             usecols=self.cols,
                             # nrows = number_of_rows
                             )
            # df = df[df['zph']!=-9999]
            df.rename(columns={'zsp': 'redshift'}, inplace=True)
            df.rename(columns={'uPmag': 'u',
                               'gPmag': 'g',
                               'rPmag': 'r',
                               'iPmag': 'i',
                               'zPmag': 'zmag'
                               }, inplace=True)
            # df = df.drop(['q_mode', 'm_SDSS12'], axis = 1)
            datasetname = 'SDSS12 with Petrosian and PSF mags'

    # %%% GLEAM x SDSS16
        if self.name.lower() == 'gleam_x_sdss16':  # 420
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
            magnames = ['Fp076', 'Fp084', 'Fp092', 'Fp099', 'Fp107',
                        'Fp115', 'Fint115', 'Fp122', 'Fint122', 'Fp130', 'Fint130', 'Fp143',
                        'Fp151', 'Fp158', 'Fp166',
                        'Fp174', 'Fp181', 'Fp189', 'Fp197',
                        'Fp204', 'Fp212', 'Fp220',
                        'Fp227', 'alpha', 'umag', 'gmag', 'rmag', 'imag', 'zmag']
            df = pd.read_csv(path + '/x-matches/gleam_x_sdss16_r10arcsec.csv',
                             sep=',',
                             # names = fields,
                             index_col=False,
                             usecols=self.cols)
            if redshift:
                df.rename(columns={'zsp': 'redshift'}, inplace=True)
            datasetname = 'GLEAM x SDSS 16, r=10"'

    # %%% GLEAM x SDSS12
        if self.name.lower() == 'gleam_x_sdss12':  # 2028
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
            magnames = ['Fp076', 'Fp084', 'Fp092', 'Fp099', 'Fp107',
                        'Fp115', 'Fp122', 'Fp130', 'Fp143',
                        'Fp151', 'Fp158', 'Fp166',
                        'Fp174', 'Fp181', 'Fp189', 'Fp197',
                        'Fp204', 'Fp212', 'Fp220',
                        'Fp227', 'alpha', 'umag', 'gmag', 'rmag', 'imag', 'zmag']
            df = pd.read_csv(path + '/x-matches/gleam_x_sdss12.csv',
                             sep=',',
                             # names = fields,
                             index_col=False,
                             usecols=self.cols)
            df = df.drop(['_RAJ2000_1'], axis=1)
            df = df.drop(['_DEJ2000_1'], axis=1)
            if redshift:
                df.rename(columns={'zsp': 'redshift'}, inplace=True)
            datasetname = 'GLEAM x SDSS 12, r=10"'
    # %%% GLEAM x SIMBAD
        if self.name.lower() == 'gleam_x_simbad':  # 33012
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

            magnames = ["Fp076", "Fint076",
                        "Fp084", "Fp092", "Fp099", "Fp107",
                        "Fp115", "Fp122", "Fp130",
                        "Fp143", "Fp151", "Fp158", "Fp166",
                        "Fp174", "Fp181", "Fp189",
                        "Fp197", "Fp204", "Fp212", "Fp220",
                        "Fp227", "B", "V", "R", "J", "H", "K", "u", "g", "r", "i", "z"]
            df = pd.read_csv(path + '/x-matches/gleamxSIMBAD.csv',
                             sep=',', index_col=False,
                             header=0
                             # names = fields
                             )
            datasetname = 'GLEAM x SIMBAD'
    # %%% SDSS12 x UKIDSS
        if self.name.lower() == 'sdss12_x_ukidss':  # 2417
            redshift = True
            fields = ['_RAJ2000_1', '_DEJ2000_1', 'RA_ICRS', 'DE_ICRS', 'mode', 'q_mode',
                      'class', 'SDSS12', 'm_SDSS12', 'ObsDate', 'Q', 'umag', 'e_umag', 'gmag',
                      'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 'zsp', 'spCl',
                      'zph', 'e_zph', '<zph>', '_RAJ2000_2', '_DEJ2000_2', 'UDXS', 'm', 'RAJ2000',
                      'DEJ2000', 'Jmag', 'e_Jmag', 'Kmag', 'e_Kmag', 'Jdiam', 'Jell', 'Jflags',
                      'Kdiam', 'Kell', 'Kflags', 'Separation'
                      ]
            magnames = ['umag', 'gmag', 'rmag', 'imag', 'zmag', 'Jmag', 'Kmag']

            df = pd.read_csv(path + '/x-matches/sdss12_x_ukidss.csv',
                             sep=',', index_col=False,
                             header=0,
                             names=fields
                             )
            if redshift:
                df.rename(columns={'zsp': 'redshift'}, inplace=True)
            datasetname = 'SDSS DR12 cross-match UKIDSS'
    # %%% GALEX all-Sky Survey
        if self.name.lower() == 'galexais':
            redshift = False
            fields = ['recno', 'RAJ2000', 'DEJ2000', 'FUV', 'e_FUV', 'NUV', 'e_NUV', 'objid',
                      'tile', 'img', 'sv', 'r.fov', 'b', 'E(B-V)', 'FUV.b', 'e_FUV.b', 'NUV.b',
                      'e_NUV.b', 'FUV.a', 'e_FUV.a', 'NUV.a', 'e_NUV.a', 'FUV.4', 'e_FUV.4',
                      'NUV.4', 'e_NUV.4', 'FUV.6', 'e_FUV.6', 'NUV.6', 'e_NUV.6', 'Fafl',
                      'Nafl', 'Fexf', 'Nexf', 'Fflux', 'e_Fflux', 'Nflux', 'e_Nflux', 'FXpos',
                      'FYpos', 'NXpos', 'NYpos', 'Fima', 'Nima', 'Fr', 'Nr', 'phID', 'fRAdeg',
                      'fDEdeg'
                      ]
            magnames = [
                # 'FUV', Very few measurements
                'NUV']

            df = pd.read_csv(path + '/GALEX/GALEX_II_312_ais.csv',
                             sep=',', index_col=False,
                             header=0,
                             names=fields
                             )
            datasetname = 'GALEX AIS (All-sky Imaging Survey)'
    # %%% GALEX x SDSS12
        if self.name.lower() == 'galex_x_sdss12':  # 73906
            redshift = True
            fields = ['recno', 'RAJ2000', 'DEJ2000', 'FUV', 'e_FUV', 'NUV', 'e_NUV', 'objid',
                      'tile', 'img', 'sv', 'r.fov', 'b', 'E(B-V)', 'FUV.b', 'e_FUV.b', 'NUV.b',
                      'e_NUV.b', 'FUV.a', 'e_FUV.a', 'NUV.a', 'e_NUV.a', 'FUV.4', 'e_FUV.4',
                      'NUV.4', 'e_NUV.4', 'FUV.6', 'e_FUV.6', 'NUV.6', 'e_NUV.6', 'Fafl',
                      'Nafl', 'Fexf', 'Nexf', 'Fflux', 'e_Fflux', 'Nflux', 'e_Nflux', 'FXpos',
                      'FYpos', 'NXpos', 'NYpos', 'Fima', 'Nima', 'Fr', 'Nr', 'phID', 'fRAdeg',
                      'fDEdeg', '_RAJ2000', '_DEJ2000', 'RA_ICRS', 'DE_ICRS', 'mode', 'q_mode',
                      'class', 'SDSS12', 'm_SDSS12', 'ObsDate', 'Q', 'umag', 'e_umag', 'gmag',
                      'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 'zsp', 'spCl',
                      'zph', 'e_zph', '<zph>', 'Separation'
                      ]
            magnames = [
                # 'FUV', Very few measurements
                'NUV', 'umag', 'gmag', 'rmag', 'imag', 'zmag']

            df = pd.read_csv(path + '/x-matches/galex_x_sdss12.csv',
                             sep=',', index_col=False,
                             header=0,
                             names=fields
                             )
            if redshift:
                df.rename(columns={'zsp': 'redshift'}, inplace=True)
            datasetname = 'GALEX All-sky Survey X SDSS12'

    # %%% GALEX x SDSS12 x GLEAM
        if self.name.lower() == 'galex_x_sdss12_x_gleam':  # 565
            redshift = True
            fields = ['recno', 'RAJ2000_1', 'DEJ2000_1', 'FUV', 'e_FUV',
                      'NUV', 'e_NUV', 'objid', 'tile', 'img', 'sv', 'r.fov', 'b',
                      'E(B-V)', 'FUV.b', 'e_FUV.b', 'NUV.b', 'e_NUV.b', 'FUV.a',
                      'e_FUV.a', 'NUV.a', 'e_NUV.a', 'FUV.4', 'e_FUV.4', 'NUV.4',
                      'e_NUV.4', 'FUV.6', 'e_FUV.6', 'NUV.6', 'e_NUV.6', 'Fafl',
                      'Nafl', 'Fexf', 'Nexf', 'Fflux', 'e_Fflux', 'Nflux', 'e_Nflux',
                      'FXpos', 'FYpos', 'NXpos', 'NYpos', 'Fima', 'Nima', 'Fr', 'Nr',
                      'phID', 'fRAdeg', 'fDEdeg', '_RAJ2000_1', '_DEJ2000_1',
                      'RA_ICRS', 'DE_ICRS', 'mode', 'q_mode', 'class', 'SDSS12',
                      'm_SDSS12', 'ObsDate', 'Q', 'umag', 'e_umag', 'gmag', 'e_gmag',
                      'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 'zsp',
                      'spCl', 'zph', 'e_zph', '<zph>', 'Separation_1', '_RAJ2000_2',
                      '_DEJ2000_2', 'GLEAM', 'RAJ2000_2', 'DEJ2000_2', 'Fpwide',
                      'Fintwide', 'eabsFpct', 'efitFpct', 'Fp076', 'Fint076', 'Fp084',
                      'Fint084', 'Fp092', 'Fint092', 'Fp099', 'Fint099', 'Fp107',
                      'Fint107', 'Fp115', 'Fint115', 'Fp122', 'Fint122', 'Fp130',
                      'Fint130', 'Fp143', 'Fint143', 'Fp151', 'Fint151', 'Fp158',
                      'Fint158', 'Fp166', 'Fint166', 'Fp174', 'Fint174', 'Fp181',
                      'Fint181', 'Fp189', 'Fint189', 'Fp197', 'Fint197', 'Fp204',
                      'Fint204', 'Fp212', 'Fint212', 'Fp220', 'Fint220', 'Fp227',
                      'Fint227', 'alpha', 'Fintfit200', 'Separation']

            magnames = ['Fp076', 'Fp084', 'Fp092', 'Fp099', 'Fp107', 'Fp115',
                        'Fp122', 'Fp130', 'Fp143', 'Fp151', 'Fp158', 'Fp166', 'Fp174',
                        'Fp181', 'Fp189', 'Fp197', 'Fp204', 'Fp212', 'Fp220', 'Fp227',
                        'alpha', 'FUV', 'NUV', 'umag', 'gmag', 'rmag', 'imag', 'zmag']

            df = pd.read_csv(path + '/x-matches/galex_x_sdss12_x_gleam.csv',
                             sep=',', index_col=False,
                             header=0,
                             names=fields
                             )
            if redshift:
                df.rename(columns={'zsp': 'redshift'}, inplace=True)
            datasetname = 'GALEX All-sky Survey X SDSS12 x GLEAM'
    # %%% Milliquas x GLEAM
        if self.name.lower() == 'milli_x_gleam':  # 6834
            redshift = True
            fields = ['angDist', '_RAJ2000_milli', '_DEJ2000_milli', 'recno', 'RAJ2000_milli',
                      'DEJ2000_milli', 'Name',
                      'Type', 'Rmag', 'Bmag', 'Comment', 'R', 'B', 'z', 'Qpct', 'XName', 'RName',
                      '_RAJ2000', '_DEJ2000', 'GLEAM', 'RAJ2000', 'DEJ2000', 'Fpwide', 'Fintwide',
                      'eabsFpct', 'efitFpct', 'Fp076', 'Fint076', 'Fp084', 'Fint084', 'Fp092',
                      'Fint092', 'Fp099', 'Fint099', 'Fp107', 'Fint107', 'Fp115', 'Fint115',
                      'Fp122', 'Fint122', 'Fp130', 'Fint130', 'Fp143', 'Fint143', 'Fp151',
                      'Fint151', 'Fp158', 'Fint158', 'Fp166', 'Fint166', 'Fp174', 'Fint174',
                      'Fp181', 'Fint181', 'Fp189', 'Fint189', 'Fp197', 'Fint197', 'Fp204',
                      'Fint204', 'Fp212', 'Fint212', 'Fp220', 'Fint220', 'Fp227', 'Fint227',
                      'alpha', 'Fintfit200'
                      ]
            magnames = ['Rmag', 'Bmag', 'Fp076', 'Fp084', 'Fp092',
                        'Fp099', 'Fp107', 'Fp115',
                        'Fp122', 'Fp130', 'Fp143', 'Fp151',
                        'Fp158', 'Fp166', 'Fp174',
                        'Fp181', 'Fp189', 'Fp197', 'Fp204',
                        'Fp212', 'Fp220', 'Fp227',
                        # 'alpha'
                        ]

            df = pd.read_csv(path + '/x-matches/milliquas_x_gleam.csv',
                             sep=',', index_col=False,
                             header=0,
                             names=fields
                             )
            df = df.drop(['_RAJ2000_milli', '_DEJ2000_milli', 'RAJ2000_milli',
                          'DEJ2000_milli', 'Name', 'XName', 'RName', '_RAJ2000',
                          '_DEJ2000', 'GLEAM'], axis=1)
            df = df[df['z'] < 5.84]

            if redshift:
                df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'Milliquas GLEAM cross-match'
    # %%% SDSS16
        if self.name.lower() == 'sdss16':  # 28000
            redshift = True
            fields = ['_RAJ2000', '_DEJ2000', 'objID', 'RA_ICRS', 'DE_ICRS', 'mode', 'class',
                      'clean', 'e_RA_ICRS', 'e_DE_ICRS', 'umag', 'gmag', 'rmag', 'imag', 'zmag',
                      'e_umag', 'e_gmag', 'e_rmag', 'e_imag', 'e_zmag', 'zsp', 'e_zsp', 'f_zsp',
                      'zph', 'e_zph', '<zph>', 'Q', 'SDSS16', 'Sp-ID', 'MJD'
                      ]
            magnames = ['umag', 'gmag', 'rmag', 'imag', 'zmag']

            df = pd.read_csv(path + '/SDSS/vizier_V_154_sdss16_20220307.csv',
                             sep=',', index_col=False,
                             header=0,
                             names=fields
                             )
            df = df.drop(['SDSS16', '_RAJ2000', '_DEJ2000', 'Sp-ID'], axis=1)

            if redshift:
                df.rename(columns={'zsp': 'redshift'}, inplace=True)
            datasetname = 'SDSS 16'
    # %%% New Fitted
        if self.name.lower() == 'new_fitted':  # 43699
            redshift = True
            fields = ['idx', 'z_spec', 'flag_TO', 'SI', 'S_400', 'S_1p4', 'S_5', 'S_8p7'
                      ]
            magnames = ['SI', 'S_400', 'S_1p4', 'S_5', 'S_8p7']

            df = pd.read_csv(path + '/out.csv',
                             sep=',', index_col=False,
                             header=0,
                             names=fields
                             )
            if redshift:
                df.rename(columns={'z_spec': 'redshift'}, inplace=True)
            datasetname = 'The Million Quasars catalogue'
    # %%% Old fitted
        if self.name.lower() == 'old_fitted':  # 43699
            sample = 'not PEAK'
            redshift = True
            fields = ['no', 'NED', 'zspec', 'z_NED', 'class', 'type', ' Qpct', 'no_radio', 'fit',
                      'X2', 'X1', 'X0', 'double_dash_1', 'root_1', 'TO', 'flag_TO', 'TO_flux', 'SI',
                      'nu_peak', 'S_peak', 'thick', 'thin', 'S_70', 'S_150', 'S_400', 'S_700', 'S_1',
                      'S_1p4', 'S2p7', 'S_5', 'S_8p7', 'S_15', 'S_20', 'u', 'g', 'r', 'i', 'z', 'W1',
                      'W2', 'NUV', 'FUV'
                      ]
            if sample == 'PEAK':
                magnames = ['SI', 'nu_peak', 'S_peak', 'thick', 'thin']

            else:
                magnames = [
                    # 'TO_flux','SI',
                    #   'nu_peak','S_peak','thick','thin','S_70','S_150','S_400','S_700','S_1',
                    #   'S_1p4','S2p7','S_5','S_8p7','S_15','S_20',
                    'u', 'g', 'r', 'i', 'zmag', 'W1', 'W2',
                    # 'NUV','FUV'
                ]

            df = pd.read_csv(path + '/new_fitted.dat',
                             sep=' ', index_col=False,
                             header=None,
                             names=fields
                             )
            df.rename(columns={'z': 'zmag'}, inplace=True)
            if redshift:
                df.rename(columns={'zspec': 'redshift'}, inplace=True)
            datasetname = 'The Million Quasars (Milliquas) catalogue'
    # %%% XLSOptn
        if self.name.lower() == 'xlsoptn':  # 31585
            redshift = True
            # fields = ['_RAJ2000','_DEJ2000','Xcatname','RACtpdeg','DECtpdeg','zspec',
            #           'uSDSSmag','gSDSSmag','rSDSSmag','iSDSSmag','zSDSSmag','uCFHTmag',
            #           'gCFHTmag','rCFHTmag','iCFHTmag','yCFHTmag','zCFHTmag','zVISTAmag',
            #           'YVISTAmag','JVISTAmag','HVISTAmag','KVISTAmag','JUKIDSSmag',
            #           'HUKIDSSmag','KUKIDSSmag','KWIRcammag','IRAC3.6mag','IRAC4.5mag',
            #           'GALEXFUVmag','GALEXNUVmag','WISE1mag','WISE2mag','WISE3mag',
            #           'WISE4mag','recno'
            #           ]
            magnames = ['u', 'g', 'r', 'i', 'zmag', 'W1', 'W2']

            df = pd.read_csv(path + '/vizier_IX_52_3xlsoptn_20220120.csv',
                             sep=',', index_col=False,
                             header=0,
                             usecols=['zspec', 'uSDSSmag', 'gSDSSmag', 'rSDSSmag',
                                      'iSDSSmag', 'zSDSSmag', 'WISE1mag', 'WISE2mag']
                             )
            df.rename(columns={'uSDSSmag': 'u',
                               'gSDSSmag': 'g',
                               'rSDSSmag': 'r',
                               'iSDSSmag': 'i',
                               'zSDSSmag': 'zmag',
                               'WISE1mag': 'W1',
                               'WISE2mag': 'W2'
                               }, inplace=True)
            if redshift:
                df.rename(columns={'zspec': 'redshift'}, inplace=True)
            datasetname = 'Spectrophotometric catalog of galaxies'
    # %%% XXGal
        if self.name.lower() == 'xxgal':  # 24336
            redshift = True
            fields = ['_RAJ2000', '_DEJ2000', 'Index', 'RAJ2000', 'DEJ2000', 'z', 'r_z', 'f_z',
                      'q_z', 'DRr200-1', 'DRr200-2', 'DRr200-3', 'DRr200-4', 'DRr200-5', 'Dv-1',
                      'Dv-2', 'Dv-3', 'Dv-4', 'Dv-5', 'XLSSC3r200', 'XLSSC3r200u', 'DRr200u',
                      'uMag', 'gMag', 'rMag', 'iMag', 'yMag', 'zMag', 'bMass', 'Mass', 'BMass',
                      'ComplSM'

                      ]
            magnames = ['uMag', 'gMag', 'rMag', 'iMag', 'yMag', 'zMag']

            df = pd.read_csv(path + '/vizier_IX_52_xxlngal_20220120.csv',
                             sep=',', index_col=False,
                             header=0,
                             usecols=self.cols
                             )
            if redshift:
                df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'Spectrophotometric catalog of galaxies'
    # %%% Milliquas
        if self.name.lower() == 'milliquas':  # 31561
            redshift = True
            fields = ['recno', 'RAJ2000', 'DEJ2000', 'Name', 'Type', 'Rmag', 'Bmag', 'Comment',
                      'R', 'B', 'z', 'Qpct', 'XName', 'RName'
                      ]
            magnames = ['Rmag', 'Bmag']

            df = pd.read_csv(path + '/Milliquas/vizier_VII_290_catalog_20211213_fixed.csv',
                             sep=',', index_col=False,
                             header=0,
                             usecols=self.cols
                             )
            df['z'] = pd.to_numeric(df['z'], errors='coerce')
            df['Rmag'] = pd.to_numeric(df['Rmag'], errors='coerce')
            df['Bmag'] = pd.to_numeric(df['Bmag'], errors='coerce')
            df = df[df['z'] < 5.84]
            df = df[df['Type'].str.contains("QR")]  # quasar + radio association

            if redshift:
                df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'The Million Quasars (Milliquas) catalogue, version 7.2'
    # %%% FIRST-NVSS
        if self.name.lower() == 'first_nvss':  # 20648
            redshift = True
            fields = ['_RAJ2000', '_DEJ2000', 'recno', 'UNIQ_ID', 'RAJ2000', 'DEJ2000',
                      'Fiflux', 'Nflux', 'Wflux', 'Gflux', 'SNgmag', 'SNimag', 'MJD',
                      'Plate', 'Fiber', 'z', 'Dist', 'WDist', 'GB6Dist', 'SNDist',
                      'SBDist', 'Plate1', 'MJD1', 'Ori'
                      ]
            magnames = ['Fiflux', 'Nflux', 'Wflux', 'Gflux', 'SNgmag', 'SNimag']

            df = pd.read_csv(path + '/vizier_J_ApJ_699_L43_catalog_20220117.csv',
                             sep=',', index_col=False,
                             header=0,
                             usecols=self.cols
                             )
            if redshift:
                df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'FIRST-NVSS-SDSS AGN sample catalog'
    # %%% GLEAM x NVSS, radius 5"
        if self.name.lower() == 'gxn':  # 69745
            redshift = False
            fields = ['angDist', '_RAJ2000', '_DEJ2000', 'NVSS', 'RAJ2000', 'DEJ2000',
                      'e_RAJ2000', 'e_DEJ2000', 'S1.4', 'e_S1.4', 'l_MajAxis', 'MajAxis',
                      'l_MinAxis', 'MinAxis', 'f_resFlux', '_RAJ2000', '_DEJ2000', 'GLEAM',
                      'RAJ2000', 'DEJ2000', 'Fpwide', 'Fintwide', 'eabsFpct', 'efitFpct',
                      'Fp076', 'Fint076', 'Fp084', 'Fint084', 'Fp092', 'Fint092', 'Fp099',
                      'Fint099', 'Fp107', 'Fint107', 'Fp115', 'Fint115', 'Fp122', 'Fint122',
                      'Fp130', 'Fint130', 'Fp143', 'Fint143', 'Fp151', 'Fint151', 'Fp158',
                      'Fint158', 'Fp166', 'Fint166', 'Fp174', 'Fint174', 'Fp181', 'Fint181',
                      'Fp189', 'Fint189', 'Fp197', 'Fint197', 'Fp204', 'Fint204', 'Fp212',
                      'Fint212', 'Fp220', 'Fint220', 'Fp227', 'Fint227', 'alpha', 'Fintfit200']
            magnames = [mag for mag in fields if mag.startswith(
                'Fp') and not mag.startswith("Fpw")]

            df = pd.read_csv(path + '/x-matches/gleam_x_nvss_5_arcsec.csv',
                             sep=',', index_col=False,
                             header=0,
                             usecols=self.cols
                             )
            if redshift:
                df.rename(columns={'zsp': 'redshift'}, inplace=True)
            datasetname = 'GLEAM x NVSS, radius 5"'
    # %%% GLEAM x NVSS x SDSS12 radius 5"
        if self.name.lower() == 'gxnxs':  # 28424
            redshift = True
            fields = ['angDist', 'angDist', '_RAJ2000', '_DEJ2000', 'NVSS', 'RAJ2000',
                      'DEJ2000', 'e_RAJ2000', 'e_DEJ2000', 'S1.4', 'e_S1.4', 'l_MajAxis',
                      'MajAxis', 'l_MinAxis', 'MinAxis', 'f_resFlux', '_RAJ2000',
                      '_DEJ2000', 'GLEAM', 'RAJ2000', 'DEJ2000', 'Fpwide', 'Fintwide',
                      'eabsFpct', 'efitFpct', 'Fp076', 'Fint076', 'Fp084', 'Fint084',
                      'Fp092', 'Fint092', 'Fp099', 'Fint099', 'Fp107', 'Fint107', 'Fp115',
                      'Fint115', 'Fp122', 'Fint122', 'Fp130', 'Fint130', 'Fp143',
                      'Fint143', 'Fp151', 'Fint151', 'Fp158', 'Fint158', 'Fp166',
                      'Fint166', 'Fp174', 'Fint174', 'Fp181', 'Fint181', 'Fp189',
                      'Fint189', 'Fp197', 'Fint197', 'Fp204', 'Fint204', 'Fp212',
                      'Fint212', 'Fp220', 'Fint220', 'Fp227', 'Fint227', 'alpha',
                      'Fintfit200', 'RAdeg', 'DEdeg', 'errHalfMaj', 'errHalfMin',
                      'errPosAng', 'objID', 'mode', 'q_mode', 'class', 'SDSS12',
                      'm_SDSS12', 'flags', 'ObsDate', 'Q', 'umag', 'e_umag', 'gmag',
                      'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 'zsp',
                      'e_zsp', 'f_zsp', 'zph', 'e_zph', 'avg_zph', 'pmRA', 'e_pmRA', 'pmDE',
                      'e_pmDE', 'SpObjID', 'spType', 'spCl', 'subClass']
            magnames1 = [mag for mag in fields if mag.startswith(
                'Fp') and not mag.startswith("Fpw")]
            magnames2 = [mag for mag in fields if (
                mag.endswith('mag') and not mag.startswith("e_"))]
            magnames = magnames1 + magnames2
            df = pd.read_csv(path + '/x-matches/gleam_x_nvss_x_sdss12_5_arcsec.csv',
                             sep=',', index_col=False,
                             header=0,
                             usecols=self.cols)
            df.rename(columns={'zsp': 'redshift'}, inplace=True)
            datasetname = 'GLEAM x NVSS x SDSS12 radius 5"'
    # %%% eMERLIN
        if self.name.lower() == 'emerlin':  # 395
            redshift = False
            fields = ['IslId', 'RAdeg', 'e_RAdeg', 'DEdeg', 'e_DEdeg', 'FluxT',
                      'e_FluxT', 'FluxP', 'e_FluxP', 'Maj', 'e_Maj', 'Min', 'e_Min',
                      'PA', 'e_PA', 'DCMaj', 'e_DCMaj', 'DCMin', 'e_DCMin', 'DCPA',
                      'e_DCPA', 'FluxTIsl', 'e_FluxTIsl', 'Islrms', 'Islmean',
                      'ResIslrms', 'ResIslmean', 'SCode', 'ScaleFlag', 'ResFlag',
                      'SMorphFlag']
            magnames = ['FluxT', 'FluxP']
            df = pd.read_csv(path + '/emerlin_vla_subaru/vizier_J_MNRAS_495_1706_emerlin_20220112.csv',
                             sep=',', index_col=False,
                             header=0,
                             usecols=self.cols)
            datasetname = 'SuperCLASS eMERLIN survey'
    # %%% VLA
        if self.name.lower() == 'vla':  # 887
            redshift = False
            fields = ['GausId', 'IslId', 'SourceId', 'WaveId', 'RAJ2000', 'DEJ2000',
                      'FluxT', 'FluxP', 'Maj', 'Min', 'PA', 'DCMaj', 'DCMin', 'DCPA',
                      'FluxTIsl', 'Islrms', 'Islmean', 'ResIslrms', 'ResIslmean',
                      'SCode', 'ResFlag', 'SMorphFlag', 'e1unc', 'e2unc', 'e1cal',
                      'e2cal', 'Valid', 'RadIm3Shape', 'e1calCorr', 'e2calCorr']
            magnames = ['FluxT', 'FluxP']
            df = pd.read_csv(path + '/emerlin_vla_subaru/vizier_J_MNRAS_495_1706_vla_20220112.csv',
                             sep=',', index_col=False,
                             header=0,
                             usecols=self.cols)
            datasetname = 'SuperCLASS VLA survey'
    # %%% Subaru
        if self.name.lower() == 'subaru':  # 376380
            redshift = False
            fields = ['RAJ2000', 'DEJ2000', 'Bmag', 'Vmag', 'rmag', 'imag', 'zmag',
                      'ymag', '[3.6]', '[4.5]', 'Id', 'za', 'chiza', 'Nfilt', 'e1',
                      'e2', 'Rad', 'RadRatio', 'BulgeA', 'DiscA', 'BulgeIndex',
                      'DiscIndex', 'BulgeFlux', 'DiscFlux', 'FluxRatio', 'snr',
                      'SourceId']
            magnames = ['Bmag', 'Vmag', 'rmag', 'imag',
                        'zmag', 'ymag', '[3.6]', '[4.5]']
            df = pd.read_csv(path + '/emerlin_vla_subaru/vizier_J_MNRAS_495_1706_subaru_20220112.csv',
                             sep=',', index_col=False,
                             header=0,
                             usecols=self.cols)
            df.rename(columns={'za': 'redshift'}, inplace=True)
            datasetname = 'SuperCLASS Subaru survey'
    # %%% GLEAM
        if self.name.lower() == 'gleam':  # 307455
            redshift = False
            fields = ['_RAJ2000', '_DEJ2000', 'GLEAM', 'RAJ2000', 'DEJ2000', 'Fpwide',
                      'Fintwide', 'eabsFpct', 'efitFpct', 'Fp076', 'Fint076', 'Fp084',
                      'Fint084', 'Fp092', 'Fint092', 'Fp099', 'Fint099', 'Fp107', 'Fint107',
                      'Fp115', 'Fint115', 'Fp122', 'Fint122', 'Fp130', 'Fint130', 'Fp143',
                      'Fint143', 'Fp151', 'Fint151', 'Fp158', 'Fint158', 'Fp166', 'Fint166',
                      'Fp174', 'Fint174', 'Fp181', 'Fint181', 'Fp189', 'Fint189', 'Fp197',
                      'Fint197', 'Fp204', 'Fint204', 'Fp212', 'Fint212', 'Fp220', 'Fint220',
                      'Fp227', 'Fint227', 'alpha', 'Fintfit200']
            magnames = ['Fp076', 'Fp084', 'Fp092', 'Fp099', 'Fp107', 'Fp115',
                        'Fp122', 'Fp130', 'Fp143', 'Fp151', 'Fp158', 'Fp166', 'Fp174',
                        'Fp181', 'Fp189', 'Fp197', 'Fp204', 'Fp212', 'Fp220', 'Fp227', 'alpha']
            df = pd.read_csv(path + '/GLEAM/vizier_VIII_100_gleamegc_20210608.csv',
                             sep=',', index_col=False,
                             header=0,
                             usecols=self.cols)
            df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'GaLactic and Extragalactic All-sky MWA survey'

    # %%% Markarian
        if self.name.lower() == 'markarian':  # 1469
            redshift = False
            fields = ['name', 'ra', 'dec', 'vmag', 'major_axis', 'minor_axis', 'redshift',
                      'bv_color', 'ub_color', 'class']
            magnames = ['vmag', 'bv_color', 'ub_color']
            df = pd.read_csv(path + '/Markarian/markarian.csv',
                             sep=',', header=0, index_col=False,
                             usecols=self.cols)
            datasetname = 'First Byurakan Survey (Markarian) Catalog of UV-Excess Galaxies'
    # %%% LBQS
        if self.name.lower() == 'lbqs':  # 1055
            redshift = True
            fields = ['name', 'ra', 'dec', 'bjmag', 'redshift', 'field_info']
            magnames = ['bjmag']
            df = pd.read_csv(path + '/lbqs.csv',
                             sep=',', header=0, index_col=False,
                             usecols=self.cols)
            datasetname = 'Large Bright Quasar Survey'
    # %%% LQAC
        if self.name.lower() == 'lqac':  # 20000
            redshift = True
            fields = ['name', 'ra', 'dec', 'vmag', 'rmag',
                      'kmag', 'flux_20_cm', 'redshift', 'abs_imag']
            magnames = ['vmag', 'rmag', 'kmag']
            df = pd.read_csv(path + '/lqac.csv',
                             sep=',', header=0, index_col=False,
                             usecols=self.cols)
            datasetname = 'Large Quasar Astrometric Catalog, 3rd Release'
    # %%% allWISE AGN
        if self.name.lower() == 'allwiseagn':  # 10000
            redshift = True
            fields = ['name', 'ra', 'dec', 'w1w2_color',
                      'w2w3_color', 'w1_mag', 'gmag', 'redshift']
            magnames = ['w1w2_color', 'w2w3_color', 'w1_mag', 'gmag']
            df = pd.read_csv(path + '/WISE/allwiseagn.csv',
                             sep=',', header=0, index_col=False,
                             usecols=self.cols)
            datasetname = 'AllWISE Catalog of Mid-IR AGNs'
    # %%% Crampton
        if self.name.lower() == 'crampton':  # 777
            redshift = False
            fields = ['_RAJ2000', '_DEJ2000', 'recno', 'Rem', 'Name', 'RA1950', 'DE1950',
                      'Bmag', 'n_Bmag', 'redshift', 'u_z', 'Nature', '_RA.icrs', '_DE.icrs']
            magnames = ['Bmag']
            cols = ['_RAJ2000', '_DEJ2000', 'recno',
                    'Bmag', 'z', '_RA.icrs', '_DE.icrs']
            df = pd.read_csv(path + '/vizier_VII_143_catalog_20201104.csv',
                             sep=',', header=0, index_col=False,
                             usecols=self.cols)
            df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'Quasar Candidates (Crampton+ 1985-1990)'
    # %%% Rood
        if self.name.lower() == 'rood':  # 3981
            redshift = False
            fields = ['_RAJ2000', '_DEJ2000', 'recno', 'Seq', 'UGC', 'OtherName', 'RA1950',
                      'DE1950', 'Pmag', 'zFlag', 'r_e_HVel', 'HVel', 'e_HVel', 'u_e_HVel',
                      '_RA.icrs', '_DE.icrs']
            magnames = ['Pmag']
            df = pd.read_csv(path + '/vizier_VII_36_catalog_20201104.csv',
                             sep=',', header=0, index_col=False,
                             usecols=self.cols)
            df.rename(columns={'HVel': 'redshift', 'e_HVel': 'e_z', 'u_e_HVel': 'uncertain z'},
                      inplace=True)
            datasetname = 'Galaxy Redshifts (Rood 1980)'
     # %%% DEEP2
        if self.name.lower() == 'deep2':  # 52989
            redshift = True
            magnames = ['Bmag', 'Rmag', 'Imag', 'EBV']
            # cols = ['Bmag','Rmag','Imag','RG','BMag', 'z']
            df = pd.read_csv(path + '/DEEP2/vizier_III_268_deep2all_20220727_color_extinction.csv',
                             # sep = ',', header = 0, index_col = False, usecols = self.cols
                             )
            df.rename(columns={'z': 'redshift'}, inplace=True)
            datasetname = 'DEEP2 Redshift Survey, DR4'

            if redshift:
                df = df[df['redshift'] > 0]
                # move redshift column to the end
                df.insert(len(df.columns)-1, 'redshift', df.pop('redshift'))

            deep2_agn = df[df['Cl'] == 'A']
            deep2_gal = df[df['Cl'] == 'G']
            deep2_stars = df[df['Cl'] == 'S']

    # %%% PanSTARRS
        if self.name.lower() == 'panstarrs':  # 999999
            redshift = False
            fields = ['_RAJ2000', '_DEJ2000', 'RAJ2000', 'DEJ2000', 'objID', 'f_objID',
                      'Qual', 'e_RAJ2000', 'e_DEJ2000', 'Epoch', 'Ns', 'Nd', 'gmag',
                      'e_gmag', 'gKmag', 'e_gKmag', 'gFlags', 'rmag', 'e_rmag', 'rKmag',
                      'e_rKmag', 'rFlags', 'imag', 'e_imag', 'iKmag', 'e_iKmag', 'iFlags',
                      'zmag', 'e_zmag', 'zKmag', 'e_zKmag', 'zFlags', 'ymag', 'e_ymag',
                      'yKmag', 'e_yKmag', 'yFlags']
            magnames = ['gmag', 'gKmag', 'rmag', 'rKmag',
                        'imag', 'iKmag', 'zmag', 'ymag', 'yKmag']
            df = pd.read_csv(path + '/vizier_II_349_ps1_20201102_10^6rows.csv',
                             sep=',', header=0, index_col=False, usecols=self.cols)
            datasetname = 'Pan-STARRS release 1 (PS1) Survey - DR1'

    # %%% VIMOS
        if self.name.lower() == 'vimos':  # 8981
            redshift = False
            print("This dataset contains infs or NaNs; neural network may fail.")
            fields = ['ID', 'redshift', 'q_z', 'phf', 'UEmag', 'B', 'V', 'R', 'I', 'J',
                      'K', 'VVDS', 'FITS', 'RAJ2000', 'DEJ2000']
            magnames = ['B', 'V', 'R', 'I', 'J', 'K']
            cols = ['B', 'V', 'R', 'I', 'J', 'K', 'redshift']
            df = pd.read_csv(path + '/vizier_III_250_vvds_dp_20200908.csv', skiprows=[0],
                             sep=',', names=fields, index_col=False, usecols=self.cols)
            datasetname = 'VIMOS VLT deep survey (VVDS-DEEP)'

    # %%% SDSS
        if self.name.lower() == 'nedfirst50k':  # 49738
            redshift = False
            #   SDSS data
            fields = ['ignore', 'NED', 'redshift', 'ez', 'type', 'class', 'no_radio', 'no_UV',
                      'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'W1', 'W2', 'W3', 'W4',
                      'SPIT_5', 'SPIT_8', 'NUV', 'FUV']
            magnames = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'W1', 'W2', 'W3', 'W4',
                        'SPIT_5', 'SPIT_8', 'NUV', 'FUV']
            cols = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'W1', 'W2', 'W3', 'W4',
                    'SPIT_5', 'SPIT_8', 'NUV', 'FUV', 'redshift']
            df = pd.read_csv(path + '/optical_data/QSOs-NED_1st_50K.dat-mags.dat',
                             sep=' ', names=fields, index_col=False, usecols=self.cols)
            # df = df.drop_duplicates(['NED'], keep='first')
            # df = df.drop(['ignore'], axis = 1)
            datasetname = 'SDSS'

    # %%% MgII
        if self.name.lower() == 'mgii':  # 29008
            redshift = True
            fields = ['no', 'NED', 'redshift', 'type', 'S_21', 'freq_21',
                      'SI_flag', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'W1', 'W2', 'W3',
                      'W4', 'L_UV', 'Q', 'flag_uv', 'n_abs', 'z_a', 'dz_a', 'EW_2803',
                      'dEW_2803', 'EW_2796', 'dEW_2796', 'SPIT_5', 'SPIT_8', 'NUV', 'FUV']
            magnames = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'W1', 'W2', 'W3',
                        'W4', 'SPIT_5', 'SPIT_8', 'NUV', 'FUV',
                        'chord'
                        ]
            # cols = ['U','B','V','R','I','J','H','K','W1','W2','W3',
            #                'W4', 'SPIT_5','SPIT_8','NUV','FUV','redshift']
            df = pd.read_csv(path + '/optical_data/MgII_mags+SPITZER+UV_0.050.dat',
                             sep=' ', names=fields, index_col=False, usecols=self.cols)
            # df = df.drop_duplicates(['NED'], keep='first')
            # df = df.drop(['SI_flag'], axis = 1)
            df['chord'] = (df['I']-df['W2']) / (df['W3']-df['U'])

            datasetname = 'MgII'
    # %%% LARGESS
        if self.name.lower() == 'largess':  # 10944
            redshift = True
            fields = ['no', 'NED', 'redshift', 'q_z', 'zsource', 'EWOIII', 'OIII_SN',
                      'VClass', 'BClass', 'SI', 'TO', 'flag_TO', 'L_21',
                      'P_21', 'L_UV', 'Q', 'flag_uv', 'U', 'B', 'V', 'R', 'I', 'J',
                              'H', 'K', 'W1', 'W2', 'W3', 'W4', 'SPIT_5', 'SPIT_8', 'NUV', 'FUV']
            magnames = ['U', 'B', 'V', 'R', 'I', 'J',
                        'H', 'K', 'W1', 'W2', 'W3', 'W4', 'SPIT_5', 'SPIT_8', 'NUV', 'FUV']
            cols = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'W1', 'W2', 'W3', 'W4',
                    'SPIT_5', 'SPIT_8', 'NUV', 'FUV', 'redshift']
            df = pd.read_csv(path + '/optical_data/LARGESS_mag_0.050.dat', sep=' ',
                             names=fields, index_col=False, usecols=cols)
            # df = df.drop_duplicates(['NED'], keep='first')
            # df = df.drop(['VClass', 'BClass'], axis = 1)
            datasetname = 'LARGESS'
    # %%% ICRF
        if self.name.lower() == 'icrf':  # 1493
            redshift = True
            fields = ['no', 'NED', 'redshift', 'SI', 'TO', 'flag_TO', 'type', 'L_21', 'P_21', 'L_UV',
                      'Q', 'flag_uv', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'W1', 'W2', 'W3',
                      'W4', 'SPIT_5', 'SPIT_8', 'NUV', 'FUV']
            magnames = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'W1', 'W2', 'W3',
                        'W4', 'SPIT_5', 'SPIT_8', 'NUV', 'FUV']
            cols = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'W1', 'W2', 'W3',
                    'W4', 'SPIT_5', 'SPIT_8', 'NUV', 'FUV', 'redshift']
            df = pd.read_csv(path + '/optical_data/Hunstead_mags+SPITZER_0.050.dat',
                             sep=' ', names=fields, index_col=False, usecols=cols)
            # df = df.drop_duplicates(['NED'], keep='first')

            datasetname = 'ICRF2'

    # %%% Test dataframe
        if self.name.lower() == 'test':
            redshift = True
            df = pd.read_csv(path + '/test_dataset.csv',
                             sep=',', index_col=False, header=0)

            datasetname = 'Test dataset'
            colours = False
            print('Colours cannot be computed for the test frame')
            magnames = df.columns[3:-1]

    # %%% Test dataframe 1
        if self.name.lower() == 'test1':
            redshift = True
            df = pd.read_csv(path + '/test_dataset1.csv',
                             sep=',', index_col=False, header=0)

            datasetname = 'Test dataset 1'
            magnames = df.columns[3:-1]

    # %%% Test dataframe 2
        if self.name.lower() == 'test2':
            redshift = True
            df = pd.read_csv(path + '/test_dataset2.csv',
                             sep=',', index_col=False, header=0)

            datasetname = 'Test dataset 2'
            magnames = df.columns[3:-1]

# %%% Tidy the data
        if redshift:
            df = df[df['redshift'] > 0]
            # move redshift column to the end
            df.insert(len(df.columns)-1, 'redshift', df.pop('redshift'))

        mgf = df[magnames]
        df = df.where(df != -999, np.nan)
        mgf = mgf.where(mgf != -999, np.nan)

        print('{2} sources loaded from {0} with the following bands:\n{1}\n'.format(
            datasetname, magnames, df.shape[0]))
        if self.dropna:
            df = df.dropna(axis=0, how='any')
            mgf = mgf.dropna(axis=0, how='any')
            print('NaNs have been dropped from the original data.')
        else:  # If selected to keep NaNs in dataset, show structure of missing data
            msno.matrix(df[magnames],
                        sort='descending',
                        color=(0, .6, .8),
                        # fontsize = 20,
                        figsize=(10, 6))
        if self.colours:
            # Compute colours using the magnitudes columns and add to the dataset
            colours = self.compute_colours(mgf)
            # compute colours and add colours to magnitudes dataframe
            colours = colours.iloc[:, len(magnames):]
            if redshift:
                # add redshift to the list of magnitude names
                allcolumns = magnames + ['redshift']
                # dataset is now magnitudes and redshift column
                df = df[allcolumns]
            else:
                df = df[magnames]
            # dataset is now magnitudes and colours
            df = pd.concat([df, colours], axis=1)
            print('Colours have been computed and added to the dataset.')

        if self.impute_method == 'max':
            # using max() assumes missing data are due to detection limit
            df = df.fillna(df.max())
            mgf = mgf.fillna(mgf.max())
            print('Missing values have been imputed with the maximum for each column.')
        elif self.impute_method == 'mean':
            impute_mean = SimpleImputer(missing_values=np.nan,
                                        strategy='mean')
            # impute_mean.fit(df)
            impute_mean.fit(mgf)
            # impute_mean.transform(df)
            mgf = impute_mean.transform(mgf)  # converts to np.array
            mgf = pd.DataFrame(mgf, columns=magnames)  # back to DataFrame
            print('Missing values have been imputed with the mean for each column.')

            # return the preprocessed dataframe
        return df, datasetname, magnames, mgf


# %% Test
if __name__ == "__main__":
    # create a DataLoader object for dataset 1
    data_loader = DataLoader(dropna=False,
                             colours=False,
                             impute_method='max')
    dataset, datasetname, magnames, mags = data_loader.load_data('test')
    skymap, skymapname, skymapmagnames, skymapmags = data_loader.load_data(
        'skymapper')
