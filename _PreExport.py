import sys
sys.path.insert(0, 'C:\\PythonArea\\Lib\\site-packages')
sys.path.insert(0, 'C:\\PythonArea\\Scripts')
import pickle
import imblearn
import re
import pandas as pd
import numpy as np
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from BorutaShap import BorutaShap

from pipiline__init__ import variables
from fastlane.data_ingestion import obj_to_pickle, obj_from_pickle
from fastlane import BinaryLane
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import f_classif,chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
dict_vars = variables()


# =============================================================================
# # <----- ----->
# # Import DataFrame and Dict of columns
# # <----- ----->
# =============================================================================

rootin = dict_vars['pathinp']
rootout = dict_vars['pathout']
subpath = '\\reporting'
subpathII = '\\objects'

df = obj_from_pickle(dict_vars['pathinp'],
                     'L0_Input_Dataframe.pkl')

df = df.loc['201906':'201912', :]

lista = [
    'AGENZIA',
    'AGE_SOSTITUITA',
    'ANNI_BONUS',
    'CLI_ANNI_CU',
    'CLI_BERSANI',
    'CLI_BM',
    'CLI_CAP_PRA',
    'CLI_CITTA_PRA',
    'CLI_CU',
    'CLI_ESPERIENZA_GUIDA',
    'CLI_ETA_PRA',
    'CLI_ISTAT_PRA',
    'CLI_PROV_PRA',
    'CLI_REGIONE_PRA',
    'CLI_SESSO_PRA',
    'CRIF_KPI_100',
    'CRIF_KPI_SINTESI',
    'ETA_VEIC',
    'FRAZIONAMENTO',
    'GAR_SLOT2',
    'GAR_SLOT3',
    'GAR_SLOT4',
    'GAR_SLOT5',
    'GAR_SLOT6',
    'GAR_SLOT7',
    'GAR_SLOT8',
    'PREMIO_ANNUO',
    'PREMIO_ANNUO_CVT',
    'PREMIO_ANNUO_RCA',
    'PREMIO_PROPOSTO',
    'PREMIO_PROPOSTO_CVT',
    'PREMIO_PROPOSTO_RCA',
    'Q4_FK_COD_MARCA',
    'Q4_FK_COD_MODELLO',
    'RETENTION',
    'SCONTO_RCA',
    'SCONTO_RCA_PROPOSTO',
    'SUBAGENZIA',
    'TAR_BLACK_BOX',
    'TAR_BONUS_PROTETTO',
    'TAR_MAX_RCA',
    'TAR_TIPO_GUIDA_SIC',
    'TAR_VEICOLO_AGG',
    'TAR_VERS_TARIFFA',
    'TIPO_AFFARE',
    'TIPO_ASS',
    'VEI_ALIMENTAZIONE',
    'VEI_CARROZZERIA',
    'VEI_CV',
    'VEI_MARCA',
    'VEI_POTENZA_KW',
    'd_TENUREALLm',
    'd_TENURERCm',
    'i_PREMI_RCA',
    'i_PREMI_RE',
    'n_GAR_RCA',
    'n_GAR_RE']

df = df[lista]

df.reset_index(inplace=True)

df.drop(columns='ID_POL', inplace=True)

transc = [
'1',
'2',
'3',
'4',
'5',
'6',
'7',
'8',
'9',
'10',
'11',
'12',
'13',
'14',
'15',
'16',
'17',
'18',
'19',
'20',
'21',
'22',
'23',
'24',
'25',
'26',
'27',
'28',
'29',
'30',
'31',
'32',
'33',
'34',
'35',
'36',
'37',
'38',
'39',
'40',
'41',
'42',
'43',
'44',
'45',
'46',
'47',
'48',
'49',
'50',
'51',
'52',
'53',
'54',
'55',
'56',
'57',
'58'
]

df.columns = transc

dfI = df.iloc[:70000, :]
dfI.to_pickle(rootout+'\\dfI.pkl')

dfII = df.iloc[70001:140000, :]
dfII.to_pickle(rootout+'\\dfII.pkl')
