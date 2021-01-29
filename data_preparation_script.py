import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit as sss

from modules.utils.general import save_arrays

TUNING_SIZE = 0.2
SPLITTER = sss(
    n_splits=1,
    test_size=TUNING_SIZE

)

DATA = pd.read_csv(
    'data\\csv\\single_patch_immortals.csv',
    dtype=np.int8
)

TUNING_ARRAYS = {}
VALIDATION_ARRAYS = {}

# ##################### ARRAY TRANSFORMATIONS ################################

X_one_hot = DATA.drop('radiant_win', axis=1).values

X_embedded = np.argwhere(X_one_hot == 1)
X_embedded = X_embedded[:, 1]
X_emebdded = X_embedded.reshape(X_one_hot.shape[0], 10)

y_static = DATA['radiant_win'].map({True: 1, False: 0}).values.reshape(-1, 1)

win_indices = np.argwhere(y_static == 1).flatten()
y_temporal = np.zeros(shape=(X_one_hot.shape[0], 10, 1))
y_temporal[win_indices, :, :] = 1

# ############################# SPLIT THE ARRAYS ############################

for validation_index, tuning_index in SPLITTER.split(X_one_hot, y_static):

    TUNING_ARRAYS['X_one_hot'] = X_one_hot[tuning_index]
    TUNING_ARRAYS['X_embedded'] = X_emebdded[tuning_index]
    TUNING_ARRAYS['y_static'] = y_static[tuning_index]
    TUNING_ARRAYS['y_temporal'] = y_temporal[tuning_index]

    VALIDATION_ARRAYS['X_one_hot'] = X_one_hot[validation_index]
    VALIDATION_ARRAYS['X_embedded'] = X_emebdded[validation_index]
    VALIDATION_ARRAYS['y_static'] = y_static[validation_index]
    VALIDATION_ARRAYS['y_temporal'] = y_temporal[validation_index]

# ######################## SAVE ARRAYS LOCALLY ##############################

save_arrays(
    arrays_to_save=TUNING_ARRAYS,
    folder_name='tuning_arrays'
)

save_arrays(
    arrays_to_save=VALIDATION_ARRAYS,
    folder_name='validation_arrays'
)
