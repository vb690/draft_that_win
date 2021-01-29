import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit as sss
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.callbacks import EarlyStopping

from modules.utils.general import load_arrays

from modules.models.baselines import RandomEstimator
from modules.models.neural_nets import HyperMLP, HyperRNN
from modules.models.sk_models import HyperSklearn

# ##################### DATA LOADING ################################

ARRAYS = load_arrays(
    arrays_to_load=[
        'X_one_hot',
        'X_embedded',
        'y_static',
        'y_temporal'
    ],
    folder_name='validation_arrays'
)

# ##################### INSTANTIATE MODELS ################################

RND = RandomEstimator(
    model_tag='RND'
)
RNN = HyperRNN(
    output_size=ARRAYS['y_temporal'].shape,
    model_tag='RNN'
)
RNN.load_m(
    model_name='RNN_best',
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)
MLP = HyperMLP(
    input_size=ARRAYS['X_one_hot'].shape,
    output_size=ARRAYS['y_static'].shape,
    model_tag='MLP'
)
MLP.load_m(
    model_name='MLP_best',
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)
SKL = HyperSklearn(
    model_tag='SKL'
)
SKL.load_m(
    model_name='RandomForestClassifier_SKL_best'
)
MODELS = {
    'RND': RND,
    'RNN': RNN,
    'MLP': MLP,
    'SKL': SKL
}

# ##################### EXPERIMENT PARAMETERS ################################

RESULTS_INDEX = 0
FOLD_NUMBER = 0
RESULTS_DF = pd.DataFrame(
    columns=['model', 'fold_number', 'metric_name', 'metric_value']
)

VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2
SPLITTER = sss(
    n_splits=30,
    test_size=TEST_SIZE
)
METRICS = {
    'Accuracy': accuracy_score,
    'F1 Score': f1_score
}

BATCH_SIZE = 256
EPOCHS = 40
OBJECTIVE = 'val_loss'
MIN_DELTA = 0.001
PATIENCE = 5

# ##################### RUN EXPERIMENT ################################

for tr_ind, ts_ind in SPLITTER.split(ARRAYS['X_one_hot'], ARRAYS['y_static']):

    for model_name, model_object in MODELS.items():

        print(f'Tesing {model_name} over fold {FOLD_NUMBER}')
        es = EarlyStopping(
            monitor=OBJECTIVE,
            min_delta=MIN_DELTA,
            patience=PATIENCE,
            restore_best_weights=True
        )
        if model_name == 'MLP':
            model_object.fit(
                X=ARRAYS['X_one_hot'][tr_ind],
                y=ARRAYS['y_static'][tr_ind],
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=[es],
                validation_split=VALIDATION_SIZE
            )
            predictions = model_object.predict(
                X=ARRAYS['X_one_hot'][ts_ind],
                batch_size=256
            )
        elif model_name == 'RNN':
            model_object.fit(
                X=ARRAYS['X_embedded'][tr_ind],
                y=ARRAYS['y_temporal'][tr_ind],
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=[es],
                validation_split=VALIDATION_SIZE
            )
            predictions = model_object.predict(
                X=ARRAYS['X_embedded'][ts_ind],
                batch_size=256
            )
        else:
            model_object.fit(
                X=ARRAYS['X_one_hot'][tr_ind],
                y=ARRAYS['y_static'][tr_ind]
            )
            predictions = model_object.predict(
                X=ARRAYS['X_one_hot'][ts_ind],
            )

        predictions = np.around(predictions)

        for metric_name, metric_function in METRICS.items():

            if metric_name == 'F1 Score':
                metric_value = metric_function(
                    y_true=ARRAYS['y_static'][ts_ind],
                    y_pred=predictions,
                    average='weighted'
                )
            else:
                metric_value = metric_function(
                    y_true=ARRAYS['y_static'][ts_ind],
                    y_pred=predictions
                )
            RESULTS_DF.loc[RESULTS_INDEX] = [
                model_name,
                FOLD_NUMBER,
                metric_name,
                metric_value
            ]
            RESULTS_INDEX += 1

    FOLD_NUMBER += 1

RESULTS_DF.to_csv('results\\tables\\results_performance_experiment.csv')
