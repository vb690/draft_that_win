from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit as sss

from kerastuner.tuners import Hyperband as hb
from kerastuner.oracles import Hyperband as hbo
from kerastuner import Objective

from tensorflow.keras.callbacks import EarlyStopping

from modules.utils.general import load_arrays

from modules.models.neural_nets import HyperMLP, HyperRNN
from modules.models.sk_models import HyperSklearn

# ########################### GLOBAL VARIABLES ###############################

HYPERBAND_MAX_EPOCHS = 40
HYPERBAND_ITERATIONS = 5
OBJECTIVE = 'val_loss'
MAX_BEST_MODELS = 3
VALIDATION_SIZE = 0.2

MIN_DELTA = 0.001
PATIENCE = 5

ARRAYS = load_arrays(
    arrays_to_load=[
        'X_one_hot',
        'X_embedded',
        'y_static',
        'y_temporal'
    ],
    folder_name='tuning_arrays'
)

# ###################### RECURRENT NEURAL NETWORK ############################

es = EarlyStopping(
    monitor=OBJECTIVE,
    min_delta=MIN_DELTA,
    patience=PATIENCE,
    restore_best_weights=True
)

model = HyperRNN(
    output_size=ARRAYS['y_temporal'].shape,
    model_tag='RNN'
)

best_models = model.tune(
    X=ARRAYS['X_embedded'],
    y=ARRAYS['y_temporal'],
    tuner=hb,
    epochs=HYPERBAND_MAX_EPOCHS,
    callbacks=[es],
    verbose=1,
    val_split=VALIDATION_SIZE,
    num_models=MAX_BEST_MODELS,
    objective=OBJECTIVE,
    max_epochs=HYPERBAND_MAX_EPOCHS,
    hyperband_iterations=HYPERBAND_ITERATIONS,
    directory='o',
    project_name='RNN'
)

model.save_m()

# ###################### MULTILAYER PERCEPTRON ###############################

es = EarlyStopping(
    monitor=OBJECTIVE,
    min_delta=MIN_DELTA,
    patience=PATIENCE,
    restore_best_weights=True
)

model = HyperMLP(
    input_size=ARRAYS['X_one_hot'].shape,
    output_size=ARRAYS['y_static'].shape,
    model_tag='MLP'
)

best_models = model.tune(
    X=ARRAYS['X_one_hot'],
    y=ARRAYS['y_static'],
    tuner=hb,
    epochs=HYPERBAND_MAX_EPOCHS,
    callbacks=[es],
    verbose=1,
    val_split=VALIDATION_SIZE,
    num_models=MAX_BEST_MODELS,
    objective=OBJECTIVE,
    max_epochs=HYPERBAND_MAX_EPOCHS,
    hyperband_iterations=HYPERBAND_ITERATIONS,
    directory='o',
    project_name='MLP'
)

model.save_m()

# ###################### SKLEARN MODELS ######################################

model = HyperSklearn(model_tag='SKL')

best_models = model.tune(
    X=ARRAYS['X_one_hot'],
    y=ARRAYS['y_static'].flatten(),
    oracle=hbo(
        max_epochs=HYPERBAND_MAX_EPOCHS,
        hyperband_iterations=HYPERBAND_ITERATIONS,
        objective=Objective(
            'score',
            direction='max'
        )
    ),
    num_models=MAX_BEST_MODELS,
    scoring=make_scorer(accuracy_score),
    cv=sss(n_splits=1, test_size=VALIDATION_SIZE),
    directory='o',
    project_name='SKL'
)

model.save_m()
