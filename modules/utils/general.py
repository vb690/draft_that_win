import os

import numpy as np

import matplotlib.pyplot as plt


def save_arrays(arrays_to_save, folder_name):
    """Function for saving list of arrays with target names

    Args:
        - arrays_to_save: dictionary, keys are string specifying the name
                          with which the array will be saved, values are
                          numpy arrays
        - folder_name:    string, location in data\\saved_arrays where the
                          arrays will be saved
    Returns:
        - None
    """
    path = 'data\\saved_arrays\\{}'.format(folder_name)
    if not os.path.exists(path):
        os.mkdir(path)

    for name, array in arrays_to_save.items():

        file_path = '{}\\{}.npy'.format(path, name)
        np.save(file_path, array)

    return None


def load_arrays(arrays_to_load, folder_name):
    """Function for loading arrays based on target names

    Args:
        - arrays_to_load: list of strings, name of the arrays to be loaded

    Returns:
        - loaded_arrays: is a dictionary, keys are arrays name
                         values are corresponding loaded_arrays
    """
    loaded_arrays = {}
    path = 'data\\saved_arrays\\{}'.format(folder_name)
    for name in arrays_to_load:

        file_path = '{}\\{}.npy'.format(path, name)
        loaded_array = np.load(file_path)
        loaded_arrays[name] = loaded_array

    return loaded_arrays


def plot_history(history, metrics):
    """
    Function for plotting and saving the training history produced by a
    fitted keras model.

    Args:
        - history: history object, object produced at the end of the training
                   process for a keras model. It contains information collected
                   during training.

        - metrics: iterable of strings, EXACT aliases of loss and metrics
                   used during training.

    Returns:
        - None
    """
    for metric in metrics:

        plt.plot(
            history.history[metric],
            label='Train'
        )
        plt.plot(
            history.history[f'val_{metric}'],
            label='Validation'
        )
        plt.title('Training History')
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend(
            loc='upper left'
        )
        plt.ylim((0., 1.1))
        plt.savefig(f'results\\figures\\{metric}.png')
        plt.clf()

    return None
