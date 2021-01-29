from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ..utils.models import AbstractHyperNeuralNet


class HyperMLP(AbstractHyperNeuralNet):
    '''
    Class implementing and hyper MultiLayer Perceptron, a MLP written
    to be optimized with kerastuner.
    '''
    def __init__(self, input_size, output_size, final_activation='sigmoid',
                 max_depth=2, loss='binary_crossentropy', metrics=['acc'],
                 model_tag='MLP', clone_dir='results\\saved_models'):
        '''
        Instatiate the class

        Args:
            - input_size: integer, the number of features in the input tensor
            - output_size: integer, the number of targets in the output tensor
            - final_activation: string or callable, final activation function
            - max_depth: integer, specifying the maximum number of layers for
                         the dense block.
            - loss: string or callable, loss function used by the model
            - metrics: iterable, collecion of metrics computed on the train
                       and validation set.
            - model_tag: string, used for describing / saving the model

        '''
        self.input_size = input_size
        self.output_size = output_size
        self.final_activation = final_activation
        self.max_depth = max_depth
        self.model_tag = model_tag
        self.loss = loss
        self.metrics = metrics
        self.clone_dir = clone_dir

    def build(self, hp):
        '''
        Kerastuner requires each Hypermodel to have a build method in charge
        of building the computational graph using the hyperparameters (hp)
        suggested byt the Oracle

        Args:
            - hp: kerastuner hyperparameter object, object passed
                  to the Oracle for keeping track of the hp space

        Returns:
            - model: a compiled keras model
        '''
        bn = hp.Boolean(
            name='batch_normalization'
        )
        do = hp.Float(
            min_value=0.0,
            max_value=0.4,
            step=0.1,
            name='dropout_rate'
        )
        model_inp = Input(shape=(self.input_size[1],))
        dense = self._fc_block(
            hp=hp,
            input_tensor=model_inp,
            bn=bn,
            do=do,
            max_depth=self.max_depth,
        )
        model_out = Dense(self.output_size[1])(dense)
        model_out = Activation(self.final_activation)(model_out)

        model = Model(model_inp, model_out)
        model.compile(
            loss=self.loss,
            metrics=self.metrics,
            optimizer=Adam(
                learning_rate=hp.Float(
                    name='learning_rate',
                    min_value=1e-6,
                    max_value=0.01,
                    sampling='log'
                )
            )
        )

        return model


class HyperRNN(AbstractHyperNeuralNet):
    """
    Class implementing and hyper RNN, a RNN written
    to be optimized with kerastuner.
    """
    def __init__(self, output_size, final_activation='sigmoid',
                 max_depth_dense=2, max_depth_recurrent=2,
                 loss='binary_crossentropy', metrics=['acc'], model_tag='RNN',
                 clone_dir='results\\saved_models'):
        '''
        Instatiate the class, input size is not needed because we want this
        model to be able to handle sequences of variable length.

        Args:
            - output_size: integer, the number of targets in the output tensor
            - final_activation: string or callable, final activation function
            - max_depth_dense: integer, specifying the maximum number of layers
                               for the dense block.
            - loss: string or callable, loss function used by the model
            - metrics: iterable, collecion of metrics computed on the train
                       and validation set.
            - model_tag: string, used for describing / saving the model

        '''
        self.output_size = output_size
        self.final_activation = final_activation
        self.max_depth_dense = max_depth_dense
        self.max_depth_recurrent = max_depth_recurrent
        self.model_tag = model_tag
        self.loss = loss
        self.metrics = metrics
        self.clone_dir = clone_dir

    def build(self, hp):
        '''
        Kerastuner requires each Hypermodel to have a build method in charge
        of building the computational graph using the hyperparameters (hp)
        suggested byt the Oracle

        Args:
            - hp: kerastuner hyperparameter object, object passed
                  to the Oracle for keeping track of the hp space

        Returns:
            - model: a compiled keras model
        '''
        bn = hp.Boolean(
            name='batch_normalization'
        )
        do = hp.Float(
            min_value=0.0,
            max_value=0.4,
            step=0.1,
            name='dropout_rate'
        )
        model_inp = Input(shape=(None,))
        embedding = self._emb_block(
            hp=hp,
            input_tensor=model_inp,
            bn=bn,
            do=do,
        )
        rnn = self._rnn_block(
            hp=hp,
            input_tensor=embedding,
            bn=bn,
            max_depth=self.max_depth_recurrent
        )
        dense = self._fc_block(
            hp=hp,
            input_tensor=rnn,
            bn=bn,
            do=do,
            max_depth=self.max_depth_dense
        )
        model_out = Dense(self.output_size[2])(dense)
        model_out = Activation(self.final_activation)(model_out)

        model = Model(model_inp, model_out)
        model.compile(
            loss=self.loss,
            metrics=self.metrics,
            optimizer=Adam(
                learning_rate=hp.Float(
                    name='learning_rate',
                    min_value=1e-6,
                    max_value=0.01,
                    sampling='log'
                )
            )
        )

        return model

    def predict(self, X, **kwargs):
        """Method warpping sklearn predict method.
        This override the inherithed predict method because a RNN will
        produce an estimation for each time step in the input, however we are
        only interested in the last step since it correspond to the probability
        of Radiant winning given the entire set of observed picks.

        Args:
            - X: numpy array, input data for making predictions with the model
            - y: numpy array, ground truth against which the model is tested
            - **kwargs: keyword arguments passed to the model's predict method
        Returns:
            - predictions: numpy array of shape
                           (y.shape[0], y.shape[1], y.shape[2]),
                           predictions produced by the model given X.
        """
        if not hasattr(self, 'fitted_model'):
            raise(AttributeError('No fitted model is present'))

        predictions = self.fitted_model.predict(X, **kwargs)
        predictions = predictions[:, -1, :]
        return predictions
