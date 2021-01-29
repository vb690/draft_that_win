import numpy as np


class RandomEstimator:
    """
    Class implementing a 'random estimator'. The estimator will retrieve
    the parameters of the Binomial Distrubution describing the radiant_win
    event from the training set.
    """
    def __init__(self, model_tag='RND', number_of_simulations=1e4):
        """
        Instatiate the RandomEstimator

        Args:
            - model_tag: string, used for describing / saving the model

            - number_of_simulations: integer, specify the number of bernoulli
              trials perfomed when producing a random estimation. The greater
              is the value of this parameter the lower is the variability in
              the estimations. With number_of_simulations -> inf the
              variability in estimations -> 0 and equals the probability of
              radiant_win event in the training set.
        """
        self.model_tag = model_tag
        self.number_of_simulations = number_of_simulations

    def fit(self, X, y, **kwargs):
        """Method for computing the parameters of the parameters of the
        Binomial Distrubution describing the radiant_win event.

        Args:
            - X: numpy array, kept only for consistency
            - y: numpy array, output data for fitting the model
            - **kwargs: keyword arguments, kept only for consistency

        Returns:
            - fitted_model: compiled sklearn model, cloned version of
                            best_model with its parameters updated with respect
                            to X and y
        """
        number_of_matches = len(y)
        probability_radiant_win = np.sum(y) / number_of_matches
        setattr(self, 'probability_radiant_win', probability_radiant_win)

    def predict(self, X, **kwarg):
        """For each element in y simulate number_of_simulations bernoulli
        trials with p equal to the p of radiant_win event in the training
        set.

        Args:
            - X: numpy array, input data for making predictions with the model
            - y: numpy array, ground truth against which the model is tested
            - **kwargs: keyword arguments kept for consistency

        Returns:
            - predictions: numpy array of shape(y.shape[0], y.shape[1]),
                           predictions produced by the model given X.
        """
        predictions = np.random.binomial(
            n=self.number_of_simulations,
            p=self.probability_radiant_win,
            size=X.shape[0]
        )
        # we divide by n for having actual probability
        predictions = predictions / self.number_of_simulations
        predictions = np.round(predictions, decimals=2)
        predictions = predictions.reshape(-1, 1)
        return predictions
