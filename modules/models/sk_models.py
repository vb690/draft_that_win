from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from ..utils.models import AbstractHyperSklearn


class HyperSklearn(AbstractHyperSklearn):
    """Class implementing a hyper sklearn model, a collection of sklearn models
    to be optimized with kerastuner.

    Differently from a keras model what is optimized here are not just
    the model's hyperparameters but also the type of model.
    """
    def __init__(self, hypermodels=['lr', 'rfc', 'abc'], model_tag='SKL'):
        """Instatiate the hypermodel

        Args:
            - model_tag: string, used for describing / saving the model
            - hypermodels: iterable of strings, aliases for the hypermodels
                           to be considered during the tuning process.
        """
        self.model_tag = model_tag
        self.hypermodels = hypermodels

    def build(self, hp):
        """Kerastuner requires  to have a build method in charge
        of instating an sklearn model which will be compiled using a set
        of hyperparameters hp suggested by an Oracle.

        In this case we also tune the type of sklearn model that is going to
        be instantiated. The subsequent hyperparameters will therefore be
        conditioned on the type of model selected.

        Args:
            - hp: kerastuner hyperparameter object, object passed
                  to the Oracle for keeping track of the hp space.

        Returns:
            - model: an instance of an sklearn model
        """
        selected_model = hp.Choice(
            name='selected_model',
            values=self.hypermodels

        )
        # logistic regression
        if selected_model == 'lr':
            model = LogisticRegression(
                penalty=hp.Choice(
                    name='penalty_type',
                    values=['l1', 'l2', 'elasticnet']
                ),
                C=hp.Float(
                    name='regularization_ammount',
                    min_value=1e-5,
                    max_value=5,
                    sampling='log'
                ),
                l1_ratio=hp.Float(
                    name='l1_ratio',
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                ),
                solver='saga',
                n_jobs=-1
            )
        # random forest classifer
        elif selected_model == 'rfc':
            model = RandomForestClassifier(
                n_estimators=hp.Int(
                    name='number_trees',
                    min_value=10,
                    max_value=500,
                    step=10,
                ),
                criterion=hp.Choice(
                    name='split_criterion',
                    values=['gini', 'entropy']
                ),
                min_samples_split=hp.Float(
                    name='minimum_samples_split',
                    min_value=1e-3,
                    max_value=0.2,
                    sampling='log',
                ),
                min_samples_leaf=hp.Float(
                    name='minimum_samples_leaf',
                    min_value=1e-3,
                    max_value=0.2,
                    sampling='log',
                ),
                n_jobs=-1
            )
        # adaboost classifier
        elif selected_model == 'abc':
            model = AdaBoostClassifier(
                n_estimators=hp.Int(
                    name='number_estimator',
                    min_value=10,
                    max_value=500,
                    step=10,
                ),
                learning_rate=hp.Float(
                    name='learning_rate',
                    min_value=1e-4,
                    max_value=5,
                    sampling='log'
                )
            )

        return model
