from sklearn.pipeline import Pipeline
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.transformations.panel.compose import ColumnConcatenator
from src.model.training_data import TrainingData
from src.model.model import Model


class ForestClassifierTrainer:
    """
    Class providing a training method for a time series forest classifier
    """

    def __init__(self, n_estimators: int = 100):
        """
        Parameters
        ----------
        n_estimators: Number of estimators: see sktime.classification.interval_based.TimeSeriesForestClassifier
        """
        self.n_estimators = n_estimators
        self.features = None
        steps = [("concatenate", ColumnConcatenator()),
                 ("classify", TimeSeriesForestClassifier(n_estimators=self.n_estimators)),
                 ]
        self.clf = Pipeline(steps)

    def fit(self, training_data: TrainingData):
        """
        Fit method to train forest classifier
        Parameters
        ----------
        training_data: TrainingData-Object

        """
        x = training_data.data.drop('label', axis=1)
        y = training_data.data['label']
        self.features = x.columns
        self.clf.fit(x, y)
        return Model(self)
