from shared_code.unpickler import *
import numpy as np

class Model:
    def __init__(self, model=None):
        self.model = model

    def read(self, path):
        """opens the model, which is a pickle file

        Args:
            path ([string]): path to the model.pkl file

        Returns:
            model: the already trained model on which we can make predictions
        """
        with open(path, 'rb') as input_file:
            self.model = unpickler_load(input_file)
        return self

    def predict(self, pred_data):
        try:
            pred_proba = np.array([max(x) for x in self.model.clf.predict_proba(pred_data[self.model.features])])
            pred_class = self.model.clf.predict(pred_data[self.model.features])
            return np.stack([pred_class, pred_proba]).T
        except KeyError:
            raise KeyError('The model expects sensors which are missing in prediction data: ' +
                           ' '.join(list(set(self.model.features) - set(pred_data.columns))))
