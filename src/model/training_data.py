import pandas as pd


class TrainingData:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data
        self.number_of_features = None
        self.number_of_timesteps = None
        if self.data is not None:
            self.get_data_parameters()
            self.check_data_format()

    def get_data_parameters(self):
        self.number_of_features = self.data.shape[1] - 1
        self.number_of_timesteps = self.data.iloc[0, 0].__len__()

    def check_data_format(self):
        # Check features and columns
        format_ok = (list(self.data.columns) ==
                     ['feature_' + str(i) for i in range(self.number_of_features)] + ['label'])
        if not format_ok:
            raise ValueError('Data is not conform to training data port definition: Column names')

        # Check timesteps
        df_ = self.data. \
            drop('label', axis=1). \
            applymap(lambda x: x.__len__())
        format_ok = (df_ == self.number_of_timesteps).all().all()
        if not format_ok:
            raise ValueError('Data is not conform to training data port definition: Number of timesteps')
