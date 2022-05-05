from src.model.model import Model
from src.preprocessing.preprocessing_helpers import build_window_samples, correct_columns, wrangle_data, \
    read_raw_csv_data

WINDOW_SIZE = 30
MODEL_EXERCISE_PATH = "model/model.pkl"
MIN_ROWS = 650
COLUMNS = ['Time Received', 'Time Measured', 'Client', 'Value', 'ValueTwo', 'ValueThree', 'Command', 'Exercise',
           'SessionId', 'UserName', 'Info']

HORIZONTAL_DIST = 5
VERTICAL_DIST_LOWER_BOUND = 0.35
VERTICAL_DIST_UPPER_BOUND = 1000  # setting the upper bound this high renders it practically inactive. This is done because we do not need the upper bound at the moment.


def construct_message(df_path):
    """
    predicts on the data and constructs a message to later return to the app
    Parameters
    ----------
    df_path : String
        path to the csv file to predict on
    Returns
    -------
    prediction : String
        the prediction message that is then send to the app
    batch_prediction : list(String)
        list of all predicted windows (currently only used as a backup data to send to the azure cloud)
    success : bool
        True when prediction was successful
    """
    try:
        df = read_raw_csv_data(df_path)
    except Exception as ex:
        return ex
    else:
        # Approximately 300 Rows of Raw Data is one Batch, for a Prediction one Batch is enough but since we want to predict sets we need at least 2 Batches (~600 Rows)
        if df.shape[0] >= MIN_ROWS:
            model = Model().read(path=MODEL_EXERCISE_PATH)
            try:
                # delete unwanted meta information and wrangle the data
                df = correct_columns(df, COLUMNS)
                df = wrangle_data(df)

                # build samples with right window size, remove samples with size smaller than window size, reset index after row removing
                df_prediction_samples = build_window_samples(df, WINDOW_SIZE)

                # make prediction
                list_of_predicted_batches = model.predict(df_prediction_samples)

                return list_of_predicted_batches
            except Exception as ex:
                return ex
        else:
            return "Not enough data"
