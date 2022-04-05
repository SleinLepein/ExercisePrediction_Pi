import numpy as np
import pandas as pd
import datetime as dt
import joblib

def read_raw_data(path):
    """ Reads in the raw data

    Parameters
    ----------
    path : PATH
    file_format : FORMAT (CSV, JSON)

    Returns
    -------
    pd.DataFrame
        Containing the raw data

    """
    return pd.read_csv(path, sep=';')

def correct_columns(df, COLUMNS):
    """

    Parameters
    ----------
    df : pd.DataFrame
        Containing the raw DataFrame

    COLUMNS
        List containing the correct column names and in order

    Returns
    -------
    pd.DataFrame
        DataFrame without unnecessary columns and columns in the right order

    """
    columns_not_needed = list(set(list(df)).symmetric_difference(set(COLUMNS)))
    df = df.drop(columns_not_needed, axis=1)
    df = df[COLUMNS]
    return df

def time_binning(df, sampling_rate_for_time_binnings='200ms'):
    """

    Parameters
    ----------
    df : pd.DataFrame
        Containing the raw Data

    sampling_rate_for_time_binning
        String describing the sample rate

    Returns
    -------
    pd.DataFrame
        Time-binned data

    """
    df['Time Measured'] = df['Time Measured'].apply(lambda x: x[:-1] if type(x) == str else x)
    df['Time Measured'] = pd.to_datetime(df['Time Measured'])
    df['Time Measured'] = df['Time Measured'].dt.round(sampling_rate_for_time_binnings)
    df['Time Measured'] = df['Time Measured'].fillna(method='ffill').fillna(method='bfill')
    return df

def get_commands_data(df):
    """ Pick the command infos,
    that means meta-infos containing
        - exercise name,
        - session start/stop,
        - user-id
        - etc
    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Only rows containing meta infos
    """
    return df[df['Command'].notna()][['Time Received', 'Time Measured', 'Command', 'Exercise', 'SessionId', 'UserName', 'Info']]

def spread_sensors_to_columns(df: pd.DataFrame):
    """ Spread sensor-values in a pivot table to make each column represent an individual sensor-dimension

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = pd.pivot_table(df, index='Time Measured', columns='Client', values=['Value', 'ValueTwo', 'ValueThree'], aggfunc=np.mean)
    df.columns = [col[1] + "_" + col[0] for col in df.columns.values]
    df = df.reset_index()
    return df

def sync_data_with_continuously_sampled_timeline(df):
    """ Syncronise data with a coherent timeline. This makes the time steps equal from row to row.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df_timeline = pd.DataFrame()
    df_timeline['Time Measured'] = np.arange(df['Time Measured'].min(), df['Time Measured'].max()+dt.timedelta(milliseconds=200), np.timedelta64(200, 'ms'))
    df = pd.merge_asof(df_timeline, df)
    return df

def add_commands_info(df, df_commands):
    """ Add meta info to preprocessed sensor data

    Parameters
    ----------
    df : pd.DataFrame
        sensor data
    df_commands : pd.DataFrame
        meta data

    Returns
    -------
    pd.DataFrame
    """
    return df.merge(df_commands, how='left')

def construct_input_conditions(pred_exercise, one_hot_encoder_path, window_size):
    """ Transforms the predicted labels of the exercise classification model into one-hot vectors and return the stacked one-hot vectors.

    Parameters
    ----------
    pred_exercise: list(str)
        List of exercise labels. Contains one label per sample in df_pred_samples.
    one_hot_encoder_path: str
        Path to load the one hot encoder.
    window_size: int
        Size of the input window.
    Returns
    -------
    np.array
    """
    one_hot_encoder = joblib.load(one_hot_encoder_path)
    exercises_one_hot = one_hot_encoder.transform(np.array(pred_exercise).reshape((-1,1))).toarray()

    exercises_one_hot_extended = []
    for i in range(exercises_one_hot.shape[0]):
        cond_list = [exercises_one_hot[i]]*window_size
        exercises_one_hot_extended.extend(cond_list)

    number_of_categories = exercises_one_hot.shape[1]
    exercises_one_hot_extended = np.asarray(exercises_one_hot_extended).reshape(-1, window_size, number_of_categories)

    return exercises_one_hot_extended


def construct_input_sensor_data(data):
    """
        Takes the sensor data arrays from the dataframe cells and puts them into a 2d numpy array, which will be used as input for the keras anomaly detection model.
        Parameters
        ----------
        data : pd.DataFrame
            Contains the windowed sensor input data.

        Returns
        ---------
        np.array
    """
    row_arrays_time_series = []
    for idx, row in data.iterrows():
        cell_arrays_time_series = []
        for col in data.columns:
            # maybe not needed in case label and exercise already filtered out.
            if col == 'Exercise' or col == 'label':
                continue

            cell_arrays_time_series.append(np.array(row[col].values))

        row_arrays_time_series.append(np.stack(cell_arrays_time_series, axis=0).T)

    return np.stack(row_arrays_time_series, axis=0)

def wrangle_data(df):
    df = time_binning(df)
    df = spread_sensors_to_columns(df)
    df = sync_data_with_continuously_sampled_timeline(df)
    df.fillna(axis=0, method='ffill', inplace=True)
    df.fillna(axis=0, method='bfill', inplace=True)
    return df


def build_window_samples(df, WINDOW_SIZE):
    """
        Builds Windows for the prediction, removes all samples with smaller size than the window size and resets index afterwards
        ----------
        df : pd.DataFrame
            Contains Dataframe
        WINDOW_SIZE: int
            The size for the windows

        Returns
        ---------
        np.array
    """
    # build samples with right window size
    df_prediction_samples = pd.DataFrame()
    df.reset_index(drop=True, inplace=True)
    for col in df.columns:
        df_prediction_samples[col] = [x[1] for x in df[col].groupby(df.index//WINDOW_SIZE)]

    # remove samples with size smaller than window size
    minimum_sample_size_this_row = df_prediction_samples.apply(lambda x: min([len(y) for y in x]), axis=1)
    df_prediction_samples = df_prediction_samples[minimum_sample_size_this_row == WINDOW_SIZE]

    # reset index after row removing
    df_prediction_samples.reset_index(inplace=True, drop=True)
    df_prediction_samples.drop(['Time Measured'], axis=1, inplace=True)

    # if labels:
    #     relevant_sensors = ['gyro_cable_left_ValueThree',
    #                         'gyro_cable_right_ValueThree',
    #                         'gyro_bar_ValueThree',
    #                         'gyro_bench_legs_ValueThree',
    #                         'label']
    # else:
    #     relevant_sensors = ['gyro_cable_left_ValueThree',
    #                         'gyro_cable_right_ValueThree',
    #                         'gyro_bar_ValueThree',
    #                         'gyro_bench_legs_ValueThree'
    #                         ]
    # df = df[relevant_sensors]
    return df_prediction_samples
    