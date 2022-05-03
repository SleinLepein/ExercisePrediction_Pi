import numpy as np
from src.model.model import Model
from src.preprocessing.preprocessing_helpers import build_window_samples, correct_columns, wrangle_data, \
    read_raw_csv_data
from src.predict.counting_reps import integrate_acceleration

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
    finished : bool
        True when set is finished
    """
    try:
        df = read_raw_csv_data(df_path)
    except Exception as ex:
        return ex, None, False, False
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
                batch_prediction = f"\n{list_of_predicted_batches}\n"

                # get exercise, probability, repetitions and bool if data should be deleted afterwards (DATA IS CURRENTLY NEVER DELETED, IMPLEMENTED LATER)
                exercise, probability, repetitions, set_finished = predict_df(list_of_predicted_batches, df)

                # start and end time of the current dataframe
                exercise_start_time = str(df.loc[df.index[0], "Time Measured"])
                exercise_end_time = str(df.loc[df.index[-1], "Time Measured"])

                # construct message
                message = f"\nStart Time: {exercise_start_time[11:19]}\nEnd Time: {exercise_end_time[11:19]}\nExercise: {exercise}\
                    \nRepetitions: {repetitions}\nProbability: {probability:.2f}\nSet finished: {set_finished}\n"
                return message, batch_prediction, True, set_finished
            except Exception as ex:
                return ex, None, False, False
        else:
            return f"Not enough data to make a Prediction [{df.shape[0]}/{MIN_ROWS} rows]\n", None, False, False


def predict_df(list_of_predicted_batches, df):
    """
    checks what exercise is the most common, the probability of the prediction,
    the amount of repetitions and if the file can be deleted
    Parameters
    ----------
    list_of_predicted_batches : list(String)
        list of all windows and the predicted exercise
    df : pd.DataFrame
        dataframe containing all sensor data (necessary for repetition count func)
    Returns
    -------
    exercise : String
        the most common exercise in the df
    probability :  float
        the probability of the prediction
    repetitions : int
        the amount of repetitions in the df
    set_finished : bool
        True when there are pauses at the end of the df so that the set might be finished
    """
    # List that contains only the predicted exercise (not the probability)
    list_of_all_exercises = [item[0] for item in list_of_predicted_batches]
    # The DF contains only 'nothing' so there is no exercise
    if len(list_of_all_exercises) == list_of_all_exercises.count('nothing'):
        return 'pause', np.NaN, np.NaN, False
    # There is a exercise in the DF, but is it finished yet?
    else:
        # Looks up which exercises is the most common
        unique_element, position = np.unique(list_of_all_exercises, return_inverse=True)
        type_of_exercise = str(unique_element[(np.bincount(position)).argmax()])
        # if the last two batches of the df are 'nothing' the set might be finished
        if list_of_all_exercises[-2] == 'nothing' and list_of_all_exercises[-1] == 'nothing':
            set_finished = True
        else:
            set_finished = False
        # The most common prediction is nothing (but since there is an exercise we need to find the second most common now)
        if type_of_exercise == 'nothing':
            list_of_all_exercises = [item[0] for item in list_of_predicted_batches if item[0] != 'nothing']
            unique_element, position = np.unique(list_of_all_exercises, return_inverse=True)
            type_of_exercise = str(unique_element[(np.bincount(position)).argmax()])
            list_of_predicted_exercises = [float(item[1]) for item in list_of_predicted_batches if
                                           item[0] == type_of_exercise]
            repetitions = integrate_acceleration(df, type_of_exercise, HORIZONTAL_DIST, VERTICAL_DIST_LOWER_BOUND,
                                                 VERTICAL_DIST_UPPER_BOUND)
            return type_of_exercise, (np.sum(list_of_predicted_exercises) / float(
                np.shape(list_of_predicted_exercises)[0])), repetitions, set_finished
        # 'nothing' is NOT the most common exercise
        else:
            list_of_predicted_exercises = [float(item[1]) for item in list_of_predicted_batches if
                                           item[0] == type_of_exercise]
            repetitions = integrate_acceleration(df, type_of_exercise, HORIZONTAL_DIST, VERTICAL_DIST_LOWER_BOUND,
                                                 VERTICAL_DIST_UPPER_BOUND)
            return type_of_exercise, (np.sum(list_of_predicted_exercises) / float(
                np.shape(list_of_predicted_exercises)[0])), repetitions, set_finished
