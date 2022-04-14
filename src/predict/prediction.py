import numpy as np
from src.model.model import Model
from src.preprocessing.preprocessing_helpers import build_window_samples, correct_columns, wrangle_data, read_raw_csv_data, read_raw_json_data
from src.predict.counting_reps import integrate_acceleration

WINDOW_SIZE = 30
MODEL_EXERCISE_PATH = "model/model.pkl"
MIN_ROWS = 650
COLUMNS = ['Time Received', 'Time Measured', 'Client', 'Value', 'ValueTwo', 'ValueThree', 'Command', 'Exercise', 'SessionId', 'UserName', 'Info']

HORIZONTAL_DIST = 5
VERTICAL_DIST_LOWER_BOUND = 0.35
VERTICAL_DIST_UPPER_BOUND = 1000 # setting the upper bound this high renders it practically inactive. This is done because we do not need the upper bound at the moment.

def construct_message(df_path):
    try:
        df_path = df_path.replace("\\", "/")[2:]
        df = read_raw_csv_data(df_path)
    except Exception as ex:
        return ex, False
    else:
        # Approximately 300 Rows of Raw Data is one Batch, for a Prediction one Batch is enough but since we want to predict sets we need atleast 2 Batches (~600 Rows)
        if df.shape[0] >= MIN_ROWS:
            # read model
            model = Model().read(path = MODEL_EXERCISE_PATH)

            try:
                # delete unwanted meta informaton and wrangle the data
                df = correct_columns(df, COLUMNS)
                df = wrangle_data(df)

                # build samples with right window size, remove samples with size smaller than window size, reset index after row removing
                df_prediction_samples = build_window_samples(df, WINDOW_SIZE)

                # make prediction
                list_of_predicted_batches = model.predict(df_prediction_samples)
                #print(f"\nPrediction:\n{list_of_predicted_batches} [{len(list_of_predicted_batches)}]\n")

                # get exercise, probability, repetitions and bool if data should be deleted afterwards (DATA IS CURRENTLY NEVER DELETED, IMPLEMENTED LATER)
                exercise, probability, repetitions, delete_file = predict_df(list_of_predicted_batches, df)

                # start and end time of the current dataframe
                exercise_start_time = str(df.loc[df.index[0], "Time Measured"])
                exercise_end_time = str(df.loc[df.index[-1], "Time Measured"])

                # send message to app
                message = f"\nStart Time: {exercise_start_time[11:19]}\nEnd Time: {exercise_end_time[11:19]}\nExercise: {exercise}\
                    \nRepetitions: {repetitions}\nProbability: {probability:.2f}\n"
                return message, True
            except Exception as ex:
                return ex, False
        else:
            return f"Not enough data to make a Prediction [{df.shape[0]}/{MIN_ROWS} rows]\n", False

# only predicts full sets which are considers full when there are at least two nothing batches following the set
def predict_set(list_of_predicted_batches, df):
    # List that contains only the predicted exercise (not the probability)
    list_of_all_exercises = [item[0] for item in list_of_predicted_batches]
    # The DF contains only 'nothing' so there is no exercise
    if len(list_of_all_exercises) == list_of_all_exercises.count('nothing'):
        return 'pause', np.NaN, np.NaN, False
    # There is a exercise in the DF, but is it finished yet?
    elif len(list_of_all_exercises) != list_of_all_exercises.count('nothing'):
        # There are 'nothing' batches at the end of the DF so the Set should be finished
        if list_of_all_exercises[-2] == 'nothing' and list_of_all_exercises[-1] == 'nothing':
            # Looks up which exercises is the most common
            unique_element, position = np.unique(list_of_all_exercises, return_inverse = True)
            type_of_exercise = str(unique_element[(np.bincount(position)).argmax()])
            # The most common prediction is nothing (but since there is an exercise we need to find the second most common now)
            if type_of_exercise == 'nothing':
                list_of_all_exercises = [item[0] for item in list_of_predicted_batches if item[0] != 'nothing']
                unique_element, position = np.unique(list_of_all_exercises, return_inverse = True)
                type_of_exercise = str(unique_element[(np.bincount(position)).argmax()])
                list_of_predicted_exercises = [float(item[1]) for item in list_of_predicted_batches if item[0] == type_of_exercise]
                repetitions = integrate_acceleration(df, type_of_exercise, HORIZONTAL_DIST, VERTICAL_DIST_LOWER_BOUND, VERTICAL_DIST_UPPER_BOUND)
                return type_of_exercise, (np.sum(list_of_predicted_exercises) / float(np.shape(list_of_predicted_exercises)[0])), repetitions, True
            # 'nothing' is NOT the most common exercise
            else:
                list_of_predicted_exercises = [float(item[1]) for item in list_of_predicted_batches if item[0] == type_of_exercise]
                repetitions = integrate_acceleration(df, type_of_exercise, HORIZONTAL_DIST, VERTICAL_DIST_LOWER_BOUND, VERTICAL_DIST_UPPER_BOUND)
                return type_of_exercise, (np.sum(list_of_predicted_exercises) / float(np.shape(list_of_predicted_exercises)[0])), repetitions, True
        # There are no 'nothing' batches at the end of the DF so the Set might still be running
        elif list_of_all_exercises[-2] != 'nothing' or list_of_all_exercises[-1] != 'nothing':
            return 'not finished', np.NaN, np.NaN, False

# does not wait for sets to be finished, just predicts everything in the df
def predict_df(list_of_predicted_batches, df):
    # List that contains only the predicted exercise (not the probability)
    list_of_all_exercises = [item[0] for item in list_of_predicted_batches]
    # The DF contains only 'nothing' so there is no exercise
    if len(list_of_all_exercises) == list_of_all_exercises.count('nothing'):
        return 'pause', np.NaN, np.NaN, False
    # There is a exercise in the DF, but is it finished yet?
    else:
        # Looks up which exercises is the most common
        unique_element, position = np.unique(list_of_all_exercises, return_inverse = True)
        type_of_exercise = str(unique_element[(np.bincount(position)).argmax()])
        # The most common prediction is nothing (but since there is an exercise we need to find the second most common now)
        if type_of_exercise == 'nothing':
            list_of_all_exercises = [item[0] for item in list_of_predicted_batches if item[0] != 'nothing']
            unique_element, position = np.unique(list_of_all_exercises, return_inverse = True)
            type_of_exercise = str(unique_element[(np.bincount(position)).argmax()])
            list_of_predicted_exercises = [float(item[1]) for item in list_of_predicted_batches if item[0] == type_of_exercise]
            repetitions = integrate_acceleration(df, type_of_exercise, HORIZONTAL_DIST, VERTICAL_DIST_LOWER_BOUND, VERTICAL_DIST_UPPER_BOUND)
            return type_of_exercise, (np.sum(list_of_predicted_exercises) / float(np.shape(list_of_predicted_exercises)[0])), repetitions, True
        # 'nothing' is NOT the most common exercise
        else:
            list_of_predicted_exercises = [float(item[1]) for item in list_of_predicted_batches if item[0] == type_of_exercise]
            repetitions = integrate_acceleration(df, type_of_exercise, HORIZONTAL_DIST, VERTICAL_DIST_LOWER_BOUND, VERTICAL_DIST_UPPER_BOUND)
            return type_of_exercise, (np.sum(list_of_predicted_exercises) / float(np.shape(list_of_predicted_exercises)[0])), repetitions, True
