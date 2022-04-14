import numpy as np
from src.predict.counting_reps import integrate_acceleration

HORIZONTAL_DIST = 5
VERTICAL_DIST_LOWER_BOUND = 0.35
VERTICAL_DIST_UPPER_BOUND = 1000 # setting the upper bound this high renders it practically inactive. This is done because we do not need the upper bound at the moment.

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
