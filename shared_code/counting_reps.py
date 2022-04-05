import numpy as np
import pandas as pd
from scipy import integrate
from scipy.signal import find_peaks

def integrate_acceleration(df, exercise, horizontal_dist, vertical_dist_lower_bound, vertical_dist_upper_bound):
    """
    df: pandas.Dataframe
        Contains the input dataframe.
    exercise: str
        Label of the exercise.
    horizontal_dist: int
        Minimum number of timesteps between to repetitions.
    vertical_dist_lower_bound: float
        Determines the minimum vertical distance between a repetition and the surrounding points in the signal.
    vertical_dist_upper_bound: float
        Determines the maximum vertical distance between a repetition and the surrounding points in the signal.
    """

    # Not all mappings are final
    exercise_sensor_mapping = { 
                                'Ausfallschritte': ['gyro_bar_Value', 'gyro_bar_ValueTwo', 'gyro_bar_ValueThree'],
                                'Brustpresse': ['gyro_bar_Value', 'gyro_bar_ValueTwo', 'gyro_bar_ValueThree'],
                                'Biceps Curl': ['gyro_cable_left_ValueThree', 'gyro_cable_right_ValueThree'],
                                'Bizeps_free': ['gyro_cable_left_Value', 'gyro_cable_left_ValueTwo', 'gyro_cable_left_ValueThree'],
                                'Crossover': ['gyro_cable_left_ValueThree', 'gyro_cable_right_ValueThree'],
                                'Squats': ['gyro_bar_ValueThree'],
                                'Trizepskabelzug': ['gyro_cable_left_ValueThree', 'gyro_cable_right_ValueThree'],
                                'Beinstrecker': ['']
                               }

    relevant_sensor_list = exercise_sensor_mapping[exercise]
    
    # select the gyro sensor with the highest standard deviation to approximate the repetitions.
    selected_sensor = ''
    max_std = 0.0
    for sensor in relevant_sensor_list:
        if 'gyro' in sensor:
            tmp_std = df[sensor].std()
            if tmp_std > max_std:
                selected_sensor = sensor
                max_std = tmp_std

    # normalize acceleration sensor values and calculate velocity values
    values_acceleration = df[selected_sensor]
    values_acceleration_normalized = values_acceleration - values_acceleration.rolling(10).mean()
    values_acceleration_normalized.dropna(axis=0, inplace=True)
    values_velocity = pd.Series(integrate.cumtrapz(values_acceleration_normalized, dx=1.0, initial=0))

    # normalize velocity values and calculate positions
    values_velocity_normalized = values_velocity - values_velocity.rolling(10).mean()
    values_velocity_normalized.dropna(axis=0, inplace=True)
    position = integrate.cumtrapz(values_velocity_normalized, dx=1.0, initial=0)

    # find the local maxima on the position curve to approximate repetitions
    local_maxima = calculate_local_maxima(position, horizontal_dist, vertical_dist_lower_bound, vertical_dist_upper_bound)

    return len(local_maxima)

def calculate_local_maxima(signal, horizontal_dist, vertical_dist_lower_bound, vertical_dist_upper_bound):
    """
    Searches for local maxima in a given signal
    Parameters
    ----------
    signal: list(float)
        Contains the signal.
    horizontal_dist: int
        Minimum number of timesteps between to repetitions.
    vertical_dist_lower_bound: float
        Determines the minimum vertical distance between a repetition and the surrounding points in the signal.
    vertical_dist_upper_bound: float
        Determines the maximum vertical distance between a repetition and the surrounding points in the signal.
    Returns
    ----------
    list(int)
        Contains all the local maxima that passed the threshold filtering.
    """

    filtered_local_maxima, props = find_peaks(np.array(signal), distance=horizontal_dist, prominence=[vertical_dist_lower_bound, vertical_dist_upper_bound])
    return filtered_local_maxima