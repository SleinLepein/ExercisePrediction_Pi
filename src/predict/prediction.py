from src.model.model import Model
from src.preprocessing.preprocessing_helpers import build_window_samples, correct_columns, wrangle_data, read_raw_csv_data, read_raw_json_data
from src.predict.probability import predict_set

WINDOW_SIZE = 30
MODEL_EXERCISE_PATH = "model/model.pkl"
MIN_ROWS = 650
COLUMNS = ['Time Received', 'Time Measured', 'Client', 'Value', 'ValueTwo', 'ValueThree', 'Command', 'Exercise', 'SessionId', 'UserName', 'Info']

def main(df_path):
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
                exercise, probability, repetitions, delete_file = predict_set(list_of_predicted_batches, df)

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
            