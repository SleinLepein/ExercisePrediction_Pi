import numpy as np

from shared_code.model import Model
from shared_code.preprocessing_helpers import build_window_samples, read_raw_data, correct_columns, wrangle_data, construct_input_conditions, construct_input_sensor_data, wrangle_data
from shared_code.prediction_helpers import predict_anomaly, load_anomaly_model
from shared_code.probability import predict_set
from shared_code.anomaly_preprocessing import normalize_df

WINDOW_SIZE = 30
MODEL_EXERCISE_PATH = "data/model_prob_14_03.pkl"
MIN_ROWS = 650

# anomaly detection params
MODEL_ANOMALY_SAVE_FOLDER = "data/"
MODEL_ANOMALY_NAME = "model_latest.ckpt"
MODEL_ANOMALY_CONFIG_PATH = "data/anomaly_model_config.json"
ANOMALY_THRESHOLD = 0.023104940428652 # currently only an educated guess.
ONE_HOT_ENCODER_PATH = "data/one_hot_pipeline.joblib"

COLUMNS = ['Time Received', 'Time Measured', 'Client', 'Value', 'ValueTwo', 'ValueThree', 'Command', 'Exercise', 'SessionId', 'UserName', 'Info']

def main():
    try:
        df_path = "raw_data/2022_03_03_Ausfallschritte_falsch.csv"
        df = read_raw_data(df_path)
    except Exception as ex:
        print(f"Error while reading the data: ({ex})")
        #msg.set(f"Error while reading the file: ({ex})")
    else:
        # Approximately 300 Rows of Raw Data is one Batch, for a Prediction one Batch is enough but since we want to predict sets we need atleast 2 Batches (~600 Rows)
        if df.shape[0] >= MIN_ROWS:
            # read model
            model = Model().read(path = MODEL_EXERCISE_PATH)

            try:
                # delete unwanted meta informaton and wrangle the data
                df = correct_columns(df, COLUMNS)
                df = wrangle_data(df)

                # create a second normalized df for keras model
                df_anomaly = df.copy()
                df_anomaly = normalize_df(df_anomaly)

                # build samples with right window size, remove samples with size smaller than window size, reset index after row removing
                df_prediction_samples = build_window_samples(df, WINDOW_SIZE)
                df_prediction_samples_normalized = build_window_samples(df_anomaly, WINDOW_SIZE)

                # drop ultrasonic sensors in the input of the anomaly detection model
                relevant_columns = [col for col in df_prediction_samples_normalized.columns if 'ultra' not in col]
                df_prediction_samples_normalized = df_prediction_samples_normalized[[col for col in df_prediction_samples_normalized if col in relevant_columns]].copy()

                # make prediction
                list_of_predicted_batches = model.predict(df_prediction_samples)
                print(f"\nPrediction:\n{list_of_predicted_batches} [{len(list_of_predicted_batches)}]\n")

                # drop rows that are tagged as nothing.
                nothing_indices = []
                not_nothing_tagged = []
                for i,x in enumerate(list_of_predicted_batches):
                    if x[0] == 'nothing':
                        nothing_indices.append(i)
                    else:
                        not_nothing_tagged.append(x[0])
                df_prediction_samples_normalized.drop(df_prediction_samples_normalized.index[nothing_indices], inplace=True)

                # get exercise, probability, repetitions and bool if data should be deleted afterwards (DATA IS CURRENTLY NEVER DELETED, IMPLEMENTED LATER)
                exercise, probability, repetitions, delete_file = predict_set(list_of_predicted_batches, df)

                # start and end time of the current dataframe
                exercise_start_time = str(df.loc[df.index[0], "Time Measured"])
                exercise_end_time = str(df.loc[df.index[-1], "Time Measured"])

                if len(not_nothing_tagged) != 0:
                    # construct anomaly detection input arrays. Only use inputs that are not labled as nothing.
                    input_sensor_data = construct_input_sensor_data(df_prediction_samples_normalized)
                    input_conditions = construct_input_conditions(not_nothing_tagged, ONE_HOT_ENCODER_PATH, WINDOW_SIZE) 
                    print(f"input_sensor_data_shape: {input_sensor_data.shape}") 
                    print(f"input_conditions_shape: {input_conditions.shape}")                 
                    
                    # load keras vae model from checkpoint and use it to classify windows
                    anomaly_model = load_anomaly_model(MODEL_ANOMALY_SAVE_FOLDER, MODEL_ANOMALY_NAME, MODEL_ANOMALY_CONFIG_PATH)
                    pred_anomalies = predict_anomaly(anomaly_model, input_sensor_data, input_conditions, ANOMALY_THRESHOLD)
                    print(f"pred_anomalies: {pred_anomalies}") 

                    # calculate precentage of anomalous time series windows
                    number_of_anomalies = pred_anomalies.count(1)
                    percentage_of_anomalies = abs((number_of_anomalies/len(pred_anomalies)-1))
                else:
                    percentage_of_anomalies = np.NaN

                # send message to namespace queue
                message = f"\nStart Time: {exercise_start_time[11:19]}\nEnd Time: {exercise_end_time[11:19]}\nExercise: {exercise}\
                    \nRepetitions: {repetitions}\nProbability: {probability:.2f}\nAnomalous: {percentage_of_anomalies}"
                print(message)
                #msg.set(message)
            except Exception as ex:
                print(f"An error occured:\n({ex})\n")
                #msg.set(f"An error occured: ({ex})")
        else:
            print(f"Not enough data to make a Prediction [{df.shape[0]}/{MIN_ROWS} rows]")
            #msg.set(f"Not enough data to make a Prediction [{df.shape[0]}/{MIN_ROWS} rows]")

if __name__ == "__main__":
    main()
