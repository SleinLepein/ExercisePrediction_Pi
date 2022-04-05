import numpy as np
import json
from sklearn import metrics
import logging
from templates.vae_model_template.forward_net_vae_model_class import ForwardNetVAE

def predict_anomaly(model, sensor_input, condition_input, threshold):
    """
     Takes the trained keras anomaly detection model and reconstructs the given input. The reconstruction error is then used to 
     determine if a winow contains an anomaly or not.

     Parameters
     ----------
    model: KerasModel
        The trained variational autoencoder model.
    sensor_input: np.array
        The input data from the different sensors in the window.
    condition_input: np.array
        The conditions for the input. Constructed using the exercise labels.
    threshold: float
        Manually chosen threshold. Determines at which reconstruction loss value to draw the line.

    Return
    --------
    list(int)
        List of anomaly prediction labels.

    """
    anomaly_labels = []
    # iterate over each sample in the batch
    for i in range(sensor_input.shape[0]):
        sens = np.expand_dims(sensor_input[i], axis=0)
        cond = np.expand_dims(condition_input[i], axis=0)
        prediction = model.mean_model.predict([sens, cond])
        mse_loss = metrics.mean_squared_error(np.squeeze(sens), np.squeeze(prediction))
        #logging.info(f"MSE Loss:   {mse_loss}")
        if mse_loss > threshold:
            anomaly_labels.append(1)
        else:
            anomaly_labels.append(0)

    return anomaly_labels


def load_anomaly_model(model_anomaly_save_folder, model_anomaly_name, model_anomaly_config_path):
    """
    Load a given keras model for anomaly detection.

    Parameters
    -----------
    model_anomaly_save_folder: str
        Path to the anomaly model save folder.
    model_anomaly_name: str
        Filename of the concrete model checkpoint to use.
    model_anomaly_config_name: str
        Path to the json file which describes the strucutre of the loaded keras model.

    Returns
    --------
    ForwardNetVAE

    """
    loaded_anomaly_config = {}
    with open(model_anomaly_config_path, 'r') as config_file:
        loaded_anomaly_config = json.load(config_file)

    loaded_anomaly_config["batch_size"] = 1
    loaded_vae = ForwardNetVAE(**loaded_anomaly_config)
    loaded_vae.load(folder_path=model_anomaly_save_folder, model_name=model_anomaly_name)

    return loaded_vae