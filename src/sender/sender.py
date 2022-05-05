import paho.mqtt.client as paho

topic = "CloudPredict"
broker = "192.168.0.100"
port = 1883


def on_publish(client, userdata, result):
    pass


client = paho.Client("Prediction")
client.on_publish = on_publish
client.connect(broker, port)


def send_to_app(msg):
    """
    sends the prediction to the app
    ----------
    msg : String
        contains the message that will be send to the app
    """
    client.publish(topic, msg, 1, False)
