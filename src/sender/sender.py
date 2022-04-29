import paho.mqtt.client as paho
topic = "CloudPredict"
broker = "192.168.0.100"
port = 1883

def on_publish(client,userdata,result):
    print("Prediction send \n")
    pass

client= paho.Client("Prediction")
client.on_publish = on_publish
client.connect(broker,port) 


def send_to_app(msg):
    client.publish(topic,msg)
    print("Published in "+topic)
    print(msg)