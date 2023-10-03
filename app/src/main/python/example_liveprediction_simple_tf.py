from keras.models import load_model
import numpy as np
from vggish_input import waveform_to_examples
import ubicoustics
from pathlib import Path

RATE = 16000
model = None
label = None
context = None

def initialize():
    global model, label, context

    model_filename = "models/example_model.hdf5"

    model = load_model(model_filename)
    context = ubicoustics.everything

    label = dict()
    for k in range(len(context)):
        label[k] = context[k]

def process_audio(audio_data):
    global model, label

    np_wav = np.frombuffer(audio_data, dtype=np.int16) / 32768.0  # Convert to [-1.0, +1.0]
    x = waveform_to_examples(np_wav, RATE)
    predictions = []

    if x.shape[0] != 0:
        x = x.reshape(len(x), 96, 64, 1)
        pred = model.predict(x)
        predictions.append(pred)

    for prediction in predictions:
        m = np.argmax(prediction[0])
        if m < len(label):
            return ("%s (%0.2f)" % (ubicoustics.to_human_labels[label[m]], prediction[0,m]))
        
    return "Prediction Failed"

