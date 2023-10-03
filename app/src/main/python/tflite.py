import numpy as np
from vggish_input import waveform_to_examples
import ubicoustics
from pathlib import Path
import tensorflow as tf
import os

RATE = 16000
interpreter = None
input_details = None
output_details = None
label = None
context = None

def initialize(model_path):
    global interpreter, input_details, output_details, label, context
    # Set model path to the extracted asset location
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    context = ubicoustics.everything
    label = dict()
    for k in range(len(context)):
        label[k] = context[k]

def process_audio(audio_data):
    global interpreter, input_details, output_details, label

    np_wav = np.frombuffer(audio_data, dtype=np.int16) / 32768.0  # Convert to [-1.0, +1.0]
    x = waveform_to_examples(np_wav, RATE)

    if x.shape[0] != 0:
        x = x.reshape(len(x), 96, 64, 1)
        
        # Set the tensor (i.e., input) data
        interpreter.set_tensor(input_details[0]['index'], x)

        # Invoke the interpreter
        interpreter.invoke()

        # Get the prediction result
        pred = interpreter.get_tensor(output_details[0]['index'])

        m = np.argmax(pred[0])
        if m < len(label):
            return ("%s (%0.2f)" % (ubicoustics.to_human_labels[label[m]], pred[0,m]))
        
    return "Prediction Failed"
