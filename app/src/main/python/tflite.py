import numpy as np
from vggish_input import waveform_to_examples
import ubicoustics
from pathlib import Path
import tensorflow as tf
import os
import logging

logging.basicConfig(level=logging.DEBUG)

RATE = 16000
interpreter = None
input_details = None
output_details = None
label = None
context = None
accumulated_audio_data = np.array([])  # Added to accumulate audio data

def initialize(model_path):
    global interpreter, input_details, output_details, label, context
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    context = ubicoustics.everything
    label = dict()
    for k in range(len(context)):
        label[k] = context[k]

def process_audio(audio_data):
    global interpreter, input_details, output_details, label, accumulated_audio_data

    np_wav = np.frombuffer(audio_data, dtype=np.int16) / 32768.0  # Convert to [-1.0, +1.0]
    
    accumulated_audio_data = np.concatenate((accumulated_audio_data, np_wav))

    if len(accumulated_audio_data) >= RATE:  # If we have 1 second of audio
        x = waveform_to_examples(accumulated_audio_data, RATE)

        # Keep only the remaining audio data after extracting 1-second frames
        accumulated_audio_data = accumulated_audio_data[len(x)*RATE//x.shape[0]:]

        if x.shape[0] != 0:
            x = x.reshape(len(x), 96, 64, 1).astype(np.float32)

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
