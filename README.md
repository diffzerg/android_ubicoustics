# android_ubicoustics

Overview

<img width="217" alt="Screenshot 2023-10-11 at 7 59 27 PM" src="https://github.com/diffzerg/android_ubicoustics/assets/50289876/6cfecc45-725d-4898-a898-c45d9f5a9b7f">

This is a android version application of FIGLAB/ubicoustics : https://github.com/FIGLAB/ubicoustics.
This application uses simple live prediction code in ubicoustics, and it has usally 300~350ms latency in Tested Android Device.

Tested Device : SM-M205 (Samsung Galaxy M20, 3GB RAM, 2019)
Tested OS : Android 10

This project integrates Python ubicoustics code directly into the Android app using the Chaquopy library. Due to incompatibility issues with the pyaudio library on Android, the microphone input code was manually implemented.

Additionally, the original model was quantized using TensorFlow Lite to run in a Android mobile device. Specifically, float16 quantization was applied to reduce its size and computational requirements. As the model size is over the limit of GitHub file capacity, it should be downloaded manually in dropbox.

Installation Steps

1. Download the Quantized Tensorflow Lite Model

download link : https://www.dropbox.com/scl/fi/cff5dwcdm5plc3lft2gvg/example_model.tflite?rlkey=6xp4i8sqs8td4ooe609gcld2g&dl=0

2. Add Model to Android Project

After downloading, please place the .tflite model in the following directory of your project: app/src/main/assets/
