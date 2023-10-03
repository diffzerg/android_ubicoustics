package com.example.ubicoustics

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    private var recordingThread: Thread? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!Python.isStarted()) {
            Log.d("Python Initialization", "Starting Python...")
            Python.start(AndroidPlatform(this))
        } else {
            Log.d("Python Initialization", "Python already started.")
        }

        val sampleRate = 16000
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT

        val assetManager = this.assets
        val modelInputStream = assetManager.open("example_model.tflite")
        val modelFile = File(this.filesDir, "example_model.tflite")
        
        modelFile.outputStream().use { it.write(modelInputStream.readBytes()) }

        val py = Python.getInstance()
        val pyObj = py.getModule("tflite")  // Replace with your module name
        pyObj.callAttr("initialize", modelFile.absolutePath)

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Log.d("Permission Check", "Requesting RECORD_AUDIO permission.")
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 1)
            return
        } else {
            Log.d("Permission Check", "RECORD_AUDIO permission granted.")
        }

        val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

        val audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, channelConfig, audioFormat, bufferSize)

        Log.d("AudioRecord", "State: ${audioRecord.state}")

        audioRecord.startRecording()

        recordingThread = Thread {
            Log.d("RecordingThread", "Recording thread started")
            readAudioData(audioRecord, bufferSize)
            Log.d("RecordingThread", "Recording thread finished")
        }

        recordingThread?.start()
    }

    private fun readAudioData(audioRecord: AudioRecord, bufferSize: Int) {
        val audioData = ByteArray(bufferSize)
        try {
            while (audioRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING && !Thread.interrupted()) {
                val bytesRead = audioRecord.read(audioData, 0, bufferSize)
                Log.d("RecordingThread", "Bytes read: $bytesRead")
                if (bytesRead > 0) {
                    val py = Python.getInstance()
                    val pyObj = py.getModule("tflite")
                    val resultString = pyObj.callAttr("process_audio", audioData)
                    Log.d("Prediction result", resultString.toString())
                }
            }
        } catch (e: InterruptedException) {
            Log.e("RecordingThread", "Recording thread was interrupted", e)
        } catch (e: Exception) {
            Log.e("RecordingThread", "Error in recording thread", e)
        }
    }
}
