import pyaudio
import wave
from scipy.io import wavfile
import scipy.io
import numpy as np

RATE = 16000 # this is the rATE of the ML model

def capture_audio(method = 2) -> np.array:
    # method = 1
    if method == 1:
        return capture_audio_with_pyaudio()
    if method ==2:
        return capture_audio_sound_device()
    
    raise Exception("not implemented")

def capture_audio_sound_device()-> np.array:
    import sounddevice as sd
    import numpy as np
    import scipy.io.wavfile as wav

    fs=RATE
    duration = 5  # seconds
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=2,dtype='float64')
    print ("Recording Audio")
    sd.wait()
    print ("Audio recording complete , Play Audio")
    return myrecording
    # sd.play(myrecording, fs)
    # sd.wait()
    # print ("Play Audio Complete")


def capture_audio_with_pyaudio() -> np.array:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    # RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "data/output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    samplerate, data = wavfile.read(WAVE_OUTPUT_FILENAME)
    return data