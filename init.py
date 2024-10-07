import whisper
import pyaudio
import wave
import numpy as np
import json
import os
import time
import librosa
import soundfile as sf
import noisereduce as nr
 
# Configurações de áudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 100  # Valor de amplitude que define silêncio
SILENCE_DURATION = 2.5  # Segundos de silêncio contínuo para parar a gravação
DELAY_TO_START_SILENCE_CHECK = 2  # Delay antes de verificar silêncio
 

audio = pyaudio.PyAudio()
 
def detect_silence(data):
    amplitude = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return np.abs(amplitude).mean()
 
def gravar_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("Gravando... Fale algo!")
    
    frames = []
    start_time = time.time()
    silence_start = None
    is_silence_checking = False
 
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        amplitude = detect_silence(data)
        elapsed_time = time.time() - start_time
 
        if elapsed_time > DELAY_TO_START_SILENCE_CHECK:
            is_silence_checking = True
 
        if is_silence_checking:
            if amplitude < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_DURATION:
                    break
            else:
                silence_start = None
 
    stream.stop_stream()
    stream.close()
 
    arquivo_audio = "audio_temp.wav"
    wf = wave.open(arquivo_audio, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
 
    print(f"Áudio salvo em {arquivo_audio}")
    return arquivo_audio
 
def preprocess_audio(audio_file):
    audio, sr = librosa.load(audio_file, sr=None)
 
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
 
    novo_arquivo_audio = 'novo_audio_temp.wav'
    sf.write(novo_arquivo_audio, reduced_noise, sr)
 
    return novo_arquivo_audio
 
def transcrever_audio(audio_file, caminho_modelo):
    audio_processado = preprocess_audio(audio_file)
 
    print(f"Carregando o modelo Whisper a partir de: {caminho_modelo}")
    model = whisper.load_model(caminho_modelo)
    
    print("Transcrevendo o áudio em português...")
    resultado = model.transcribe(audio_processado, language='pt')
    
    if os.path.exists(audio_file):
        os.remove(audio_file)
    if os.path.exists(audio_processado):
        os.remove(audio_processado)
 
    return resultado['text']
 
if __name__ == "__main__":
    arquivo_audio = gravar_audio()
    caminho_modelo = "whisper-small/small.pt"
    texto_transcrito = transcrever_audio(arquivo_audio, caminho_modelo)
    print("Transcrição:", texto_transcrito)